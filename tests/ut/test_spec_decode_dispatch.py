# SPDX-License-Identifier: Apache-2.0
"""Check copied sampler calls against the selected upstream kernel signature."""

from __future__ import annotations

import inspect
from types import SimpleNamespace as NS

import pytest

from tests.support.source import function


class Tensor:
    def __init__(self, *shape, dtype="float32", contiguous=True):
        self.shape, self.dtype = shape, dtype
        self.ndim = len(shape)
        self._contiguous = contiguous

    def stride(self, axis):
        return 1 if self._contiguous else 2

    def size(self, axis):
        return self.shape[axis]

    def new_empty(self, *shape, dtype):
        return Tensor(*shape, dtype=dtype)

    def is_contiguous(self):
        return self._contiguous

    def contiguous(self):
        return Tensor(*self.shape, dtype=self.dtype)

    def view(self, dtype):
        return Tensor(*self.shape, dtype=dtype)


class Kernel:
    def __init__(self, call=lambda *a, **kw: None):
        self.call = call

    def __getitem__(self, grid):
        return self.call


@pytest.mark.upstream_source
@pytest.mark.parametrize("mask_layout", [None, "contiguous", "strided"])
def test_resample_dispatch_binds_the_upstream_mask_argument(
    upstream_source_root, mask_layout
):
    upstream = function(
        upstream_source_root
        / "vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py",
        "_resample_kernel",
        {},
    )
    calls = []

    def resample(*args, **kwargs):
        calls.append(inspect.signature(upstream).bind(*args, **kwargs).arguments)

    namespace = dict(
        torch=NS(
            **{
                name: name
                for name in ("int32", "int64", "float32", "float64", "bool", "uint8")
            }
        ),
        triton=NS(
            cdiv=lambda a, b: (a + b - 1) // b,
            next_power_of_2=lambda n: 1 << (n - 1).bit_length(),
        ),
        _compute_local_logits_stats_kernel=Kernel(),
        _rejection_kernel=Kernel(),
        _resample_kernel=Kernel(resample),
        _insert_resampled_kernel=Kernel(),
    )
    sampler = function(
        "vllm_sail/patch/enhancement/runtime.py", "rejection_sample", namespace
    )
    masked = mask_layout is not None
    sampler(
        Tensor(2, 32),
        None,
        Tensor(2),
        Tensor(2),
        Tensor(2),
        Tensor(1),
        Tensor(2),
        Tensor(2),
        Tensor(1),
        Tensor(1),
        1,
        contexts=Tensor(2, 2, dtype="int32") if masked else None,
        watermarking=Tensor(1, dtype="bool") if masked else None,
        watermarking_skip_mask=Tensor(
            2, dtype="bool", contiguous=mask_layout != "strided"
        )
        if masked
        else None,
        watermark_key=123 if masked else None,
    )
    assert len(calls) == 1
    bound = calls[0]
    assert bound["DEDUPLICATE_CONTEXTS"] is masked
    pointer = bound["watermarking_skip_mask_ptr"]
    if masked:
        assert pointer.shape == (2,)
        assert pointer.dtype == "uint8" and pointer.is_contiguous()
    else:
        assert pointer is None


@pytest.mark.upstream_source
@pytest.mark.parametrize("shards", [None, 1, 4])
@pytest.mark.parametrize(
    "mode", ["ordinary", "block", "watermark", "fly", "synthetic", "no-draft"]
)
def test_all_sampler_kernel_calls_bind_the_upstream_draft_shard_contract(
    upstream_source_root, shards, mode
):
    upstream = (
        upstream_source_root
        / "vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py"
    )
    symbols = [
        "_compute_local_logits_stats_kernel",
        "_compute_cumulative_log_p_kernel",
        "_compute_local_residual_mass_kernel",
        "_rejection_kernel",
        "_resample_kernel",
        "_insert_resampled_kernel",
    ]
    calls = {}

    def bind(symbol):
        signature = inspect.signature(function(upstream, symbol, {}))

        def launch(*args, **kwargs):
            arguments = {
                key: value for key, value in kwargs.items() if key != "num_warps"
            }
            calls[symbol] = (signature.bind(*args, **arguments).arguments, kwargs)

        return Kernel(launch)

    class Logits(Tensor):
        def __getitem__(self, index):
            rows, columns = index
            assert rows == slice(None)
            assert columns.start is None and columns.step is None
            return Tensor(self.shape[0], columns.stop)

    namespace = {
        "torch": NS(
            **{
                name: name
                for name in ("int32", "int64", "float32", "float64", "bool", "uint8")
            }
        ),
        "triton": NS(
            cdiv=lambda a, b: (a + b - 1) // b,
            next_power_of_2=lambda n: 1 << (n - 1).bit_length(),
        ),
        "compute_fly_entropy": lambda data, top_k, from_logits: Tensor(data.shape[0]),
        **{symbol: bind(symbol) for symbol in symbols},
    }
    sampler = function(
        "vllm_sail/patch/enhancement/runtime.py", "rejection_sample", namespace
    )
    kwargs = {} if shards is None else {"draft_logits_num_shards": shards}
    if mode == "block":
        kwargs["use_block_verification"] = True
    elif mode == "watermark":
        kwargs.update(
            contexts=Tensor(13, 2, dtype="int32"),
            watermarking=Tensor(32, dtype="bool"),
            watermarking_skip_mask=Tensor(13, dtype="bool", contiguous=False),
            watermark_key=0x123456789ABCDEF,
        )
    elif mode == "fly":
        kwargs["fly_window_size"] = 2
    elif mode == "synthetic":
        kwargs["synthetic_conditional_rates"] = Tensor(7)
    sample, count = sampler(
        Logits(13, 129),
        None if mode == "no-draft" else Tensor(8, 7, 129),
        Tensor(13),
        Tensor(4),
        Tensor(13),
        Tensor(3),
        Tensor(13),
        Tensor(13),
        Tensor(32),
        Tensor(32),
        7,
        **kwargs,
    )
    expected = set(symbols) if mode == "block" else set(symbols) - set(symbols[1:3])
    assert set(calls) == expected
    for symbol, (bound, _) in calls.items():
        if symbol != "_insert_resampled_kernel":
            assert bound["DRAFT_LOGITS_NUM_SHARDS"] == (shards or 1)
    assert calls["_insert_resampled_kernel"][1]["num_warps"] == 1
    assert sample.shape == (3, 8) and count.shape == (3,)
