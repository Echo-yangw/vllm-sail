# SPDX-License-Identifier: Apache-2.0
"""Speculative steps and total logits rows are different launch dimensions."""

import inspect
from types import SimpleNamespace as NS

import pytest

from tests.support.source import function
from tests.support.spec_decode import Kernel, Tensor

PATCH = "vllm_sail/patch/_upstream/runtime.py"
UPSTREAM = "vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py"


def check_launches(body, block, draft, fp64, upstream=None):
    calls = {name: [] for name in ("stats", "residual", "resample")}

    def capture(name):
        def record(*args, **kwargs):
            calls[name].append((args, kwargs))
            if upstream is not None and name != "resample":
                symbol = (
                    "_compute_local_logits_stats_kernel"
                    if name == "stats"
                    else "_compute_local_residual_mass_kernel"
                )
                kernel = function(upstream, symbol, {})
                assert (
                    inspect.signature(kernel)
                    .bind(*args, **kwargs)
                    .arguments["num_logits"]
                    == 13
                )

        return record

    namespace = {
        "torch": NS(
            **{name: name for name in ("int32", "int64", "float32", "float64")}
        ),
        "triton": NS(
            cdiv=lambda a, b: (a + b - 1) // b,
            next_power_of_2=lambda n: 1 << (n - 1).bit_length(),
        ),
        "_compute_local_logits_stats_kernel": Kernel(capture("stats")),
        "_compute_local_residual_mass_kernel": Kernel(capture("residual")),
        "_compute_cumulative_log_p_kernel": Kernel(),
        "_rejection_kernel": Kernel(),
        "_resample_kernel": Kernel(capture("resample")),
        "_insert_resampled_kernel": Kernel(),
    }
    sampler = function(body, "rejection_sample", namespace)
    sampled, count = sampler(
        Tensor(13, 8193),
        NS(values=Tensor(2, 7, 8193), row_mapping=None) if draft else None,
        Tensor(13),
        Tensor(3),
        Tensor(13),
        Tensor(2),
        Tensor(13),
        Tensor(13),
        Tensor(2),
        Tensor(2),
        7,
        use_block_verification=block,
        use_fp64=fp64,
    )
    assert len(calls["stats"]) == 1
    assert calls["stats"][0][0][-1] == 13
    assert calls["stats"][0][1]["HAS_DRAFT_LOGITS"] is draft
    assert len(calls["residual"]) == int(block and draft)
    if calls["residual"]:
        assert calls["residual"][0][0][-2:] == (13, 2)
    assert sampled.shape == (2, 8) and count.shape == (2,)
    assert calls["resample"][0][1]["USE_FP64"] is fp64
    assert calls["resample"][0][1]["USE_BLOCK_VERIFICATION"] is block


@pytest.mark.parametrize("block", [False, True])
@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("fp64", [False, True])
def test_copied_sampler_passes_total_rows_to_both_statistics_kernels(
    block, draft, fp64
):
    check_launches(PATCH, block, draft, fp64)


@pytest.mark.upstream_source
@pytest.mark.parametrize("copied", [False, True])
@pytest.mark.parametrize("block", [False, True])
@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("fp64", [False, True])
def test_real_kernel_signatures_receive_total_logits_rows(
    upstream_source_root, copied, block, draft, fp64
):
    upstream = upstream_source_root / UPSTREAM
    check_launches(PATCH if copied else upstream, block, draft, fp64, upstream)
