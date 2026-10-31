# SPDX-License-Identifier: Apache-2.0
"""Forward FLY options through the actual speculative sampler launch."""

import inspect
from types import SimpleNamespace as NS

import pytest

from tests.support.source import function
from tests.ut.test_spec_decode_dispatch import Kernel, Tensor

PATCH = "vllm_sail/patch/enhancement/runtime.py"
UPSTREAM = "vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py"


def check_fly(path, window, incompatible=None, upstream=None):
    entropy, sampled_slice = object(), object()
    entropy_calls, launches = [], []

    class Logits(Tensor):
        def __getitem__(self, index):
            assert index == (slice(None), slice(None, 8000))
            return sampled_slice

    def compute(values, top_k, **kwargs):
        entropy_calls.append((values, top_k, kwargs))
        return entropy

    def rejection(*args, **kwargs):
        launches.append((args, kwargs))
        if upstream is not None:
            kernel = function(upstream, "_rejection_kernel", {})
            assert kwargs["num_warps"] == 1
            # Triton consumes launch configuration before binding kernel args.
            arguments = {
                key: value for key, value in kwargs.items() if key != "num_warps"
            }
            bound = inspect.signature(kernel).bind(*args, **arguments).arguments
            assert bound["fly_entropy_ptr"] is (entropy if window else None)
            assert bound["fly_entropy_threshold"] == 0.42
            assert bound["FLY_WINDOW_SIZE"] == window

    namespace = {
        "torch": NS(**{key: key for key in ("int32", "int64", "float32", "float64")}),
        "triton": NS(
            cdiv=lambda a, b: (a + b - 1) // b,
            next_power_of_2=lambda n: 1 << (n - 1).bit_length(),
        ),
        "compute_fly_entropy": compute,
        "_compute_local_logits_stats_kernel": Kernel(),
        "_compute_local_residual_mass_kernel": Kernel(),
        "_compute_cumulative_log_p_kernel": Kernel(),
        "_rejection_kernel": Kernel(rejection),
        "_resample_kernel": Kernel(),
        "_insert_resampled_kernel": Kernel(),
    }
    sampler = function(path, "rejection_sample", namespace)
    args = (
        Logits(13, 8193),
        NS(values=Tensor(2, 7, 8000), row_mapping=None),
        Tensor(13),
        Tensor(3),
        Tensor(13),
        Tensor(2),
        Tensor(13),
        Tensor(13),
        Tensor(2),
        Tensor(2),
        7,
    )
    options = dict(
        fly_window_size=window, fly_entropy_threshold=0.42, fly_entropy_top_k=5
    )
    if incompatible == "block":
        options["use_block_verification"] = True
    elif incompatible == "synthetic":
        options["synthetic_conditional_rates"] = Tensor(7)
    if incompatible:
        with pytest.raises(AssertionError):
            sampler(*args, **options)
        assert not entropy_calls and not launches
        return
    sampled, count = sampler(*args, **options)
    assert sampled.shape == (2, 8) and count.shape == (2,)
    assert entropy_calls == (
        [(sampled_slice, 5, {"from_logits": True})] if window else []
    )
    assert len(launches) == 1
    arguments, keywords = launches[0]
    assert arguments[-3] is (entropy if window else None)
    assert arguments[-2:] == (0.42, 1)
    assert keywords["FLY_WINDOW_SIZE"] == window
    assert keywords["SYNTHETIC_MODE"] is False
    assert keywords["USE_BLOCK_VERIFICATION"] is False


@pytest.mark.parametrize(
    "window,incompatible", [(0, None), (3, None), (3, "block"), (3, "synthetic")]
)
def test_copied_sampler_forwards_fly_options(window, incompatible):
    check_fly(PATCH, window, incompatible)


@pytest.mark.upstream_source
@pytest.mark.parametrize("copied", [False, True])
@pytest.mark.parametrize(
    "window,incompatible", [(0, None), (3, None), (3, "block"), (3, "synthetic")]
)
def test_actual_rejection_kernel_receives_fly_contract(
    upstream_source_root, copied, window, incompatible
):
    upstream = upstream_source_root / UPSTREAM
    check_fly(PATCH if copied else upstream, window, incompatible, upstream)
