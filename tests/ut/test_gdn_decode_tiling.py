# SPDX-License-Identifier: Apache-2.0
"""Keep CUDA-specific tuning separate from PPU packed decode launches."""

from __future__ import annotations

import math
from types import SimpleNamespace as NS

import pytest

from tests.support.source import function

NAME = "fused_recurrent_gated_delta_rule_packed_decode"
PATCH = "vllm_sail/patch/enhancement/attention/gdn.py"
UPSTREAM = "vllm/third_party/flash_linear_attention/ops/fused_recurrent.py"
CASES = [
    ("ppu", 80, 16, "fp32", 128, 32),
    ("ppu", 89, 17, "fp32", 128, 32),
    ("cuda", 90, 16, "fp32", 128, 4),
    ("cuda", 90, 17, "fp32", 128, 4),
    ("cuda", 90, 16, "bf16", 128, 32),
    ("cuda", 90, 17, "bf16", 128, 32),
    ("cuda", 90, 0, "fp32", 128, 4),
    ("ppu", 89, 0, "fp32", 128, 32),
    ("cuda", 100, 16, "fp32", 128, 4),
    ("cuda", 100, 17, "fp32", 128, 8),
    ("cuda", 100, 16, "bf16", 128, 4),
    ("cuda", 100, 17, "bf16", 128, 32),
    ("cuda", 100, 16, "fp32", 64, 32),
    ("rocm", 0, 16, "fp32", 128, 32),
    ("ppu", 89, 16384, "fp32", 128, 32),
]


class Tensor:
    def __init__(self, shape, dtype="bf16"):
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = NS(type="cuda", index=3)

    def stride(self, dimension):
        if dimension < 0:
            dimension += self.ndim
        return math.prod(self.shape[dimension + 1 :])

    def numel(self):
        return math.prod(self.shape)

    def is_contiguous(self):
        return True


def run_decode(path, modules, device, capability, batch, dtype, width, expected):
    launches, probes = [], []

    class Kernel:
        def __getitem__(self, grid):
            return lambda **kwargs: launches.append((grid, kwargs))

    def family(value, *, device_id):
        probes.append((value, device_id))
        return capability // 10 == value // 10

    def exact(value, *, device_id):
        probes.append((value, device_id))
        return capability == value

    modules(
        "vllm_sail.attention.pla_decode",
        get_sail_cuda_pla_k_last_packed=lambda: None,
    )
    platform = NS(
        is_cuda=lambda: device in ("cuda", "ppu"),
        is_ppu=lambda: device == "ppu",
        is_device_capability_family=family,
        is_device_capability=exact,
    )
    namespace = {
        "torch": NS(float32="fp32", int32="int32"),
        "triton": NS(
            next_power_of_2=lambda value: 1 << (value - 1).bit_length(),
            cdiv=lambda value, divisor: (value + divisor - 1) // divisor,
        ),
        "current_platform": platform,
        NAME + "_kernel": Kernel(),
    }
    heads, value_heads = 2, 4
    mixed = Tensor((batch, 2 * heads * width + value_heads * width))
    a, b = Tensor((batch, value_heads)), Tensor((batch, value_heads))
    log, bias = Tensor((value_heads,)), Tensor((value_heads,))
    state = Tensor((max(batch, 1), value_heads, width, width), dtype)
    out, indices = Tensor((batch, 1, value_heads, width)), Tensor((batch,), "int32")

    result = function(path, NAME, namespace)(
        mixed, a, b, log, bias, 0.25, state, out, indices, True
    )

    assert result[0] is out and result[1] is state
    assert len(launches) == 1
    grid, kwargs = launches[0]
    nv = math.ceil(width / expected)
    split = batch * value_heads > 65535
    assert grid == ((nv, value_heads, batch) if split else (nv, batch * value_heads))
    assert kwargs["BV"] == expected and kwargs["BK"] == width
    assert kwargs["h0"] is kwargs["ht"] is state
    assert kwargs["o"] is out and kwargs["ssm_state_indices"] is indices
    assert kwargs["scale"] == 0.25 and kwargs["USE_QK_L2NORM_IN_KERNEL"] is True
    assert kwargs["SPLIT_BATCH_HEAD_GRID"] is split
    assert kwargs["num_warps"] == 1 and kwargs["num_stages"] == 3
    assert all(device_id == 3 for _, device_id in probes)


@pytest.mark.parametrize("capability", [90, 100])
def test_ppu_vendor_keeps_its_fallback_tile_under_cuda_capability_aliases(
    modules, capability
):
    run_decode(PATCH, modules, "ppu", capability, 16, "fp32", 128, 32)


@pytest.mark.parametrize("device,capability,batch,dtype,width,expected", CASES)
def test_ppu_copy_preserves_upstream_decode_tiling(
    modules, device, capability, batch, dtype, width, expected
):
    run_decode(PATCH, modules, device, capability, batch, dtype, width, expected)


@pytest.mark.upstream_source
@pytest.mark.parametrize("device,capability,batch,dtype,width,expected", CASES)
def test_real_upstream_uses_the_same_launch_contract(
    modules, upstream_source_root, device, capability, batch, dtype, width, expected
):
    run_decode(
        upstream_source_root / UPSTREAM,
        modules,
        device,
        capability,
        batch,
        dtype,
        width,
        expected,
    )
