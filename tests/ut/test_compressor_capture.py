# SPDX-License-Identifier: Apache-2.0
"""Use the actual capture state for DeepSeek V4 compression decisions."""

from __future__ import annotations

import ast
from types import MethodType
from types import SimpleNamespace as NS
from typing import cast

import pytest

from tests.support.source import ROOT, definition, function

PATCH = "vllm_sail/patch/enhancement/models/deepseek_v4_compressor.py"
UPSTREAM = "vllm/models/deepseek_v4/compressor.py"
FORWARD_FIXTURE = "tests/fixtures/deepseek_v4_compressor_forward.py"
CASES = [
    (False, False, "FULL", False),
    (False, False, "NONE", False),
    (True, False, "FULL", True),
    (True, False, "NONE", True),
    (False, True, "FULL", True),
    (True, True, "FULL", True),
    (False, None, "FULL", True),
    (True, None, "NONE", True),
]


def run_forward(path, name, modules, capturing, boundary, mode, expected):
    saved, compressed = [], []
    kv, score, state = object(), object(), NS(shape=(4, 256, 1024))
    slots = NS(shape=(2,))
    metadata = NS(
        token_to_req_indices=object(),
        slot_mapping=slots,
        block_table=object(),
        block_size=256,
        c128_boundary=boundary,
    )
    cached = NS(kv_cache=NS(dtype="uint8"))
    context = NS(
        attn_metadata={"state": metadata, "key": object()},
        cudagraph_runtime_mode=mode,
    )
    platform = NS(
        is_cuda=lambda: True,
        is_ppu=lambda: True,
        is_rocm=lambda: False,
        is_xpu=lambda: False,
    )
    namespace = {
        "__package__": "vllm.models.deepseek_v4",
        "torch": NS(
            cuda=NS(is_current_stream_capturing=lambda: capturing),
            uint8="uint8",
            float8_e4m3fn="fp8",
        ),
        "current_platform": platform,
        "get_forward_context": lambda: context,
        "cast": cast,
        "CompressorMetadata": object,
        "Any": object,
        "CUDAGraphMode": NS(FULL="FULL"),
        "_SAVE_PARTIAL_STATES_KERNEL": lambda **kwargs: saved.append(kwargs),
        "compress_norm_rope_store_triton": lambda **kwargs: compressed.append(kwargs),
    }
    # The upstream CUDA implementation uses CuTeDSL; the PPU copy must use
    # Triton. Both launch recorders let the real forward policy execute.
    modules(
        "vllm.models.deepseek_v4.nvidia.ops.sparse_attn_compress_cutedsl",
        _SPARSE_ATTN_COMPRESSOR_CUTEDSL_KERNEL=lambda **kwargs: compressed.append(
            kwargs
        ),
    )
    layer = NS(
        coff=1,
        head_dim=512,
        compress_ratio=128,
        state_cache=NS(prefix="state", kv_cache=state),
        ape=object(),
        k_cache_prefix="key",
        _static_forward_context={"key": cached},
        _use_two_stage_fused_compressor=False,
        rope_head_dim=64,
        overlap=False,
        use_fp4_cache=False,
        norm=NS(weight=object()),
        rms_norm_eps=1e-6,
        _quant_block=object(),
        _token_stride=object(),
        _scale_dim=object(),
    )
    layer._compress_norm_rope_store = MethodType(
        function(PATCH, "_compress_norm_rope_store", namespace), layer
    )
    positions, rope = object(), NS(cos_sin_cache=object())

    function(path, name, namespace)(
        layer,
        NS(split=lambda widths, dim: (kv, score)),
        positions,
        rope,
    )

    assert len(saved) == 1
    assert saved[0]["kv"] is kv and saved[0]["score"] is score
    assert saved[0]["state_cache"] is state
    assert saved[0]["pdl_kwargs"] == {"launch_pdl": False}
    assert len(compressed) == int(expected)
    if expected:
        assert compressed[0]["state_cache"] is state
        assert compressed[0]["slot_mapping"] is slots
        assert compressed[0]["positions"] is positions
        assert compressed[0]["kv_cache"] is cached.kv_cache
        assert compressed[0]["block_size"] == 256


@pytest.mark.parametrize("capturing,boundary,mode,expected", CASES)
def test_ppu_compressor_uses_capture_state(
    modules, capturing, boundary, mode, expected
):
    run_forward(
        FORWARD_FIXTURE, "forward", modules, capturing, boundary, mode, expected
    )


@pytest.mark.upstream_source
@pytest.mark.parametrize("capturing,boundary,mode,expected", CASES)
def test_upstream_compressor_uses_the_same_capture_policy(
    modules, upstream_source_root, capturing, boundary, mode, expected
):
    run_forward(
        upstream_source_root / UPSTREAM,
        "DeepseekCompressor.forward",
        modules,
        capturing,
        boundary,
        mode,
        expected,
    )


@pytest.mark.upstream_source
def test_cpu_forward_fixture_matches_the_actual_provider(upstream_source_root):
    donor = definition(ROOT / FORWARD_FIXTURE, "forward")
    actual = definition(upstream_source_root / UPSTREAM, "DeepseekCompressor.forward")
    assert ast.dump(donor) == ast.dump(actual)
