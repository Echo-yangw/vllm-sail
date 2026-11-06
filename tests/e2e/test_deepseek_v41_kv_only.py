# SPDX-License-Identifier: Apache-2.0
"""KV-only replay writes must match the cache written by ordinary attention."""

import pytest

pytestmark = pytest.mark.ppu


@pytest.mark.parametrize("mode", ["packed", "mxfp8", "bf16", "fp8"])
@pytest.mark.parametrize("tokens", [0, 1, 17, 129])
def test_kv_only_insert_matches_attention_cache(mode, tokens):
    import torch

    from vllm_sail.native import install
    from vllm_sail.native.extensions import import_kernels

    import_kernels(strict=True)
    install()
    generator = torch.Generator(device="cuda").manual_seed(41)
    kv = torch.randn(
        tokens, 512, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    original = kv.clone()
    q = torch.zeros(tokens, 16, 512, dtype=torch.bfloat16, device="cuda")
    positions = torch.arange(tokens, dtype=torch.int64, device="cuda") * 3
    frequency = 10000.0 ** (-torch.arange(0, 64, 2, device="cuda").float() / 64)
    angles = torch.arange(4096, device="cuda").float()[:, None] * frequency
    rope = torch.cat((angles.cos(), angles.sin()), dim=-1)
    slots = torch.arange(tokens, dtype=torch.int64, device="cuda").flip(0) + 7
    slots[2::5] = -1
    if tokens > 1:
        slots = slots[:-1]
    block_size = 128
    blocks = (tokens + 7 + block_size - 1) // block_size + 1
    packed = mode in ("packed", "mxfp8")
    width = (528 if mode == "mxfp8" else 584) if packed else 512
    dtype = (
        torch.uint8
        if packed
        else (torch.bfloat16 if mode == "bf16" else torch.float8_e4m3fn)
    )
    cache = torch.zeros(blocks, block_size, width, dtype=dtype, device="cuda")
    expected = cache.clone()
    scale = torch.tensor([0.25], device="cuda") if mode == "fp8" else None
    torch.ops._C.fused_deepseek_v4_kv_rope_insert(
        kv, cache, slots, positions, rope, block_size, scale, mode == "mxfp8"
    )
    if tokens == 0:
        # The older attention entry points cannot launch an empty grid.
        torch.testing.assert_close(cache.view(torch.uint8), expected.view(torch.uint8))
        torch.testing.assert_close(kv, original, rtol=0, atol=0)
        return
    if packed:
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            q,
            kv,
            expected.view(blocks, -1),
            slots,
            positions,
            rope,
            16,
            1e-6,
            block_size,
            False,
            mode == "mxfp8",
        )
    elif mode == "bf16":
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert(
            q, kv, expected, slots, positions, rope, 1e-6, block_size, False
        )
    else:
        q_out = torch.empty_like(q, dtype=dtype)
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert(
            q,
            kv,
            q_out,
            expected,
            slots,
            positions,
            rope,
            scale,
            torch.ones_like(scale),
            1e-6,
            block_size,
            False,
        )
    torch.cuda.synchronize()
    torch.testing.assert_close(cache.view(torch.uint8), expected.view(torch.uint8))
    torch.testing.assert_close(kv, original, rtol=0, atol=0)
