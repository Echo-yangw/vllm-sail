# SPDX-License-Identifier: Apache-2.0
"""Device oracles for the native ops used by DeepSeek V4 Flash.

Run against an installed native wheel with ``pytest`` (or an editable native
build). Missing ops must fail these tests, never silently skip or use a fallback.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.ppu


@pytest.fixture(scope="module")
def runtime():
    import torch

    from vllm_sail.native import install
    from vllm_sail.native.extensions import import_kernels

    import_kernels(strict=True)
    install()
    return torch


@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize(
    "shape", [(1, 256, 6), (17, 256, 6), (33, 384, 6), (17, 256, 8)]
)
@pytest.mark.parametrize("routing", ["bias", "hash32", "hash64"])
@pytest.mark.parametrize("renormalize", [False, True])
def test_topk_softplus_sqrt(runtime, dtype_name, shape, routing, renormalize):
    torch = runtime
    tokens, experts, topk = shape
    generator = torch.Generator(device="cuda").manual_seed(7)
    logits = torch.randn(
        (tokens, experts),
        generator=generator,
        device="cuda",
        dtype=getattr(torch, dtype_name),
    )
    padding = torch.arange(tokens, device="cuda") % 4 == 3
    logits[padding] = float("nan")
    bias = input_ids = table = None
    index_dtype = torch.int64 if routing == "hash64" else torch.int32
    if routing == "bias":
        bias = torch.randn(experts, device="cuda", generator=generator)
    else:
        # Distinct experts per vocabulary row, with nontrivial token indirection.
        table = torch.stack(
            [
                torch.randperm(experts, device="cuda", generator=generator)[:topk]
                for _ in range(19)
            ]
        ).to(index_dtype)
        input_ids = (torch.arange(tokens, device="cuda") * 7 % 19).to(index_dtype)
    weights = torch.empty(tokens, topk, device="cuda", dtype=torch.float32)
    indices = torch.empty(tokens, topk, device="cuda", dtype=index_dtype)
    source_rows = torch.empty(tokens, topk, device="cuda", dtype=torch.int32)

    torch.ops._moe_C.topk_softplus_sqrt(
        weights,
        indices,
        source_rows,
        logits,
        renormalize,
        1.5,
        bias,
        input_ids,
        table,
        padding,
    )
    torch.cuda.synchronize()

    scores = torch.nn.functional.softplus(logits.float()).sqrt()
    expected_ids = (
        table[input_ids.long()].long()
        if table is not None
        else (scores + bias).topk(topk, dim=-1).indices
    )
    expected = scores.gather(1, expected_ids)
    if renormalize:
        expected /= expected.sum(-1, keepdim=True)
    expected *= 1.5
    # Keep oracle masking independent of torch's device-side nonzero kernel.
    padding = padding.cpu()
    indices, weights = indices.cpu(), weights.cpu()
    expected_ids, expected = expected_ids.cpu(), expected.cpu()
    torch.testing.assert_close(indices[~padding].long(), expected_ids[~padding])
    torch.testing.assert_close(
        weights[~padding], expected[~padding], rtol=2e-5, atol=2e-6
    )
    assert bool((weights[padding] == 0).all())
    assert bool((indices[padding] == -1).all())


@pytest.mark.parametrize("hash_routing", [False, True])
def test_topk_nan_padding_without_mask(runtime, hash_routing):
    torch = runtime
    logits = torch.full((1, 256), float("nan"), device="cuda")
    weights = torch.empty(1, 6, device="cuda")
    indices = torch.empty(1, 6, device="cuda", dtype=torch.int32)
    rows = torch.empty_like(indices)
    ids = torch.zeros(1, device="cuda", dtype=torch.int32) if hash_routing else None
    table = (
        torch.arange(6, device="cuda", dtype=torch.int32).view(1, 6)
        if hash_routing
        else None
    )
    torch.ops._moe_C.topk_softplus_sqrt(
        weights, indices, rows, logits, True, 1.5, None, ids, table, None
    )
    torch.cuda.synchronize()
    assert bool(torch.isfinite(weights).all())
    assert bool((weights == 0).all())


def _rope(torch, value, positions, cache):
    value = value.float()
    cs = cache[positions]
    cos, sin = cs[:, :32], cs[:, 32:]
    if value.ndim == 3:
        cos, sin = cos[:, None], sin[:, None]
    even, odd = value[..., 448::2], value[..., 449::2]
    result = value.clone()
    result[..., 448::2] = torch.addcmul(-odd * sin, even, cos)
    result[..., 449::2] = torch.addcmul(odd * cos, even, sin)
    return result


def _assert_ulp(torch, actual, expected, max_ulp=1):
    """Allow only rounding-boundary differences, including values near zero."""
    assert actual.dtype == expected.dtype
    storage = torch.int8 if actual.element_size() == 1 else torch.int16
    sign = 1 << (actual.element_size() * 8 - 1)
    a = actual.detach().cpu().contiguous().view(storage).to(torch.int32)
    b = expected.detach().cpu().contiguous().view(storage).to(torch.int32)
    a = torch.where(a < 0, -sign - a, a)
    b = torch.where(b < 0, -sign - b, b)
    distance = (a - b).abs().max().item()
    assert distance <= max_ulp, f"maximum storage ULP distance: {distance}"


@pytest.mark.parametrize("mode", ["packed", "packed_no_qnorm", "bf16", "fp8"])
@pytest.mark.parametrize("tokens", [1, 17, 1025])
def test_qnorm_rope_cache_insert(runtime, mode, tokens):
    torch = runtime
    heads, padded_heads, block_size, eps = 16, 64, 16, 1e-6
    generator = torch.Generator(device="cuda").manual_seed(11)
    q = torch.randn(
        tokens, heads, 512, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    kv = torch.randn(
        tokens, 512, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    q_original, kv_original = q.clone(), kv.clone()
    positions = torch.arange(tokens, device="cuda", dtype=torch.int64) * 3 % 4096
    frequency = 10000.0 ** (-torch.arange(0, 64, 2, device="cuda").float() / 64)
    angles = torch.arange(4096, device="cuda").float()[:, None] * frequency
    rope_cache = torch.cat((angles.cos(), angles.sin()), dim=-1)
    # Nonsequential slots cross block boundaries. Negative slots and a shorter
    # insertion list exercise DP padding without suppressing the Q output.
    slots = torch.arange(tokens, device="cuda", dtype=torch.int64).flip(0) + 7
    slots[2::5] = -1
    if tokens > 1:
        slots = slots[:-1]
    blocks = (tokens + 7 + block_size - 1) // block_size + 1
    # Integer indices preserve the oracle rows without device-side nonzero.
    valid_rows = (slots.cpu() >= 0).nonzero(as_tuple=True)[0].to(slots.device)
    block_ids = slots[valid_rows] // block_size
    offsets = slots[valid_rows] % block_size
    q_float = q.float()
    q_norm = q_float * torch.rsqrt(q_float.square().mean(-1, keepdim=True) + eps)
    q_ref = _rope(
        torch, q_float if mode == "packed_no_qnorm" else q_norm, positions, rope_cache
    )
    kv_ref = _rope(torch, kv, positions, rope_cache)

    if mode.startswith("packed"):
        cache = torch.full(
            (blocks, block_size * 584), 93, device="cuda", dtype=torch.uint8
        )
        expected = cache.clone()
        args = (cache, slots, positions, rope_cache, padded_heads, eps, block_size)
        output = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            q, kv, *args, apply_q_norm=mode != "packed_no_qnorm"
        )
        torch.cuda.synchronize()
        _assert_ulp(torch, output[:, :heads], q_ref.to(q.dtype))
        assert bool((output[:, heads:] == 0).all())
        torch.testing.assert_close(q, q_original, rtol=0, atol=0)

        # Decode the documented 576-byte token payload and separate 8-byte
        # scale tail, independently of vLLM's Triton cache helpers.
        rounded_kv = kv_ref.to(kv.dtype)[: slots.numel()][valid_rows]
        groups = rounded_kv[:, :448].float().reshape(-1, 7, 64)
        exponent = torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(1e-4) / 448))
        quantized = (groups * torch.exp2(-exponent[..., None])).clamp(-448, 448)
        payload = expected[:, : block_size * 576].view(blocks, block_size, 576)
        scales = expected[:, block_size * 576 :].view(blocks, block_size, 8)
        payload[block_ids, offsets, :448] = (
            quantized.to(torch.float8_e4m3fn).view(torch.uint8).reshape(-1, 448)
        )
        payload[block_ids, offsets, 448:] = (
            rounded_kv[:, 448:].contiguous().view(torch.uint8)
        )
        scales[block_ids, offsets, :7] = (exponent + 127).to(torch.uint8)
        scales[block_ids, offsets, 7] = 0
        actual_rope = cache[:, : block_size * 576].view(blocks, block_size, 576)[
            block_ids, offsets, 448:
        ]
        _assert_ulp(torch, actual_rope.contiguous().view(kv.dtype), rounded_kv[:, 448:])
        # Compare every other byte exactly, including untouched slots and scale
        # padding. The RoPE region was already checked within one BF16 ULP.
        payload[block_ids, offsets, 448:] = actual_rope
        torch.testing.assert_close(cache, expected, rtol=0, atol=0)
    else:
        dtype = torch.bfloat16 if mode == "bf16" else torch.float8_e4m3fn
        cache = torch.zeros(blocks, block_size, 512, device="cuda", dtype=dtype)
        expected = torch.zeros(
            blocks, block_size, 512, device="cuda", dtype=torch.float32
        )
        if mode == "bf16":
            torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert(
                q, kv, cache, slots, positions, rope_cache, eps, block_size
            )
            output, output_ref = q, q_ref.to(dtype)
            kv_encoded = kv_ref.to(dtype)
        else:
            kv_scale = torch.tensor([0.25], device="cuda")
            q_scale_inv = torch.tensor([2.0], device="cuda")
            output = torch.empty_like(q, dtype=dtype)
            torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert(
                q,
                kv,
                output,
                cache,
                slots,
                positions,
                rope_cache,
                kv_scale,
                q_scale_inv,
                eps,
                block_size,
            )
            output_ref = (q_ref * q_scale_inv).clamp(-448, 448).to(dtype)
            # Full-cache FP8 keeps the KV rotation in fp32 until quantization.
            kv_encoded = (kv_ref / kv_scale).clamp(-448, 448).to(dtype)
            torch.testing.assert_close(q, q_original, rtol=0, atol=0)
        torch.cuda.synchronize()
        expected[block_ids, offsets] = kv_encoded.float()[: slots.numel()][valid_rows]
        _assert_ulp(torch, output, output_ref)
        _assert_ulp(torch, cache, expected.to(dtype))
        torch.testing.assert_close(
            cache[..., :448].float(), expected[..., :448], rtol=0, atol=0
        )
    torch.testing.assert_close(kv, kv_original, rtol=0, atol=0)


def test_qnorm_rope_cache_insert_supports_split_pcp_writes(runtime):
    torch = runtime
    tokens, heads, padded_heads, block_size = 17, 16, 32, 16
    generator = torch.Generator(device="cuda").manual_seed(19)
    q = torch.randn(
        tokens, heads, 512, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    kv = torch.randn(
        tokens, 512, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    positions = torch.arange(tokens, device="cuda", dtype=torch.int64)
    frequency = 10000.0 ** (-torch.arange(0, 64, 2, device="cuda").float() / 64)
    angles = torch.arange(tokens, device="cuda").float()[:, None] * frequency
    rope_cache = torch.cat((angles.cos(), angles.sin()), dim=-1)
    slots = torch.arange(tokens, device="cuda", dtype=torch.int64)
    slots[3::5] = -1
    cache = torch.full((2, block_size * 584), 93, device="cuda", dtype=torch.uint8)

    kv_only = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        kv.unsqueeze(1).contiguous(),
        kv,
        cache,
        slots,
        positions,
        rope_cache,
        0,
        1e-6,
        block_size,
        apply_q_norm=False,
        apply_q_rope=False,
    )
    torch.cuda.synchronize()
    assert kv_only.numel() == 0
    assert bool((cache != 93).any())

    written = cache.clone()
    q_only = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        q,
        kv,
        cache,
        slots[:0],
        positions,
        rope_cache,
        padded_heads,
        1e-6,
        block_size,
    )
    torch.cuda.synchronize()
    assert q_only.shape == (tokens, padded_heads, 512)
    assert bool(torch.isfinite(q_only).all())
    torch.testing.assert_close(cache, written, rtol=0, atol=0)
