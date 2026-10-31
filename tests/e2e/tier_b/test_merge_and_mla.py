# SPDX-License-Identifier: Apache-2.0
"""Numerical and graph-replay oracles for the public upstream op schemas."""

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


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("head_dim", [128, 192])
def test_merge_partials_strides_empty_context_and_graph(runtime, dtype, fp8, head_dim):
    torch = runtime
    torch.manual_seed(37)
    tokens, heads = 7, 4
    dtype = getattr(torch, dtype)
    # Padded token/head strides and transposed LSE are used by MLA backends.
    prefix = torch.randn(tokens, heads + 2, head_dim + 16, device="cuda", dtype=dtype)[
        :, :heads, :head_dim
    ]
    suffix = torch.randn_like(prefix)
    pl = torch.randn(tokens, heads, device="cuda").T
    sl = torch.randn(tokens, heads, device="cuda").T
    pl[:, 0], sl[:, 1] = float("-inf"), float("inf")
    pl[:, 2], sl[:, 2] = float("-inf"), float("-inf")
    prefix[0], prefix[2], suffix[1], suffix[2] = (
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
    )
    # Exercise LSE-normalized merge for four tokens; copy suffix thereafter.
    output_dtype = torch.float8_e4m3fn if fp8 else dtype
    storage = torch.empty(
        tokens, heads + 1, head_dim + 8, device="cuda", dtype=output_dtype
    )
    output = storage[:, :heads, :head_dim]
    output_lse = torch.empty(tokens, heads, device="cuda").T
    scale = torch.tensor([0.25], device="cuda") if fp8 else None

    def run():
        torch.ops._C.merge_attn_states(
            output, output_lse, prefix, pl, suffix, sl, 4, scale
        )

    run()
    torch.cuda.synchronize()
    p, s, lp, ls = prefix.cpu().float(), suffix.cpu().float(), pl.cpu().T, sl.cpu().T
    lp[lp == float("inf")] = float("-inf")
    ls[ls == float("inf")] = float("-inf")
    merged_lse = torch.logaddexp(lp, ls)
    p = torch.where(torch.isneginf(lp)[..., None], 0, p)
    s_clean = torch.where(torch.isneginf(ls)[..., None], 0, s)
    expected = (
        p * torch.exp(lp - merged_lse)[..., None]
        + s_clean * torch.exp(ls - merged_lse)[..., None]
    )
    expected = torch.where(torch.isneginf(merged_lse)[..., None], 0, expected)
    expected[4:] = s[4:]
    merged_lse[4:] = sl.cpu().T[4:]
    if fp8:
        expected = (expected / 0.25).clamp(-448, 448).to(output_dtype)
    else:
        expected = expected.to(dtype)
    torch.testing.assert_close(
        output.cpu().float(), expected.float(), rtol=0.01, atol=0.002
    )
    torch.testing.assert_close(output_lse.cpu(), merged_lse.T, rtol=1e-5, atol=1e-6)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output.cpu().float(), expected.float(), rtol=0.01, atol=0.002
    )


def test_merge_optional_lse_empty_batch_and_aliases(runtime):
    torch = runtime
    p = torch.full((2, 1, 128), 2.0, device="cuda")
    s = torch.full_like(p, 6.0)
    lse = torch.zeros(1, 2, device="cuda")
    torch.ops._C.merge_attn_states(p, None, p, lse, s, lse, None)
    torch.testing.assert_close(p, torch.full_like(p, 4.0))
    empty = p[:0]
    torch.ops._C.merge_attn_states(
        empty, None, empty, lse[:, :0], empty, lse[:, :0], None
    )


@pytest.mark.cap89
@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
def test_mla_cache_skip_scale_and_legacy_default(runtime, dtype_name):
    """Exercise the new FP8 specialization and preserve six-argument callers."""
    torch = runtime
    dtype = getattr(torch, dtype_name)
    rank, rope = 512, 64
    # Binary fractions keep the independent CPU FP8 oracle unambiguous.
    rows = (
        (torch.arange(6 * (rank + rope), device="cpu") % 65 - 32).float() / 16
    ).reshape(6, rank + rope)
    latent = rows[:, :rank].to(device="cuda", dtype=dtype).contiguous()
    positional = rows[:, rank:].to(device="cuda", dtype=dtype).contiguous()
    # Two graph-padding rows have no mapping and one mapped row is null.
    slots = torch.tensor([2, 10, -1, 40], device="cuda", dtype=torch.int64)
    scale = torch.tensor([0.25], device="cuda")
    op = torch.ops._C_cache_ops.concat_and_cache_mla
    outputs = {}
    for mode in ("legacy", "scaled", "unscaled"):
        cache = torch.full((2, 32, rank + rope), 0x5A, device="cuda", dtype=torch.uint8)
        args = (latent, positional, cache, slots, "fp8", scale)
        if mode == "legacy":
            op(*args)
        else:
            op(*args, mode == "unscaled")
        torch.cuda.synchronize()
        outputs[mode] = cache.cpu()
        expected = torch.full_like(outputs[mode], 0x5A)
        divisor = 1.0 if mode == "unscaled" else 0.25
        for token, slot in enumerate((2, 10, -1, 40)):
            if slot >= 0:
                expected.view(-1, rank + rope)[slot] = (
                    (rows[token] / divisor).to(torch.float8_e4m3fn).view(torch.uint8)
                )
        torch.testing.assert_close(outputs[mode], expected, rtol=0, atol=0)
    torch.testing.assert_close(outputs["legacy"], outputs["scaled"], rtol=0, atol=0)
    assert not torch.equal(outputs["scaled"], outputs["unscaled"])


def test_mla_empty_slot_mapping_does_not_launch_or_modify_inputs(runtime):
    torch = runtime
    q = torch.ones(3, 2, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.ones(3, 64, device="cuda", dtype=torch.bfloat16)
    latent = torch.ones(3, 512, device="cuda", dtype=torch.bfloat16)
    cache = torch.ones(2, 8, 576, device="cuda", dtype=torch.bfloat16)
    torch.ops._C_cache_ops.concat_and_cache_mla_rope_fused(
        torch.zeros(3, device="cuda", dtype=torch.int64),
        q,
        k,
        latent,
        torch.ones(1, 64, device="cuda", dtype=torch.bfloat16),
        True,
        torch.empty(0, device="cuda", dtype=torch.int64),
        cache,
        "auto",
        torch.ones(1, device="cuda"),
    )
    torch.cuda.synchronize()
    for tensor in (q, k, latent, cache):
        torch.testing.assert_close(tensor, torch.ones_like(tensor), rtol=0, atol=0)


@pytest.mark.parametrize("neox", [False, True])
@pytest.mark.parametrize("cache_kind", ["auto", "fp8"])
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_mla_rope_cache_padding_and_scale(runtime, neox, cache_kind, dtype_name):
    torch = runtime
    torch.manual_seed(17)
    dtype = getattr(torch, dtype_name)
    tokens, heads, rank, rope = 6, 4, 512, 64
    q = torch.randn(tokens, heads, rope, device="cuda", dtype=dtype)
    k = torch.randn(tokens, rope, device="cuda", dtype=dtype)
    latent = torch.randn(tokens, rank, device="cuda", dtype=dtype)
    positions = torch.tensor([1, 7, 13, 3, 9, 10], device="cuda")
    # Two graph-padding rows have no mapping; one mapped row is explicitly null.
    slots = torch.tensor([3, -1, 9, 0], device="cuda")
    phase = torch.outer(
        torch.arange(32, device="cuda").float(),
        10000.0 ** (-torch.arange(0, rope, 2, device="cuda").float() / rope),
    )
    cos_sin = torch.cat((phase.cos(), phase.sin()), -1).to(dtype)
    scale = torch.tensor([0.25], device="cuda")
    cache_dtype = dtype if cache_kind == "auto" else torch.uint8
    cache = torch.zeros(2, 8, rank + rope, device="cuda", dtype=cache_dtype)
    before_q, before_k = q.clone(), k.clone()
    torch.ops._C_cache_ops.concat_and_cache_mla_rope_fused(
        positions, q, k, latent, cos_sin, neox, slots, cache, cache_kind, scale
    )
    torch.cuda.synchronize()
    expected_q, expected_k = before_q.cpu(), before_k.cpu()
    expected_cache = torch.zeros_like(cache, device="cpu")
    for token, slot in enumerate(slots.cpu().tolist()):
        if slot < 0:
            continue
        cos, sin = cos_sin.cpu()[positions[token].item()].chunk(2)
        for values in (expected_q, expected_k):
            x = values[token].clone()
            left, right = (
                (slice(0, 32), slice(32, 64))
                if neox
                else (slice(0, None, 2), slice(1, None, 2))
            )
            # Native arithmetic uses the input scalar type at each operation.
            values[token, ..., left] = x[..., left] * cos - x[..., right] * sin
            values[token, ..., right] = x[..., right] * cos + x[..., left] * sin
        row = torch.cat((latent.cpu()[token], expected_k[token]))
        if cache_kind == "fp8":
            row = (
                (row.float() / 0.25)
                .clamp(-448, 448)
                .to(torch.float8_e4m3fn)
                .view(torch.uint8)
            )
        expected_cache.view(-1, rank + rope)[slot] = row
    torch.testing.assert_close(q.cpu(), expected_q, rtol=0.01, atol=0.016)
    torch.testing.assert_close(k.cpu(), expected_k, rtol=0.01, atol=0.016)
    untouched = torch.ones(16, dtype=torch.bool)
    untouched[slots.cpu()[slots.cpu() >= 0]] = False
    unwritten_rows = cache.cpu().view(16, rank + rope)[untouched]
    torch.testing.assert_close(
        unwritten_rows, torch.zeros_like(unwritten_rows), rtol=0, atol=0
    )
    if cache_kind == "fp8":
        actual = cache.cpu().view(torch.float8_e4m3fn).float()
        expected = expected_cache.view(torch.float8_e4m3fn).float()
        torch.testing.assert_close(actual, expected, rtol=0.15, atol=0.063)
    else:
        torch.testing.assert_close(cache.cpu(), expected_cache, rtol=0.01, atol=0.016)
