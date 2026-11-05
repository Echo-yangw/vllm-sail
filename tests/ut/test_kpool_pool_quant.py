# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the PPU kpool pooling, quantization and cache-layout helpers.

These cover the three places that can silently diverge from the CUDA Triton
kernels: the softmax pool weighting, the 132-byte INT8 cache layout and the
floor-divided logits width.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vllm_sail.attention.ops import mla_sparse_kpool as kpool  # noqa: E402
from vllm_sail.models.glm5_next.ops import kpool_quant  # noqa: E402

KPOOL = 4
HEAD_DIM = 128


@pytest.fixture(autouse=True)
def _deterministic_torch():
    torch.manual_seed(0)


def _fake_quant(x: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    assert group_size == HEAD_DIM
    rows = x.shape[0]
    return (
        torch.zeros((rows, HEAD_DIM), dtype=torch.int8),
        torch.ones((rows, 1), dtype=torch.float32),
    )


def test_hadamard_is_orthogonal_and_preserves_dot_products():
    h = kpool_quant.hadamard128("cpu")
    assert h.shape == (128, 128) and h.dtype == torch.float32
    torch.testing.assert_close(h @ h.T, torch.eye(128), atol=1e-5, rtol=0)

    q = torch.randn(4, 128).to(torch.bfloat16)
    k = torch.randn(4, 128).to(torch.bfloat16)
    before = (q.float() * k.float()).sum()
    after = (kpool_quant.rotate128(q) * kpool_quant.rotate128(k)).sum()
    torch.testing.assert_close(after, before, rtol=2e-2, atol=2e-2)


def test_pool_weighted_matches_softmax_reference():
    k = torch.randn(5, KPOOL, HEAD_DIM).to(torch.bfloat16)
    gate = torch.randn(5, KPOOL, HEAD_DIM).to(torch.bfloat16)
    ape = torch.randn(KPOOL, HEAD_DIM)

    out = kpool._pool_weighted(k, gate, ape, KPOOL)

    score = gate.float() + ape
    prob = torch.exp(score - score.max(dim=1, keepdim=True).values)
    expected = ((k.float() * prob).sum(dim=1) / prob.sum(dim=1)).to(torch.bfloat16)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_compress_insert_writes_only_complete_pools(monkeypatch):
    recorded: dict = {}

    def fake_scatter(kv_cache, values, scales, slots):
        recorded["values"] = values
        recorded["slots"] = slots.clone()

    monkeypatch.setattr(kpool, "_scatter_int8_cache", fake_scatter)
    monkeypatch.setattr(kpool, "per_token_group_quant_int8", _fake_quant)

    n = 10
    k = torch.randn(n, HEAD_DIM).to(torch.bfloat16)
    gate = torch.randn(n, HEAD_DIM).to(torch.bfloat16)
    ape = torch.randn(KPOOL, HEAD_DIM)
    kv_cache = torch.zeros((4, 8, 132), dtype=torch.uint8)
    slots = torch.full((n,), -1, dtype=torch.int64)
    slots[3] = 100
    slots[7] = 101

    kpool._kpool_compress_insert_int8(k, gate, ape, kv_cache, slots, KPOOL, HEAD_DIM)

    # All rows are pooled with static shapes (capture-safe); only pool-ending
    # rows carry a non-negative slot and reach the INT8 scatter kernel.
    assert recorded["slots"].tolist() == [-1, -1, -1, 100, -1, -1, -1, 101, -1, -1]
    assert recorded["values"].shape == (n, HEAD_DIM)


def test_compress_insert_skips_batches_shorter_than_one_pool(monkeypatch):
    called: list = []
    monkeypatch.setattr(kpool, "_scatter_int8_cache", lambda *args: called.append(args))
    monkeypatch.setattr(kpool, "per_token_group_quant_int8", _fake_quant)

    kpool._kpool_compress_insert_int8(
        torch.zeros((3, HEAD_DIM), dtype=torch.bfloat16),
        torch.zeros((3, HEAD_DIM), dtype=torch.bfloat16),
        torch.zeros((KPOOL, HEAD_DIM)),
        torch.zeros((4, 8, 132), dtype=torch.uint8),
        torch.tensor([7, 8, 9], dtype=torch.int64),
        KPOOL,
        HEAD_DIM,
    )

    assert not called


def test_int8_cache_layout_round_trip():
    # Mirrors indexer_k_quant_and_cache_kernel and _scatter_indexer_int8_cache:
    # value plane then fp32 scale plane inside each page.
    page_size, num_blocks = 32, 3
    cache = torch.zeros((num_blocks, page_size, HEAD_DIM + 4), dtype=torch.uint8)
    values = torch.randint(-127, 128, (5, HEAD_DIM), dtype=torch.int8)
    scales = torch.rand(5)
    slots = [0, 31, 32, 63, 70]

    flat = cache.reshape(-1)
    block_stride = cache.stride(0)
    for row, slot in enumerate(slots):
        page, offset = divmod(slot, page_size)
        base = page * block_stride
        value_base = base + offset * HEAD_DIM
        flat[value_base : value_base + HEAD_DIM] = values[row].view(torch.uint8)
        scale_base = base + page_size * HEAD_DIM + offset * 4
        flat[scale_base : scale_base + 4] = scales[row : row + 1].view(torch.uint8)

    for row, slot in enumerate(slots):
        page, offset = divmod(slot, page_size)
        base = page * block_stride
        got_values = flat[base + offset * HEAD_DIM : base + (offset + 1) * HEAD_DIM]
        assert torch.equal(got_values.view(torch.int8), values[row])
        scale_base = base + page_size * HEAD_DIM + offset * 4
        got_scale = flat[scale_base : scale_base + 4].view(torch.float32)
        assert got_scale.item() == scales[row].item()


def test_non_divisible_max_model_len_keeps_floor_tail():
    # The live Indexer derives max_pool_len = max_model_len // index_kpool
    # (floor). A non-divisible max_model_len leaves a token-granular tail that
    # must survive pool expansion: 4097 tokens -> 1024 pools + 1 tail token,
    # and the last logits column is max_pool_len - 1 (= 1023).
    max_model_len = 4097
    max_pool_len = max_model_len // KPOOL
    assert max_pool_len == 1024

    pool_ids = torch.tensor([[0, max_pool_len - 1]], dtype=torch.int64)
    seq_lens = torch.tensor([max_model_len], dtype=torch.int32)

    out = kpool._expand_pools(pool_ids, seq_lens, KPOOL, 8)

    # 2 pools * 4 tokens + (4 - 1) tail columns.
    assert out.shape == (1, 11)
    assert out[0].tolist() == [0, 1, 2, 3, 4092, 4093, 4094, 4095, 4096, -1, -1]


def test_expand_pools_keeps_incomplete_tail_for_non_divisible_seq_len():
    pool_ids = torch.tensor([[0, 2]], dtype=torch.int64)
    # 10 % 4 = 2: the trailing two tokens are the incomplete tail pool.
    seq_lens = torch.tensor([10], dtype=torch.int32)

    out = kpool._expand_pools(pool_ids, seq_lens, KPOOL, 8)

    assert out.shape == (1, 11)
    assert out[0].tolist() == [0, 1, 2, 3, 8, 9, 10, 11, 8, 9, -1]
