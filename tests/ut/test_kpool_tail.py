# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the PPU kpool tail-cache semantics.

The tail cache is stateful across steps: the reference is the CUDA
``_kpool_tail_seed_kernel`` / ``_kpool_decode_update_batched_kernel`` pair.
These tests pin the two properties that are easy to break: only the request's
trailing pool is seeded, and a decode completion reads the ring *after* the
earlier tokens of the same step were stashed while the current token is taken
live.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vllm_sail.attention.ops import mla_sparse_kpool as kpool  # noqa: E402
from vllm_sail.models.glm5_next.ops import kpool_quant  # noqa: E402

KPOOL = 4
HEAD_DIM = 128


def _torch_stash(tail, blk, phys, key, gate, valid):
    """Reference for ``_stash_tail_kernel``: masked ring write."""
    rows = valid.nonzero(as_tuple=False).flatten()
    tail[blk[rows], 0, phys[rows]] = key[rows]
    tail[blk[rows], 1, phys[rows]] = gate[rows]


def _torch_stash_t(tail, blk, phys, key, gate, valid, step):
    _torch_stash(tail, blk, phys, key[:, step, :], gate[:, step, :], valid)


@pytest.fixture(autouse=True)
def _patch_stash(monkeypatch):
    # CPU tests exercise the gather/mask semantics; the Triton store itself is
    # covered by the decode device probe.
    monkeypatch.setattr(kpool, "_stash_tail", _torch_stash)
    monkeypatch.setattr(kpool, "_stash_tail_t", _torch_stash_t)


def test_seed_writes_only_the_trailing_pool():
    # Production uses one pool per tail block (ring == kpool); a larger ring
    # makes the "earlier tokens must not be seeded" property observable, since
    # the trailing pool then covers only part of the block.
    ring = 2 * KPOOL
    tail = torch.zeros((8, 2, ring, HEAD_DIM), dtype=torch.bfloat16)
    block = 5
    n = 12
    tslot = torch.tensor(
        [block * ring + pos % ring for pos in range(n)], dtype=torch.int64
    )
    key = torch.randn(n, HEAD_DIM).to(torch.bfloat16)
    gate = torch.randn(n, HEAD_DIM).to(torch.bfloat16)

    kpool._kpool_seed_tail(tail, key, gate, tslot, KPOOL, HEAD_DIM)

    # Tokens 0..7 are not in the request's trailing pool: the second half of
    # the block stays empty.
    assert torch.count_nonzero(tail[block, :, KPOOL:]) == 0
    for pos in range(n - KPOOL, n):
        torch.testing.assert_close(tail[block, 0, pos % ring], key[pos])
        torch.testing.assert_close(tail[block, 1, pos % ring], gate[pos])


def test_decode_update_completion_reads_stashed_order(monkeypatch):
    recorded: dict = {}

    def fake_quant(x: torch.Tensor, group_size: int):
        assert group_size == HEAD_DIM
        recorded["x"] = x
        rows = x.shape[0]
        return (
            torch.zeros((rows, HEAD_DIM), dtype=torch.int8),
            torch.ones((rows, 1), dtype=torch.float32),
        )

    def fake_scatter(kv_cache, values, scales, slots):
        recorded["slots"] = slots.clone()

    monkeypatch.setattr(kpool, "_scatter_int8_cache", fake_scatter)
    monkeypatch.setattr(kpool, "per_token_group_quant_int8", fake_quant)

    ring = KPOOL
    block = 3
    tail = torch.zeros((8, 2, ring, HEAD_DIM), dtype=torch.bfloat16)
    # Position 4 was stashed by a previous step into ring slot 0.
    k4 = torch.randn(HEAD_DIM).to(torch.bfloat16)
    g4 = torch.randn(HEAD_DIM).to(torch.bfloat16)
    tail[block, 0, 0] = k4
    tail[block, 1, 0] = g4
    prior = [torch.randn(HEAD_DIM).to(torch.bfloat16) for _ in range(3)]
    prior_gate = [torch.randn(HEAD_DIM).to(torch.bfloat16) for _ in range(3)]
    key = torch.stack(prior).unsqueeze(0)  # positions 5, 6, 7
    gate = torch.stack(prior_gate).unsqueeze(0)
    positions = torch.tensor([[5, 6, 7]], dtype=torch.int32)
    tail_slot = torch.tensor(
        [[block * ring + 1, block * ring + 2, block * ring + 3]],
        dtype=torch.int32,
    )
    # Pool-granular slot mapping: only the completing token (pos 7) writes.
    slot_mapping = torch.tensor([[-1, -1, 42]], dtype=torch.int64)
    ape = torch.randn(KPOOL, HEAD_DIM)
    cache = torch.zeros((4, 8, 132), dtype=torch.uint8)

    kpool._decode_update_torch(
        cache,
        tail,
        tail_slot,
        key,
        gate,
        ape,
        slot_mapping,
        positions,
        KPOOL,
        HEAD_DIM,
    )

    # Every batch row is pooled (static shapes, capture-safe); the scatter
    # drops rows whose pool-granular slot is negative.
    assert recorded["slots"].tolist() == [-1, -1, 42]
    expected = kpool._pool_weighted(
        torch.stack([k4, *prior]).unsqueeze(0),
        torch.stack([g4, *prior_gate]).unsqueeze(0),
        ape,
        KPOOL,
    )
    torch.testing.assert_close(
        recorded["x"][-1], kpool_quant.rotate128(expected)[0], rtol=0, atol=0
    )
    # Stash happens after the completion read, in position order.
    for offset, token_key in enumerate(prior, start=1):
        torch.testing.assert_close(tail[block, 0, offset], token_key)
        torch.testing.assert_close(tail[block, 1, offset], prior_gate[offset - 1])


def test_decode_update_skips_padded_tokens(monkeypatch):
    recorded: dict = {}
    monkeypatch.setattr(
        kpool,
        "_scatter_int8_cache",
        lambda kv_cache, values, scales, slots: recorded.update(slots=slots.clone()),
    )

    def fake_quant(x: torch.Tensor, group_size: int):
        assert group_size == HEAD_DIM
        rows = x.shape[0]
        return (
            torch.zeros((rows, HEAD_DIM), dtype=torch.int8),
            torch.ones((rows, 1), dtype=torch.float32),
        )

    monkeypatch.setattr(kpool, "per_token_group_quant_int8", fake_quant)
    tail = torch.zeros((4, 2, KPOOL, HEAD_DIM), dtype=torch.bfloat16)
    key = torch.zeros((1, 1, HEAD_DIM), dtype=torch.bfloat16)
    gate = torch.zeros((1, 1, HEAD_DIM), dtype=torch.bfloat16)
    positions = torch.tensor([[-1]], dtype=torch.int32)
    tail_slot = torch.tensor([[12]], dtype=torch.int32)
    slot_mapping = torch.tensor([[7]], dtype=torch.int64)

    kpool._decode_update_torch(
        torch.zeros((2, 8, 132), dtype=torch.uint8),
        tail,
        tail_slot,
        key,
        gate,
        torch.zeros((KPOOL, HEAD_DIM)),
        slot_mapping,
        positions,
        KPOOL,
        HEAD_DIM,
    )

    assert recorded["slots"].tolist() == [-1]
    assert torch.count_nonzero(tail) == 0
