# SPDX-License-Identifier: Apache-2.0
"""Dispatch guards for the PPU kpool indexer (no device required)."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from vllm_sail.attention.ops import mla_sparse_kpool as kpool  # noqa: E402


def test_ppu_module_contains_no_cuda_fp8_calls():
    source = Path(kpool.__file__).read_text(encoding="utf-8")

    assert "int8_mqa_logits" in source
    assert "int8_paged_mqa_logits" in source
    # Match call sites only: the docstrings name the CUDA entry points on
    # purpose to document the ported semantics.
    assert not re.search(r"\bfp8_fp4_", source)
    assert not re.search(r"\bindexer_k_quant_and_cache\(", source)
    assert not re.search(r"\bkpool_compress_and_write_cache\(", source)
    assert not re.search(r"\bexpand_pools_and_append_tail\(", source)
    assert not re.search(r"\bfwht128_quant_fp8\(", source)
    # The model_executor module registers the same CustomOp name as the live
    # variant; importing it raises ``Duplicate op name``. Only imports matter:
    # the docstring names the module to explain why it must stay unimported.
    assert not re.search(
        r"^\s*(?:from|import)\s+vllm\.model_executor\.layers\."
        r"sparse_attn_indexer_kpool",
        source,
        re.MULTILINE,
    )


def test_registered_op_annotations_survive_breakable_cudagraph_wrapper():
    """Schema inference must not depend on the op module's globals.

    ``ppu_sparse_attn_indexer_kpool`` is wrapped by
    ``@eager_break_during_capture``, whose wrapper lives in
    ``vllm.compilation.breakable_cudagraph``. torch's ``infer_schema`` follows
    ``__wrapped__`` for the signature but evaluates string annotations in the
    *wrapper's* globals, so ``from __future__ import annotations`` makes
    ``LayerNameType`` unresolvable there (EngineCore startup NameError).
    """
    import functools

    from torch._library.infer_schema import infer_schema

    def foreign_wrapper(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        return wrapper

    wrapped = foreign_wrapper(kpool.ppu_sparse_attn_indexer_kpool)
    schema = infer_schema(wrapped, mutates_args=("kv_cache",))
    assert "k_cache_prefix" in schema


def test_decode_defaults_to_non_paged_backend(monkeypatch):
    from vllm_sail import envs as ppu_envs

    monkeypatch.delenv("VLLM_SAIL_KPOOL_DECODE_LOGITS", raising=False)
    monkeypatch.delenv("VLLM_PPU_KPOOL_DECODE_LOGITS", raising=False)
    assert ppu_envs.VLLM_SAIL_KPOOL_DECODE_LOGITS == "non_paged"

    monkeypatch.setenv("VLLM_SAIL_KPOOL_DECODE_LOGITS", "paged")
    with pytest.raises(ValueError, match="Expected one of non_paged"):
        _ = ppu_envs.VLLM_SAIL_KPOOL_DECODE_LOGITS

    monkeypatch.setenv("VLLM_SAIL_KPOOL_DECODE_LOGITS", "bogus")
    with pytest.raises(ValueError, match="Expected one of"):
        _ = ppu_envs.VLLM_SAIL_KPOOL_DECODE_LOGITS


def test_decode_logits_layout_uniform_and_padded_rows():
    decode_lens = torch.tensor([2, 1])
    next_n = 2
    # Request 0: two queries of 8/8 pools; request 1: 3 pools, second row padded.
    seq_lens = torch.tensor([[8, 8], [3, 0]])
    cu, ks, ke = kpool._decode_logits_layout(seq_lens, decode_lens, 2, next_n)
    # cu is the cumulative *context* (pool) length, never the new-token count.
    assert cu.tolist() == [0, 8, 11]
    assert ks.tolist() == [0, 0, 8, 8]
    assert ke.tolist() == [8, 8, 11, 8]  # padded row: ks == ke


def test_decode_logits_layout_broadcasts_1d_seq_lens():
    decode_lens = torch.tensor([1, 1])
    seq_lens = torch.tensor([5, 2])
    cu, ks, ke = kpool._decode_logits_layout(seq_lens, decode_lens, 2, 1)
    assert cu.tolist() == [0, 5, 7]
    assert ks.tolist() == [0, 5]
    assert ke.tolist() == [5, 7]


def test_decode_non_paged_glue(monkeypatch):
    calls = {}

    def fake_gather(kv_cache, k_quant, k_scale, block_table, cu):
        calls["cu"] = cu.tolist()
        calls["block_table_shape"] = tuple(block_table.shape)

    def fake_logits(q, kv, weights, ks, ke, clean_logits):
        calls["q_shape"] = tuple(q.shape)
        calls["logits_rows"] = tuple(ks.shape)
        calls["ks"] = ks.tolist()
        calls["ke"] = ke.tolist()
        calls["clean_logits"] = clean_logits
        return torch.zeros((ks.shape[0], q.shape[0]), dtype=torch.float32)

    class _Workspace:
        def get_simultaneous(self, values_spec, scales_spec):
            v_shape, v_dtype = values_spec
            s_shape, s_dtype = scales_spec
            calls["values_shape"] = v_shape
            return torch.zeros(v_shape, dtype=v_dtype), torch.zeros(
                s_shape, dtype=s_dtype
            )

    monkeypatch.setattr(
        kpool, "ops", SimpleNamespace(cp_gather_indexer_k_quant_cache=fake_gather)
    )
    monkeypatch.setattr(kpool, "int8_mqa_logits", fake_logits)
    monkeypatch.setattr(kpool, "current_workspace_manager", lambda: _Workspace())

    batch_size, next_n, head_dim = 2, 1, 128
    padded_q = torch.zeros((batch_size, next_n, 32, head_dim), dtype=torch.int8)
    padded_w = torch.zeros((batch_size, 32), dtype=torch.float32)
    logits, ks, ke = kpool._decode_logits_non_paged(
        torch.zeros((2, 8, head_dim + 4), dtype=torch.uint8),
        padded_q,
        padded_w,
        torch.tensor([5, 3]),
        torch.tensor([5, 3]),
        torch.zeros((batch_size, 4), dtype=torch.int32),
        batch_size,
        next_n,
        batch_size * next_n,
        20,  # token-granular batch max (metadata int, no host sync)
        4,  # kpool: workspace = 2 * ceil(20 / 4) = 10 pools
        head_dim,
    )

    assert calls["cu"] == [0, 5, 8]
    assert calls["values_shape"] == (10, head_dim)
    assert calls["q_shape"] == (2, 32, head_dim)
    assert calls["ks"] == [0, 5]
    assert calls["ke"] == [5, 8]
    assert calls["clean_logits"] is False
    assert logits.shape == (2, 2)
    assert ks.tolist() == [0, 5] and ke.tolist() == [5, 8]


def test_rejects_non_int8_q(monkeypatch):
    monkeypatch.setattr(
        kpool,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=None),
    )
    with pytest.raises(RuntimeError, match="INT8 q_quant"):
        kpool.ppu_sparse_attn_indexer_kpool(
            SimpleNamespace(shape=(1,)),
            "layer",
            object(),
            torch.zeros(1, dtype=torch.bfloat16),  # not int8
            None,
            None,
            None,
            128,
            "ue8m0",
            2048,
            128,
            8192,
            1024,
            None,
            False,
        )


def test_rejects_q_scale(monkeypatch):
    monkeypatch.setattr(
        kpool,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=None),
    )
    with pytest.raises(RuntimeError, match="q_scale must be None"):
        kpool.ppu_sparse_attn_indexer_kpool(
            SimpleNamespace(shape=(1,)),
            "layer",
            object(),
            torch.zeros(1, dtype=torch.int8),
            torch.ones(1),
            None,
            None,
            128,
            "ue8m0",
            2048,
            128,
            8192,
            1024,
            None,
            False,
        )


def test_rejects_fp4_cache(monkeypatch):
    monkeypatch.setattr(
        kpool,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=None),
    )
    with pytest.raises(NotImplementedError, match="MXFP4"):
        kpool.ppu_sparse_attn_indexer_kpool(
            SimpleNamespace(shape=(1,)),
            "layer",
            object(),
            torch.zeros(1, dtype=torch.int8),
            None,
            None,
            None,
            128,
            "ue8m0",
            2048,
            128,
            8192,
            1024,
            None,
            False,
            use_fp4_cache=True,
        )


def test_fake_impl_returns_input_buffer():
    buffer = torch.zeros((4, 2176), dtype=torch.int32)

    out = kpool.ppu_sparse_attn_indexer_kpool_fake(
        torch.zeros((1, 4), dtype=torch.bfloat16),
        "layer",
        torch.zeros((2, 8, 132), dtype=torch.uint8),
        torch.zeros((1, 128), dtype=torch.int8),
        None,
        torch.zeros((1, 128), dtype=torch.bfloat16),
        torch.zeros((1, 32)),
        128,
        "ue8m0",
        2048,
        128,
        8192,
        1024,
        buffer,
        False,
    )

    assert out is buffer
