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


def test_decode_logits_use_the_paged_kernel_only(monkeypatch):
    """Decode must call int8_paged_mqa_logits and never gather the K cache."""
    calls = {}

    def fake_paged(
        q, kv_cache, weights, seq_lens, block_table, schedule_metadata,
        max_model_len, clean_logits,
    ):
        calls.update(
            q_shape=tuple(q.shape),
            kv_shape=tuple(kv_cache.shape),
            weights_shape=tuple(weights.shape),
            seq_lens=seq_lens.tolist(),
            block_table=block_table.tolist(),
            sched_shape=tuple(schedule_metadata.shape),
            max_model_len=max_model_len,
            clean_logits=clean_logits,
        )
        rows = q.shape[0] * q.shape[1]
        return torch.zeros((rows, max_model_len), dtype=torch.float32)

    def _forbidden(*args, **kwargs):
        raise AssertionError("decode must not gather the K cache")

    monkeypatch.setattr(kpool, "int8_paged_mqa_logits", fake_paged)
    monkeypatch.setattr(kpool, "_gather_workspace_shapes", _forbidden)
    monkeypatch.setattr(kpool, "int8_mqa_logits", _forbidden)

    kv_cache_view = torch.zeros((4, 64, 1, 132), dtype=torch.uint8)
    padded_q = torch.zeros((2, 1, 32, 128), dtype=torch.int8)
    padded_w = torch.zeros((2, 32), dtype=torch.float32)
    seq_lens = torch.tensor([[5], [3]], dtype=torch.int32)
    block_table = torch.zeros((2, 1), dtype=torch.int32)
    schedule_metadata = torch.zeros((4,), dtype=torch.int32)

    logits = kpool._decode_logits_paged(
        kv_cache_view,
        padded_q,
        padded_w,
        seq_lens,
        block_table,
        schedule_metadata,
        64,
    )

    assert logits.shape == (2, 64)
    assert calls["max_model_len"] == 64
    assert calls["clean_logits"] is False
    assert calls["kv_shape"] == (4, 64, 1, 132)
    assert calls["q_shape"] == (2, 1, 32, 128)
    assert calls["weights_shape"] == (2, 32)
    assert calls["seq_lens"] == [[5], [3]]
    assert calls["block_table"] == [[0], [0]]
    assert calls["sched_shape"] == (4,)


def test_decode_path_keeps_no_dense_fallback():
    source = Path(kpool.__file__).read_text(encoding="utf-8")

    assert "_decode_logits_paged(" in source
    assert "_decode_logits_non_paged" not in source
    assert "_decode_logits_layout" not in source
    assert "KPOOL_DECODE_LOGITS" not in source
    assert "KPOOL_MAX_DENSE_POOLS" not in source


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
