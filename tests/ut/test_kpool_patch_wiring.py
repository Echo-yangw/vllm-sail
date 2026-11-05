# SPDX-License-Identifier: Apache-2.0
"""Wiring tests for the glm5_next kpool PPU patches.

The live glm5next model (``common.attention``) imports ``SparseAttnIndexerKpool``
from the package selector ``vllm.models.glm5next.sparse_indexer``, which
resolves to the ``nvidia/sparse_indexer`` variant on PPU. That is the class this
plugin patches; if the model ever switches to another variant the PPU patches
would silently stop applying, so the identity is asserted here (and at patch
import time).

Upstream also ships ``vllm.models.glm5next.nvidia.attention`` importing
``vllm.model_executor.layers.sparse_attn_indexer_kpool``. That module registers
the same CustomOp name and cannot be imported after the glm5next package
(``Duplicate op name``), so nothing here (or in the patch) may import it.
"""

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")


def test_live_model_uses_the_patched_class():
    from vllm.model_executor.custom_op import CustomOp
    from vllm.models.glm5next import sparse_indexer as selector
    from vllm.models.glm5next.common import attention as model_attention
    from vllm.models.glm5next.nvidia import sparse_indexer as live

    import vllm_sail.patch.enhancement.attention.sparse_attn_indexer_kpool  # noqa: F401

    assert model_attention.SparseAttnIndexerKpool is live.SparseAttnIndexerKpool
    assert selector.SparseAttnIndexerKpool is live.SparseAttnIndexerKpool
    assert live.SparseAttnIndexerKpool.forward_ppu is not CustomOp.forward_ppu
    # forward_ppu must accept the same keyword set as the live forward_cuda,
    # including the topk dispatcher choice (ignored on PPU).
    params = inspect.signature(live.SparseAttnIndexerKpool.forward_ppu).parameters
    assert "topk_backend" in params


def test_class_patch_only_targets_the_live_variant():
    from vllm_sail.patch.enhancement.attention import (
        sparse_attn_indexer_kpool as patch_module,
    )

    source = Path(patch_module.__file__).read_text(encoding="utf-8")
    assert patch_module._MODULE == "vllm.models.glm5next.nvidia.sparse_indexer"
    assert not re.search(
        r"^\s*(?:from|import)\s+vllm\.model_executor\.layers\."
        r"sparse_attn_indexer_kpool",
        source,
        re.MULTILINE,
    )


def test_live_path_guard_fails_loudly(monkeypatch):
    from vllm_sail.patch.enhancement.attention import (
        sparse_attn_indexer_kpool as patch_module,
    )

    for name in patch_module._EXPECTED_CONSUMERS:
        monkeypatch.setitem(
            sys.modules,
            name,
            SimpleNamespace(SparseAttnIndexerKpool=object()),
        )
        with pytest.raises(RuntimeError, match="no longer"):
            patch_module._check_live_consumer()
        monkeypatch.undo()


def test_fwht_redirect_returns_int8_on_ppu(monkeypatch):
    import torch

    from vllm_sail.models.glm5_next.ops import kpool_quant
    from vllm_sail.patch.enhancement.models import glm5_next as patch_module

    sentinel = (
        torch.zeros((2, 128), dtype=torch.int8),
        torch.ones((2, 1)),
    )
    monkeypatch.setattr(kpool_quant, "fwht128_quant_int8", lambda _q: sentinel)
    monkeypatch.setattr(
        type(patch_module.current_platform), "is_ppu", lambda self: True
    )
    monkeypatch.setattr(
        type(patch_module.current_platform),
        "supports_fp8",
        lambda self: False,
    )

    out = patch_module.fwht128_quant_fp8(torch.zeros((2, 128), dtype=torch.bfloat16))

    assert out is sentinel


def test_fwht_provider_is_patched_in_place():
    from vllm.models.glm5next.nvidia.ops import kpool_compress

    import vllm_sail.patch.enhancement.models.glm5_next  # noqa: F401
    from vllm_sail.patch.utils import PATCH_MARKER

    markers = getattr(kpool_compress.fwht128_quant_fp8, PATCH_MARKER, None)
    assert markers is not None
    assert "vllm.models.glm5next.nvidia.ops.kpool_compress.fwht128_quant_fp8" in markers
