# SPDX-License-Identifier: Apache-2.0
"""fp8-capable PPU (890P/SM89) gating for the kpool PPU patches.

890P reports PPU capability ``(8, 9)`` and supports native fp8. The kpool INT8
port only applies to PPUs without native fp8 (810E/SM80). On a fp8-capable PPU
every redirect must fall back to the upstream fp8 implementation instead of
forcing INT8.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

import torch  # noqa: E402


def _patch_module():
    from vllm_sail.patch.enhancement.attention import (
        sparse_attn_indexer_kpool as module,
    )

    return module


class _ForwardCudaSpy:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def forward_cuda(self, hidden_states, q_quant, k, weights, **kwargs):
        self.calls.append(kwargs)
        return "upstream-cuda"


class _Upstream:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, self_obj, hidden_states, q_quant, k, weights, **kwargs):
        self.calls.append(kwargs)
        return "upstream"


_PIN_KEYS = {
    "self",
    "hidden_states",
    "q_quant",
    "k",
    "weights",
    "gate_score",
    "compress_ape",
    "index_kpool",
    "positions",
}


def _fp8_ppu():
    return SimpleNamespace(is_ppu=lambda: True, supports_fp8=lambda: True)


def test_forward_ppu_delegates_to_forward_cuda_on_fp8_ppu(monkeypatch) -> None:
    module = _patch_module()
    spy = _ForwardCudaSpy()
    monkeypatch.setattr(module, "current_platform", _fp8_ppu())
    monkeypatch.setattr(module, "_UPSTREAM_FORWARD_CUDA_KEYS", frozenset(_PIN_KEYS))
    q = torch.zeros(1)
    out = module.forward_ppu(spy, q, q, q, q, gate_score=None, topk_backend="triton")
    assert out == "upstream-cuda"
    assert spy.calls == [
        {
            "gate_score": None,
            "compress_ape": None,
            "index_kpool": 1,
            "positions": None,
        }
    ]


def test_forward_native_delegates_upstream_on_fp8_ppu(monkeypatch) -> None:
    module = _patch_module()
    upstream = _Upstream()
    monkeypatch.setattr(module, "current_platform", _fp8_ppu())
    monkeypatch.setattr(module, "_upstream_forward_native", upstream)
    monkeypatch.setattr(module, "_UPSTREAM_FORWARD_NATIVE_KEYS", frozenset(_PIN_KEYS))
    q = torch.zeros(1)
    assert (
        module.forward_native(object(), q, q, q, q, topk_backend="triton") == "upstream"
    )
    assert upstream.calls == [
        {
            "gate_score": None,
            "compress_ape": None,
            "index_kpool": 1,
            "positions": None,
        }
    ]


def test_fwht_redirect_delegates_original_on_fp8_ppu(monkeypatch) -> None:
    from vllm_sail.patch.enhancement.models import glm5_next as patch_module
    from vllm_sail.patch.utils import PATCH_MARKER

    sentinel = object()
    markers = getattr(patch_module.fwht128_quant_fp8, PATCH_MARKER)
    key = "vllm.models.glm5next.nvidia.ops.kpool_compress.fwht128_quant_fp8"
    monkeypatch.setitem(markers, key, lambda _q: sentinel)
    monkeypatch.setattr(
        type(patch_module.current_platform), "is_ppu", lambda self: True
    )
    monkeypatch.setattr(
        type(patch_module.current_platform),
        "supports_fp8",
        lambda self: True,
    )

    q = torch.zeros((2, 128), dtype=torch.bfloat16)
    assert patch_module.fwht128_quant_fp8(q) is sentinel
