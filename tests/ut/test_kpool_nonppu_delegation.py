# SPDX-License-Identifier: Apache-2.0
"""Non-PPU delegation tests for the kpool ``forward_native`` patch.

The pin (db9527a46) upstream ``SparseAttnIndexerKpool.forward_native`` has no
``topk_backend`` keyword; newer fork builds do. The patch must keep its own
signature stable while forwarding only the keywords the live upstream accepts,
otherwise enabling vllm-sail on NVIDIA raises ``TypeError`` in the eager path.
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


class _Upstream:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, self_obj, hidden_states, q_quant, k, weights, **kwargs):
        self.calls.append(kwargs)
        return "upstream"


def _call(module, monkeypatch, upstream: _Upstream, keys, topk_backend="triton"):
    monkeypatch.setattr(module, "_upstream_forward_native", upstream)
    monkeypatch.setattr(module, "_UPSTREAM_FORWARD_NATIVE_KEYS", frozenset(keys))
    monkeypatch.setattr(
        module, "current_platform", SimpleNamespace(is_ppu=lambda: False)
    )
    q = torch.zeros(1)
    return module.forward_native(object(), q, q, q, q, topk_backend=topk_backend)


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


def test_non_ppu_omits_topk_backend_when_upstream_lacks_it(monkeypatch) -> None:
    module = _patch_module()
    upstream = _Upstream()
    assert _call(module, monkeypatch, upstream, _PIN_KEYS) == "upstream"
    assert upstream.calls == [
        {
            "gate_score": None,
            "compress_ape": None,
            "index_kpool": 1,
            "positions": None,
        }
    ]


def test_non_ppu_forwards_topk_backend_when_upstream_accepts_it(monkeypatch) -> None:
    module = _patch_module()
    upstream = _Upstream()
    keys = _PIN_KEYS | {"topk_backend"}
    assert _call(module, monkeypatch, upstream, keys) == "upstream"
    assert upstream.calls[0]["topk_backend"] == "triton"
