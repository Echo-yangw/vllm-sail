# SPDX-License-Identifier: Apache-2.0
"""The PPU MXFP4 indexer cache stays opt-in on the v0.31 line.

The plugin replaces upstream's ``dsa_indexer_uses_fp4`` because MXFP4 on PPU
comes from SAIL DeepGEMM rather than CUDA SM100. Unlike main, this branch keeps
upstream's ``indexer_kv_dtype="auto"`` default of fp8 on every capability: the
MXFP4 cache is only selected when the server is started with
``--attention-config '{"indexer_kv_dtype": "mxfp4"}'`` (sm_89 only). Exercised
here without torch, vLLM or a device by executing the patch's own function body.
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest

PATCH_PATH = (
    Path(__file__).parents[2] / "vllm_sail/patch/enhancement/attention/mla_indexer.py"
)
FUNCTION = "dsa_indexer_uses_fp4"


class _AttentionConfig:
    def __init__(self, indexer_kv_dtype: str) -> None:
        self.indexer_kv_dtype = indexer_kv_dtype

    def resolve_indexer_kv_dtype(self, default: str) -> str:
        # Same contract as vllm.config.attention.AttentionConfig.
        return default if self.indexer_kv_dtype == "auto" else self.indexer_kv_dtype


def _config(indexer_kv_dtype: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(attention_config=_AttentionConfig(indexer_kv_dtype))


def _install_platform(monkeypatch, capability: int, *, ppu: bool = True):
    platform = types.SimpleNamespace(
        is_ppu=lambda: ppu,
        is_device_capability=lambda cap: capability == cap,
    )
    vllm = types.ModuleType("vllm")
    vllm.__path__ = []
    platforms = types.ModuleType("vllm.platforms")
    platforms.current_platform = platform
    vllm.platforms = platforms
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
    return platform


def _load(monkeypatch, capability: int, *, ppu: bool = True):
    _install_platform(monkeypatch, capability, ppu=ppu)
    tree = ast.parse(PATCH_PATH.read_text(encoding="utf-8"), filename=str(PATCH_PATH))
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == FUNCTION
    ]
    assert len(definitions) == 1, f"{PATCH_PATH}: expected one {FUNCTION} definition"
    definition = definitions[0]
    # The patch decorator is the registration, not the behaviour under test.
    definition.decorator_list = []
    calls = []

    def upstream_uses_fp4(vllm_config):
        calls.append(vllm_config)
        return "upstream"

    namespace = {
        "_upstream_uses_fp4": upstream_uses_fp4,
        "_indexer_module": types.SimpleNamespace(
            DSA_INDEXER_KV_DTYPES=("fp8", "mxfp4")
        ),
    }
    module = ast.Module(body=[definition], type_ignores=[])
    exec(compile(module, str(PATCH_PATH), "exec"), namespace)
    return namespace[FUNCTION], calls


@pytest.mark.parametrize(
    "capability,configured,expected",
    [
        # "auto" keeps upstream's fp8 default, on sm_89 as well as everywhere
        # else: starting the server without the indexer arg must not switch the
        # cache format.
        (89, "auto", False),
        (80, "auto", False),
        # The MXFP4 cache is reachable only through the explicit config value.
        (89, "mxfp4", True),
        # An explicit fp8 always wins, whatever the capability.
        (89, "fp8", False),
        (80, "fp8", False),
    ],
)
def test_indexer_cache_dtype_resolution(monkeypatch, capability, configured, expected):
    uses_fp4, calls = _load(monkeypatch, capability)
    assert uses_fp4(_config(configured)) is expected
    assert calls == []


def test_mxfp4_stays_rejected_below_sm_89(monkeypatch):
    uses_fp4, _ = _load(monkeypatch, 80)
    with pytest.raises(ValueError, match="requires capability sm_89"):
        uses_fp4(_config("mxfp4"))


def test_non_ppu_delegates_to_upstream(monkeypatch):
    uses_fp4, calls = _load(monkeypatch, 100, ppu=False)
    config = _config("auto")
    assert uses_fp4(config) == "upstream"
    assert calls == [config]
