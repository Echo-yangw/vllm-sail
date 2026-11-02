# SPDX-License-Identifier: Apache-2.0
"""A dense Kimi K3 DSpark draft must decode with TRITON_MLA on PPU.

Upstream's ``configure_dense_draft`` pins a dense EAGLE3/DSpark MLA draft to
``FLASHINFER_MLA`` on an fp8 KV cache, because the draft keeps plain fp8 rows at
the head of the target's merged pages instead of owning a cache. PPU can honour
neither half of that pin: FlashInfer is absent and the dense FlashMLA has no
FP8 decode path. ``PPUPlatform.apply_config_platform_defaults`` therefore runs
after the model config hooks and moves the draft onto ``TRITON_MLA``, leaving
the KV format, the page binding and ``kv_cache_dtype="fp8"`` untouched.

Exercised here without torch, vLLM or a device by executing the hook's own
function body.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

PLATFORM_PATH = Path(__file__).parents[2] / "vllm_sail" / "platform.py"
FUNCTION = "_retarget_dense_draft_backend"


class _Enum:
    """Stand-in for AttentionBackendEnum (identity comparisons are the contract)."""

    FLASHINFER_MLA = "FLASHINFER_MLA"
    TRITON_MLA = "TRITON_MLA"


def _parse() -> ast.Module:
    return ast.parse(PLATFORM_PATH.read_text(encoding="utf-8"), filename=str(PLATFORM_PATH))


def _load_hook():
    definitions = [
        node
        for node in _parse().body
        if isinstance(node, ast.FunctionDef) and node.name == FUNCTION
    ]
    assert len(definitions) == 1, f"{PLATFORM_PATH}: expected one {FUNCTION} definition"
    namespace = {"AttentionBackendEnum": _Enum}
    module = ast.Module(body=definitions, type_ignores=[])
    exec(compile(module, str(PLATFORM_PATH), "exec"), namespace)
    return namespace[FUNCTION]


def _spec(backend, kv_cache_dtype: str) -> SimpleNamespace:
    return SimpleNamespace(attention_backend=backend, kv_cache_dtype=kv_cache_dtype)


@pytest.mark.parametrize("kv_cache_dtype", ["fp8", "fp8_e4m3"])
def test_dense_mla_draft_switches_to_triton_mla(kv_cache_dtype: str) -> None:
    spec = _spec(_Enum.FLASHINFER_MLA, kv_cache_dtype)
    _load_hook()(spec)
    assert spec.attention_backend == _Enum.TRITON_MLA
    # The dtype is upstream's decision and must survive the retarget.
    assert spec.kv_cache_dtype == kv_cache_dtype


def test_repeated_hook_runs_are_idempotent() -> None:
    # VllmConfig post-init runs on every replace(), including DSpark's draft
    # config, so the retarget must be a no-op the second time around.
    hook = _load_hook()
    spec = _spec(_Enum.FLASHINFER_MLA, "fp8")
    hook(spec)
    hook(spec)
    assert spec.attention_backend == _Enum.TRITON_MLA


@pytest.mark.parametrize("backend", [None, _Enum.TRITON_MLA, "FLASHMLA"])
def test_other_backends_are_untouched(backend) -> None:
    spec = _spec(backend, "fp8")
    _load_hook()(spec)
    assert spec.attention_backend == backend


def test_non_fp8_draft_cache_is_untouched() -> None:
    # configure_dense_draft rejects this pairing before the platform hook runs,
    # so the hook must not paper over a GQA draft's own cache.
    spec = _spec(_Enum.FLASHINFER_MLA, "auto")
    _load_hook()(spec)
    assert spec.attention_backend == _Enum.FLASHINFER_MLA


def test_platform_hook_wires_the_retarget() -> None:
    platform_class = next(
        node
        for node in _parse().body
        if isinstance(node, ast.ClassDef) and node.name == "PPUPlatform"
    )
    hook = next(
        node
        for node in platform_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "apply_config_platform_defaults"
    )
    source = ast.unparse(hook)
    assert f"{FUNCTION}(vllm_config.speculative_config)" in source
