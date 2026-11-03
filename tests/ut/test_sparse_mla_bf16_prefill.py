# SPDX-License-Identifier: Apache-2.0
"""Tests for the forced-BF16 sparse MLA prefill patch module.

The patch is a pure delegation to upstream while
``VLLM_SAIL_SPARSE_MLA_BF16_PREFILL`` is unset, so most of these tests pin that
"switch off means upstream" contract next to the switch-on behaviour.
"""

from __future__ import annotations

import importlib

import pytest

pytest.importorskip("torch", reason="requires torch")
pytest.importorskip("vllm", reason="requires vLLM")

import vllm_sail.patch as patch_pkg  # noqa: E402
from vllm_sail import envs as sail_envs  # noqa: E402
from vllm_sail.patch.enhancement.attention import (  # noqa: E402
    sparse_mla_bf16_prefill as plugin,
)
from vllm_sail.patch.utils import PATCH_MARKER, PATCH_REGISTRY  # noqa: E402

_BACKEND = importlib.import_module("vllm.v1.attention.backends.mla.flashmla_sparse")
_ENV = "VLLM_SAIL_SPARSE_MLA_BF16_PREFILL"
_TARGETS = (
    f"{plugin._MODULE}.use_fp8_mixed_batch",
    f"{plugin._MODULE}.FlashMLASparseImpl.__init__",
    f"{plugin._MODULE}.FlashMLASparseImpl.prefill_padding",
)


@pytest.fixture(scope="module")
def installed():
    import vllm_sail

    vllm_sail.register()
    patch_pkg.install()
    return PATCH_REGISTRY


@pytest.fixture(autouse=True)
def switch_off(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        _ENV,
        "VLLM_PPU_SPARSE_MLA_BF16_PREFILL",
    ):
        monkeypatch.delenv(name, raising=False)


def _new_impl():
    """A bare impl instance: the property under test needs no real __init__."""
    from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseImpl

    return object.__new__(FlashMLASparseImpl)


# ---------------------------------------------------------------------------
# Environment variable
# ---------------------------------------------------------------------------
def test_switch_defaults_off() -> None:
    assert sail_envs.VLLM_SAIL_SPARSE_MLA_BF16_PREFILL is False


# ---------------------------------------------------------------------------
# Patch wiring
# ---------------------------------------------------------------------------
def test_three_targets_are_registered(installed) -> None:
    registered = {record.target for record in installed}
    assert set(_TARGETS) <= registered


def test_markers_land_on_the_replacement(installed) -> None:
    markers = getattr(plugin.use_fp8_mixed_batch, PATCH_MARKER, None)
    assert isinstance(markers, dict) and _TARGETS[0] in markers
    markers = getattr(plugin.impl_init, PATCH_MARKER, None)
    assert isinstance(markers, dict) and _TARGETS[1] in markers
    markers = getattr(plugin._get_prefill_padding, PATCH_MARKER, None)
    assert isinstance(markers, dict) and _TARGETS[2] in markers


def test_prefill_padding_installs_a_data_descriptor(installed) -> None:
    descriptor = vars(_BACKEND.FlashMLASparseImpl)["prefill_padding"]
    assert isinstance(descriptor, property)
    assert descriptor.fset is not None, "the intrinsic assignment must be intercepted"


# ---------------------------------------------------------------------------
# switch off == upstream
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("num_heads", "use_hisparse", "pcp_dcp_kv_gather"),
    [(16, False, False), (32, False, False), (8, True, False), (8, False, True)],
)
def test_off_delegates_the_mixed_batch_heuristic(
    installed, num_heads: int, use_hisparse: bool, pcp_dcp_kv_gather: bool
) -> None:
    upstream_fn = plugin._upstream_use_fp8_mixed_batch
    args = (num_heads, use_hisparse, pcp_dcp_kv_gather)
    assert plugin.use_fp8_mixed_batch(*args) == upstream_fn(*args)


def test_off_keeps_the_upstream_prefill_padding(installed) -> None:
    impl = _new_impl()
    impl.num_heads = 16
    impl.prefill_padding = 64
    assert impl.prefill_padding == 64


def test_unset_padding_reads_as_upstream_default(installed) -> None:
    assert _new_impl().prefill_padding == 64


def test_off_leaves_shardq_alone(installed, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_upstream(self, *args, **kwargs):
        self.use_shardq = True

    monkeypatch.setattr(plugin, "_upstream_impl_init", fake_upstream)
    impl = _new_impl()
    plugin.impl_init(impl, 16, 576, 0.1, 1, None, None, "fp8_ds_mla", None, "", None)
    assert impl.use_shardq is True


# ---------------------------------------------------------------------------
# switch on
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("num_heads", [8, 16, 32, 64, 128])
def test_on_forces_the_separate_prefill_path(
    installed, monkeypatch: pytest.MonkeyPatch, num_heads: int
) -> None:
    monkeypatch.setenv(_ENV, "1")
    assert plugin.use_fp8_mixed_batch(num_heads, False, False) is False


@pytest.mark.parametrize(
    ("num_heads", "expected"),
    [(8, 16), (16, 16), (32, 32), (48, 64), (64, 64), (128, 64)],
)
def test_on_padding_follows_the_kernel_head_tile(
    installed, monkeypatch: pytest.MonkeyPatch, num_heads: int, expected: int
) -> None:
    monkeypatch.setenv(_ENV, "1")
    impl = _new_impl()
    impl.num_heads = num_heads
    impl.prefill_padding = 64
    assert impl.prefill_padding == expected


def test_on_clears_shardq(installed, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ENV, "1")

    def fake_upstream(self, *args, **kwargs):
        self.use_shardq = True

    monkeypatch.setattr(plugin, "_upstream_impl_init", fake_upstream)
    impl = _new_impl()
    plugin.impl_init(impl, 16, 576, 0.1, 1, None, None, "fp8_ds_mla", None, "", None)
    assert impl.use_shardq is False


# DCP-without-PCP keeps upstream's rejection: the metadata builder already
# refuses ``not fp8_use_mixed_batch and not pcp_dcp_kv_gather``, and the switch
# only flips the first flag, so no plugin-side guard is needed.
