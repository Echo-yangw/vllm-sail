# SPDX-License-Identifier: Apache-2.0
"""QSA launch-config table regression tests without Triton, torch or vLLM.

Guards the two measured regressions in upstream's GB300 default table: the
``base_programs > 2048`` prefill entry, and the ``base_programs <= 256`` band
that production decode (attention TP=1, ``num_kv_heads = 2``) reaches at up to
32 concurrent sequences with MTP=3.
"""

from __future__ import annotations

import ast
import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest

MODULE = Path(__file__).parents[2] / "vllm_sail/patch/performance/qsa_select_config.py"
FAKE_OPS = "_test_qsa_ops"

#: Qwen3.8-Flash-Next: topk 2048 + compress_ratio 4 - 1, the selection width
#: ``qsa_sparse_paged_attention`` passes as ``num_columns``.
NUM_COLUMNS = 2051


@pytest.fixture
def qsa_patch(monkeypatch, patch_utils_module):
    """Load the patch module against a fake upstream ops module."""
    logger = types.ModuleType("vllm.logger")
    logger.init_logger = logging.getLogger
    platforms = types.ModuleType("vllm.platforms")
    platforms.current_platform = types.SimpleNamespace(is_ppu=lambda: True)
    ops = types.ModuleType(FAKE_OPS)
    ops._select_config = lambda *args: ("upstream", args)
    for name, module in {
        "vllm.logger": logger,
        "vllm.platforms": platforms,
        "vllm_sail.patch.utils": patch_utils_module,
        FAKE_OPS: ops,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("_test_qsa_select_config", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._QSA_OPS_MODULE = FAKE_OPS
    module._on_ppu.cache_clear()
    return module, ops, platforms


def test_tier_table_matches_the_measured_values(qsa_patch):
    """Every tier, as (block_n, num_warps, num_tiles, num_splits)."""
    module, _, _ = qsa_patch
    select = module.ppu_select_config
    # base_programs = num_rows * num_kv_heads; num_kv_heads=2 is TP=1 decode.
    expected = {
        4: (16, 4, 129, 64),
        8: (16, 4, 129, 32),
        16: (16, 4, 129, 32),
        31: (16, 4, 129, 32),
        32: (64, 2, 33, 8),
        64: (64, 2, 33, 8),
        128: (64, 2, 33, 8),
        160: (64, 2, 33, 8),
        256: (64, 2, 33, 8),
        257: (64, 2, 33, 4),
        512: (64, 2, 33, 4),
        513: (64, 2, 33, 1),
        2048: (64, 2, 33, 1),
        4096: (64, 2, 33, 1),
        8192: (64, 2, 33, 1),
    }
    for base_programs, want in expected.items():
        assert select(base_programs, 1, False, NUM_COLUMNS) == want, base_programs


@pytest.mark.parametrize("use_prefill_config", [False, True])
@pytest.mark.parametrize("is_fp8", [False, True])
def test_regressed_bands_never_use_block_n_32_or_one_warp(
    qsa_patch, use_prefill_config, is_fp8
):
    """The two upstream entries that measured 1.5-2.7x slow on Zhenwu M890."""
    module, _, _ = qsa_patch
    for base_programs in (8, 16, 32, 64, 128, 160, 256, 4096, 8192):
        block_n, num_warps, num_tiles, num_splits = module.ppu_select_config(
            base_programs, 1, use_prefill_config, NUM_COLUMNS, is_fp8
        )
        assert block_n != 32, base_programs
        assert num_warps != 1, base_programs
        # num_tiles must follow the chosen block_n, not upstream's.
        assert num_tiles == -(-NUM_COLUMNS // block_n)
        assert 1 <= num_splits <= num_tiles


def test_prefill_and_decode_share_one_table(qsa_patch):
    """>2048 is where upstream splits on use_prefill_config; PPU does not."""
    module, _, _ = qsa_patch
    for base_programs in (8, 256, 8192):
        decode = module.ppu_select_config(base_programs, 2, False, NUM_COLUMNS)
        prefill = module.ppu_select_config(base_programs, 2, True, NUM_COLUMNS)
        assert decode == prefill
    assert module.ppu_select_config(8192, 1, True, NUM_COLUMNS) == (64, 2, 33, 1)


def test_splits_are_clamped_by_tiles_not_target(qsa_patch):
    """Upstream's invariant: never more splits than tiles, never empty."""
    module, _, _ = qsa_patch
    # 3 columns -> 1 tile at block_n=16, so the 64-split target collapses to 1.
    assert module.ppu_select_config(4, 1, False, 3) == (16, 4, 1, 1)
    assert module.ppu_select_config(0, 1, False, 0)[2:] == (0, 0)


def test_small_profile_limit_is_four_not_eight(qsa_patch):
    """0.28 keyed this on block_m, which _select_config is not passed."""
    module, _, _ = qsa_patch
    assert module._SMALL_PROFILE_LIMIT == 4
    # base_programs 5..8 take the <32 branch, not the <=limit one.
    for base_programs in (5, 6, 7, 8):
        assert module.ppu_select_config(base_programs, 1, False, NUM_COLUMNS)[3] == 32
    assert module.ppu_select_config(4, 1, False, NUM_COLUMNS)[3] == 64


def test_install_replaces_select_config_and_records_the_original(qsa_patch):
    module, ops, _ = qsa_patch
    original = ops._select_config

    module.install()

    installed = ops._select_config
    assert installed is not original
    assert installed(8192, 1, True, NUM_COLUMNS) == (64, 2, 33, 1)
    marker = getattr(installed, "__vllm_sail_patch__", None)
    assert marker is not None
    assert marker[f"{FAKE_OPS}._select_config"] is original
    assert module._installed is True


def test_install_delegates_to_upstream_off_ppu(qsa_patch):
    module, ops, platforms = qsa_patch
    platforms.current_platform.is_ppu = lambda: False
    original = ops._select_config

    module.install()

    args = (128, 2, False, NUM_COLUMNS, False)
    assert ops._select_config(*args) == ("upstream", args)
    assert original(*args) == ("upstream", args)


def test_install_is_idempotent_but_the_framework_rejects_reapplication(qsa_patch):
    module, ops, _ = qsa_patch
    module.install()
    installed = ops._select_config

    module.install()
    assert ops._select_config is installed

    # The guard is what makes that safe; without it the framework must refuse.
    module._installed = False
    with pytest.raises(RuntimeError, match="already patched"):
        module.install()


def test_unimportable_ops_module_warns_and_installs_nothing(qsa_patch, caplog):
    module, ops, _ = qsa_patch
    module._QSA_OPS_MODULE = "_test_qsa_ops_absent"
    original = ops._select_config

    with caplog.at_level(logging.WARNING):
        module.install()

    assert ops._select_config is original
    assert module._installed is False
    assert "Skipping the QSA launch-config patch" in caplog.text


def test_module_scope_has_no_vllm_imports():
    """CI runs tests/ut without torch, vLLM or a device."""
    for node in ast.parse(MODULE.read_text()).body:
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] != "vllm" for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "vllm"


@pytest.mark.upstream_source
def test_upstream_still_misses_ppu_and_still_tunes_for_gb300(upstream_source_root):
    """The patch is dead code if upstream grows a PPU branch or retunes."""
    source = (
        upstream_source_root / "vllm/models/qwen4_exp/nvidia/ops/qsa.py"
    ).read_text()
    # Exact-equality capability checks that Zhenwu M890's (8,9) satisfies
    # neither of.
    assert "get_device_capability() == (12, 0)" in source
    assert "get_device_capability() == (9, 0)" in source
    # The regressed prefill value must still be there, and no PPU branch added.
    assert "(32, 1, 1) if use_prefill_config else (64, 1, 2)" in source
    assert "is_ppu" not in source
