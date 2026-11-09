# SPDX-License-Identifier: Apache-2.0
"""Tests for patch installation as a whole.

The most important test here is :func:`test_patches_actually_land`. vLLM-MetaX
ships a patch package whose ``__init__`` does ``from utils import patch`` — an
absolute import that raises ``ModuleNotFoundError`` on a clean interpreter, so
none of its patches install. That failure is invisible unless something asserts
the patches took effect. This module is that assertion for us.
"""

from __future__ import annotations

import sys

import pytest

import vllm_sail.patch as patch_pkg


def test_importing_the_package_has_no_side_effects() -> None:
    """Importing vllm_sail.patch must not install anything.

    The whole restructure away from install-on-import depends on this, and it is
    what makes `from vllm_sail.patch.utils import patch` safe inside a patch module.
    """
    assert "vllm_sail.patch" in sys.modules
    # No vLLM in this environment, so if importing had installed patches the
    # import above would have raised. Reaching here at all proves the point;
    # assert the flag too so the intent is explicit.
    if "vllm" not in sys.modules:
        assert not patch_pkg.installed()


def test_utils_importable_without_vllm() -> None:
    """The patch framework itself must not require vLLM."""
    from vllm_sail.patch.utils import PATCH_REGISTRY, patch, patch_value

    assert callable(patch)
    assert callable(patch_value)
    assert isinstance(PATCH_REGISTRY, list)


def test_all_categories_are_declared() -> None:
    """Every category directory must be in the documented apply order."""
    import pathlib

    package_dir = pathlib.Path(patch_pkg.__file__).parent
    on_disk = {
        child.name
        for child in package_dir.iterdir()
        if child.is_dir()
        and (child / "__init__.py").exists()
        and child.name != "template"
        and not child.name.startswith("_")
    }
    assert on_disk == set(patch_pkg._CATEGORIES), (
        "A patch category exists on disk but is not in _CATEGORIES, so its "
        "patches would never be applied."
    )


def test_bugfix_category_imports_cleanly() -> None:
    """bugfix defers target imports until install(), so bare CPU imports work.

    `performance` gained the Phase-4 FLA patches, whose patch modules import
    their vLLM targets at import time, so it moved into the vLLM-gated test
    below (as the previous docstring of this test instructed).
    """
    import importlib

    importlib.import_module("vllm_sail.patch.bugfix")


# ---------------------------------------------------------------------------
# Everything below needs a real vLLM to patch against, so it is gated on the
# `installed_patches` fixture rather than a module-level skip: the tests above
# must keep running on a bare CPU runner.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def installed_patches():
    """Install every patch against a real vLLM, or skip."""
    pytest.importorskip("torch", reason="requires torch")
    pytest.importorskip("vllm", reason="requires vLLM")

    patch_pkg.install()
    return patch_pkg.PATCH_REGISTRY


def test_install_is_idempotent(installed_patches) -> None:
    """register_out_of_tree runs in every process and may run twice.

    A second install() must be a no-op, not a RuntimeError from the
    double-patch guard.
    """
    count = len(installed_patches)
    patch_pkg.install()
    patch_pkg.install()
    assert len(patch_pkg.PATCH_REGISTRY) == count
    assert patch_pkg.installed()


def test_performance_category_imports_with_vllm(installed_patches) -> None:
    """performance holds the Phase-4 FLA patches and needs vLLM to import.

    Moved here from the bare-CPU test above when the FLA patches landed; the
    installed_patches fixture already ran install(), which imports the
    category, so this asserts the import stays clean when repeated.
    """
    import importlib

    importlib.import_module("vllm_sail.patch.performance")


def test_patches_actually_land(installed_patches) -> None:
    """Assert each Phase-2 patch is present on its real upstream target."""
    from vllm.model_executor.custom_op import CustomOp
    from vllm.platforms.interface import Platform
    from vllm.utils import import_utils

    from vllm_sail.patch.utils import PATCH_MARKER

    # Additive predicates exist.
    assert hasattr(Platform, "is_ppu")
    assert hasattr(CustomOp, "forward_ppu")

    # Replacements carry the marker recording what they replaced.
    assert getattr(import_utils.has_cutedsl, PATCH_MARKER, None) is not None
    assert getattr(import_utils.has_humming, PATCH_MARKER, None) is not None
    assert getattr(CustomOp.dispatch_forward, PATCH_MARKER, None) is not None


def test_every_patch_has_metadata(installed_patches) -> None:
    for record in installed_patches:
        assert record.reason.strip(), record.target
        assert record.affected_versions.strip(), record.target
        assert record.remove_when.strip(), record.target
        # "when upstream fixes it" is not a verifiable condition.
        assert record.remove_when.strip().lower() != "todo", record.target


def test_is_ppu_false_for_non_ppu_platforms(installed_patches) -> None:
    """The additive predicate must be safe for every platform, not just PPU."""
    from vllm.platforms.interface import Platform

    class _NotPPU(Platform):
        device_name = "cuda"

    assert Platform.is_ppu(_NotPPU()) is False


def test_is_ppu_true_for_the_ppu_platform(installed_patches) -> None:
    """PPUPlatform's identity predicates -- the core of decision D1.

    Importing vllm_sail.platform pulls in vllm.platforms.cuda, which imports
    vLLM's compiled `_C_stable_libtorch` extension. A pure-source vLLM checkout
    does not have it, so skip rather than fail: the absence is an environment
    property, not a defect in this plugin.
    """
    try:
        from vllm_sail.platform import PPUPlatform
    except ModuleNotFoundError as exc:
        if "_C" not in str(exc):
            raise
        pytest.skip(f"vLLM compiled extensions unavailable: {exc}")

    platform = PPUPlatform()
    assert platform.is_ppu() is True
    # The compatibility declaration that makes upstream's CUDA code paths work.
    assert platform.is_cuda() is True
    assert platform.is_cuda_alike() is True
    # False is required: it is what keeps the four upstream is_out_of_tree()
    # short-circuits from pre-empting the CUDA logic PPU needs.
    assert platform.is_out_of_tree() is False
    assert platform.is_sleep_mode_available() is True
