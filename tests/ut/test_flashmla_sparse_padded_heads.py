# SPDX-License-Identifier: Apache-2.0
"""FP8 decode head-padding patch on ``FlashMLASparseImpl``.

Upstream pads the FP8 sparse decode kernel's query heads to a flat 64/128
(the NVIDIA kernel's constraint). The PPU kernel folds the heads into the
query length and picks its M tile (kBlockM = 16/32/64) from the folded
length, so the patch narrows the padding to the tile covering the actual
head count.

Everything here needs an importable vLLM; on a bare CPU runner the whole
module skips.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

pytest.importorskip("torch", reason="requires torch")
pytest.importorskip("vllm", reason="requires vLLM")

import vllm_sail.patch as patch_pkg  # noqa: E402
from tests.conftest import FakePPUPlatform  # noqa: E402
from vllm_sail.patch.utils import PATCH_MARKER, PATCH_REGISTRY  # noqa: E402

_MODULE = "vllm.v1.attention.backends.mla.flashmla_sparse"
_TARGET = f"{_MODULE}.FlashMLASparseImpl._compute_fp8_decode_padded_heads"


@pytest.fixture(scope="module")
def installed():
    import vllm_sail

    vllm_sail.register()
    patch_pkg.install()
    return PATCH_REGISTRY


@pytest.fixture
def backend_cls():
    from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseImpl

    return FlashMLASparseImpl


@pytest.fixture
def patch_module():
    from vllm_sail.patch.enhancement.attention import flashmla_sparse_backend

    return flashmla_sparse_backend


@pytest.fixture
def ppu_platform(monkeypatch, patch_module):
    platform = FakePPUPlatform("810e")
    monkeypatch.setattr(patch_module, "current_platform", platform)
    return platform


@pytest.fixture
def non_ppu_platform(monkeypatch, patch_module):
    platform = SimpleNamespace(is_ppu=lambda: False)
    monkeypatch.setattr(patch_module, "current_platform", platform)
    return platform


# ---------------------------------------------------------------------------
# Patch wiring
# ---------------------------------------------------------------------------
def test_patch_is_registered(installed) -> None:
    assert _TARGET in {record.target for record in installed}


def test_markers_land_on_the_replacement(installed) -> None:
    from vllm.v1.attention.backends.mla import flashmla_sparse

    replacement = inspect.getattr_static(
        flashmla_sparse.FlashMLASparseImpl, "_compute_fp8_decode_padded_heads"
    )
    assert isinstance(replacement, staticmethod)
    markers = getattr(replacement.__func__, PATCH_MARKER, None)
    assert isinstance(markers, dict) and _TARGET in markers


# ---------------------------------------------------------------------------
# Padding behavior
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "num_heads,expected",
    (
        (1, 16),
        (16, 16),
        (17, 32),
        (32, 32),
        (33, 64),
        (64, 64),
        (65, 128),
        (128, 128),
    ),
)
def test_ppu_pads_to_the_covering_tile(
    installed, ppu_platform, backend_cls, num_heads, expected
) -> None:
    assert backend_cls._compute_fp8_decode_padded_heads(num_heads) == expected


@pytest.mark.parametrize("num_heads", (1, 16, 32, 64))
def test_non_ppu_keeps_upstream_padding(
    installed, non_ppu_platform, patch_module, backend_cls, num_heads
) -> None:
    upstream = patch_module._upstream_fp8_decode_padded_heads
    assert backend_cls._compute_fp8_decode_padded_heads(num_heads) == upstream(
        num_heads
    )


def test_non_positive_heads_delegate_to_upstream(
    installed, ppu_platform, patch_module, backend_cls
) -> None:
    upstream = patch_module._upstream_fp8_decode_padded_heads
    assert backend_cls._compute_fp8_decode_padded_heads(0) == upstream(0)
