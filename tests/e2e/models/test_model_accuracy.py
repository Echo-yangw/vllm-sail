# SPDX-License-Identifier: Apache-2.0
"""GSM8K accuracy E2E tests for PPU models (server-client mode).

Each model in ``scripts/ci/ppu_e2e_models.json`` gets one test function
that starts ``vllm serve``, runs evalscope GSM8K, and checks the accuracy
score against a per-chip golden baseline.

evalscope is imported at **runtime** inside the eval script, not at
collection time, so ``pytest --collect-only`` succeeds even without the
package installed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

PPU_CI_MIN_DEVICES = 8

# --- Guard: skip the whole module without a real PPU device. ---------------
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip(
        "requires a real PPU/CUDA device",
        allow_module_level=True,
    )
pytest.importorskip("vllm")

from vllm.platforms import current_platform  # noqa: E402

if not current_platform.is_ppu():
    pytest.skip(
        "requires a PPU platform (is_ppu() is False)",
        allow_module_level=True,
    )

from tests.e2e.models._offline_runner import (  # noqa: E402
    chip_key_for_capability,
)

pytestmark = [pytest.mark.ppu, pytest.mark.model_e2e]

# --- Lazy-load the eval script by file path (no package on sys.path). ------
_EVAL_SCRIPT = (
    Path(__file__).resolve().parents[3] / "scripts" / "eval" / "ppu_gsm8k_eval.py"
)


def _load_eval() -> Any:
    spec = importlib.util.spec_from_file_location("ppu_gsm8k_eval", _EVAL_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_model(
    model_key: str,
    chip_capability: Any,
    golden_store: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Shared helper: resolve chip, call the eval pipeline."""
    eval_mod = _load_eval()
    chip = chip_key_for_capability(chip_capability)
    result = eval_mod.run_accuracy_eval(
        model_key,
        chip=chip,
        golden_dir=golden_store.root,
        update_golden=golden_store.update,
        **kwargs,
    )
    assert result["status"] in {
        "pass",
        "updated",
        "created",
    }, f"{model_key}: {result['status']} — {result.get('error', '')}"
    return result


# --- One test per model (matches ppu_e2e_models.json). ---


def test_deepseek_v4_int8_accuracy(chip_capability: Any, golden_store: Any) -> None:
    _run_model("deepseek_v4_int8", chip_capability, golden_store)


@pytest.mark.cap89
def test_deepseek_v4_mxfp4_accuracy(chip_capability: Any, golden_store: Any) -> None:
    _run_model("deepseek_v4_mxfp4", chip_capability, golden_store)


@pytest.mark.cap89
def test_minimax_m3_fp8_accuracy(chip_capability: Any, golden_store: Any) -> None:
    _run_model("minimax-m3-fp8", chip_capability, golden_store)


@pytest.mark.cap89
def test_qwen3_8_flash_next_accuracy(chip_capability: Any, golden_store: Any) -> None:
    _run_model("qwen3.8-flash-next", chip_capability, golden_store)
