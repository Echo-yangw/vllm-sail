# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the accuracy gate functions in ``_accuracy_gate``.

Schema v2 replaces per-token logprob goldens with a single accuracy score
from evalscope GSM8K evaluation.  This module tests every gate branch:

1. :func:`check_score_threshold` — primary ``score >= threshold`` gate.
2. :func:`check_accuracy_golden` — three-stage golden regression gate
   (completeness, hard floor, regression lower bound).
3. :func:`build_accuracy_golden` — golden record construction.
4. Naming helpers (``golden_filename``, ``pending_golden_filename``).
5. I/O round-trip (``write_golden``, ``read_golden``).

The module under test imports neither ``torch`` nor ``vllm``, so everything
here runs on a plain CPU interpreter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tests.e2e.models import _accuracy_gate as gate

# ---------------------------------------------------------------------------
# Constants.
# ---------------------------------------------------------------------------


def test_default_threshold_is_0_9() -> None:
    assert gate.DEFAULT_THRESHOLD == 0.9


def test_golden_schema_version_is_2() -> None:
    assert gate.GOLDEN_SCHEMA_VERSION == 2


def test_benchmark_constant() -> None:
    assert gate.BENCHMARK_GSM8K == "gsm8k"


def test_gsm8k_full_size() -> None:
    assert gate.GSM8K_FULL_SIZE == 1319


def test_default_tolerance_is_a_symmetric_band() -> None:
    t = gate.DEFAULT_TOLERANCE
    assert t["mode"] == "band"
    assert t["abs"] == 0.02
    # The threshold is the fallback for a checkpoint with no golden yet; it
    # must not also cap goldens that legitimately sit below it.
    assert "hard_floor" not in t


# ---------------------------------------------------------------------------
# File naming helpers.
# ---------------------------------------------------------------------------


def test_golden_filename_format() -> None:
    assert (
        gate.golden_filename("deepseek_v4_int8", "810e")
        == "accuracy_deepseek_v4_int8_810e.json"
    )


def test_pending_golden_filename_format() -> None:
    name = gate.pending_golden_filename("deepseek_v4_mxfp4", "890p")
    assert name.startswith("accuracy_deepseek_v4_mxfp4_890p")
    assert name.endswith(".pending-review.json")


# ---------------------------------------------------------------------------
# check_score_threshold.
# ---------------------------------------------------------------------------


def test_score_above_threshold_passes() -> None:
    ok, err = gate.check_score_threshold(0.95)
    assert ok is True
    assert err is None


def test_score_exactly_at_threshold_passes() -> None:
    ok, err = gate.check_score_threshold(0.9)
    assert ok is True
    assert err is None


def test_score_below_threshold_fails() -> None:
    ok, err = gate.check_score_threshold(0.89)
    assert ok is False
    assert err is not None
    assert "below threshold" in err


def test_custom_threshold() -> None:
    ok, _ = gate.check_score_threshold(0.8, threshold=0.7)
    assert ok is True
    ok2, err2 = gate.check_score_threshold(0.6, threshold=0.7)
    assert ok2 is False
    assert "0.7" in err2


def test_invalid_scores_are_rejected() -> None:
    for score in (float("nan"), float("inf"), float("-inf"), -0.1, 1.1):
        ok, err = gate.check_score_threshold(score)
        assert ok is False
        assert err == f"invalid accuracy score: {score!r}"


# ---------------------------------------------------------------------------
# build_accuracy_golden — schema v2 record construction.
# ---------------------------------------------------------------------------


#: The sampling contract every golden in these tests was recorded under.
_GENERATION: dict[str, Any] = {
    "temperature": 1.0,
    "top_p": 0.95,
    "max_tokens": 10240,
}


def _make_golden(
    score: float = 0.92,
    score_num: int | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Helper: build a golden record with sensible defaults."""
    kwargs: dict[str, Any] = {
        "model_key": "demo_model",
        "chip": "890p",
        "checkpoint": "/ckpt/v4/Model-W8A8",
        "dataset_cfg": {
            "dataset_id": "/data/gsm8k",
            "few_shot_num": 4,
            "few_shot_random": False,
            "subset_list": ["main"],
            "num_expected": 1319,
        },
        "serving_cfg": {
            "tensor_parallel_size": 8,
            "gpu_memory_utilization": 0.9,
            "seed": 0,
        },
        "generation_cfg": dict(_GENERATION),
        "score": score,
        "score_num": round(score * 1319) if score_num is None else score_num,
    }
    kwargs.update(overrides)
    return gate.build_accuracy_golden(**kwargs)


def test_build_golden_has_required_fields() -> None:
    record = _make_golden()
    assert record["schema_version"] == 2
    assert record["model"] == "demo_model"
    assert record["chip"] == "890p"
    assert record["benchmark"] == "gsm8k"
    assert record["score"] == 0.92
    # Correct answers, not samples asked.
    assert record["score_num"] == round(0.92 * 1319)
    assert record["generation"] == _GENERATION
    assert "measured_at" in record
    assert "tolerance" in record
    assert record["checkpoint_tail"] is not None


def test_build_golden_custom_tolerance() -> None:
    custom = {"mode": "lower_bound", "rel": 0.95, "abs": 0.05, "hard_floor": 0.8}
    record = _make_golden(tolerance=custom)
    assert record["tolerance"] == custom


# ---------------------------------------------------------------------------
# check_accuracy_golden — completeness, hard floor, regression.
# ---------------------------------------------------------------------------


def test_completeness_gate_passes_with_matching_count() -> None:
    golden = _make_golden(score=0.92, score_num=1319)
    ok, err = gate.check_accuracy_golden(0.91, 1319, golden)
    assert ok is True
    assert err is None


def test_completeness_gate_fails_on_count_mismatch() -> None:
    golden = _make_golden(score=0.92, score_num=1319)
    ok, err = gate.check_accuracy_golden(0.92, 1000, golden)
    assert ok is False
    assert "completeness" in err
    assert "1319" in err


def test_completeness_gate_skipped_when_num_expected_absent() -> None:
    """If num_expected is not in the golden dataset, completeness is skipped."""
    golden = _make_golden(score=0.92, score_num=1319)
    golden["dataset"].pop("num_expected", None)
    ok, err = gate.check_accuracy_golden(0.92, 500, golden)
    assert ok is True


def test_band_gate_fails_below_lower_bound() -> None:
    golden = _make_golden(score=0.92, score_num=1319)
    # band = [0.90, 0.94]
    ok, err = gate.check_accuracy_golden(0.85, 1319, golden)
    assert ok is False
    assert "band gate failed" in err


def test_band_gate_fails_above_upper_bound() -> None:
    golden = _make_golden(score=0.80, score_num=1319)
    # band = [0.78, 0.82]; a jump this large means the golden is stale.
    ok, err = gate.check_accuracy_golden(0.90, 1319, golden)
    assert ok is False
    assert "band gate failed" in err
    assert "0.8200" in err


def test_band_gate_passes_at_lower_bound() -> None:
    golden = _make_golden(score=0.92, score_num=1319)
    ok, err = gate.check_accuracy_golden(0.9, 1319, golden)
    assert ok is True
    assert err is None


def test_band_gate_passes_at_upper_bound() -> None:
    golden = _make_golden(score=0.92, score_num=1319)
    ok, err = gate.check_accuracy_golden(0.94, 1319, golden)
    assert ok is True
    assert err is None


def test_band_gate_ignores_global_threshold() -> None:
    """A golden below the fallback threshold still governs its checkpoint."""
    golden = _make_golden(score=0.60, score_num=1319)
    ok, err = gate.check_accuracy_golden(0.61, 1319, golden)
    assert ok is True
    assert err is None


def test_band_gate_fails_on_large_drop() -> None:
    golden = _make_golden(score=0.95, score_num=1319)
    # band = [0.93, 0.97]
    ok, err = gate.check_accuracy_golden(0.92, 1319, golden)
    assert ok is False
    assert "band gate failed" in err


def test_band_gate_passes_within_tolerance() -> None:
    golden = _make_golden(score=0.95, score_num=1319)
    # band = [0.93, 0.97]
    ok, err = gate.check_accuracy_golden(0.935, 1319, golden)
    assert ok is True
    assert err is None


def test_band_gate_uses_custom_tolerance() -> None:
    golden = _make_golden(score=0.95, score_num=1319)
    golden["tolerance"] = {"mode": "band", "abs": 0.01}
    # band = [0.94, 0.96]
    ok, err = gate.check_accuracy_golden(0.935, 1319, golden)
    assert ok is False
    assert "band gate failed" in err


def test_legacy_lower_bound_golden_still_gates() -> None:
    """Records written before the band keep their one-sided semantics."""
    golden = _make_golden(score=0.95, score_num=1319)
    golden["tolerance"] = {
        "mode": "lower_bound",
        "rel": 0.99,
        "abs": 0.02,
        "hard_floor": 0.9,
    }
    ok, err = gate.check_accuracy_golden(0.94, 1319, golden)
    assert ok is True
    assert err is None
    ok, err = gate.check_accuracy_golden(0.92, 1319, golden)
    assert ok is False
    assert "regression" in err


def test_gates_are_evaluated_in_order() -> None:
    """Completeness fires before the band."""
    golden = _make_golden(score=0.95, score_num=1319)
    # score=0.5 would fail both; completeness should be reported first.
    ok, err = gate.check_accuracy_golden(0.5, 100, golden)
    assert ok is False
    assert "completeness" in err


# ---------------------------------------------------------------------------
# check_generation_config — the golden and the run must sample alike.
# ---------------------------------------------------------------------------


def test_generation_gate_passes_on_the_same_contract() -> None:
    golden = _make_golden()
    assert gate.check_generation_config(golden, dict(_GENERATION)) == (True, None)


def test_generation_gate_tolerates_int_float_spelling() -> None:
    """A hand-edited ``1`` must compare equal to the recorded ``1.0``."""
    golden = _make_golden(
        generation_cfg={"temperature": 1, "max_tokens": 10240},
    )
    ok, err = gate.check_generation_config(
        golden, {"temperature": 1.0, "max_tokens": 10240.0}
    )
    assert ok is True
    assert err is None


def test_generation_gate_fails_on_a_drifted_value() -> None:
    golden = _make_golden()
    ok, err = gate.check_generation_config(golden, {**_GENERATION, "temperature": 0.0})
    assert ok is False
    assert "temperature" in err
    assert "run=0.0" in err


def test_generation_gate_fails_when_a_knob_is_added_or_lost() -> None:
    """Either direction of drift makes the two scores incomparable."""
    golden = _make_golden()
    ok, err = gate.check_generation_config(golden, {**_GENERATION, "top_k": 20})
    assert ok is False
    assert "top_k" in err
    assert "<absent>" in err

    ok, err = gate.check_generation_config(golden, {"temperature": 1.0})
    assert ok is False
    assert "top_p" in err
    assert "max_tokens" in err


def test_generation_gate_rejects_a_golden_without_the_block() -> None:
    """A golden that predates the contract cannot gate a run at all."""
    golden = _make_golden()
    golden.pop("generation")
    ok, err = gate.check_generation_config(golden, dict(_GENERATION))
    assert ok is False
    assert "--update-golden" in err


# ---------------------------------------------------------------------------
# I/O round-trip: write_golden → read_golden.
# ---------------------------------------------------------------------------


def test_write_and_read_golden_round_trip(tmp_path: Path) -> None:
    record = _make_golden()
    path = tmp_path / "golden" / "test.json"
    gate.write_golden(record, path)
    assert path.exists()
    loaded = gate.read_golden(path)
    assert loaded["schema_version"] == record["schema_version"]
    assert loaded["score"] == record["score"]
    assert loaded["model"] == record["model"]


def test_write_golden_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "nested" / "dir" / "golden.json"
    gate.write_golden({"test": 1}, path)
    assert path.exists()


# ---------------------------------------------------------------------------
# Integration: the accuracy test module exists and has the right marks.
# ---------------------------------------------------------------------------

MODELS_DIR = Path(__file__).resolve().parents[1] / "e2e" / "models"
ACCURACY_TEST = MODELS_DIR / "test_model_accuracy.py"


def test_accuracy_test_file_exists() -> None:
    assert ACCURACY_TEST.exists()


def test_accuracy_test_uses_correct_marks() -> None:
    text = ACCURACY_TEST.read_text()
    assert "pytest.mark.ppu" in text
    assert "pytest.mark.model_e2e" in text
    assert text.count("@pytest.mark.cap89") == 3


def test_accuracy_test_requires_eight_devices() -> None:
    text = ACCURACY_TEST.read_text()
    assert "PPU_CI_MIN_DEVICES = 8" in text


def test_accuracy_test_guards_against_no_device() -> None:
    text = ACCURACY_TEST.read_text()
    assert "importorskip" in text
    assert "is_ppu()" in text


def test_accuracy_test_references_eval_script() -> None:
    text = ACCURACY_TEST.read_text()
    assert "ppu_gsm8k_eval" in text
    assert "run_accuracy_eval" in text


def test_accuracy_test_uses_current_model_keys() -> None:
    text = ACCURACY_TEST.read_text()
    for model_key in (
        "minimax-m3-fp8",
        "qwen3.8-flash-next",
    ):
        assert model_key in text
    for old_key in ("minimax_m3", "kimi_k3", "qwen3_8_27b"):
        assert f'"{old_key}"' not in text


def test_no_other_model_test_files_remain() -> None:
    """Old per-model test files should have been deleted."""
    old_files = [
        "test_model_deepseek_v4.py",
        "test_model_kimi_k3.py",
        "test_model_minimax_m3.py",
        "test_model_qwen3_next.py",
    ]
    for name in old_files:
        assert not (MODELS_DIR / name).exists(), f"old model test {name} still exists"
