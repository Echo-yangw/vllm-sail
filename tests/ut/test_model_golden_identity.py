# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for golden identity binding and the pending-review flow.

Covers the schema-v2 accuracy golden behaviours:

1. **Identity binding** — :func:`check_golden_identity` verifies chip,
   checkpoint tail, and benchmark before any numeric comparison.
2. **Full golden lifecycle** — build → write → read → compare round-trip.
3. **Pending-review separation** — naming and promotion semantics.
4. **Checkpoint identity helpers** — :func:`_checkpoint_tail`,
   :func:`_quantization_info`, :func:`_checkpoint_identity` from the
   shared runner.

The module under test imports neither ``torch`` nor ``vllm``, so everything
here runs on a plain CPU interpreter.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

from tests.e2e.models import _accuracy_gate as gate
from tests.e2e.models import _offline_runner as runner

CHECKPOINT_A = "/nas_aisw/datasets/checkpoints/LLM/DeepSeek/v4/DeepSeek-V4"
CHECKPOINT_B = (
    "/nas_aisw/datasets/checkpoints/LLM/DeepSeek/v4/DeepSeek-V4-W8A8-INT8-Flash"
)
CHIP = "890p"
EVAL_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "eval" / "ppu_gsm8k_eval.py"
)


def _load_evaluator():
    spec = importlib.util.spec_from_file_location("ppu_gsm8k_eval", EVAL_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # GSM8K_DATASET_PATH is required at module level; supply a
    # placeholder so import succeeds in CPU-only UT.
    _sentinel = "GSM8K_DATASET_PATH" not in os.environ
    if _sentinel:
        os.environ["GSM8K_DATASET_PATH"] = "/unused/gsm8k"
    try:
        spec.loader.exec_module(module)
    finally:
        if _sentinel:
            del os.environ["GSM8K_DATASET_PATH"]
    return module


#: The sampling contract every golden in these tests was recorded under.
_GENERATION: dict[str, Any] = {
    "temperature": 1.0,
    "top_p": 0.95,
    "max_tokens": 10240,
}


def _make_golden(
    chip: str = CHIP,
    checkpoint: str = CHECKPOINT_A,
    score: float = 0.92,
    **overrides: Any,
) -> dict[str, Any]:
    """Build a schema-v2 golden record with sensible defaults."""
    kwargs: dict[str, Any] = {
        "model_key": "demo_model",
        "chip": chip,
        "checkpoint": checkpoint,
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
        # Correct answers, i.e. round(score * num).
        "score_num": round(score * 1319),
    }
    kwargs.update(overrides)
    return gate.build_accuracy_golden(**kwargs)


# ---------------------------------------------------------------------------
# Checkpoint identity helpers (from _offline_runner).
# ---------------------------------------------------------------------------


def test_checkpoint_tail_keeps_last_two_segments() -> None:
    assert runner._checkpoint_tail(CHECKPOINT_B) == "v4/DeepSeek-V4-W8A8-INT8-Flash"
    assert runner._checkpoint_tail("relative/ckpt") == "relative/ckpt"
    assert runner._checkpoint_tail("ckpt") == "ckpt"
    assert runner._checkpoint_tail(None) is None


def test_current_model_keys_bind_to_distinct_checkpoint_identities() -> None:
    checkpoints = {
        "minimax-m3-fp8": (
            "/ppusw/datasets/checkpoints/LLM/T-HEAD/m3/"
            "MiniMax-M3-Quant-W-FP8-PerChannel-A-FP8-PerToken"
        ),
        "qwen3.8-flash-next": (
            "/ppusw/datasets/checkpoints/LLM/Qwen/v3.8/Qwen3.8-Flash-Next"
        ),
    }
    identities = set()
    for model_key, checkpoint in checkpoints.items():
        golden = _make_golden(checkpoint=checkpoint, model_key=model_key)
        checkpoint_tail = runner._checkpoint_tail(checkpoint)
        identities.add(checkpoint_tail)
        assert golden["model"] == model_key
        assert golden["checkpoint_tail"] == checkpoint_tail
        assert gate.check_golden_identity(
            golden, chip=CHIP, checkpoint_tail=checkpoint_tail
        ) == (True, None)
    assert len(identities) == len(checkpoints)


class _FakeHFConfig:
    quantization_config = {"quant_method": "w8a8_int8"}


class _FakeModelConfig:
    hf_config = _FakeHFConfig()
    quantization = "w8a8_int8"


class _FakeEngine:
    model_config = _FakeModelConfig()


class _FakeLLM:
    llm_engine = _FakeEngine()


class _ExplodingEngine:
    @property
    def model_config(self) -> Any:
        raise RuntimeError("engine introspection blew up")


class _ExplodingLLM:
    llm_engine = _ExplodingEngine()


def test_quantization_probe_is_best_effort() -> None:
    info = runner._quantization_info(_FakeLLM())
    assert info is not None
    assert info["quant_method"] == "w8a8_int8"
    assert info["hf_quantization_config"] == {"quant_method": "w8a8_int8"}
    assert runner._quantization_info(None) is None
    assert runner._quantization_info(_ExplodingLLM()) is None
    assert runner._quantization_info(object()) is None


# ---------------------------------------------------------------------------
# check_golden_identity — chip / checkpoint_tail / benchmark.
# ---------------------------------------------------------------------------


def test_identity_match_passes() -> None:
    golden = _make_golden()
    tail = runner._checkpoint_tail(CHECKPOINT_A)
    ok, err = gate.check_golden_identity(golden, chip=CHIP, checkpoint_tail=tail)
    assert ok is True
    assert err is None


def test_identity_chip_mismatch_fails() -> None:
    golden = _make_golden(chip="890p")
    tail = runner._checkpoint_tail(CHECKPOINT_A)
    ok, err = gate.check_golden_identity(golden, chip="810e", checkpoint_tail=tail)
    assert ok is False
    assert "chip mismatch" in err
    assert "810e" in err


def test_identity_checkpoint_mismatch_fails() -> None:
    golden = _make_golden(checkpoint=CHECKPOINT_A)
    tail_b = runner._checkpoint_tail(CHECKPOINT_B)
    ok, err = gate.check_golden_identity(golden, chip=CHIP, checkpoint_tail=tail_b)
    assert ok is False
    assert "checkpoint mismatch" in err


def test_identity_benchmark_mismatch_fails() -> None:
    golden = _make_golden()
    golden["benchmark"] = "humaneval"
    tail = runner._checkpoint_tail(CHECKPOINT_A)
    ok, err = gate.check_golden_identity(
        golden, chip=CHIP, checkpoint_tail=tail, benchmark="gsm8k"
    )
    assert ok is False
    assert "benchmark mismatch" in err


def test_identity_missing_checkpoint_tail_is_ok() -> None:
    """A golden without checkpoint_tail should pass identity."""
    golden = _make_golden()
    golden.pop("checkpoint_tail", None)
    ok, err = gate.check_golden_identity(
        golden, chip=CHIP, checkpoint_tail="something/else"
    )
    assert ok is True


def test_identity_missing_benchmark_is_ok() -> None:
    """A golden without benchmark field should pass identity."""
    golden = _make_golden()
    golden.pop("benchmark", None)
    ok, err = gate.check_golden_identity(
        golden,
        chip=CHIP,
        checkpoint_tail=runner._checkpoint_tail(CHECKPOINT_A),
    )
    assert ok is True


# ---------------------------------------------------------------------------
# Full golden lifecycle: build → write → read → compare.
# ---------------------------------------------------------------------------


def test_golden_round_trip(tmp_path: Path) -> None:
    record = _make_golden(score=0.93)
    path = tmp_path / "golden" / "demo_model_890p.json"
    gate.write_golden(record, path)
    assert path.exists()

    loaded = gate.read_golden(path)
    assert loaded["schema_version"] == 2
    assert loaded["chip"] == CHIP
    assert loaded["score"] == 0.93
    assert loaded["checkpoint"] == CHECKPOINT_A
    assert loaded["checkpoint_tail"] is not None


def test_golden_comparison_passes_on_matching_score(
    tmp_path: Path,
) -> None:
    golden = _make_golden(score=0.93)
    path = tmp_path / "golden" / "demo.json"
    gate.write_golden(golden, path)
    loaded = gate.read_golden(path)

    ok, err = gate.check_accuracy_golden(0.93, 1319, loaded)
    assert ok is True


def test_golden_comparison_fails_on_regression(
    tmp_path: Path,
) -> None:
    golden = _make_golden(score=0.95)
    path = tmp_path / "golden" / "demo.json"
    gate.write_golden(golden, path)
    loaded = gate.read_golden(path)

    ok, err = gate.check_accuracy_golden(0.92, 1319, loaded)
    assert ok is False
    assert "band gate failed" in err


# ---------------------------------------------------------------------------
# Pending-review naming and separation.
# ---------------------------------------------------------------------------


def test_pending_path_naming() -> None:
    assert runner._pending_path(Path("/g/demo_model_890p.json")) == Path(
        "/g/demo_model_890p.pending-review.json"
    )


def test_pending_golden_filename_contains_suffix() -> None:
    name = gate.pending_golden_filename("demo_model", "890p")
    assert name == ("accuracy_demo_model_890p.pending-review.json")


def test_golden_filename_is_formal() -> None:
    name = gate.golden_filename("demo_model", "890p")
    assert name == "accuracy_demo_model_890p.json"
    assert not name.endswith(runner.PENDING_REVIEW_SUFFIX)


def test_first_run_writes_pending_only(tmp_path: Path) -> None:
    """Simulate a first run: only a pending-review file should exist."""
    golden_dir = tmp_path / "golden"
    golden_dir.mkdir(parents=True)
    formal = golden_dir / gate.golden_filename("demo_model", CHIP)
    pending = golden_dir / gate.pending_golden_filename("demo_model", CHIP)
    record = _make_golden()
    # No formal golden — write pending.
    assert not formal.exists()
    gate.write_golden(record, pending)
    assert pending.exists()
    assert not formal.exists()


def test_incomplete_run_writes_no_golden(tmp_path: Path) -> None:
    evaluator = _load_evaluator()
    record = _make_golden()
    formal = tmp_path / gate.golden_filename("demo_model", CHIP)
    pending = tmp_path / gate.pending_golden_filename("demo_model", CHIP)

    result = evaluator._golden_compare(
        record=record,
        score=record["score"],
        num=20,
        chip=CHIP,
        ckpt_tail=record["checkpoint_tail"],
        gf=formal,
        pending_gf=pending,
        golden_root=tmp_path,
        update_golden=False,
        model_key="demo_model",
    )

    assert result["status"] == "fail_completeness"
    assert not formal.exists()
    assert not pending.exists()


def test_update_golden_below_threshold_preserves_baselines(
    tmp_path: Path,
) -> None:
    evaluator = _load_evaluator()
    formal = tmp_path / gate.golden_filename("demo_model", CHIP)
    pending = tmp_path / gate.pending_golden_filename("demo_model", CHIP)
    original = _make_golden(score=0.95)
    rejected = _make_golden(score=0.5)
    gate.write_golden(original, formal)
    gate.write_golden(original, pending)

    result = evaluator._golden_compare(
        record=rejected,
        score=rejected["score"],
        num=1319,
        chip=CHIP,
        ckpt_tail=rejected["checkpoint_tail"],
        gf=formal,
        pending_gf=pending,
        golden_root=tmp_path,
        update_golden=True,
        model_key="demo_model",
    )

    assert result["status"] == "fail_threshold"
    assert gate.read_golden(formal)["score"] == original["score"]
    assert pending.exists()


def test_committed_golden_below_threshold_passes_inside_its_band(
    tmp_path: Path,
) -> None:
    """A 0.60 golden must gate at 0.60, not at the 0.9 fallback.

    Before the band gate, every checkpoint was held to ``score >= 0.9`` even
    when its own baseline was lower, so a legitimately lower-accuracy model
    could never pass.
    """
    evaluator = _load_evaluator()
    formal = tmp_path / gate.golden_filename("demo_model", CHIP)
    pending = tmp_path / gate.pending_golden_filename("demo_model", CHIP)
    gate.write_golden(_make_golden(score=0.60), formal)

    result = evaluator._golden_compare(
        record=_make_golden(score=0.61),
        score=0.61,
        num=1319,
        chip=CHIP,
        ckpt_tail=_make_golden()["checkpoint_tail"],
        gf=formal,
        pending_gf=pending,
        golden_root=tmp_path,
        update_golden=False,
        model_key="demo_model",
    )

    assert result["status"] == "pass"
    assert result["error"] is None


def test_committed_golden_below_threshold_fails_outside_its_band(
    tmp_path: Path,
) -> None:
    """Below the fallback threshold, the band still decides the verdict."""
    evaluator = _load_evaluator()
    formal = tmp_path / gate.golden_filename("demo_model", CHIP)
    pending = tmp_path / gate.pending_golden_filename("demo_model", CHIP)
    gate.write_golden(_make_golden(score=0.60), formal)

    result = evaluator._golden_compare(
        record=_make_golden(score=0.55),
        score=0.55,
        num=1319,
        chip=CHIP,
        ckpt_tail=_make_golden()["checkpoint_tail"],
        gf=formal,
        pending_gf=pending,
        golden_root=tmp_path,
        update_golden=False,
        model_key="demo_model",
    )

    assert result["status"] == "fail_accuracy"
    assert "band gate failed" in result["error"]


def test_update_golden_promotes_and_deletes_pending(
    tmp_path: Path,
) -> None:
    golden_dir = tmp_path / "golden"
    golden_dir.mkdir(parents=True)
    formal = golden_dir / gate.golden_filename("demo_model", CHIP)
    pending = golden_dir / gate.pending_golden_filename("demo_model", CHIP)

    record = _make_golden()
    gate.write_golden(record, pending)
    assert pending.exists()

    # Simulate --update-golden: write formal, delete pending.
    gate.write_golden(record, formal)
    if pending.exists():
        pending.unlink()
    assert formal.exists()
    assert not pending.exists()

    loaded = gate.read_golden(formal)
    assert loaded["score"] == record["score"]


def test_golden_from_another_sampling_contract_is_a_config_failure(
    tmp_path: Path,
) -> None:
    """A drifted knob is not a regression: it invalidates the comparison.

    ``temperature`` alone moves GSM8K by more than the band, so reporting the
    score delta here would name the wrong cause.  The score below sits inside
    the band, which proves the contract is settled before the band is.
    """
    evaluator = _load_evaluator()
    formal = tmp_path / gate.golden_filename("demo_model", CHIP)
    pending = tmp_path / gate.pending_golden_filename("demo_model", CHIP)
    gate.write_golden(_make_golden(score=0.95), formal)
    record = _make_golden(
        score=0.95,
        generation_cfg={"temperature": 0.0, "top_p": 0.95, "max_tokens": 10240},
    )

    result = evaluator._golden_compare(
        record=record,
        score=record["score"],
        num=1319,
        chip=CHIP,
        ckpt_tail=record["checkpoint_tail"],
        gf=formal,
        pending_gf=pending,
        golden_root=tmp_path,
        update_golden=False,
        model_key="demo_model",
    )

    assert result["status"] == "fail_config"
    assert "temperature" in result["error"]


# ---------------------------------------------------------------------------
# _golden/README.md documents the new schema.
# ---------------------------------------------------------------------------

GOLDEN_README = (
    Path(__file__).resolve().parents[1] / "e2e" / "models" / "_golden" / "README.md"
)


def test_golden_readme_documents_schema_v2() -> None:
    text = GOLDEN_README.read_text()
    assert "schema_version" in text
    assert "score" in text
    assert "tolerance" in text


def test_golden_readme_documents_the_sampling_contract() -> None:
    text = GOLDEN_README.read_text()
    assert "fail_config" in text
    assert "generation" in text
    # The uniform output budget and the window it has to stay under.
    assert "10240" in text
    assert "16384" in text


def test_golden_readme_documents_threshold() -> None:
    text = GOLDEN_README.read_text()
    assert "threshold" in text or "0.9" in text
