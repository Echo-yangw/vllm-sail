# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for CV-adaptive thresholds, noise floor, and warmup constant.

Covers the three anti-fluctuation improvements added to the perf regression
baseline infrastructure:

1. CV-based adaptive thresholds in ``classify`` / ``compare`` / ``BaselineStore.record``
2. Noise floor (``NOISE_FLOOR = 0.03``) clamping small regressions to zero
3. Warmup iterations bumped from 5 to 50

All tests are pure-Python (no device dependency), matching the policy that
``_baseline_store.py`` stays importable on a CPU-only host.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import tests.e2e.perf._perf_harness as perf_harness_module
from tests.e2e.perf._baseline_store import (
    HIGHER_BETTER,
    LOWER_BETTER,
    NOISE_FLOOR,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_PENDING,
    STATUS_UNSTABLE,
    STATUS_WARN,
    BaselineStore,
    ComparisonRow,
    MetricSample,
    apply_noise_floor,
    classify,
    compare,
    compute_cv,
)
from tests.e2e.perf._perf_harness import PerfHarness

# ---------------------------------------------------------------------------
# Warmup constant
# ---------------------------------------------------------------------------


class TestWarmupConstant:
    """改动 2: WARMUP_ITERS should be 50."""

    def test_warmup_value(self):
        pytest.importorskip("torch")
        from tests.e2e.perf._perf_harness import WARMUP_ITERS

        assert WARMUP_ITERS == 50


# ---------------------------------------------------------------------------
# Noise floor
# ---------------------------------------------------------------------------


class TestNoiseFloor:
    """改动 3: regressions within NOISE_FLOOR (3 %) are clamped to zero."""

    def test_noise_floor_constant(self):
        assert NOISE_FLOOR == pytest.approx(0.03)

    def test_within_noise_band(self):
        """2 % regression → 0.0 after noise floor."""
        assert apply_noise_floor(0.02) == pytest.approx(0.0)

    def test_at_noise_band(self):
        """Exactly 3 % → 0.0."""
        assert apply_noise_floor(0.03) == pytest.approx(0.0)

    def test_above_noise_band(self):
        """5 % → 2 % effective regression."""
        assert apply_noise_floor(0.05) == pytest.approx(0.02)

    def test_zero_regression(self):
        assert apply_noise_floor(0.0) == pytest.approx(0.0)

    def test_noise_floor_in_compare(self):
        """compare() subtracts noise floor before classifying."""
        # 12 % raw regression, minus 3 % floor → 9 % effective, below WARN 10 %
        baseline = 100.0
        current = 88.0  # 12 % lower (higher_better)
        status, reg, _ = compare(current, baseline, HIGHER_BETTER)
        assert reg == pytest.approx(0.09)  # 0.12 - 0.03
        assert status == STATUS_PASS


# ---------------------------------------------------------------------------
# classify() with dynamic thresholds
# ---------------------------------------------------------------------------


class TestClassify:
    """改动 1 (partial): classify() accepts dynamic thresholds."""

    def test_pass_default(self):
        assert classify(0.05) == STATUS_PASS

    def test_warn_default(self):
        assert classify(0.15) == STATUS_WARN

    def test_fail_default(self):
        assert classify(0.30) == STATUS_FAIL

    def test_custom_thresholds_pass(self):
        # 18 % with relaxed 20 % warn → still PASS
        assert classify(0.18, warn_threshold=0.20, fail_threshold=0.35) == STATUS_PASS

    def test_custom_thresholds_warn(self):
        # 25 % with relaxed 20 % warn / 35 % fail → WARN
        assert classify(0.25, warn_threshold=0.20, fail_threshold=0.35) == STATUS_WARN

    def test_custom_thresholds_fail(self):
        # 40 % with relaxed 35 % fail → FAIL
        assert classify(0.40, warn_threshold=0.20, fail_threshold=0.35) == STATUS_FAIL


# ---------------------------------------------------------------------------
# CV-adaptive compare()
# ---------------------------------------------------------------------------


class TestCompareCV:
    """改动 1 (core): compare() uses CV to pick threshold band."""

    def test_low_cv_uses_default_thresholds(self):
        """CV < 8 % → default bands (WARN 10 %, FAIL 25 %)."""
        # 18 % regression, with default bands → WARN
        baseline, current = 100.0, 82.0  # 18 % lower (higher_better)
        status, reg, _ = compare(current, baseline, HIGHER_BETTER, cv=0.05)
        # 0.18 raw - 0.03 floor = 0.15 effective → WARN (> 0.10)
        assert status == STATUS_WARN

    def test_medium_cv_relaxes_thresholds(self):
        """CV 8 %~15 % → relaxed bands (WARN 20 %, FAIL 35 %)."""
        baseline, current = 100.0, 82.0  # 18 % raw regression
        status, reg, _ = compare(current, baseline, HIGHER_BETTER, cv=0.10)
        # 0.18 - 0.03 = 0.15 effective; relaxed WARN is 20 % → PASS
        assert status == STATUS_PASS

    def test_medium_cv_warn(self):
        """CV in medium band, regression exceeds relaxed WARN."""
        baseline, current = 100.0, 74.0  # 26 % raw
        status, reg, _ = compare(current, baseline, HIGHER_BETTER, cv=0.12)
        # 0.26 - 0.03 = 0.23 → relaxed WARN 20 % exceeded → WARN
        assert status == STATUS_WARN

    def test_medium_cv_fail(self):
        """CV in medium band, regression exceeds relaxed FAIL."""
        baseline, current = 100.0, 60.0  # 40 % raw
        status, reg, _ = compare(current, baseline, HIGHER_BETTER, cv=0.10)
        # 0.40 - 0.03 = 0.37 → relaxed FAIL 35 % exceeded → FAIL
        assert status == STATUS_FAIL

    def test_high_cv_unstable(self):
        """CV > 15 % → UNSTABLE regardless of regression magnitude."""
        baseline, current = 100.0, 50.0  # 50 % raw regression
        status, reg, _ = compare(current, baseline, HIGHER_BETTER, cv=0.20)
        assert status == STATUS_UNSTABLE

    def test_high_cv_no_regression_still_unstable(self):
        """CV > 15 % with zero regression is still UNSTABLE (noise taints all)."""
        baseline, current = 100.0, 110.0  # improvement
        status, reg, _ = compare(current, baseline, HIGHER_BETTER, cv=0.18)
        assert status == STATUS_UNSTABLE
        assert reg == pytest.approx(0.0)  # no regression (improved)

    def test_cv_none_uses_defaults(self):
        """cv=None behaves like the original pre-CV code."""
        baseline, current = 100.0, 82.0
        status, _, _ = compare(current, baseline, HIGHER_BETTER, cv=None)
        # 0.18 - 0.03 = 0.15 → WARN
        assert status == STATUS_WARN

    def test_lower_better_direction(self):
        """CV-adaptive logic works with lower_better metrics too."""
        baseline, current = 100.0, 118.0  # 18 % raw regression (lower better)
        status, reg, _ = compare(current, baseline, LOWER_BETTER, cv=0.10)
        # medium CV → relaxed, 0.15 effective < 0.20 WARN → PASS
        assert status == STATUS_PASS


# ---------------------------------------------------------------------------
# MetricSample.cv field
# ---------------------------------------------------------------------------


class TestMetricSampleCV:
    """MetricSample carries an optional cv field."""

    def test_cv_default_none(self):
        s = MetricSample(key="k", value=1.0, direction=HIGHER_BETTER)
        assert s.cv is None

    def test_cv_explicit(self):
        s = MetricSample(key="k", value=1.0, direction=HIGHER_BETTER, cv=0.12)
        assert s.cv == pytest.approx(0.12)


# ---------------------------------------------------------------------------
# BaselineStore.record() with CV and UNSTABLE
# ---------------------------------------------------------------------------


class TestBaselineStoreRecord:
    """Integration: BaselineStore.record() propagates CV through compare()."""

    @pytest.fixture()
    def store_with_baseline(self, tmp_path: Path) -> BaselineStore:
        """Create a store with one confirmed baseline metric."""
        chip, suite = "test", "kernel"
        doc = {
            "schema_version": 1,
            "chip": chip,
            "suite": suite,
            "metrics": {
                "latency": {
                    "direction": LOWER_BETTER,
                    "unit": "us",
                    "status": "confirmed",
                    "median": 100.0,
                    "p90": 110.0,
                    "samples": 20,
                    "history": [100.0],
                    "captured": {"commit": "abc", "date": "2026-01-01", "config": {}},
                }
            },
        }
        path = tmp_path / chip / f"{suite}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc))
        return BaselineStore(tmp_path, chip, suite)

    def test_record_low_cv_pass(self, store_with_baseline: BaselineStore):
        """Low CV + small regression → PASS."""
        row = store_with_baseline.record(
            MetricSample(key="latency", value=102.0, direction=LOWER_BETTER, cv=0.04)
        )
        # 2 % raw regression, minus 3 % floor → 0 effective → PASS
        assert row.status == STATUS_PASS

    def test_record_high_cv_unstable(self, store_with_baseline: BaselineStore):
        """High CV → UNSTABLE even with large regression."""
        row = store_with_baseline.record(
            MetricSample(key="latency", value=150.0, direction=LOWER_BETTER, cv=0.20)
        )
        assert row.status == STATUS_UNSTABLE

    def test_record_medium_cv_relaxes(self, store_with_baseline: BaselineStore):
        """Medium CV relaxes threshold so moderate regression → PASS."""
        row = store_with_baseline.record(
            MetricSample(key="latency", value=115.0, direction=LOWER_BETTER, cv=0.10)
        )
        # 15 % raw - 3 % floor = 12 % effective; relaxed WARN is 20 % → PASS
        assert row.status == STATUS_PASS

    def test_record_pending_overrides_unstable(self, tmp_path: Path):
        """Pending baseline → status stays PENDING even with high CV."""
        chip, suite = "test", "kernel"
        doc = {
            "schema_version": 1,
            "chip": chip,
            "suite": suite,
            "metrics": {
                "lat": {
                    "direction": LOWER_BETTER,
                    "unit": "us",
                    "status": "pending",
                    "median": 100.0,
                    "p90": 110.0,
                    "samples": 20,
                    "history": [100.0],
                    "captured": {"commit": "abc", "date": "2026-01-01", "config": {}},
                }
            },
        }
        path = tmp_path / chip / f"{suite}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc))
        store = BaselineStore(tmp_path, chip, suite)
        row = store.record(
            MetricSample(key="lat", value=200.0, direction=LOWER_BETTER, cv=0.25)
        )
        assert row.status == STATUS_PENDING

    def test_unstable_not_in_failures(self, store_with_baseline: BaselineStore):
        """UNSTABLE rows are NOT included in failures()."""
        store_with_baseline.record(
            MetricSample(key="latency", value=200.0, direction=LOWER_BETTER, cv=0.20)
        )
        assert len(store_with_baseline.failures()) == 0

    def test_noise_floor_prevents_warn(self, store_with_baseline: BaselineStore):
        """Regression within noise floor (< 3 %) → PASS, never WARN."""
        row = store_with_baseline.record(
            MetricSample(key="latency", value=102.5, direction=LOWER_BETTER, cv=0.05)
        )
        assert row.status == STATUS_PASS
        assert row.regression == pytest.approx(0.0)  # 2.5 % - 3 % floor → 0


class TestDerivedMetricCV:
    def test_high_cv_marks_latency_tflops_and_throughput_unstable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "890p" / "kernel.json"
        path.parent.mkdir(parents=True)
        metrics = {
            "case.latency_us": (LOWER_BETTER, 100.0),
            "case.tflops": (HIGHER_BETTER, 1.0),
            "case.tokens_per_s": (HIGHER_BETTER, 10_000.0),
        }
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "chip": "890p",
                    "suite": "kernel",
                    "metrics": {
                        key: {
                            "direction": direction,
                            "unit": "",
                            "status": "confirmed",
                            "median": median,
                            "p90": median,
                            "samples": 20,
                            "history": [median],
                            "captured": {
                                "commit": "abc",
                                "date": "2026-01-01",
                                "config": {},
                            },
                        }
                        for key, (direction, median) in metrics.items()
                    },
                }
            )
        )
        monkeypatch.setattr(
            perf_harness_module,
            "bench_kernel",
            lambda call: (200.0, 220.0, 0.20),
        )
        harness = PerfHarness(BaselineStore(tmp_path, "890p", "kernel"))

        with pytest.warns(UserWarning, match="UNSTABLE") as caught:
            measurement = harness.record_kernel("case", lambda: None, flops=2e6)
            harness.record_rate("case.tokens_per_s", 1.0, measurement, unit="tok/s")

        assert len(caught) == 3
        assert [row.status for row in harness.store.rows] == [
            STATUS_UNSTABLE,
            STATUS_UNSTABLE,
            STATUS_UNSTABLE,
        ]
        assert measurement.cv == pytest.approx(0.20)
        assert harness.store.failures() == []


# ---------------------------------------------------------------------------
# compute_cv()
# ---------------------------------------------------------------------------


class TestComputeCV:
    """Unit tests for the compute_cv helper (pure stdlib, in _baseline_store)."""

    def test_constant_values(self):
        assert compute_cv([5.0, 5.0, 5.0, 5.0]) == pytest.approx(0.0)

    def test_known_cv(self):
        # [1, 2, 3] → mean=2, std=sqrt(2/3)≈0.8165, cv≈0.4082
        cv = compute_cv([1.0, 2.0, 3.0])
        assert cv == pytest.approx(0.4082, abs=0.001)

    def test_single_sample(self):
        assert compute_cv([42.0]) == 0.0

    def test_empty(self):
        assert compute_cv([]) == 0.0


# ---------------------------------------------------------------------------
# summarize() now returns (median, p90, cv)
# ---------------------------------------------------------------------------


class TestSummarizeCV:
    """summarize() returns a 3-tuple including cv."""

    def test_three_tuple(self):
        pytest.importorskip("torch")
        from tests.e2e.perf._perf_harness import summarize

        median, p90, cv = summarize([10.0, 10.0, 10.0, 10.0, 10.0])
        assert median == pytest.approx(10.0)
        assert p90 == pytest.approx(10.0)
        assert cv == pytest.approx(0.0)

    def test_cv_nonzero(self):
        pytest.importorskip("torch")
        from tests.e2e.perf._perf_harness import summarize

        _, _, cv = summarize([100.0, 200.0])
        assert cv > 0.0


# ---------------------------------------------------------------------------
# STATUS_UNSTABLE constant
# ---------------------------------------------------------------------------


class TestStatusUnstable:
    """STATUS_UNSTABLE is a new status constant."""

    def test_value(self):
        assert STATUS_UNSTABLE == "UNSTABLE"

    def test_comparison_row_can_hold(self):
        """ComparisonRow.status can represent UNSTABLE."""
        row = ComparisonRow(
            key="k",
            unit="us",
            direction=LOWER_BETTER,
            current=1.0,
            baseline=1.0,
            delta_pct=0.0,
            regression=0.0,
            status=STATUS_UNSTABLE,
        )
        assert row.status == STATUS_UNSTABLE


# ---------------------------------------------------------------------------
# Read-only golden and writable candidate separation
# ---------------------------------------------------------------------------


class TestReadOnlyGolden:
    def test_compare_and_teardown_write_only_artifact_candidate(
        self, tmp_path: Path
    ) -> None:
        golden_root = tmp_path / "golden"
        candidate_root = tmp_path / "artifacts" / "baselines"
        golden_path = golden_root / "890p" / "kernel.json"
        golden_path.parent.mkdir(parents=True)
        golden = {
            "schema_version": 1,
            "chip": "890p",
            "suite": "kernel",
            "metrics": {
                "latency": {
                    "direction": LOWER_BETTER,
                    "unit": "us",
                    "status": "confirmed",
                    "median": 100.0,
                    "p90": 110.0,
                    "samples": 20,
                    "history": [90.0, 100.0, 110.0],
                    "captured": {"commit": "abc", "date": "2026-01-01", "config": {}},
                }
            },
        }
        original = json.dumps(golden)
        golden_path.write_text(original)
        golden_path.chmod(0o444)
        golden_path.parent.chmod(0o555)
        golden_root.chmod(0o555)
        try:
            store = BaselineStore(
                golden_root, "890p", "kernel", output_root=candidate_root
            )
            row = store.record(
                MetricSample(
                    key="latency", value=105.0, direction=LOWER_BETTER, cv=0.01
                )
            )
            candidate = store.save()  # mirrors the fixture's session teardown
        finally:
            golden_root.chmod(0o755)
            golden_path.parent.chmod(0o755)
            golden_path.chmod(0o644)

        assert row.status == STATUS_PASS
        assert golden_path.read_text() == original
        assert candidate == candidate_root / "890p" / "kernel.json"
        saved = json.loads(candidate.read_text())
        assert saved["metrics"]["latency"]["median"] == 100.0
        assert saved["metrics"]["latency"]["history"] == [90.0, 100.0, 110.0]
        report = store.report_json()
        assert report["baseline_path"] == str(golden_path)
        assert report["candidate_path"] == str(candidate)


# ---------------------------------------------------------------------------
# Auto-update removal
# ---------------------------------------------------------------------------


class TestAutoUpdateRemoval:
    """Verify that the automatic baseline update path has been removed."""

    def test_no_update_constructor_param(self):
        """BaselineStore no longer accepts an 'update' parameter."""
        import inspect

        sig = inspect.signature(BaselineStore.__init__)
        assert "update" not in sig.parameters

    def test_no_refresh_method(self):
        """_refresh() method has been removed."""
        assert not hasattr(BaselineStore, "_refresh")

    def test_record_does_not_modify_history(self, tmp_path: Path):
        """record() in comparison mode must NOT append to history."""
        chip, suite = "test", "kernel"
        doc = {
            "schema_version": 1,
            "chip": chip,
            "suite": suite,
            "metrics": {
                "lat": {
                    "direction": LOWER_BETTER,
                    "unit": "us",
                    "status": "confirmed",
                    "median": 100.0,
                    "p90": 110.0,
                    "samples": 20,
                    "history": [100.0],
                    "captured": {"commit": "abc", "date": "2026-01-01", "config": {}},
                }
            },
        }
        path = tmp_path / chip / f"{suite}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc))
        store = BaselineStore(tmp_path, chip, suite)
        store.record(MetricSample(key="lat", value=105.0, direction=LOWER_BETTER))
        # History must remain untouched (no rolling append)
        assert store.metrics()["lat"]["history"] == [100.0]
        # Median must remain untouched (no refresh)
        assert store.metrics()["lat"]["median"] == 100.0

    def test_report_json_update_is_always_false(self, tmp_path: Path):
        """report_json() always reports update=False."""
        chip, suite = "test", "kernel"
        path = tmp_path / chip / f"{suite}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = {"schema_version": 1, "chip": chip, "suite": suite, "metrics": {}}
        path.write_text(json.dumps(doc))
        store = BaselineStore(tmp_path, chip, suite)
        assert store.report_json()["update"] is False

    def test_report_markdown_mode_is_compare(self, tmp_path: Path):
        """report_markdown() always says 'mode: compare'."""
        chip, suite = "test", "kernel"
        path = tmp_path / chip / f"{suite}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = {"schema_version": 1, "chip": chip, "suite": suite, "metrics": {}}
        path.write_text(json.dumps(doc))
        store = BaselineStore(tmp_path, chip, suite)
        md = store.report_markdown()
        assert "mode: compare" in md
        assert "update" not in md.lower()
