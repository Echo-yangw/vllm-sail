# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-chip performance baseline storage and regression comparison.

Baseline layout (CI plan PART 6.1)::

    tests/e2e/perf/baselines/<chip>/<suite>.json

``chip`` is ``810e`` / ``890p`` and ``suite`` is ``kernel`` or ``model``.  One
JSON document holds every metric of that suite for that chip; the two chips are
never cross-compared because they take different quantization and TP paths.

Document schema (``SCHEMA_VERSION = 1``)::

    {
      "schema_version": 1,
      "chip": "810e",
      "suite": "kernel",
      "metrics": {
        "<metric key>": {
          "direction": "higher_better" | "lower_better",
          "unit": "us" | "tflops" | ...,
          "status": "pending" | "confirmed",
          "median": 123.4,          # authoritative comparison value
          "p90": 130.0,             # informational tail
          "samples": 20,            # timed iterations behind the median
          "history": [123.4, ...],  # reviewed historical run medians
          "captured": {"commit": "...", "date": "...", "config": {...}}
        }
      }
    }

Regression policy (PART 6.1 thresholds, centralised here):

* regression fraction ``r`` is ``0.0`` when the metric improved;
* ``r > FAIL_THRESHOLD (25%)`` -> ``FAIL`` (the CI job must fail);
* ``WARN_THRESHOLD (10%) < r <= FAIL_THRESHOLD`` -> ``WARN`` (job summary only);
* a metric whose baseline ``status`` is ``"pending"`` (written by the first run
  on a chip, never reviewed) reports ``PENDING`` and never fails.

Baseline promotion flow:

1. First run on a chip: no file -> results are written with
   ``status="pending"`` and every metric reports ``PENDING`` (not a failure).
2. A human reviews the pending numbers (CI artifact), copies the JSON into the
   repo and commits it, flipping ``status`` to ``"confirmed"``.
3. Golden baselines are maintained by hand: a developer runs the suite on a
   real chip, reviews numbers, and commits the updated JSON via a dedicated PR.

This module is pure standard library: no torch, no vLLM, no device.  It stays
importable (and unit-testable) on a CPU-only host, which is what keeps
``pytest --collect-only`` green in the PR gate.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple, Optional

SCHEMA_VERSION = 1

#: Relative regression that still passes silently.
WARN_THRESHOLD = 0.10
#: Relative regression that fails the CI job.
FAIL_THRESHOLD = 0.25
#: Reviewed run medians used by optional human-maintenance tooling (PART 6.1:
#: "median of the last 7 nightly runs").  No automated CI path consumes or
#: modifies this window.
REFRESH_WINDOW = 7
#: Run medians kept in ``history`` for trend inspection.  Retained for
#: human-maintenance tooling; no longer consumed by any automated CI path.
HISTORY_LIMIT = 30

HIGHER_BETTER = "higher_better"
LOWER_BETTER = "lower_better"

STATUS_PASS = "PASS"
STATUS_WARN = "WARN"
STATUS_FAIL = "FAIL"
STATUS_PENDING = "PENDING"
#: Metric whose measurement CV exceeds the noise ceiling; treated like
#: PENDING for CI purposes (surfaces in the report, never fails the job).
STATUS_UNSTABLE = "UNSTABLE"

#: Regression within this band (3 %) is treated as measurement noise and
#: clamped to zero so it never triggers WARN / FAIL.
NOISE_FLOOR = 0.03

_VALID_DIRECTIONS = (HIGHER_BETTER, LOWER_BETTER)


class MetricSample(NamedTuple):
    """One measured metric handed to :meth:`BaselineStore.record`."""

    key: str
    value: float
    direction: str
    unit: str = ""
    #: Tail latency of the timed iterations; ``None`` when not meaningful
    #: (e.g. a single wall-clock model measurement).
    p90: Optional[float] = None  # noqa: UP045  (py3.9 runtime: keep Optional)
    #: Number of timed iterations behind ``value``.
    samples: int = 1
    #: Fixed shape/config of the case, archived for traceability.
    config: Optional[Mapping[str, Any]] = None  # noqa: UP045
    #: Coefficient of variation (std/mean) of the timed iterations; ``None``
    #: when not available (e.g. single wall-clock measurement).
    cv: Optional[float] = None  # noqa: UP045


class ComparisonRow(NamedTuple):
    """One line of the post-run comparison table."""

    key: str
    unit: str
    direction: str
    current: float
    baseline: Optional[float]  # noqa: UP045
    #: Signed relative change of the current value vs the baseline
    #: (``0.1`` = +10%), ``None`` while the baseline is pending.
    delta_pct: Optional[float]  # noqa: UP045
    #: Regression fraction in the *bad* direction, clamped at ``0.0``.
    regression: float
    status: str


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(v) for v in values)
    count = len(ordered)
    if count == 0:
        raise ValueError("cannot take the median of an empty sequence")
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _p90(values: Sequence[float]) -> float:
    """Nearest-rank p90; small samples fall back to the maximum."""
    ordered = sorted(float(v) for v in values)
    if not ordered:
        raise ValueError("cannot take the p90 of an empty sequence")
    index = int(0.9 * (len(ordered) - 1) + 0.5)
    return ordered[min(index, len(ordered) - 1)]


def git_commit(root: Optional[Path] = None) -> str:  # noqa: UP045
    """Best-effort short commit hash of the checkout; ``"unknown"`` off-git."""
    try:
        probe = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(root) if root is not None else None,
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return "unknown"
    if probe.returncode != 0:
        return "unknown"
    return probe.stdout.strip() or "unknown"


def apply_noise_floor(regression: float) -> float:
    """Clamp regression within :data:`NOISE_FLOOR` to zero (noise band)."""
    return max(regression - NOISE_FLOOR, 0.0)


def compute_cv(values: Sequence[float]) -> float:
    """Coefficient of variation (std / mean); ``0.0`` for < 2 samples or zero mean."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    if mean <= 0.0:
        return 0.0
    variance = sum((x - mean) ** 2 for x in values) / n
    return variance**0.5 / mean


def regression_fraction(current: float, baseline: float, direction: str) -> float:
    """Fraction by which ``current`` regressed against ``baseline``.

    ``0.0`` means "no regression" (equal or improved); ``0.25`` means the
    metric moved 25% into the bad direction.  A non-positive baseline makes a
    relative band meaningless, so the metric is treated as unregressed and the
    caller is expected to keep such baselines out of the store.
    """
    if direction not in _VALID_DIRECTIONS:
        raise ValueError(
            f"unknown direction {direction!r}; expected one of {_VALID_DIRECTIONS}"
        )
    if baseline <= 0.0:
        return 0.0
    if direction == HIGHER_BETTER:
        fraction = (baseline - float(current)) / baseline
    else:
        fraction = (float(current) - baseline) / baseline
    return max(fraction, 0.0)


def classify(
    regression: float,
    *,
    warn_threshold: float = WARN_THRESHOLD,
    fail_threshold: float = FAIL_THRESHOLD,
) -> str:
    """Map a regression fraction onto PASS / WARN / FAIL.

    Accepts optional *warn_threshold* / *fail_threshold* so that
    CV-adaptive callers can relax the bands for noisy metrics.
    """
    if regression > fail_threshold:
        return STATUS_FAIL
    if regression > warn_threshold:
        return STATUS_WARN
    return STATUS_PASS


def compare(
    current: float,
    baseline: float,
    direction: str,
    *,
    cv: Optional[float] = None,  # noqa: UP045
) -> tuple[str, float, float]:
    """Return ``(status, regression_fraction, signed_delta_pct)``.

    When *cv* is provided the function applies CV-adaptive thresholds:

    * ``cv > 0.15`` → :data:`STATUS_UNSTABLE` (too noisy to judge)
    * ``0.08 < cv <= 0.15`` → relaxed bands (WARN 20 %, FAIL 35 %)
    * ``cv <= 0.08`` (or *None*) → default bands

    A :data:`NOISE_FLOOR` (3 %) is subtracted from the raw regression
    fraction before classification so that trivial fluctuation is ignored.
    """
    regression = apply_noise_floor(regression_fraction(current, baseline, direction))
    if baseline > 0.0:
        delta_pct = (float(current) - baseline) / baseline
    else:
        delta_pct = 0.0

    # CV-adaptive thresholds
    if cv is not None and cv > 0.15:
        return STATUS_UNSTABLE, regression, delta_pct
    if cv is not None and cv > 0.08:
        status = classify(regression, warn_threshold=0.20, fail_threshold=0.35)
    else:
        status = classify(regression)
    return status, regression, delta_pct


class BaselineStore:
    """Reads one golden and writes its run candidate without coupling roots."""

    def __init__(
        self,
        root: Path,
        chip: str,
        suite: str,
        *,
        output_root: Optional[Path] = None,  # noqa: UP045
    ):
        if not chip:
            raise ValueError("chip key must not be empty")
        self.root = Path(root)
        self.output_root = Path(output_root) if output_root is not None else self.root
        self.chip = chip
        self.suite = suite
        self.path = self.root / chip / f"{suite}.json"
        self.output_path = self.output_root / chip / f"{suite}.json"
        self._doc: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "chip": chip,
            "suite": suite,
            "metrics": {},
        }
        self._existed = False
        if self.path.exists():
            self._doc = self._parse(self.path.read_text())
            self._existed = True
        #: Rows produced by :meth:`record` during this run, in record order.
        self.rows: list[ComparisonRow] = []
        #: Metric keys whose baseline is pending human review.
        self.pending: list[str] = []

    # -- loading ------------------------------------------------------------

    def _parse(self, text: str) -> dict[str, Any]:
        doc = json.loads(text)
        if not isinstance(doc, dict) or "metrics" not in doc:
            raise ValueError(f"{self.path} is not a baseline document")
        chip = doc.get("chip", self.chip)
        if chip != self.chip:
            raise ValueError(
                f"baseline chip mismatch: {self.path} records chip={chip!r} but "
                f"the current device is {self.chip!r}; per-chip baselines must "
                "never be cross-compared"
            )
        doc.setdefault("schema_version", SCHEMA_VERSION)
        doc.setdefault("suite", self.suite)
        return doc

    @property
    def existed(self) -> bool:
        """Whether a baseline document was present when the store opened."""
        return self._existed

    def metrics(self) -> dict[str, Any]:
        return self._doc["metrics"]

    # -- capture metadata -----------------------------------------------------

    def _capture(self, sample: MetricSample) -> dict[str, Any]:
        return {
            "commit": os.environ.get("VLLM_SAIL_PERF_COMMIT")
            or git_commit(self.root.parent.parent.parent),
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "config": dict(sample.config or {}),
        }

    # -- write ----------------------------------------------------------------

    def save(self) -> Path:
        """Atomically write the run candidate; never writes outside ``output_root``."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.output_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(self._doc, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(str(tmp), str(self.output_path))
        return self.output_path

    # -- record ---------------------------------------------------------------

    def _create(self, sample: MetricSample) -> dict[str, Any]:
        return {
            "direction": sample.direction,
            "unit": sample.unit,
            "status": "pending",
            "median": sample.value,
            "p90": sample.p90 if sample.p90 is not None else sample.value,
            "samples": sample.samples,
            "history": [sample.value],
            "captured": self._capture(sample),
        }

    def record(self, sample: MetricSample) -> ComparisonRow:
        """Compare one metric against its baseline (or establish it).

        Read-only comparison: the authoritative ``median`` in the baseline is
        never modified by this method.  History is no longer rolled; golden
        baselines are maintained by hand via dedicated PRs.
        """
        if sample.direction not in _VALID_DIRECTIONS:
            raise ValueError(
                f"metric {sample.key!r}: unknown direction {sample.direction!r}"
            )
        metrics = self._doc["metrics"]
        entry = metrics.get(sample.key)

        if entry is None:
            metrics[sample.key] = self._create(sample)
            self.pending.append(sample.key)
            row = ComparisonRow(
                key=sample.key,
                unit=sample.unit,
                direction=sample.direction,
                current=sample.value,
                baseline=None,
                delta_pct=None,
                regression=0.0,
                status=STATUS_PENDING,
            )
            self.rows.append(row)
            return row

        baseline_median = float(entry["median"])

        status, regression, delta_pct = compare(
            sample.value, baseline_median, sample.direction, cv=sample.cv
        )
        if entry.get("status") != "confirmed":
            # Pending baseline: still surface the delta, never fail on it.
            status = STATUS_PENDING
            self.pending.append(sample.key)
        elif status == STATUS_UNSTABLE:
            # High-CV: surface in the report but never fail the job.
            pass  # keep STATUS_UNSTABLE from compare()
        row = ComparisonRow(
            key=sample.key,
            unit=sample.unit,
            direction=sample.direction,
            current=sample.value,
            baseline=baseline_median,
            delta_pct=delta_pct,
            regression=regression,
            status=status,
        )
        self.rows.append(row)
        return row

    # -- reporting --------------------------------------------------------------

    def failures(self) -> list[ComparisonRow]:
        return [row for row in self.rows if row.status == STATUS_FAIL]

    def warnings(self) -> list[ComparisonRow]:
        return [row for row in self.rows if row.status == STATUS_WARN]

    def report_json(self) -> dict[str, Any]:
        """Machine-readable run report (CI artifact ``perf_report.json``)."""
        return {
            "schema_version": SCHEMA_VERSION,
            "chip": self.chip,
            "suite": self.suite,
            "update": False,
            "baseline_path": str(self.path),
            "candidate_path": str(self.output_path),
            "thresholds": {"warn": WARN_THRESHOLD, "fail": FAIL_THRESHOLD},
            "rows": [
                {
                    "key": row.key,
                    "unit": row.unit,
                    "direction": row.direction,
                    "current": row.current,
                    "baseline": row.baseline,
                    "delta_pct": row.delta_pct,
                    "regression": row.regression,
                    "status": row.status,
                }
                for row in self.rows
            ],
            "pending": list(self.pending),
        }

    def report_markdown(self) -> str:
        """Human-readable comparison table (CI artifact / job summary)."""
        lines = [
            f"# perf regression report - {self.chip} / {self.suite}",
            "",
            f"baseline: `{self.path}`  ",
            f"candidate: `{self.output_path}`  ",
            f"thresholds: WARN > {WARN_THRESHOLD:.0%}, FAIL > {FAIL_THRESHOLD:.0%} "
            "relative regression  ",
            "mode: compare",
            "",
            "| metric | current | baseline | delta | status |",
            "| --- | ---: | ---: | ---: | :--: |",
        ]
        for row in self.rows:
            baseline = "-" if row.baseline is None else f"{row.baseline:.4g}"
            delta = "-" if row.delta_pct is None else f"{row.delta_pct:+.1%}"
            unit = f" {row.unit}" if row.unit else ""
            lines.append(
                f"| `{row.key}` | {row.current:.4g}{unit} | {baseline} | {delta} "
                f"| {row.status} |"
            )
        if self.pending:
            lines.extend(
                [
                    "",
                    f"pending-review baselines (first run writes, never fails): "
                    f"{len(self.pending)} metric(s)",
                ]
            )
        failures = self.failures()
        if failures:
            lines.extend(["", f"FAILED metrics: {len(failures)}"])
            lines.extend(
                f"- `{row.key}` regressed {row.regression:.1%}" for row in failures
            )
        lines.append("")
        return "\n".join(lines)

    def emit(self, report_dir: Path) -> tuple[Path, Path]:
        """Write ``perf_report.json`` / ``perf_report.md`` into ``report_dir``."""
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / "perf_report.json"
        md_path = report_dir / "perf_report.md"
        json_path.write_text(
            json.dumps(self.report_json(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        md_path.write_text(self.report_markdown(), encoding="utf-8")
        return json_path, md_path
