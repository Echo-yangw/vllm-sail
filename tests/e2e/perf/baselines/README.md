# Performance baselines

Per-chip performance baselines for the regression suites in `tests/e2e/perf`.

## Layout

```
baselines/<chip>/<suite>.json
```

* `<chip>` — `810e` (ZW-810E, capability `(8, 0)`) or `890p` (ZW-890P,
  capability `(8, 9)`). The two chips take different quantization and tensor
  -parallel paths, so their baselines are archived separately and **never**
  cross-compared (the store raises on a chip mismatch).
* `<suite>` — `kernel` (`test_kernel_perf.py`) or `model`
  (`test_model_perf.py`).

One JSON document holds every metric of that suite for that chip; see
`tests/e2e/perf/_baseline_store.py` for the authoritative schema
(`SCHEMA_VERSION = 1`).

## Why this directory ships empty

Baseline numbers are only meaningful when captured on the real PPU hardware the
CI lane runs on, so **no `*.json` is committed here**. `.gitkeep` keeps the
directory in the tree; the JSON files are produced by the first CI run on
each chip and promoted by a human (see below). A `.gitignore` next to them
enforces that: a generated baseline never shows up as an uncommitted change and
can never be swept into a commit by `git add -A`.

## Promotion flow

1. **First run on a chip** — no baseline file exists. The store writes the
   measured medians with `status="pending"` and every metric reports `PENDING`
   (the lane stays green; a pending baseline never fails).
2. **Human review** — the CI job uploads the pending baseline plus the
   comparison report (`perf_report.json` / `perf_report.md`) as CI artifacts. A
   reviewer sanity-checks the numbers against the hardware spec, then copies the
   JSON into `baselines/<chip>/<suite>.json`, flips `status` to `"confirmed"`
   and commits it with `git add -f` (baselines are **never** auto-committed —
   the `-f` is the deliberate human act).
3. **Steady state** — every run compares against the confirmed baseline:
   * relative regression `> 10%` → `WARN` (job summary only),
   * relative regression `> 25%` → `FAIL` (the lane fails),
   * a still-`pending` metric keeps reporting `PENDING`.

## Updating a baseline

Baselines are maintained by hand. When hardware, firmware, or SDK changes
shift the performance envelope, a developer must:

1. Run the perf suite on the target chip (via `ppu-debug.yaml` with
   `target=kernel-perf` or the weekly `perf-regression.yaml` schedule).
2. Download the `perf_report.md` and the emitted baseline JSON from the CI
   artifacts.
3. Review the numbers against the hardware spec and prior baseline.
4. Copy the reviewed JSON into `baselines/<chip>/<suite>.json`, set every
   metric's `status` to `"confirmed"`, and commit via a dedicated PR with
   `git add -f`.

There is no automatic baseline update. Every baseline change goes through
code review.
