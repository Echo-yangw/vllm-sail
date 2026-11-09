# SPDX-License-Identifier: Apache-2.0
"""Lightly-adapted upstream vLLM tests (CI category 5, ``mode = "adapted"``).

An ``adapted`` entry reuses an upstream test file whose *skeleton and
assertions* are valid for vllm-sail, but which mixes CPU-runnable logic with
device / distributed / network nodes, or carries a handful of known-failing
cases. The adaptation is data-driven and **never edits upstream source**:

* node narrowing (``markers`` / ``keywords`` / ``deselect``) and ``env``
  overrides are declared in ``REUSE_MANIFEST.toml`` and translated into pytest
  argv / environment by :class:`~tests.upstream_reuse._reuse_harness.UpstreamRunner`;
* expected failures (``xfail``) are reconciled here: a declared-xfail node that
  fails is tolerated, any *other* failure is a real regression;
* ``skip_reason`` parks an entry whose upstream prerequisites are not yet met
  (e.g. a warm offline tokenizer cache) without deleting it, keeping the intent
  auditable.

A fourth path is conditional on the runtime instead of declared in the manifest:
``offline_hub_gap`` skips an entry that would have to resolve a checkpoint from
the Hugging Face hub when this runtime has no route to it and no local copy,
which is a hang rather than a failure otherwise. Unlike ``skip_reason`` it costs
nothing when resources are available; strict CI fails on any such gap.

All of that reconciliation logic lives in this module, so following an upstream
bump only the manifest data changes -- never the upstream files.
"""

from __future__ import annotations

import pytest

from tests.upstream_reuse._reuse_harness import (
    ReuseEntry,
    RunResult,
    UpstreamRunner,
    conftest_import_error,
    entries_for_mode,
    param_for_entry,
)

ADAPTED_ENTRIES = entries_for_mode("adapted")
ADAPTED_PARAMS = [param_for_entry(entry) for entry in ADAPTED_ENTRIES]


def reconcile_xfail(
    result: RunResult, entry: ReuseEntry
) -> tuple[list[str], list[str]]:
    """Split failing nodes into (unexpected, matched_xfail).

    ``unexpected`` are real regressions; ``matched_xfail`` are declared-known
    failures that actually reproduced. A declared xfail pattern that matched
    nothing is *not* an error -- upstream may have fixed it -- but it is
    reported so the whitelist can be pruned on the next bump.
    """
    unexpected: list[str] = []
    matched: list[str] = []
    for name in result.failed_names:
        if any(pattern and pattern in name for pattern in entry.xfail):
            matched.append(name)
        else:
            unexpected.append(name)
    return unexpected, matched


def stale_xfail_patterns(result: RunResult, entry: ReuseEntry) -> list[str]:
    """Declared xfail patterns that matched no failing node this run."""
    hit = {
        pattern
        for pattern in entry.xfail
        if pattern and any(pattern in name for name in result.failed_names)
    }
    return [pattern for pattern in entry.xfail if pattern not in hit]


# ---------------------------------------------------------------------------
# CPU-only structural checks for adapted entries (run everywhere)
#
# Marked ``upstream_source`` so the CI reuse lanes collect them (mirrors
# test_reuse_direct.py); the directory conftest still skips them without
# VLLM_SOURCE_ROOT.
# ---------------------------------------------------------------------------
@pytest.mark.upstream_source
def test_adapted_entries_justify_their_adaptation() -> None:
    """An ``adapted`` row must actually adapt something.

    Otherwise it should be ``direct``. Valid justifications: a narrowing knob
    (markers/keywords/deselect), an env override, a declared xfail, a parking
    skip_reason, or being device-only (excluded from the CPU lane).
    """
    unjustified = []
    for entry in ADAPTED_ENTRIES:
        narrows = bool(entry.markers or entry.keywords or entry.deselect)
        if not (
            narrows
            or entry.env
            or entry.xfail
            or entry.skip_reason
            or entry.device_only
        ):
            unjustified.append(entry.id)
    assert not unjustified, (
        f"adapted entries with no adaptation; reclassify as direct: {unjustified}"
    )


@pytest.mark.upstream_source
def test_adapted_skip_reasons_are_documented() -> None:
    """A parked entry must explain why, so it is not silently dead weight."""
    for entry in ADAPTED_ENTRIES:
        if entry.skip_reason:
            assert len(entry.skip_reason.strip()) >= 12, entry.id


# ---------------------------------------------------------------------------
# Parameterized execution of adapted upstream tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("entry", ADAPTED_PARAMS)
def test_reuse_adapted(entry: ReuseEntry, upstream_runner: UpstreamRunner) -> None:
    """Run one adapted upstream file, honouring declared narrowing and xfails."""
    result = upstream_runner.run(entry)
    if result.skipped_by_manifest:
        pytest.skip(f"parked in REUSE_MANIFEST.toml: {result.tail}")
    if result.environment_gap:
        pytest.skip(result.environment_gap)
    environment = conftest_import_error(result)
    if environment is not None:
        pytest.skip(
            "upstream's own conftest cannot be imported by this runtime, so no "
            f"upstream test was collected: {environment}"
        )

    assert result.errors == 0, (
        f"adapted upstream {entry.path} failed to import/collect "
        f"({result.errors} error(s)); the narrowing knobs or an optional "
        f"dependency need review:\n{result.tail}"
    )
    assert result.collected >= entry.min_tests, (
        f"adapted upstream {entry.path} collected {result.collected} test(s) "
        f"after narrowing, below min_tests={entry.min_tests}. The -m/-k/"
        f"--deselect filters may be over-broad or the file was refactored:\n"
        f"{result.tail}"
    )

    unexpected, matched = reconcile_xfail(result, entry)
    stale = stale_xfail_patterns(result, entry)
    if stale:
        # Advisory only: surfaced with -s / in the report, never a failure.
        print(
            f"\n[upstream_reuse] {entry.id}: declared xfail pattern(s) did not "
            f"reproduce (consider pruning): {stale}"
        )
    assert not unexpected, (
        f"adapted reuse of upstream {entry.path} regressed with "
        f"{len(unexpected)} unexpected failure(s) (declared xfails that "
        f"reproduced: {matched[:10]}):\n  "
        + "\n  ".join(unexpected[:20])
        + f"\n{result.tail}"
    )
