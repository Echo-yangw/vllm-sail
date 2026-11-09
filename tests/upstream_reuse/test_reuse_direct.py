# SPDX-License-Identifier: Apache-2.0
"""Directly-reused upstream vLLM tests (CI category 5, ``mode = "direct"``).

Two layers live here:

1. **CPU-only structural checks** over ``REUSE_MANIFEST.toml`` that run
   everywhere -- no upstream checkout, no torch, no vLLM. They guarantee the
   whitelist stays well-formed (unique ids, valid modes/chips, sane paths) so a
   malformed manifest fails fast even on a bare PR runner.

2. **Parameterized execution** of every ``direct`` entry: each upstream test
   file is run verbatim, as an isolated pytest subprocess rooted at the pinned
   checkout, inside the ambient vllm-sail runtime. "Direct" means zero edits and
   zero expected failures. Strict CI requires actual passing tests and only
   permits reviewed, bounded skips; optional local runs remain permissive.

Without ``VLLM_SOURCE_ROOT`` the directory conftest skips every item; with a
checkout but no installed vLLM/torch the ``upstream_runner`` fixture skips. So a
bare host yields "all skipped, 0 errors" while CI (which installs the runtime)
actually executes the whitelist.  Two further skips are properties of the runtime
rather than of the manifest, so they leave "direct" meaning zero edits: one for
an installed runtime that cannot import upstream's own ``tests/conftest.py`` (a
missing upstream test-only dependency; see ``conftest_import_error``), and one
for a runtime with no route to the Hugging Face hub when the entry resolves a
checkpoint through it (see ``offline_hub_gap``).  Both are inert on a runner that
has the dependency and the network, which is what CI provides.
"""

from __future__ import annotations

import pytest

from tests.upstream_reuse._reuse_harness import (
    MANIFEST_PATH,
    VALID_CHIPS,
    VALID_MODES,
    ReuseEntry,
    UpstreamRunner,
    conftest_import_error,
    entries_for_mode,
    load_manifest,
    param_for_entry,
)

ALL_ENTRIES = load_manifest()
DIRECT_ENTRIES = entries_for_mode("direct")
DIRECT_PARAMS = [param_for_entry(entry) for entry in DIRECT_ENTRIES]


# ---------------------------------------------------------------------------
# CPU-only manifest structural checks (run everywhere; no upstream needed)
#
# Marked ``upstream_source`` so the CI reuse lanes (-m "upstream_source and not
# ppu" / -m upstream_source) collect them instead of silently filtering them
# out; without VLLM_SOURCE_ROOT the directory conftest still skips every item.
# ---------------------------------------------------------------------------
@pytest.mark.upstream_source
def test_manifest_file_is_present() -> None:
    """The committed whitelist must exist next to this suite."""
    assert MANIFEST_PATH.is_file(), MANIFEST_PATH


@pytest.mark.upstream_source
def test_manifest_entries_are_well_formed() -> None:
    """Every row must be internally consistent and point at an upstream test."""
    entries = load_manifest()
    assert entries, "REUSE_MANIFEST.toml declares no entries"
    ids = [entry.id for entry in entries]
    assert len(ids) == len(set(ids)), "duplicate manifest ids"
    for entry in entries:
        assert entry.mode in VALID_MODES, entry.id
        assert entry.chips, entry.id
        for chip in entry.chips:
            assert chip in VALID_CHIPS, f"{entry.id}: {chip}"
        assert entry.path.startswith("tests/"), f"{entry.id}: {entry.path}"
        assert entry.path.endswith(".py"), f"{entry.id}: {entry.path}"
        assert ".." not in entry.path.split("/"), f"{entry.id}: {entry.path}"
        assert entry.reason.strip(), f"{entry.id}: empty reason"
        assert entry.min_tests >= 1, f"{entry.id}: min_tests must be >= 1"


@pytest.mark.upstream_source
def test_direct_entries_carry_no_adaptation_knobs() -> None:
    """A ``direct`` entry is verbatim: no markers/-k/deselect/xfail narrowing.

    Adaptation belongs to ``mode = "adapted"`` rows executed by
    ``test_reuse_adapted.py``. Keeping direct rows knob-free preserves the
    "zero edits, must be green" contract that makes them trustworthy.
    """
    offenders = []
    for entry in entries_for_mode("direct"):
        if entry.markers or entry.keywords or entry.deselect or entry.xfail:
            offenders.append(entry.id)
    assert not offenders, f"direct entries must not narrow the run: {offenders}"


@pytest.mark.upstream_source
def test_manifest_covers_both_modes() -> None:
    """The whitelist must exercise both reuse strategies."""
    assert entries_for_mode("direct"), "no direct entries"
    assert entries_for_mode("adapted"), "no adapted entries"


# ---------------------------------------------------------------------------
# Parameterized execution of directly-reused upstream tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("entry", DIRECT_PARAMS)
def test_reuse_direct(entry: ReuseEntry, upstream_runner: UpstreamRunner) -> None:
    """Run one upstream test file verbatim; require a clean, non-empty result."""
    result = upstream_runner.run(entry)
    if result.skipped_by_manifest:
        pytest.skip(result.tail)
    if result.environment_gap:
        pytest.skip(result.environment_gap)
    environment = conftest_import_error(result)
    if environment is not None:
        pytest.skip(
            "upstream's own conftest cannot be imported by this runtime, so no "
            f"upstream test was collected: {environment}"
        )

    assert result.errors == 0, (
        f"upstream {entry.path} failed to import/collect inside the vllm-sail "
        f"runtime ({result.errors} error(s)). This is manifest drift or a "
        f"missing optional dependency:\n{result.tail}"
    )
    assert result.collected >= entry.min_tests, (
        f"upstream {entry.path} collected {result.collected} test(s), below the "
        f"min_tests={entry.min_tests} guard -- the file was likely refactored:\n"
        f"{result.tail}"
    )
    assert result.failures == 0, (
        f"direct reuse of upstream {entry.path} regressed: {result.failures} "
        f"failure(s): {list(result.failed_names)[:20]}\n{result.tail}"
    )
