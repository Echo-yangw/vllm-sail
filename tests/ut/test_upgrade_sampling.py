# SPDX-License-Identifier: Apache-2.0
"""Calendar sampling must preserve forward progress from a qualified main pin."""

from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

SCRIPT = (
    Path(__file__).parents[2]
    / ".agents/skills/vllm-main-upgrade/scripts/sample_commits.py"
)
spec = importlib.util.spec_from_file_location("upgrade_sampling", SCRIPT)
sampling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampling)


def epoch(day):
    return int(datetime(2026, 10, day, 12, tzinfo=timezone.utc).timestamp())


def test_release_before_baseline_does_not_regress_or_resample_old_history():
    history = [(f"day-{day}", epoch(day)) for day in range(17, 31)]
    times = {"baseline": epoch(24), "release": epoch(17), "target": epoch(30)}
    points = sampling.samples(
        history, times, "baseline", "release", "target", ZoneInfo("Asia/Shanghai")
    )
    assert points[0]["sha"] == "baseline"
    assert points[-1]["sha"] == "target"
    assert "release" not in [point["sha"] for point in points]
    assert [point["sha"] for point in points[1:-1]] == [
        f"day-{day}" for day in range(25, 31)
    ]


def test_in_range_off_main_release_remains_mandatory():
    history = [(f"day-{day}", epoch(day)) for day in range(24, 31)]
    times = {"baseline": epoch(24), "release": epoch(27), "target": epoch(30)}
    points = sampling.samples(
        history, times, "baseline", "release", "target", ZoneInfo("Asia/Shanghai")
    )
    release = next(point for point in points if point["sha"] == "release")
    assert release["roles"] == ["release"]
    assert points[0]["sha"] == "baseline"
    assert points[-1]["sha"] == "target"
    assert len({point["sha"] for point in points}) == len(points)


@pytest.mark.parametrize("start,release,target", [(30, 27, 29), (24, 30, 29)])
def test_rejects_endpoints_after_frozen_target(start, release, target):
    with pytest.raises(ValueError):
        sampling.samples(
            [],
            {"s": epoch(start), "r": epoch(release), "t": epoch(target)},
            "s",
            "r",
            "t",
            ZoneInfo("UTC"),
        )
