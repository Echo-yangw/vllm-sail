# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""Freeze calendar samples and mandatory release endpoints without editing Git."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


def samples(history, times, start, release, target, zone, before_days=2, after_days=1):
    if min(before_days, after_days) < 1:
        raise ValueError("Sampling intervals must be positive calendar days")
    if times[start] > times[target] or times[release] > times[target]:
        raise ValueError("Start and release must precede the frozen target in time")

    def day_of(sha):
        return datetime.fromtimestamp(times[sha], zone).date()

    result, seen = [], {}

    def add(sha, day, phase, role):
        if sha in seen:
            if role not in seen[sha]["roles"]:
                seen[sha]["roles"].append(role)
            return
        point = {
            "index": len(result),
            "sha": sha,
            "date": day.isoformat(),
            "phase": phase,
            "roles": [role],
        }
        result.append(point)
        seen[sha] = point

    add(start, day_of(start), "baseline", "baseline")
    phases = (
        (start, release, before_days, "before-release", "release"),
        (release, target, after_days, "after-release", "frozen-main"),
    )
    if times[release] <= times[start]:
        # The latest release may predate an already-qualified main pin. Do not
        # insert that older release or sample backwards from the baseline.
        phases = ((start, target, after_days, "after-release", "frozen-main"),)
    for left, right, stride, phase, endpoint_role in phases:
        day = day_of(left) + timedelta(days=stride)
        while day <= day_of(right):
            next_day = int(
                datetime.combine(day + timedelta(days=1), time(), zone).timestamp()
            )
            cutoff = min(times[right] + 1, next_day)
            eligible = [sha for sha, epoch in history if times[left] < epoch < cutoff]
            if eligible:
                add(eligible[-1], day, phase, "calendar-sample")
            day += timedelta(days=stride)
        # A release branch endpoint need not belong to main's first-parent chain.
        add(right, day_of(right), phase, endpoint_role)
    return result


def freeze(
    repo, start, release, target, zone_name="Asia/Shanghai", before_days=2, after_days=1
):
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], text=True
        ).strip()

    start, release, target = (
        git("rev-parse", "--verify", ref + "^{commit}")
        for ref in (start, release, target)
    )
    history = [
        (sha, int(epoch))
        for sha, epoch in (
            line.split()
            for line in git(
                "log", "--first-parent", "--reverse", "--format=%H %ct", target
            ).splitlines()
        )
    ]
    times = {
        sha: int(git("show", "-s", "--format=%ct", sha))
        for sha in (start, release, target)
    }
    points = samples(
        history,
        times,
        start,
        release,
        target,
        ZoneInfo(zone_name),
        before_days,
        after_days,
    )
    return {
        "schema": 1,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "upstream_start": start,
        "release_endpoint": release,
        "release_in_range": times[start] <= times[release] <= times[target],
        "vllm_target": target,
        "timezone": zone_name,
        "before_release_days": before_days,
        "after_release_days": after_days,
        "tag_topology": "Mandatory release endpoints may be outside main first-parent history",
        "samples": points,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--before-days", type=int, default=2)
    parser.add_argument("--after-days", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = freeze(
        args.repo,
        args.start,
        args.release,
        args.target,
        args.timezone,
        args.before_days,
        args.after_days,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(plan, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "samples": len(plan["samples"]),
                "target": plan["vllm_target"],
            }
        )
    )


if __name__ == "__main__":
    main()
