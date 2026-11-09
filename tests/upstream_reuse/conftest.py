# SPDX-License-Identifier: Apache-2.0
"""Keep local collection optional while requiring real upstream execution in CI."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.upstream_reuse._reuse_harness import (
    UpstreamRunner,
    load_manifest,
    resolve_source_root,
    runtime_available,
)

HERE = Path(__file__).resolve().parent


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("upstream reuse")
    group.addoption("--strict-upstream-reuse", action="store_true", default=False)
    group.addoption("--upstream-reuse-target", choices=("cpu", "890p"))
    group.addoption("--upstream-reuse-report-dir", type=Path)


def pytest_configure(config: pytest.Config) -> None:
    if not config.getoption("--strict-upstream-reuse"):
        return
    if config.getoption("--upstream-reuse-target") is None:
        raise pytest.UsageError(
            "strict upstream reuse requires --upstream-reuse-target"
        )
    if resolve_source_root(os.environ.get("VLLM_SOURCE_ROOT")) is None:
        raise pytest.UsageError(
            "strict upstream reuse requires a valid VLLM_SOURCE_ROOT"
        )
    available, reason = runtime_available()
    if not available:
        raise pytest.UsageError(
            f"strict upstream reuse requires an installed runtime: {reason}"
        )


def _under_this_directory(item: pytest.Item) -> bool:
    return HERE in Path(str(item.fspath)).resolve().parents


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    target = config.getoption("--upstream-reuse-target")
    selected = []
    deselected = []
    for item in items:
        entry = getattr(item, "callspec", None)
        entry = entry.params.get("entry") if entry is not None else None
        if (
            target is not None
            and _under_this_directory(item)
            and entry is not None
            and target not in entry.chips
        ):
            deselected.append(item)
        else:
            selected.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected
    if resolve_source_root(os.environ.get("VLLM_SOURCE_ROOT")) is not None:
        return
    skip = pytest.mark.skip(
        reason="VLLM_SOURCE_ROOT not set to a valid upstream vLLM checkout"
    )
    for item in items:
        if _under_this_directory(item):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def upstream_reuse_root(request: pytest.FixtureRequest) -> Path:
    required = request.config.getoption("--require-upstream-source")
    root = resolve_source_root(os.environ.get("VLLM_SOURCE_ROOT"))
    if root is None:
        if required:
            raise pytest.UsageError(
                "--require-upstream-source requires a valid VLLM_SOURCE_ROOT"
            )
        pytest.skip("set VLLM_SOURCE_ROOT to run upstream reuse tests")
    return root


@pytest.fixture(scope="session")
def upstream_runner(request: pytest.FixtureRequest, upstream_reuse_root: Path):
    available, reason = runtime_available()
    if not available:
        pytest.skip(f"upstream reuse needs an installed runtime: {reason}")
    runner = UpstreamRunner(
        upstream_reuse_root,
        strict=request.config.getoption("--strict-upstream-reuse"),
        execution_target=request.config.getoption("--upstream-reuse-target") or "cpu",
        report_dir=request.config.getoption("--upstream-reuse-report-dir"),
    )
    request.config._upstream_reuse_runner = runner
    try:
        yield runner
    finally:
        runner.close()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    config = session.config
    if not config.getoption("--strict-upstream-reuse"):
        return
    root = resolve_source_root(os.environ.get("VLLM_SOURCE_ROOT"))
    if root is None:
        return
    runner = getattr(config, "_upstream_reuse_runner", None)
    if runner is None:
        runner = UpstreamRunner(
            root,
            strict=True,
            execution_target=config.getoption("--upstream-reuse-target"),
            report_dir=config.getoption("--upstream-reuse-report-dir"),
        )
    report = runner.write_report(load_manifest())
    if not report["qualified"] and exitstatus == 0:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
    reporter = config.pluginmanager.getplugin("terminalreporter")
    if reporter is not None:
        reporter.write_sep(
            "=",
            f"upstream reuse: {report['passed_entries']}/{report['active_entries']} active entries passed",
        )
