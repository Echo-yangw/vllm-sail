# SPDX-License-Identifier: Apache-2.0
"""Exercise CPU phase orchestration without installing optional dependencies."""

import subprocess

import pytest

from scripts.ci import cpu_tests


@pytest.fixture
def commands(monkeypatch):
    calls = []

    def run(args, *, check, env):
        assert check is True
        calls.append((args, dict(env)))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(cpu_tests.subprocess, "run", run)
    return calls


@pytest.mark.parametrize("version", cpu_tests.VERSIONS)
def test_unit_cli_uses_isolated_interpreter_and_coverage(
    version, tmp_path, monkeypatch, commands
):
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "VLLM_SOURCE_ROOT"):
        monkeypatch.setenv(name, "/unrelated/environment")
    cpu_tests.main(["unit", "--version", version, "--work-dir", str(tmp_path)])
    directory = tmp_path / f"unit-{version}"
    assert commands[0][0] == [
        f"python{version}",
        "-m",
        "venv",
        "--clear",
        str(directory),
    ]
    assert {args[0] for args, _ in commands[1:]} == {str(directory / "bin/python")}
    for _, env in commands:
        assert (
            not {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "VLLM_SOURCE_ROOT"}
            & env.keys()
        )
        assert env["PYTHONNOUSERSITE"] == "1" and env["PIP_NO_CACHE_DIR"] == "1"
    installs = [
        args[1:] for args, _ in commands if args[1:4] == ["-m", "pip", "install"]
    ]
    assert installs == [["-m", "pip", "install", "-r", "requirements/dev.txt"]]
    tests, env = commands[-1]
    assert tests[1:5] == ["-m", "pytest", "tests/ut", "-q"]
    assert f"--cov-report=xml:{directory / 'coverage.xml'}" in tests
    assert env["COVERAGE_FILE"] == str(directory / ".coverage")


def test_source_and_reuse_share_source_but_have_separate_packages(tmp_path, commands):
    upstream = tmp_path / "vllm"
    cpu_tests.run_tests("source", tmp_path, upstream=upstream)
    source = list(commands)
    commands.clear()
    cpu_tests.run_tests("reuse", tmp_path, upstream=upstream)
    assert source[-1][0][0] != commands[-1][0][0]
    assert (
        source[-1][1]["VLLM_SOURCE_ROOT"]
        == commands[-1][1]["VLLM_SOURCE_ROOT"]
        == str(upstream)
    )
    assert source[-1][0][1:] == [
        "-m",
        "pytest",
        "tests/ut",
        "-q",
        "-rs",
        "-m",
        "upstream_source",
        "--require-upstream-source",
    ]
    assert commands[-1][0][1:] == [
        "-m",
        "pytest",
        "tests/upstream_reuse",
        "-q",
        "-rs",
        "-m",
        "upstream_source and not ppu",
        "--require-upstream-source",
    ]
    installs = [
        (args, env) for args, env in commands if args[1:4] == ["-m", "pip", "install"]
    ]
    torch = next(args for args, _ in installs if "torch" in args)
    assert torch[-3:] == [
        "--extra-index-url",
        "https://download.pytorch.org/whl/cpu",
        "torch",
    ]
    build, env = next((args, env) for args, env in installs if "-e" in args)
    assert build[-3:] == ["--no-build-isolation", "-e", str(upstream)]
    assert env["VLLM_TARGET_DEVICE"] == "empty"


@pytest.mark.parametrize("stage", ["unit", "source", "reuse"])
@pytest.mark.parametrize("failure", ["install", "pytest"])
def test_phase_failure_is_propagated(stage, failure, tmp_path, monkeypatch):
    calls = []

    def run(args, *, check, env):
        assert check is True
        calls.append(args)
        if failure in args:
            raise subprocess.CalledProcessError(7, args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(cpu_tests.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError) as caught:
        cpu_tests.run_tests(stage, tmp_path, upstream=tmp_path / "vllm")
    assert caught.value.returncode == 7
    assert failure in calls[-1]
    if failure == "install":
        assert not any("pytest" in args for args in calls)


@pytest.mark.parametrize("stage", ["source", "reuse"])
def test_source_required_before_any_commands(stage, tmp_path, commands):
    with pytest.raises(ValueError, match="upstream source"):
        cpu_tests.run_tests(stage, tmp_path)
    assert commands == []
