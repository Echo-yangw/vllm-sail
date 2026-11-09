# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the manual PPU operator pipeline, aggregation, and future gate truth tables."""

from __future__ import annotations

import copy
import importlib.util
import itertools
import json
import os
import re
import shlex
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "relative_path",
    [
        ".github/workflows/_ppu-ops-tests.yaml",
        "scripts/ci/ppu_ops_environment.json",
        "scripts/ci/ppu_ops_runner.py",
        "scripts/ci/ppu_ops_select.py",
        "scripts/ci/ppu_ops_test.sh",
        "scripts/ci/ppu_ops_tests.json",
        "tests/ut/test_ppu_ops_pipeline.py",
        "tests/ut/test_ppu_ops_runner.py",
        "tests/ut/test_ppu_ops_select.py",
    ],
)
def test_ops_asset_names(relative_path):
    assert (ROOT / relative_path).is_file(), relative_path


@pytest.mark.parametrize(
    "relative_path",
    [
        ".github/workflows/ppu-diagnostics.yaml",
        "scripts/ci/ppu_ops_probe.py",
    ],
)
def test_standalone_diagnostic_assets_are_removed(relative_path):
    assert not (ROOT / relative_path).exists(), relative_path


def test_ops_assets_have_no_legacy_namespace():
    legacy = re.compile(r"ppu[-_]dev" + r"ice[-_]")
    violations = []
    for directory in (".github/workflows", "scripts/ci"):
        for path in sorted((ROOT / directory).iterdir()):
            if path.is_file() and (
                legacy.search(path.name) or legacy.search(path.read_text())
            ):
                violations.append(str(path.relative_to(ROOT)))
    assert not violations, violations


def test_ops_workflow_artifact_contract():
    path = ROOT / ".github/workflows/_ppu-ops-tests.yaml"
    assert path.is_file()
    caller = yaml.safe_load((ROOT / ".github/workflows/ppu-ops-tests.yaml").read_text())
    workflow = yaml.safe_load(path.read_text())
    jobs = caller["jobs"]
    selection = next(
        step["with"]["name"]
        for step in jobs["select-ops-tests"]["steps"]
        if step.get("uses") == "actions/upload-artifact@v4"
    )
    assert (
        selection == "ppu-ops-selection-${{ github.run_id }}-${{ github.run_attempt }}"
    )
    assert jobs["ops-tests"]["with"]["selection_artifact"] == selection
    assert jobs["ops-tests"]["uses"] == "./.github/workflows/_ppu-ops-tests.yaml"
    results = next(
        step["with"]["name"]
        for step in workflow["jobs"]["ops-tests"]["steps"]
        if step.get("uses") == "actions/upload-artifact@v4"
    )
    assert results == (
        "ppu-ops-results-${{ github.run_id }}-${{ github.run_attempt }}-${{ matrix.group_id }}"
    )
    downloads = [
        step["with"]
        for step in workflow["jobs"]["ops-summary"]["steps"]
        if step.get("uses") == "actions/download-artifact@v4"
    ]
    assert any(
        step.get("pattern") == results.replace("${{ matrix.group_id }}", "*")
        for step in downloads
    )


@pytest.fixture
def runner():
    path = ROOT / "scripts/ci/ppu_ops_runner.py"
    spec = importlib.util.spec_from_file_location("ppu_ops_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "selector,mode,device,complete",
    itertools.product(
        ["success", "failure", "cancelled", "skipped"],
        ["none", "subset", "full", "invalid"],
        ["success", "failure", "cancelled", "skipped"],
        [True, False],
    ),
)
def test_gate_truth_table(runner, selector, mode, device, complete):
    expected = selector == "success" and (
        (mode == "none" and device == "skipped")
        or (mode in ("subset", "full") and device == "success" and complete)
    )
    assert (
        runner.gate_passes(
            selector, mode, device, has_tests=mode != "none", reports_complete=complete
        )
        is expected
    )
    assert (
        not runner.gate_passes(
            selector, mode, device, has_tests=False, reports_complete=complete
        )
        or mode == "none"
    )


@pytest.fixture
def evidence(runner, tmp_path):
    config = json.loads((ROOT / "scripts/ci/ppu_ops_tests.json").read_bytes())
    config["test_catalog"] = [
        {
            "id": "sample",
            "test": "tests/e2e/test_sample.py",
            "boards": ["ppu10", "ppu15"],
            "requires": ["torch", "triton", "vllm", "vllm_sail"],
            "board_overrides": {},
        }
    ]
    config_bytes = json.dumps(config).encode()
    env_bytes = (ROOT / "scripts/ci/ppu_ops_environment.json").read_bytes()
    image = "registry/image@sha256:" + "a" * 64
    fingerprint = dict(
        sdk_sha256="a" * 64,
        torch_sha256="b" * 64,
        pytorch_sail_arch=["ppu_10", "ppu_15"],
        build_image=image,
        vllm_commit="c" * 40,
    )
    selection = {
        "schema": 1,
        "mode": "full",
        "reasons": ["forced_full"],
        "matched_rule_ids": [],
        "changed_files": [],
        "unmapped_files": [],
        "test_ids": ["sample"],
        "files": ["tests/e2e/test_sample.py"],
        **runner.select.route_tests(config, ["sample"]),
        "base_sha": None,
        "head_sha": None,
        "tested_sha": "d" * 40,
        "build_run_id": "123",
        "config_sha256": runner.select.checksum(config_bytes),
        "environment_sha256": runner.select.checksum(env_bytes),
        "build_fingerprint": fingerprint,
        "runtime_image": image,
    }
    raw = json.dumps(selection).encode()
    dirs = []
    for group in selection["groups"]:
        directory = tmp_path / group["group_id"]
        directory.mkdir()
        dirs.append(directory)
        identity = dict(
            run_id="123",
            run_attempt="1",
            tested_sha="d" * 40,
            board=group["board"],
            group_id=group["group_id"],
        )
        profile = dict(
            capability=[8, 0],
            sdk_identity="2.2",
            driver_identity="1.2",
            packages=dict(torch="2", triton="3", pla="1"),
            runtime_env={"VLLM_SAIL_USE_PLA": True},
        )
        documents = {
            "selection.json": selection,
            "effective-selection.json": runner.effective_selection(
                config, selection, raw, group["board"], True
            ),
            "environment.json": {
                "schema": 1,
                **identity,
                "profile": profile,
                "expanded": True,
                "expansion_reason": "runtime_environment_changed",
                "runtime_image": image,
            },
            "summary.json": {
                "schema": 1,
                **identity,
                "status": "success",
                "failure_stage": None,
                "reason": None,
            },
            "wheel-manifest.json": {
                "schema": 1,
                "commits": {"vllm": "c" * 40, "vllm-sail": "d" * 40},
            },
            "timings.json": {
                "schema": 1,
                **identity,
                "config_sha256": selection["config_sha256"],
                "build_fingerprint": fingerprint,
                "runtime_image": image,
                "nproc_per_node": 1,
                "phases": {
                    k: 1
                    for k in (
                        "queue_seconds",
                        "install_seconds",
                        "preflight_seconds",
                        "device_occupied_seconds",
                        "total_seconds",
                    )
                },
                "files": [
                    {
                        "test_id": "sample",
                        "file": "tests/e2e/test_sample.py",
                        "duration_seconds": 1,
                        "outcome": "success",
                    }
                ],
            },
        }
        for name, value in documents.items():
            (directory / name).write_text(json.dumps(value))
        diagnostics = {
            **fingerprint,
            "pytorch_sail_arch": "ppu_15;ppu_10",
            "vllm_sail_commit": "d" * 40,
            "status": "succeeded",
        }
        (directory / "build-manifest.txt").write_text(
            "".join(f"{k}={v}\n" for k, v in diagnostics.items())
        )
        for name in ("logs", "nodes", "junit"):
            (directory / name).mkdir()
        node = "tests/e2e/test_sample.py::test_ok"
        report = dict(
            schema=1,
            board=group["board"],
            file="tests/e2e/test_sample.py",
            status="success",
            exitstatus=0,
            errors=[],
            collected=[node],
            applicable=[node],
            hardware_excluded=[],
            nodes={
                node: {
                    phase: dict(outcome="passed", wasxfail=None, duration_seconds=1)
                    for phase in ("setup", "call", "teardown")
                }
            },
            runtime={
                "isolated_root": "/tmp/isolated",
                "checkout_root": "/workspace/source",
                "runtime_env": {"VLLM_SAIL_USE_PLA": True},
                "paths": {
                    **{
                        name: "/site-packages/" + name.replace(".", "/") + ".py"
                        for name in runner.load(
                            ROOT / "tests/support/ppu_ci.py"
                        ).RUNTIME_MODULES
                        | {"triton"}
                    },
                    "plugin": "/tmp/isolated/tests/support/ppu_ci.py",
                    "conftest": "/tmp/isolated/tests/conftest.py",
                },
            },
        )
        (directory / "nodes/sample.json").write_text(json.dumps(report))
        (directory / "junit/sample.xml").write_text(
            '<testsuites><testsuite><testcase classname="tests.e2e.test_sample" name="test_ok"/></testsuite></testsuites>'
        )
        (directory / "logs/sample.log").write_text("pytest output")
    return config, raw, dirs


def test_summary_requires_all_groups(runner, evidence):
    config, raw, dirs = evidence
    runner.validate_results(config, raw, dirs, "123", "1")
    for broken in (dirs[:1], dirs + dirs[:1]):
        with pytest.raises(ValueError):
            runner.validate_results(config, raw, broken, "123", "1")


@pytest.mark.parametrize(
    "name,mutation",
    [
        ("summary.json", lambda s: s.update(run_attempt="2")),
        ("summary.json", lambda s: s.update(status="failure")),
        ("summary.json", lambda s: s.update(board="ppu15")),
        ("effective-selection.json", lambda s: s["groups"][0]["files"].clear()),
        (
            "effective-selection.json",
            lambda s: s["groups"].append(copy.deepcopy(s["groups"][0])),
        ),
        ("timings.json", lambda s: s["files"].clear()),
        ("nodes/sample.json", lambda s: s["nodes"].clear()),
        ("environment.json", lambda s: s.update(runtime_image="wrong")),
        ("wheel-manifest.json", lambda s: s["commits"].update(vllm="e" * 40)),
    ],
)
def test_incomplete_or_stale_group_rejected(runner, evidence, name, mutation):
    config, raw, dirs = evidence
    path = dirs[0] / name
    document = json.loads(path.read_bytes())
    mutation(document)
    path.write_text(json.dumps(document))
    with pytest.raises((ValueError, KeyError)):
        runner.validate_results(config, raw, dirs, "123", "1")


def test_summary_cli_rejects_extra_partial_artifact(
    runner, evidence, tmp_path, monkeypatch
):
    config, raw, _ = evidence
    orphan = tmp_path / "unknown-artifact"
    orphan.mkdir()
    (orphan / "build-manifest.txt").write_text("only diagnostics")
    monkeypatch.setattr(
        runner, "read_inputs", lambda env: (raw, None, config, None, None)
    )
    monkeypatch.setenv("CI_RUN_ID", "123")
    monkeypatch.setenv("CI_RUN_ATTEMPT", "1")
    monkeypatch.setattr(
        runner.sys, "argv", ["runner", "summary", "--results", str(tmp_path)]
    )
    with pytest.raises((ValueError, OSError)):
        runner.main()


@pytest.mark.parametrize(
    "filename",
    [
        "ppu-ops-tests.yaml",
        "_ppu-model-tests.yaml",
        "_ppu-ops-tests.yaml",
    ],
)
def test_device_workflows_have_no_chinese_text(filename):
    text = (ROOT / ".github/workflows" / filename).read_text()
    for line_number, line in enumerate(text.splitlines(), start=1):
        assert not re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", line), (
            f"{filename}:{line_number}: translate workflow text to English: {line}"
        )


@pytest.mark.parametrize("filename", ["ppu-ops-tests.yaml", "_ppu-ops-tests.yaml"])
def test_device_workflows_do_not_expose_qualification_mode(filename):
    text = (ROOT / ".github/workflows" / filename).read_text()
    assert "qualification_mode" not in text.lower()


def test_manual_ops_image_input_contract():
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/ppu-ops-tests.yaml").read_text()
    )
    inputs = workflow.get("on", workflow.get(True))["workflow_dispatch"]["inputs"]
    assert "ops_image" in inputs
    assert "device_image" not in inputs
    assert inputs["ops_image"]["type"] == "string"
    assert inputs["ops_image"]["required"] is True
    assert "default" not in inputs["ops_image"]
    assert (
        "download.t-head.cn/docker_release/pytorch" in inputs["build_image"]["default"]
    )
    assert re.fullmatch(r".+@sha256:[0-9a-f]{64}", inputs["build_image"]["default"])
    step = next(
        s
        for s in workflow["jobs"]["detect-ops-changes"]["steps"]
        if s.get("id") == "detect"
    )
    assert step["env"]["OPS_IMAGE_OVERRIDE"] == "${{ inputs.ops_image }}"
    assert "DEVICE_IMAGE_OVERRIDE" not in step["env"]


def test_manual_detection_accepts_unapproved_environment(tmp_path, monkeypatch):
    from scripts.ci import ppu_ops_select as selector

    path = ROOT / ".github/workflows/ppu-ops-tests.yaml"
    workflow = yaml.safe_load(path.read_text())
    step = next(
        s
        for s in workflow["jobs"]["detect-ops-changes"]["steps"]
        if s.get("id") == "detect"
    )
    script = step["run"].removeprefix("python - <<'PY'\n").removesuffix("PY\n")
    image = "registry/image@sha256:" + "a" * 64
    ops_image = "registry/ops@sha256:" + "b" * 64
    output = tmp_path / "github-output"
    environment_path = ROOT / "scripts/ci/ppu_ops_environment.json"
    original_environment = environment_path.read_bytes()
    assert json.loads(original_environment)["qualified"] is False
    monkeypatch.chdir(ROOT)
    monkeypatch.delenv("QUALIFICATION_MODE", raising=False)
    monkeypatch.delenv("DEVICE_IMAGE_OVERRIDE", raising=False)
    for name, value in {
        "BUILD_IMAGE_OVERRIDE": image,
        "OPS_IMAGE_OVERRIDE": ops_image,
        "VLLM_REF_OVERRIDE": "c" * 40,
        "GITHUB_SHA": "d" * 40,
        "GITHUB_OUTPUT": str(output),
    }.items():
        monkeypatch.setenv(name, value)
    write_json = selector.write_json
    monkeypatch.setattr(
        selector, "write_json", lambda p, value: write_json(tmp_path / p, value)
    )
    exec(compile(script, str(path), "exec"), {})
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert values == {
        "build_image": image,
        "runtime_image": ops_image,
        "expected_vllm_commit": "c" * 40,
    }
    assert json.loads((tmp_path / "changes.json").read_bytes())["tested_sha"] == (
        "d" * 40
    )
    assert environment_path.read_bytes() == original_environment


def test_manual_workflow_does_not_change_pr_gate():
    pipeline = yaml.safe_load(
        (ROOT / ".github/workflows/ppu-build-and-test.yaml").read_text()
    )["jobs"]
    assert pipeline["gate"]["needs"] == [
        "runtime-image",
        "build",
        "pin-unit-tests",
        "model-tests",
        "upstream-reuse",
    ]
    assert not any("ops-" in job_id for job_id in pipeline)
    assert pipeline["build"]["needs"] == "runtime-image"
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/ppu-ops-tests.yaml").read_text()
    )
    jobs = workflow["jobs"]
    assert "gate" not in jobs
    assert "detect-ops-changes" in jobs
    assert "if" not in jobs["detect-ops-changes"]
    assert set(workflow.get(True, workflow.get("on", {}))) == {"workflow_dispatch"}
    assert jobs["ops-tests"]["uses"] == "./.github/workflows/_ppu-ops-tests.yaml"
    assert (
        jobs["ops-build"]["container"]["image"]
        == "${{ needs.detect-ops-changes.outputs.build_image }}"
    )
    assert (
        jobs["ops-build"]["env"]["VLLM_SAIL_BUILD_IMAGE"]
        == jobs["ops-build"]["container"]["image"]
    )


@pytest.mark.parametrize(
    "owner,run_id,attempt",
    [
        ("t-head", "12345678901", "1"),
        ("t-head", "9" * 19, "9"),
        ("Long_Organization_Name_123", "9" * 13, "9"),
    ],
)
def test_scheduler_names_retain_unique_group_suffix(runner, owner, run_id, attempt):
    groups = [{"group_id": "ppu10-0"}, {"group_id": "ppu15-0"}]
    names = runner.scheduler_names(groups, owner, run_id, attempt)
    normalized_owner = "t-head" if owner == "t-head" else "long-organization-na"
    assert names == [
        f"ppu-{normalized_owner}-{run_id}-{attempt}-ops-{g['group_id']}" for g in groups
    ]
    assert len(set(names)) == 2
    assert all(len(name) <= 52 for name in names)


def test_prepare_rejects_scheduler_name_truncation_before_nas(
    runner, evidence, monkeypatch
):
    config, raw, _ = evidence
    monkeypatch.setattr(
        runner, "read_inputs", lambda env: (raw, None, config, None, None)
    )
    monkeypatch.setenv("GITHUB_REPOSITORY_OWNER", "t-head")
    monkeypatch.setenv("CI_RUN_ID", "9" * 40)
    monkeypatch.setenv("CI_RUN_ATTEMPT", "1")
    monkeypatch.setenv("DEVICE_GROUP_ID", "ppu10-0")
    monkeypatch.setattr(runner.sys, "argv", ["runner", "prepare"])

    def unexpected_nas(_):
        pytest.fail("Invalid job names must not prepare NAS or dispatch devices")

    monkeypatch.setattr(runner, "require_nas", unexpected_nas)
    with pytest.raises(ValueError, match="job name"):
        runner.main()


def test_reusable_workflow_contract():
    path = ROOT / ".github/workflows/_ppu-ops-tests.yaml"
    assert path.is_file(), "Missing reusable workflow"
    workflow = yaml.safe_load(path.read_text())
    assert workflow["permissions"] == {"contents": "read", "actions": "read"}
    jobs = workflow["jobs"]
    device = jobs["ops-tests"]
    assert device["runs-on"] == "8-ppu-m890p"
    assert device["container"]["image"] == "${{ inputs.image }}"
    assert device["timeout-minutes"] == 90
    assert device["strategy"]["fail-fast"] is False
    assert device["strategy"]["max-parallel"] == 2
    scheduler = next(
        s for s in device["steps"] if "ppu-scheduler-action" in s.get("uses", "")
    )
    assert scheduler["uses"] == (
        "t-head/ppu-scheduler-action@a4e03cbbdb2624f4871fff30d7081b34cbc4b7d4"
    )
    assert scheduler["with"]["timeout_minutes"] == 60
    assert scheduler["with"]["job_suffix"] == "ops-${{ matrix.group_id }}"
    assert (
        scheduler["with"]["node_selector"]
        == "${{ steps.prepare.outputs.node_selector }}"
    )
    assert scheduler["with"]["cleanup_policy"] == "always"
    uploads = [s for s in device["steps"] if "upload-artifact" in s.get("uses", "")]
    assert all(
        s["if"] == "always()" and s["with"]["if-no-files-found"] == "error"
        for s in uploads
    )
    assert all(s["with"]["retention-days"] == 14 for s in uploads)
    assert all("/wl_nas/devops/" in s["with"]["path"] for s in uploads)
    assert jobs["ops-summary"]["if"] == "always()"
    assert "continue-on-error" not in path.read_text()


@pytest.mark.parametrize(
    "filename,job_id",
    [
        ("ppu-ops-tests.yaml", "ops-build"),
        ("_ppu-ops-tests.yaml", "ops-tests"),
    ],
)
def test_container_checkout_uses_isolated_regular_git_config(
    filename, job_id, tmp_path
):
    workflow = yaml.safe_load((ROOT / ".github/workflows" / filename).read_text())
    job = workflow["jobs"][job_id]
    job_env = {**workflow.get("env", {}), **job.get("env", {})}
    assert job_env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert "GIT_CONFIG_GLOBAL" not in job_env
    steps = job["steps"]
    prepare = next((s for s in steps if s.get("id") == "git-config"), None)
    assert prepare is not None, (
        "Prepare a writable, isolated Git config before checkout"
    )
    checkout = next(s for s in steps if s.get("uses") == "actions/checkout@v4")
    assert steps.index(prepare) < steps.index(checkout)
    assert "if" not in prepare
    assert prepare["shell"] == "bash"

    # Simulate image and temporary checkout HOME settings without changing user or system configuration.
    home = tmp_path / "home"
    home.mkdir()
    original = '[url "https://mirror.invalid/"]\n\tinsteadOf = https://github.com/\n'
    global_config = home / ".gitconfig"
    global_config.write_text(original)
    system_config = tmp_path / "system.gitconfig"
    system_config.write_text(original)
    runner_temp = tmp_path / "runner temp"
    runner_temp.mkdir()
    exports = tmp_path / "github-env"
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("GIT_") and k not in {"BASH_ENV", "ENV"}
    }
    env.update(
        HOME=str(home),
        XDG_CONFIG_HOME=str(home),
        GIT_CONFIG_SYSTEM=str(system_config),
        RUNNER_TEMP=str(runner_temp),
        GITHUB_ENV=str(exports),
    )
    url = "https://github.com/t-head/vllm-sail"

    def git(*args, environment):
        return subprocess.run(
            ["git", *args],
            cwd=tmp_path,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    assert git("ls-remote", "--get-url", url, environment=env) != url
    env["GIT_CONFIG_NOSYSTEM"] = job_env["GIT_CONFIG_NOSYSTEM"]
    paths = []
    for _ in range(2):
        exports.write_text("")
        subprocess.run(
            ["bash", "--noprofile", "--norc", "-euo", "pipefail", "-c", prepare["run"]],
            cwd=tmp_path,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        values = dict(line.split("=", 1) for line in exports.read_text().splitlines())
        assert set(values) == {"GIT_CONFIG_GLOBAL"}
        path = Path(values["GIT_CONFIG_GLOBAL"])
        assert path.is_file() and not path.is_symlink()
        assert path.parent == runner_temp
        assert path.read_text() == ""
        assert path.stat().st_mode & 0o777 == 0o600
        paths.append(path)
        isolated_env = {**env, **values}
        assert git("ls-remote", "--get-url", url, environment=isolated_env) == url
        # Simulate safe.directory with a regular file and verify temporary HOME cannot override the isolated config.
        path.write_text("[safe]\n\tdirectory = /workspace/source\n")
        assert (
            git(
                "config",
                "--global",
                "--get-all",
                "safe.directory",
                environment=isolated_env,
            )
            == "/workspace/source"
        )
    assert paths[0] != paths[1]
    assert global_config.read_text() == system_config.read_text() == original


@pytest.mark.parametrize("job_id", ["ops-tests", "ops-summary"])
@pytest.mark.parametrize("override", [None, "https://packages.example/simple/"])
def test_cpu_dependencies_use_explicit_test_index(job_id, override, tmp_path):
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/_ppu-ops-tests.yaml").read_text()
    )
    assert (
        workflow["env"].get("SAIL_PIP_INDEX_URL")
        == "https://mirrors.aliyun.com/pypi/simple/"
    )
    job = workflow["jobs"][job_id]
    step = next(s for s in job["steps"] if "pip install" in s.get("run", ""))
    index = override or workflow["env"]["SAIL_PIP_INDEX_URL"]
    # Run the actual YAML shell; intercept only the Python boundary to avoid network installs.
    result = subprocess.run(
        [
            "bash",
            "--noprofile",
            "--norc",
            "-euo",
            "pipefail",
            "-c",
            "python() { printf '%s\\n' \"$*\"; }\n" + step["run"],
        ],
        env={
            "PATH": os.defpath,
            "SAIL_PIP_INDEX_URL": index,
            "PIP_INDEX_URL": "https://unreachable.example/simple/",
            "DEVICE_JOB_RESULT": "success",
        },
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    commands = [shlex.split(line) for line in result.stdout.splitlines()]
    assert commands[0] == [
        "-m",
        "pip",
        "install",
        "--index-url",
        index,
        "-r",
        "requirements/dev.txt",
    ]
    assert commands[1][0] == "scripts/ci/ppu_ops_runner.py"


def test_scheduler_forwards_test_index_without_changing_image():
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/_ppu-ops-tests.yaml").read_text()
    )
    device = workflow["jobs"]["ops-tests"]
    scheduler = next(
        s for s in device["steps"] if "ppu-scheduler-action" in s.get("uses", "")
    )
    extra_env = dict(
        item.split("=", 1) for item in scheduler["with"]["extra_env"].split(",")
    )
    assert extra_env.get("SAIL_PIP_INDEX_URL") == "${{ env.SAIL_PIP_INDEX_URL }}"
    assert (
        device["container"]["image"]
        == scheduler["with"]["image"]
        == "${{ inputs.image }}"
    )


@pytest.fixture
def worker_launch(tmp_path):
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/_ppu-ops-tests.yaml").read_text()
    )
    scheduler = next(
        s
        for s in workflow["jobs"]["ops-tests"]["steps"]
        if "ppu-scheduler-action" in s.get("uses", "")
    )
    # Run the actual worker command and record exports at Python entry, without installing packages or using devices.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"${CUDA_VISIBLE_DEVICES-<unset>}\" "
        '"${NVIDIA_VISIBLE_DEVICES-<unset>}" "$@"\n'
    )
    python.chmod(0o755)
    command = scheduler["with"]["command"].replace(
        "/workspace/source", shlex.quote(str(ROOT))
    )

    def launch(env):
        return subprocess.run(
            ["bash", "--noprofile", "--norc", "-c", command],
            env={"PATH": str(bin_dir) + os.pathsep + os.defpath, **env},
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=10,
        )

    return launch


@pytest.mark.parametrize("allocated", ["0", "2", "4", "15"])
@pytest.mark.parametrize("existing_cuda", [None, "", "0"])
def test_worker_exports_allocated_device_before_python(
    worker_launch, allocated, existing_cuda
):
    env = {"NVIDIA_VISIBLE_DEVICES": allocated}
    if existing_cuda is not None:
        env["CUDA_VISIBLE_DEVICES"] = existing_cuda
    result = worker_launch(env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[:2] == [allocated, allocated]
    assert result.stdout.splitlines()[-1] == "run"


@pytest.mark.parametrize("allocated", [None, ""])
@pytest.mark.parametrize("existing_cuda", [None, "0"])
def test_worker_rejects_missing_allocation_before_python(
    worker_launch, allocated, existing_cuda
):
    env = {}
    if allocated is not None:
        env["NVIDIA_VISIBLE_DEVICES"] = allocated
    if existing_cuda is not None:
        env["CUDA_VISIBLE_DEVICES"] = existing_cuda
    result = worker_launch(env)
    assert result.returncode != 0
    assert "NVIDIA_VISIBLE_DEVICES" in result.stderr
    assert result.stdout == "", "Python must not start without allocation metadata"


def test_summary_does_not_inherit_container_git_config():
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/_ppu-ops-tests.yaml").read_text()
    )
    job = workflow["jobs"]["ops-summary"]
    env = {**workflow.get("env", {}), **job.get("env", {})}
    assert "GIT_CONFIG_GLOBAL" not in env
    assert "GIT_CONFIG_NOSYSTEM" not in env


@pytest.mark.parametrize(
    "filename,job_id",
    [
        ("ppu-ops-tests.yaml", "detect-ops-changes"),
        ("ppu-ops-tests.yaml", "ops-build"),
        ("ppu-ops-tests.yaml", "select-ops-tests"),
        ("_ppu-ops-tests.yaml", "ops-tests"),
        ("_ppu-ops-tests.yaml", "ops-summary"),
    ],
)
def test_device_checkout_is_bound_to_workflow_sha(filename, job_id):
    workflow = yaml.safe_load((ROOT / ".github/workflows" / filename).read_text())
    checkout = next(
        s
        for s in workflow["jobs"][job_id]["steps"]
        if s.get("uses") == "actions/checkout@v4"
    )
    assert checkout["with"]["ref"] == "${{ github.sha }}"
    assert checkout["with"]["persist-credentials"] is False


def test_device_identity_cannot_be_overridden_by_reusable_input():
    caller = yaml.safe_load((ROOT / ".github/workflows/ppu-ops-tests.yaml").read_text())
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/_ppu-ops-tests.yaml").read_text()
    )
    triggers = workflow.get(True, workflow.get("on", {}))
    assert "tested_sha" not in triggers["workflow_call"]["inputs"]
    assert "tested_sha" not in caller["jobs"]["ops-tests"]["with"]
    assert workflow["env"]["EXPECTED_SAIL_COMMIT"] == "${{ github.sha }}"
    scheduler = next(
        s
        for s in workflow["jobs"]["ops-tests"]["steps"]
        if "ppu-scheduler-action" in s.get("uses", "")
    )
    assert "EXPECTED_SAIL_COMMIT=${{ github.sha }}," in scheduler["with"]["extra_env"]
