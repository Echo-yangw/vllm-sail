# SPDX-License-Identifier: Apache-2.0
import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github/workflows"
DEFAULT_BRANCH = "${{ github.event.repository.default_branch }}"
EXPECTED_SHA = "${{ inputs.pr_head_sha || github.sha }}"
DEFAULT_REF_GUARD = (
    "github.ref == format('refs/heads/{0}', github.event.repository.default_branch)"
)
BUILD_IMAGE = (
    "download.t-head.cn/docker_release/pytorch@sha256:"
    "a33a67832f7ec442c481421e94255518dd67e464406f8e610c15d2b594b21f80"
)


def workflow(name):
    path = WORKFLOWS / name
    assert path.is_file(), f"Missing workflow: {name}"
    return yaml.safe_load(path.read_text())


def checkouts(job):
    return [
        step
        for step in job["steps"]
        if step.get("uses", "").startswith("actions/checkout@")
    ]


@pytest.mark.parametrize(
    "name,job_id,ref",
    [
        ("run-ci-command.yml", "command", DEFAULT_BRANCH),
        ("notify-ci-authorized.yml", "notify", DEFAULT_BRANCH),
        ("ppu-pr-ci.yaml", "validate", "${{ github.sha }}"),
        ("ppu-pr-ci.yaml", "report", "${{ github.sha }}"),
    ],
)
def test_control_jobs_only_execute_trusted_code(name, job_id, ref):
    job = workflow(name)["jobs"][job_id]
    assert job["runs-on"] == "ubuntu-latest"
    assert len(checkouts(job)) == 1
    checkout = checkouts(job)[0]
    assert checkout["with"]["ref"] == ref
    assert checkout["with"]["persist-credentials"] is False
    if name == "ppu-pr-ci.yaml":
        assert DEFAULT_REF_GUARD in job.get("if", checkout.get("if", "")) or (
            job_id == "report"
            and all(DEFAULT_REF_GUARD in step.get("if", "") for step in job["steps"])
        )


def test_ppu_jobs_use_hosted_build_and_self_hosted_runtime_runners():
    assert workflow("_ppu-build.yaml")["jobs"]["build"]["runs-on"] == "ubuntu-24.04"
    assert (
        workflow("_ppu-model-tests.yaml")["jobs"]["prepare"]["runs-on"]
        == "ubuntu-latest"
    )
    for filename, job_id in (
        ("_ppu-pin-ut.yaml", "pin-unit-tests"),
        ("_ppu-model-tests.yaml", "e2e"),
        ("_ppu-ops-tests.yaml", "ops-tests"),
        ("upstream-reuse.yaml", "nightly-ppu-reuse"),
    ):
        assert workflow(filename)["jobs"][job_id]["runs-on"] == "8-ppu-m890p"


def test_controller_filters_comments_serializes_all_commands_and_reports_failure():
    config = workflow("run-ci-command.yml")
    assert config[True] == {"issue_comment": {"types": ["created"]}}
    job = config["jobs"]["command"]
    assert "github.event.issue.pull_request" in job["if"]
    assert "github.event.comment.user.type == 'User'" in job["if"]
    assert "contains(github.event.comment.body, '/ci')" in job["if"]
    assert job["concurrency"] == {
        "group": "ppu-ci-command-${{ github.event.issue.number }}",
        "cancel-in-progress": False,
        "queue": "max",
    }
    fallback = next(step for step in job["steps"] if step.get("if") == "failure()")
    assert fallback["uses"].startswith("actions/github-script@")
    assert "ppu-ci-command:" in fallback["with"]["script"]
    assert "context.runId" in fallback["with"]["script"]


def test_notify_only_receives_label_events_with_minimum_permissions():
    config = workflow("notify-ci-authorized.yml")
    assert config[True] == {"pull_request_target": {"types": ["labeled"]}}
    job = config["jobs"]["notify"]
    assert "ready-run-all-tests" in job["if"]


# GITHUB_TOKEN needs pull-requests: write to comment or react on a PR;
# issues: write alone returns 403 "Resource not accessible by integration".
@pytest.mark.parametrize(
    "name,job_id,expected",
    [
        (
            "run-ci-command.yml",
            "command",
            {
                "contents": "read",
                "actions": "write",
                "issues": "write",
                "pull-requests": "write",
                "statuses": "write",
            },
        ),
        (
            "notify-ci-authorized.yml",
            "notify",
            {"contents": "read", "issues": "write", "pull-requests": "write"},
        ),
        (
            "ppu-pr-ci.yaml",
            "report",
            {
                "contents": "read",
                "actions": "read",
                "statuses": "write",
                "issues": "write",
                "pull-requests": "write",
            },
        ),
    ],
)
def test_pr_commenting_jobs_can_write_pull_request_comments(name, job_id, expected):
    assert workflow(name)["jobs"][job_id]["permissions"] == expected


def test_executor_uses_only_dispatch_with_a_stable_request_identity():
    config = workflow("ppu-pr-ci.yaml")
    assert set(config[True]) == {"workflow_dispatch"}
    inputs = config[True]["workflow_dispatch"]["inputs"]
    required_inputs = {"pr_number", "head_sha", "requester", "comment_id"}
    assert set(inputs) == required_inputs
    for name in required_inputs:
        assert inputs[name]["type"] == "string" and inputs[name]["required"]
    assert config["run-name"] == (
        "PPU PR #${{ inputs.pr_number }} @ ${{ inputs.head_sha }} "
        "by ${{ inputs.requester }} (comment ${{ inputs.comment_id }})"
    )
    assert config["permissions"] == {}
    assert config["cache-mode"] == "none"
    assert config["concurrency"] == {
        "group": "ppu-pr-ci-${{ inputs.pr_number }}",
        "cancel-in-progress": False,
        "queue": "max",
    }
    assert set(config["jobs"]) == {"validate", "ppu", "report"}


def test_executor_separates_read_only_pr_execution_and_always_reporting():
    jobs = workflow("ppu-pr-ci.yaml")["jobs"]
    ppu = jobs["ppu"]
    assert ppu["needs"] == "validate"
    assert ppu["uses"] == "./.github/workflows/ppu-build-and-test.yaml"
    assert ppu["permissions"] == {"contents": "read", "actions": "read"}
    assert "secrets" not in ppu
    assert ppu["with"] == {
        "pr_head_sha": "${{ inputs.head_sha }}",
        "skip_ppu": "${{ needs.validate.outputs.skip_ppu == 'true' }}",
        "existing_build_run_id": "${{ needs.validate.outputs.existing_build_run_id }}",
        "native_reuse_run_id": "${{ needs.validate.outputs.native_reuse_run_id }}",
        "wheel_source_sha": "${{ needs.validate.outputs.wheel_source_sha }}",
    }
    validate = jobs["validate"]
    assert validate["permissions"] == {
        "contents": "read",
        "actions": "read",
        "pull-requests": "read",
        "statuses": "write",
    }
    step = next(step for step in validate["steps"] if "run" in step)
    assert step["run"] == "python -m scripts.ci.ppu_ci_command validate"
    assert "continue-on-error" not in step
    assert validate["outputs"] == {
        key: "${{ steps." + step["id"] + ".outputs." + key + " }}"
        for key in (
            "skip_ppu",
            "existing_build_run_id",
            "native_reuse_run_id",
            "wheel_source_sha",
        )
    }
    report = jobs["report"]
    assert report["if"] == "always()"
    assert report["needs"] == ["validate", "ppu"]
    assert all("always()" in step["if"] for step in report["steps"])
    step = next(step for step in report["steps"] if "run" in step)
    assert step["env"]["NEEDS"] == "${{ toJSON(needs) }}"
    assert "continue-on-error" not in step
    assert step["run"] == "python -m scripts.ci.ppu_ci_command report"


@pytest.mark.parametrize(
    "name",
    [
        "ppu-build-and-test.yaml",
        "_ppu-build.yaml",
        "_ppu-pin-ut.yaml",
        "_ppu-model-tests.yaml",
        "ppu-ops-tests.yaml",
        "ppu-debug.yaml",
    ],
)
def test_untrusted_workflows_have_no_write_permissions_or_inherited_secrets(name):
    config = workflow(name)
    assert config["cache-mode"] == "none"
    assert config["permissions"] == {"contents": "read", "actions": "read"}
    for job in config["jobs"].values():
        assert "secrets" not in job
        assert all(value != "write" for value in job.get("permissions", {}).values())
        for step in checkouts(job) if "steps" in job else []:
            assert step["with"]["persist-credentials"] is False


@pytest.mark.parametrize(
    "name,job_ids",
    [
        ("_ppu-build.yaml", ["build"]),
        ("_ppu-pin-ut.yaml", ["pin-unit-tests"]),
        ("_ppu-model-tests.yaml", ["prepare", "e2e"]),
    ],
)
def test_every_pr_checkout_verifies_the_requested_sha_before_execution(name, job_ids):
    config = workflow(name)
    inputs = config[True]["workflow_call"]["inputs"]
    if name == "_ppu-build.yaml":
        assert "pr_mode" not in inputs
    else:
        assert inputs["pr_mode"]["type"] == "boolean"
        assert inputs["pr_mode"]["default"] is False
    assert inputs["pr_head_sha"]["type"] == "string"
    assert inputs["pr_head_sha"]["default"] == ""
    for job_id in job_ids:
        job = config["jobs"][job_id]
        checkout = checkouts(job)[0]
        assert checkout["with"]["ref"] == EXPECTED_SHA
        index = job["steps"].index(checkout)
        verify = job["steps"][index + 1]
        assert verify["env"]["EXPECTED_SHA"] == EXPECTED_SHA
        if "SOURCE_DIR" in verify["env"]:
            assert checkout["with"]["path"] == "${{ steps.runtime.outputs.source_rel }}"
            assert verify["env"]["SOURCE_DIR"] == "${{ steps.runtime.outputs.source }}"
            assert (
                'test "$(git -c safe.directory="$SOURCE_DIR" -C "$SOURCE_DIR" rev-parse HEAD)" = "$EXPECTED_SHA"'
                == verify["run"]
            )
        else:
            assert (
                'test "$(git -c safe.directory="$PWD" rev-parse HEAD)" = "$EXPECTED_SHA"'
                == verify["run"]
            )
        guard = next(
            step
            for step in job["steps"]
            if step.get("name") == "Require disabled cache access"
        )
        assert job["steps"].index(guard) < index
        assert "process.env.ACTIONS_CACHE_MODE !== 'none'" in guard["with"]["script"]


@pytest.mark.parametrize(
    "name,job_id",
    [
        ("_ppu-build.yaml", "build"),
        ("_ppu-pin-ut.yaml", "pin-unit-tests"),
        ("_ppu-model-tests.yaml", "prepare"),
        ("_ppu-model-tests.yaml", "e2e"),
    ],
)
def test_sha_verification_allows_different_owner_but_rejects_wrong_sha(
    name, job_id, tmp_path
):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    sha = "a" * 40
    (tmp_path / ".git" / "HEAD").write_text(sha + "\n")
    env = {
        **os.environ,
        "LC_ALL": "C",
        "GIT_TEST_ASSUME_DIFFERENT_OWNER": "1",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "safe.directory",
        "GIT_CONFIG_VALUE_0": "",
        "EXPECTED_SHA": sha,
    }
    untrusted = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert untrusted.returncode != 0
    assert "dubious ownership" in untrusted.stderr
    job = workflow(name)["jobs"][job_id]
    command = next(
        step["run"]
        for step in job["steps"]
        if step.get("name") == "Verify checked-out revision"
    )
    for expected, returncode in ((sha, 0), ("b" * 40, 1)):
        result = subprocess.run(
            ["sh", "-e", "-c", command],
            cwd=tmp_path,
            env={**env, "EXPECTED_SHA": expected, "SOURCE_DIR": str(tmp_path)},
            capture_output=True,
            text=True,
        )
        assert result.returncode == returncode, result.stderr
        assert not result.stderr


def test_pr_pipeline_structure():
    config = workflow("ppu-build-and-test.yaml")
    assert set(config[True]) == {"workflow_call", "pull_request"}
    inputs = config[True]["workflow_call"]["inputs"]
    assert set(inputs) == {
        "pr_head_sha",
        "skip_ppu",
        "existing_build_run_id",
        "native_reuse_run_id",
        "skip_device_tests",
        "wheel_source_sha",
    }
    assert inputs["pr_head_sha"]["type"] == "string"
    assert inputs["pr_head_sha"]["required"] is True
    assert inputs["skip_ppu"] == {
        "description": "Report success without allocating PPU runners for non-impacting changes",
        "type": "boolean",
        "default": False,
    }
    assert inputs["skip_device_tests"] == {
        "description": "Build wheels only, without the pin-UT and model-smoke lanes",
        "type": "boolean",
        "default": False,
    }
    for name in (
        "existing_build_run_id",
        "native_reuse_run_id",
        "wheel_source_sha",
    ):
        assert inputs[name]["type"] == "string"
        assert inputs[name]["default"] == ""
        assert not inputs[name].get("required", False)
    jobs = config["jobs"]
    assert set(jobs) == {
        "runtime-image",
        "build",
        "pin-unit-tests",
        "model-tests",
        "gate",
    }
    runtime = jobs["runtime-image"]
    assert runtime["runs-on"] == "8-ppu-m890p"
    assert "!inputs.skip_ppu" in runtime["if"]
    assert "outputs" not in runtime
    resolve = next(
        step
        for step in runtime["steps"]
        if step.get("name") == "Resolve the runner-configured CUDA-compatible runtime"
    )
    assert "ppu_runtime_image.sh" in resolve["run"]
    assert "PPU_CI_RUNTIME_IMAGE" not in resolve.get("env", {})
    build = jobs["build"]
    assert "ci-preview" in build["if"]
    assert "inputs.existing_build_run_id || ''" in build["if"]
    assert build["uses"] == "./.github/workflows/_ppu-build.yaml"
    assert build["with"] == {
        "pr_head_sha": "${{ inputs.pr_head_sha || github.sha }}",
        "native_reuse_run_id": "${{ inputs.native_reuse_run_id || '' }}",
    }
    assert build["needs"] == "runtime-image"
    assert jobs["gate"]["needs"] == [
        "runtime-image",
        "build",
        "pin-unit-tests",
        "model-tests",
    ]
    assert "always()" in jobs["gate"]["if"]
    assert "ci-preview" in jobs["gate"]["if"]
    gate = jobs["gate"]["steps"][0]
    assert gate["env"] == {
        "JOB_RESULTS": "${{ toJSON(needs) }}",
        "SKIP_PPU": "${{ inputs.skip_ppu }}",
        "SKIP_DEVICE_TESTS": "${{ inputs.skip_device_tests }}",
        "EXISTING_BUILD_RUN_ID": "${{ inputs.existing_build_run_id || '' }}",
        "WHEEL_SOURCE_SHA": "${{ inputs.wheel_source_sha || '' }}",
    }
    assert "PR_MODE" not in gate["run"]
    assert "smoke" not in gate["run"]
    for job_id, name, target in (
        ("pin-unit-tests", "PPU pin UT", "_ppu-pin-ut.yaml"),
        ("model-tests", "PPU model tests", "_ppu-model-tests.yaml"),
    ):
        job = jobs[job_id]
        assert job["name"] == name
        assert job["needs"] == ["runtime-image", "build"]
        assert job["uses"] == f"./.github/workflows/{target}"
        assert "!cancelled()" in job["if"]
        assert "ci-preview" in job["if"]
        assert "needs.build.result == 'success'" in job["if"]
        assert job["with"] == {
            "pr_mode": True,
            "pr_head_sha": "${{ inputs.pr_head_sha || github.sha }}",
            "wheel_source_sha": "${{ inputs.wheel_source_sha || '' }}",
            "build_run_id": "${{ inputs.existing_build_run_id || github.run_id }}",
        }


def test_docs_only_call_uses_hosted_gate_without_ppu_jobs():
    jobs = workflow("ppu-build-and-test.yaml")["jobs"]
    context = {
        "github.event_name": "workflow_dispatch",
        "inputs.skip_ppu": True,
        "inputs.existing_build_run_id": "",
        "inputs.wheel_source_sha": "",
        "needs.runtime-image.result": "skipped",
        "needs.build.result": "skipped",
    }
    for job_id in ("runtime-image", "build", "pin-unit-tests", "model-tests"):
        assert not expression(jobs[job_id]["if"], context)
    assert expression(jobs["gate"]["if"], context)
    gate = jobs["gate"]["steps"][0]["run"].replace(
        "python - <<'PYTHON'", f"{sys.executable} - <<'PYTHON'"
    )
    result = subprocess.run(
        ["bash", "-e", "-c", gate],
        env={
            **os.environ,
            "JOB_RESULTS": json.dumps(
                {
                    job: {"result": "skipped"}
                    for job in (
                        "runtime-image",
                        "build",
                        "pin-unit-tests",
                        "model-tests",
                    )
                }
            ),
            "SKIP_PPU": "true",
            "SKIP_DEVICE_TESTS": "false",
            "EXISTING_BUILD_RUN_ID": "",
            "WHEEL_SOURCE_SHA": "",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "PPU CI skipped: no PPU-impacting changes" in result.stdout


def test_skip_device_tests_guards_the_device_lanes():
    jobs = workflow("ppu-build-and-test.yaml")["jobs"]
    context = {
        "github.event_name": "workflow_dispatch",
        "inputs.skip_ppu": False,
        "inputs.skip_device_tests": True,
        "inputs.existing_build_run_id": "",
        "inputs.wheel_source_sha": "",
        "needs.runtime-image.result": "success",
        "needs.build.result": "success",
    }
    # Building wheels for a lane that tests elsewhere must not also allocate
    # the pin-UT and model-smoke hosts.
    for job_id in ("pin-unit-tests", "model-tests"):
        assert not expression(jobs[job_id]["if"], context)
    assert expression(jobs["build"]["if"], context)
    assert expression(jobs["gate"]["if"], context)


@pytest.mark.parametrize("skip_device_tests,returncode", [("true", 0), ("false", 1)])
def test_gate_honours_skip_device_tests(skip_device_tests, returncode):
    step = workflow("ppu-build-and-test.yaml")["jobs"]["gate"]["steps"][0]
    script = step["run"].replace(
        "python - <<'PYTHON'", f"{sys.executable} - <<'PYTHON'"
    )
    env = {
        **os.environ,
        "JOB_RESULTS": json.dumps(
            {
                "runtime-image": {"result": "success"},
                "build": {"result": "success"},
                "pin-unit-tests": {"result": "skipped"},
                "model-tests": {"result": "skipped"},
            }
        ),
        "SKIP_PPU": "false",
        "SKIP_DEVICE_TESTS": skip_device_tests,
        "EXISTING_BUILD_RUN_ID": "",
        "WHEEL_SOURCE_SHA": "",
    }
    result = subprocess.run(
        ["bash", "-e", "-c", script], env=env, capture_output=True, text=True
    )
    # Only the explicit switch tolerates the skipped lanes; the default stays
    # fail-closed so a broken device lane cannot hide behind it.
    assert result.returncode == returncode, result.stderr


def test_ops_manual_entry_structure():
    config = workflow("ppu-ops-tests.yaml")
    assert config["name"] == "PPU ops tests (manual)"
    assert set(config[True]) == {"workflow_dispatch"}
    assert config[True]["workflow_dispatch"]["inputs"] == {
        "build_image": {
            "description": "Digest-pinned SDK image; CI installs the checksum-pinned SAIL Torch",
            "required": False,
            "default": BUILD_IMAGE,
            "type": "string",
        },
        "ops_image": {
            "description": "Digest-pinned operator runtime image (required; value is not stored in the repository)",
            "required": True,
            "type": "string",
        },
        "ops_test_scope": {
            "description": "Initial manual runs always use full; auto remains disabled until qualification",
            "default": "full",
            "type": "choice",
            "options": ["auto", "full"],
        },
        "vllm_ref": {
            "description": "vLLM ref (blank uses .github/vllm-main-verified.commit)",
            "required": False,
            "default": "",
            "type": "string",
        },
        "pytorch_sail_arch": {
            "description": "Semicolon-separated HG architectures",
            "required": True,
            "default": "ppu_15;ppu_10",
            "type": "string",
        },
    }
    assert config["concurrency"] == {
        "group": "ppu-ops-tests-${{ github.ref }}",
        "cancel-in-progress": True,
    }
    jobs = config["jobs"]
    assert set(jobs) == {
        "detect-ops-changes",
        "ops-build",
        "select-ops-tests",
        "ops-tests",
    }
    assert "if" not in jobs["detect-ops-changes"]
    assert jobs["ops-build"]["needs"] == "detect-ops-changes"
    assert "if" not in jobs["ops-build"]
    assert jobs["select-ops-tests"]["needs"] == ["detect-ops-changes", "ops-build"]
    assert "if" not in jobs["select-ops-tests"]
    assert jobs["ops-tests"]["needs"] == ["detect-ops-changes", "select-ops-tests"]
    assert (
        jobs["ops-tests"]["if"] == "needs.select-ops-tests.outputs.has_tests == 'true'"
    )
    assert jobs["ops-tests"]["uses"] == "./.github/workflows/_ppu-ops-tests.yaml"


def test_debug_structure():
    config = workflow("ppu-debug.yaml")
    assert config["name"] == "PPU debug"
    assert set(config[True]) == {"workflow_dispatch"}
    inputs = config[True]["workflow_dispatch"]["inputs"]
    assert set(inputs) == {
        "build_run_id",
        "target",
        "models",
        "accuracy_models",
    }
    assert inputs["build_run_id"]["type"] == "string"
    assert inputs["build_run_id"]["required"] is True
    assert inputs["target"]["type"] == "choice"
    assert inputs["target"]["options"] == [
        "both",
        "pin-ut",
        "model-tests",
        "accuracy-tests",
    ]
    assert inputs["target"]["default"] == "both"
    assert inputs["models"]["type"] == "string"
    assert inputs["models"]["default"] == "qwen3.8-27b-m890p"
    assert inputs["accuracy_models"]["type"] == "string"
    assert inputs["accuracy_models"]["default"] == "deepseek-v4-mxfp4"
    assert "concurrency" not in config
    jobs = config["jobs"]
    expected_jobs = {
        "pin-unit-tests": (
            "inputs.target == 'both' || inputs.target == 'pin-ut'",
            "_ppu-pin-ut.yaml",
            {},
        ),
        "model-tests": (
            "inputs.target == 'both' || inputs.target == 'model-tests'",
            "_ppu-model-tests.yaml",
            {"models": "${{ inputs.models }}"},
        ),
        "accuracy-tests": (
            "inputs.target == 'accuracy-tests'",
            "_ppu-accuracy-tests.yaml",
            {"models": "${{ inputs.accuracy_models }}"},
        ),
    }
    assert set(jobs) == set(expected_jobs)
    for job_id, (condition, target, extra_inputs) in expected_jobs.items():
        job = jobs[job_id]
        assert job["if"] == condition
        assert "needs" not in job
        assert job["uses"] == f"./.github/workflows/{target}"
        assert job["with"] == {
            "pr_mode": False,
            "build_run_id": "${{ inputs.build_run_id }}",
            **extra_inputs,
        }


def test_build_leaf_preserves_retry_identity_and_image_output():
    config = workflow("_ppu-build.yaml")
    assert set(config[True]) == {"workflow_call"}
    call = config[True]["workflow_call"]
    assert set(call["inputs"]) == {
        "pr_head_sha",
        "build_image",
        "vllm_ref",
        "pytorch_sail_arch",
        "native_reuse_run_id",
    }
    for value in call["inputs"].values():
        assert value["type"] == "string"
        assert value["default"] == ""
        assert not value.get("required", False)
    assert set(call["outputs"]) == {"image"}
    assert call["outputs"]["image"]["value"] == "${{ jobs.build.outputs.image }}"
    assert set(config["jobs"]) == {"build"}
    build = config["jobs"]["build"]
    # The controller identifies successful wheel builds by the leaf job name on retry.
    assert build["name"] == "PPU wheel build"
    assert build["outputs"] == {"image": "${{ steps.image.outputs.image }}"}
    assert "if" not in build
    assert "needs" not in build
    caller = workflow("ppu-build-and-test.yaml")["jobs"]["build"]
    assert caller["uses"] == "./.github/workflows/_ppu-build.yaml"
    assert caller["with"] == {
        "pr_head_sha": "${{ inputs.pr_head_sha || github.sha }}",
        "native_reuse_run_id": "${{ inputs.native_reuse_run_id || '' }}",
    }
    download = next(
        step
        for step in build["steps"]
        if step.get("uses") == "actions/download-artifact@v4"
    )
    assert download["if"] == "inputs.native_reuse_run_id != ''"
    assert download["continue-on-error"] is True
    assert download["with"] == {
        "name": "ppu-wheels-${{ inputs.native_reuse_run_id }}",
        "path": "artifacts/native-reuse",
        "github-token": "${{ github.token }}",
        "run-id": "${{ inputs.native_reuse_run_id }}",
    }
    package = next(
        step
        for step in build["steps"]
        if step.get("name") == "Build or repackage wheels"
    )
    assert "VLLM_SAIL_NATIVE_REUSE_DIR" in package["env"]


def test_workflow_filenames_distinguish_reusable_calls_from_entrypoints():
    for path in WORKFLOWS.glob("*.y*ml"):
        triggers = workflow(path.name)[True]
        if path.name == "ppu-build-and-test.yaml":
            assert set(triggers) == {"workflow_call", "pull_request"}
        elif path.name.startswith("_"):
            assert set(triggers) == {"workflow_call"}, path.name
        else:
            assert "workflow_call" not in triggers, path.name


@pytest.mark.parametrize(
    "name", ["build-ppu-wheels.yaml", "ppu-pin-ut.yaml", "e2e-ppu.yaml"]
)
def test_obsolete_workflow_paths_are_absent(name):
    assert not (WORKFLOWS / name).exists(), name


def test_pr_pipeline_is_the_only_ppu_pull_request_route():
    for path in WORKFLOWS.glob("*.y*ml"):
        triggers = workflow(path.name)[True]
        if path.name == "ppu-build-and-test.yaml":
            assert set(triggers) == {"workflow_call", "pull_request"}
        elif path.name.startswith(("_ppu-", "ppu-")):
            assert "pull_request" not in triggers, path.name


def test_preview_route_requires_the_ci_preview_label():
    config = workflow("ppu-build-and-test.yaml")
    assert config["name"] == "PPU build and test"
    assert config["run-name"] == (
        "${{ github.event_name == 'pull_request' && "
        "format('PPU preview PR #{0} @ {1}', github.event.pull_request.number, "
        "github.sha) || 'PPU build and test' }}"
    )
    assert config[True]["pull_request"] == {
        "branches": ["main"],
        "types": [
            "opened",
            "synchronize",
            "reopened",
            "ready_for_review",
            "labeled",
            "unlabeled",
        ],
        "paths": [".github/workflows/**", "scripts/ci/**"],
    }
    assert not (WORKFLOWS / "ppu-pr-preview.yaml").exists()
    for job in config["jobs"].values():
        guard = job["if"]
        assert "github.event_name != 'pull_request'" in guard
        assert "!github.event.pull_request.draft" in guard
        assert "ci-preview" in guard
        assert "github.event.label.name == 'ci-preview'" in guard


def test_preview_label_is_distinct_from_formal_ready_authorization():
    text = (WORKFLOWS / "ppu-build-and-test.yaml").read_text()
    assert "ci-preview" in text
    assert "ready and ready-run-all-tests continue to authorize only" in text
    for label in ("ready", "ready-run-all-tests"):
        assert label not in config_preview_guards()


def config_preview_guards():
    return "\n".join(
        job["if"] for job in workflow("ppu-build-and-test.yaml")["jobs"].values()
    )


def test_preview_cancellation_is_per_pr_and_formal_calls_are_isolated():
    concurrency = workflow("ppu-build-and-test.yaml")["concurrency"]
    cancel = concurrency["cancel-in-progress"]
    assert "github.event.label.name == 'ci-preview'" in cancel

    groups = []
    for number in (49, 50):
        context = {
            "github.event_name": "pull_request",
            "github.event.action": "synchronize",
            "github.event.label.name": None,
            "github.event.pull_request.number": number,
            "github.run_id": "900",
        }
        group = re.sub(
            r"\$\{\{.*?\}\}",
            lambda match, context=context: str(expression(match[0], context)),
            concurrency["group"],
        )
        assert group == f"ppu-build-and-test-{number}"
        assert expression(cancel, context) is True
        groups.append(group)
    assert len(set(groups)) == 2

    formal = {
        "github.event_name": "workflow_dispatch",
        "github.event.pull_request.number": 0,
        "github.run_id": "901",
    }
    group = re.sub(
        r"\$\{\{.*?\}\}",
        lambda match: str(expression(match[0], formal)),
        concurrency["group"],
    )
    assert group == "ppu-build-and-test-901"
    assert expression(cancel, formal) is False


def test_unrelated_label_events_do_not_cancel_or_start_preview():
    config = workflow("ppu-build-and-test.yaml")
    concurrency = config["concurrency"]
    context = {
        "github.event_name": "pull_request",
        "github.event.action": "labeled",
        "github.event.label.name": "ready",
        "github.event.pull_request.number": 49,
        "github.event.pull_request.draft": False,
        "github.event.pull_request.labels.*.name": ["ci-preview", "ready"],
        "github.run_id": "902",
        "inputs.existing_build_run_id": "",
        "inputs.wheel_source_sha": "",
        "needs.runtime-image.result": "skipped",
        "needs.build.result": "skipped",
    }
    group = re.sub(
        r"\$\{\{.*?\}\}",
        lambda match: str(expression(match[0], context)),
        concurrency["group"],
    )
    assert group == "ppu-build-and-test-902"
    assert expression(concurrency["cancel-in-progress"], context) is False
    assert all(not expression(job["if"], context) for job in config["jobs"].values())


def test_removing_ci_preview_cancels_active_preview_without_starting_jobs():
    config = workflow("ppu-build-and-test.yaml")
    concurrency = config["concurrency"]
    context = {
        "github.event_name": "pull_request",
        "github.event.action": "unlabeled",
        "github.event.label.name": "ci-preview",
        "github.event.pull_request.number": 49,
        "github.event.pull_request.draft": False,
        "github.event.pull_request.labels.*.name": [],
        "github.run_id": "903",
        "inputs.existing_build_run_id": "",
        "inputs.wheel_source_sha": "",
        "needs.runtime-image.result": "skipped",
        "needs.build.result": "skipped",
    }
    group = re.sub(
        r"\$\{\{.*?\}\}",
        lambda match: str(expression(match[0], context)),
        concurrency["group"],
    )
    assert group == "ppu-build-and-test-49"
    assert expression(concurrency["cancel-in-progress"], context) is True
    assert all(not expression(job["if"], context) for job in config["jobs"].values())


@pytest.mark.parametrize(
    "caller_id,leaf_name,job_ids",
    [
        ("build", "_ppu-build.yaml", ["build"]),
        ("pin-unit-tests", "_ppu-pin-ut.yaml", ["pin-unit-tests"]),
        ("model-tests", "_ppu-model-tests.yaml", ["prepare", "e2e"]),
    ],
)
def test_labeled_preview_propagates_merge_sha_to_build_and_test_checkouts(
    caller_id, leaf_name, job_ids
):
    merge_sha, head_sha = "a" * 40, "b" * 40
    context = {
        "github.event_name": "pull_request",
        "github.event.action": "synchronize",
        "github.event.label.name": None,
        "github.event.pull_request.draft": False,
        "github.event.pull_request.labels.*.name": ["ci-preview"],
        "github.sha": merge_sha,
        "github.event.pull_request.head.sha": head_sha,
        "github.run_id": "200",
        "inputs.pr_head_sha": "",
        "inputs.existing_build_run_id": "",
        "inputs.wheel_source_sha": "",
        "needs.runtime-image.result": "success",
        "needs.build.result": "success",
    }
    pipeline = workflow("ppu-build-and-test.yaml")["jobs"]
    assert expression(pipeline["build"]["if"], context)
    caller = pipeline[caller_id]
    assert caller["uses"] == f"./.github/workflows/{leaf_name}"
    context["inputs.pr_head_sha"] = expression(caller["with"]["pr_head_sha"], context)
    assert context["inputs.pr_head_sha"] == merge_sha
    if caller_id != "build":
        assert caller["with"]["pr_mode"] is True
        assert expression(caller["with"]["wheel_source_sha"], context) == ""
    for job_id in job_ids:
        job = workflow(leaf_name)["jobs"][job_id]
        checkout = checkouts(job)[0]
        assert expression(checkout["with"]["ref"], context) == merge_sha
        verify = job["steps"][job["steps"].index(checkout) + 1]
        assert expression(verify["env"]["EXPECTED_SHA"], context) == merge_sha


@pytest.mark.parametrize(
    "labels,draft", [([], False), (["ready"], False), (["ci-preview"], True)]
)
def test_preview_jobs_skip_without_an_active_ci_preview_label(labels, draft):
    jobs = workflow("ppu-build-and-test.yaml")["jobs"]
    context = {
        "github.event_name": "pull_request",
        "github.event.action": "synchronize",
        "github.event.label.name": None,
        "github.event.pull_request.draft": draft,
        "github.event.pull_request.labels.*.name": labels,
        "inputs.existing_build_run_id": "",
        "inputs.wheel_source_sha": "",
        "needs.runtime-image.result": "skipped",
        "needs.build.result": "skipped",
    }
    assert all(not expression(job["if"], context) for job in jobs.values())


@pytest.mark.parametrize(
    "target,expected",
    [("both", (True, True)), ("pin-ut", (True, False)), ("model-tests", (False, True))],
)
def test_debug_targets_select_requested_consumers(target, expected):
    jobs = workflow("ppu-debug.yaml")["jobs"]
    context = {"inputs.target": target}
    assert (
        tuple(
            bool(expression(jobs[job_id]["if"], context))
            for job_id in ("pin-unit-tests", "model-tests")
        )
        == expected
    )


def test_pr_wheel_verification_runs_inside_the_runtime_container():
    source = "${{ inputs.wheel_source_sha || inputs.pr_head_sha || github.sha }}"
    for workflow_name, job_id in (
        ("_ppu-pin-ut.yaml", "pin-unit-tests"),
        ("_ppu-model-tests.yaml", "e2e"),
    ):
        job = workflow(workflow_name)["jobs"][job_id]
        assert not any(
            "ppu_wheel_manifest.py" in step.get("run", "") for step in job["steps"]
        )
        runtime = next(
            step
            for step in job["steps"]
            if "ppu_runtime_container.sh" in step.get("run", "")
        )
        assert runtime["env"]["PPU_CI_REQUIRE_PR_ARTIFACT"] == "${{ inputs.pr_mode }}"
        assert runtime["env"]["PPU_CI_WHEEL_SOURCE_SHA"] == source

    pin_script = (ROOT / "scripts/ci/ppu_pin_ut.sh").read_text()
    model_script = (ROOT / "scripts/ci/ppu_e2e_smoke.sh").read_text()
    for script in (pin_script, model_script):
        assert '--vllm-commit "${EXPECTED_VLLM_COMMIT}"' in script
        assert '--sail-commit "${EXPECTED_SAIL_COMMIT}"' in script
    assert "PPU PIN UT SKIP" in pin_script


def test_retry_reuses_build_artifacts_but_keeps_pin_diagnostics_per_attempt():
    for name, job_id in (
        ("_ppu-build.yaml", "build"),
        ("ppu-ops-tests.yaml", "ops-build"),
    ):
        uploads = {
            step["with"]["name"]: step["with"]
            for step in workflow(name)["jobs"][job_id]["steps"]
            if step.get("uses") == "actions/upload-artifact@v4"
        }
        assert uploads["ppu-wheels-${{ github.run_id }}"]["overwrite"] is True
        assert (
            uploads["ppu-build-diagnostics-${{ github.run_id }}"]["overwrite"] is True
        )
    pin = workflow("_ppu-pin-ut.yaml")["jobs"]["pin-unit-tests"]
    upload = next(
        step
        for step in pin["steps"]
        if step.get("uses") == "actions/upload-artifact@v4"
    )
    assert (
        upload["with"]["name"]
        == "ppu-pin-main-verified-${{ github.run_id }}-${{ github.run_attempt }}"
    )


def test_cpu_checks_still_run_automatically_with_read_only_checkout():
    config = workflow("ut.yaml")
    assert set(config[True]) == {"pull_request", "push", "workflow_dispatch"}
    assert config["permissions"] == {"contents": "read"}
    for job in config["jobs"].values():
        for checkout in checkouts(job):
            assert checkout["with"]["persist-credentials"] is False


def test_ops_partial_retry_downloads_diagnostics_from_successful_previous_attempt():
    build = workflow("ppu-ops-tests.yaml")["jobs"]
    producer = next(
        step["with"]["name"]
        for step in build["ops-build"]["steps"]
        if step.get("uses") == "actions/upload-artifact@v4"
        and step["with"]["name"].startswith("ppu-build-diagnostics-")
    )
    uploaded = producer.replace("${{ github.run_id }}", "100").replace(
        "${{ github.run_attempt }}", "1"
    )
    consumers = [
        build["select-ops-tests"],
        *workflow("_ppu-ops-tests.yaml")["jobs"].values(),
    ]
    for job in consumers:
        name = next(
            step["with"]["name"]
            for step in job["steps"]
            if step.get("uses") == "actions/download-artifact@v4"
            and step["with"].get("name", "").startswith("ppu-build-diagnostics-")
        )
        downloaded = (
            name.replace("${{ github.run_id }}", "100")
            .replace("${{ inputs.build_run_id }}", "100")
            .replace("${{ github.run_attempt }}", "2")
        )
        assert downloaded == uploaded


@pytest.mark.parametrize(
    "name", ["ppu-build-and-test.yaml", "_ppu-pin-ut.yaml", "_ppu-model-tests.yaml"]
)
def test_wheel_source_is_optional_internal_input_only(name):
    triggers = workflow(name)[True]
    source = triggers["workflow_call"]["inputs"].get("wheel_source_sha")
    assert source is not None
    assert source["type"] == "string"
    assert source["default"] == ""
    assert not source.get("required", False)
    assert "wheel_source_sha" not in triggers.get("workflow_dispatch", {}).get(
        "inputs", {}
    )


def test_all_local_reusable_calls_use_declared_inputs():
    for path in WORKFLOWS.iterdir():
        if path.suffix not in {".yaml", ".yml"}:
            continue
        for job_id, job in workflow(path.name)["jobs"].items():
            target = job.get("uses", "")
            if not target.startswith("./.github/workflows/"):
                continue
            declared = workflow(Path(target).name)[True]["workflow_call"].get(
                "inputs", {}
            )
            supplied = job.get("with", {})
            assert supplied.keys() <= declared.keys(), (path.name, job_id)
            required = {key for key, value in declared.items() if value.get("required")}
            assert required <= supplied.keys(), (path.name, job_id)


def expression(value, context):
    """Evaluate only the Boolean/context subset used by these workflow guards."""
    value = value.removeprefix("${{").removesuffix("}}").strip()
    value = re.sub(
        r"\b(?:inputs|github|needs|steps)(?:\.[\w*-]+)+",
        lambda match: repr(context.get(match[0])),
        value,
    )
    value = value.replace("always()", "True").replace("cancelled()", "False")
    value = value.replace("contains(", "contains_value(")
    value = value.replace("&&", " and ").replace("||", " or ")
    value = re.sub(r"!(?!=)", "not ", value)
    value = re.sub(r"\btrue\b", "True", value)
    value = re.sub(r"\bfalse\b", "False", value)

    def contains_value(values, item):
        return item in values

    return eval(
        value.strip(),
        {"__builtins__": {}, "contains_value": contains_value},
        {},
    )


@pytest.mark.parametrize(
    "pr_mode,event",
    [
        (True, "workflow_dispatch"),
        (False, "workflow_dispatch"),
    ],
)
def test_e2e_concurrency_isolates_reuse_consumers_only_in_pr_mode(pr_mode, event):
    concurrency = workflow("_ppu-model-tests.yaml")["concurrency"]
    assert concurrency["cancel-in-progress"] is False
    assert "queue" not in concurrency
    refs = ("refs/heads/main", "refs/heads/release")
    groups_by_ref = []
    for ref in refs:
        groups_by_attempt = []
        for attempt in (1, 2):
            groups = {}
            for run_id in ("901", "902", "903"):
                context = {
                    "inputs.pr_mode": pr_mode,
                    "inputs.build_run_id": "555",
                    "github.ref": ref,
                    "github.event_name": event,
                    "github.run_id": run_id,
                    "github.run_attempt": attempt,
                }
                groups[run_id] = re.sub(
                    r"\$\{\{.*?\}\}",
                    lambda match, context=context: str(expression(match[0], context)),
                    concurrency["group"],
                )
            # One running and two pending consumers must not share a PR-mode key.
            assert len(set(groups.values())) == (3 if pr_mode else 1)
            assert groups == {
                run_id: f"e2e-ppu-{ref}-{run_id if pr_mode else '555'}"
                for run_id in groups
            }
            groups_by_attempt.append(groups)
        assert groups_by_attempt[0] == groups_by_attempt[1]
        groups_by_ref.append(set(groups_by_attempt[0].values()))
    assert groups_by_ref[0].isdisjoint(groups_by_ref[1])


@pytest.fixture(scope="module")
def build_jobs():
    return workflow("ppu-build-and-test.yaml")["jobs"]


@pytest.mark.parametrize("build", ["success", "failure", "cancelled", "skipped"])
@pytest.mark.parametrize(
    "run_id,source", [("", ""), ("123", "a" * 40), ("999", ""), ("", "a" * 40)]
)
def test_consumers_require_build_or_valid_reuse(build_jobs, build, run_id, source):
    context = {
        "github.event_name": "workflow_dispatch",
        "github.event.pull_request.draft": False,
        "github.event.pull_request.labels.*.name": [],
        "inputs.skip_ppu": False,
        "inputs.existing_build_run_id": run_id,
        "inputs.wheel_source_sha": source,
        "needs.runtime-image.result": "success",
        "needs.build.result": build,
    }
    reuse = bool(run_id) and bool(source)
    eligible = build == "success" or (build == "skipped" and reuse)
    for job_id in ("pin-unit-tests", "model-tests"):
        assert bool(expression(build_jobs[job_id]["if"], context)) == eligible
    assert bool(expression(build_jobs["build"]["if"], context)) == (not run_id)
    # Gate must execute even after failed/cancelled/skipped dependencies.
    assert bool(expression(build_jobs["gate"]["if"], context)) is True


@pytest.mark.parametrize(
    "states",
    itertools.product(["success", "failure", "cancelled", "skipped"], repeat=3),
)
@pytest.mark.parametrize(
    "run_id,source", [("", ""), ("123", "a" * 40), ("999", ""), ("", "a" * 40)]
)
def test_gate_executes_fail_closed_truth_table(
    build_jobs, monkeypatch, states, run_id, source
):
    step = build_jobs["gate"]["steps"][0]
    context = {
        "inputs.skip_ppu": False,
        "inputs.skip_device_tests": False,
        "inputs.existing_build_run_id": run_id,
        "inputs.wheel_source_sha": source,
    }
    for key, value in step["env"].items():
        if key != "JOB_RESULTS":
            resolved = expression(value, context)
            monkeypatch.setenv(
                key, str(resolved).lower() if isinstance(resolved, bool) else resolved
            )
    results = {"runtime-image": {"result": "success"}}
    results.update(
        zip(
            ("build", "pin-unit-tests", "model-tests"),
            ({"result": state} for state in states),
            strict=False,
        )
    )
    monkeypatch.setenv("JOB_RESULTS", json.dumps(results))
    script = step["run"].removeprefix("python - <<'PYTHON'\n").removesuffix("PYTHON\n")
    try:
        exec(compile(script, "ppu-ci-gate", "exec"), {})
        passed = True
    except SystemExit as error:
        assert error.code
        passed = False
    expected = (
        states[0] == "success"
        or (states[0] == "skipped" and bool(run_id) and bool(source))
    ) and states[1:] == ("success", "success")
    assert passed is expected


def test_runtime_workflows_use_the_checked_in_docker_wrapper():
    pin = workflow("_ppu-pin-ut.yaml")["jobs"]["pin-unit-tests"]
    model = workflow("_ppu-model-tests.yaml")["jobs"]["e2e"]
    pin_step = next(
        step
        for step in pin["steps"]
        if "ppu_runtime_container.sh" in step.get("run", "")
    )
    model_step = next(
        step
        for step in model["steps"]
        if "ppu_runtime_container.sh" in step.get("run", "")
    )
    assert (
        pin_step["run"]
        == 'bash "${PPU_CI_SOURCE_DIR}/scripts/ci/ppu_runtime_container.sh" pin-ut'
    )
    assert pin_step["env"]["PPU_CI_DEVICE_COUNT"] == "1"
    assert "PPU_RUNTIME_IMAGE" not in pin_step["env"]
    assert pin_step["env"]["PPU_CI_REQUIRE_PR_ARTIFACT"] == "${{ inputs.pr_mode }}"
    assert (
        model_step["run"]
        == 'bash "${PPU_CI_SOURCE_DIR}/scripts/ci/ppu_runtime_container.sh" model-test'
    )
    assert model_step["env"]["PPU_CI_DEVICE_COUNT"] == "${{ matrix.model.ppu_count }}"
    # Smoke cases keep their parallelism inside one process, so the tensor
    # parallel size is the card count they asked for.
    assert model_step["env"]["TP_SIZE"] == "${{ matrix.model.ppu_count }}"
    assert model_step["env"]["PPU_CI_REQUIRE_PR_ARTIFACT"] == "${{ inputs.pr_mode }}"
    assert pin_step["env"]["PPU_CI_SOURCE_DIR"] == "${{ steps.runtime.outputs.source }}"
    assert (
        model_step["env"]["PPU_CI_SOURCE_DIR"] == "${{ steps.runtime.outputs.source }}"
    )
    assert model_step["env"]["MODEL_ID"] == "${{ matrix.model.model_id }}"
    assert "MODEL_PATH" not in model_step["env"]
    assert "ppu-scheduler-action" not in (WORKFLOWS / "_ppu-pin-ut.yaml").read_text()
    assert (
        "ppu-scheduler-action" not in (WORKFLOWS / "_ppu-model-tests.yaml").read_text()
    )
