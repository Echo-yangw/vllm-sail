# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "ci" / "build_ppu_wheels.sh"
WORKFLOW = ROOT / ".github" / "workflows" / "_ppu-build.yaml"


def test_build_script_has_valid_bash_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_build_script_preserves_required_build_contract() -> None:
    text = SCRIPT.read_text()
    required = (
        "set -Eeuo pipefail",
        "ppu_image_stack.py",
        "status=running",
        'export PIP_CONSTRAINT="${WORK_DIR}/image-constraints.txt"',
        ".github/vllm-main-verified.commit",
        "unset TORCH_CUDA_ARCH_LIST",
        "command -v hgcc",
        "decision = decide(probe())",
        "python tools/use_existing_torch.py",
        "VLLM_TARGET_DEVICE=empty",
        "VLLM_REQUIRE_RUST_FRONTEND=1",
        "CARGO_BUILD_JOBS",
        "rust-toolchain.toml",
        "rustc --version",
        "cargo --version",
        "--no-build-isolation --no-deps",
        'git -C "${VLLM_SRC}" fetch --tags',
        "check_vllm_compatibility()",
        "ppu_wheel_manifest.py",
        "VERBOSE=1",
        "build-manifest.txt",
        "printf 'build_image=%s\\n' \"${VLLM_SAIL_BUILD_IMAGE:-}\"",
    )
    for marker in required:
        assert marker in text


def test_build_script_uses_configurable_aliyun_pip_mirror() -> None:
    text = SCRIPT.read_text()
    forced = (
        'export PIP_INDEX_URL="${SAIL_PIP_INDEX_URL:-'
        'https://mirrors.aliyun.com/pypi/simple/}"'
    )
    assert forced in text
    # The PPU SDK envsetup.sh repoints pip at the unreachable internal mirror,
    # so the public mirror must be forced AFTER sourcing it.
    assert text.index("source /usr/local/PPU_SDK/envsetup.sh") < text.index(forced)


def test_build_script_does_not_install_runtime_only_ppu_dependencies() -> None:
    text = SCRIPT.read_text()
    assert '-r "${ROOT_DIR}/requirements/build.txt"' in text
    assert '-r "${ROOT_DIR}/requirements/ppu.txt"' not in text
    assert "separate CUDA-compatible runtime image" in text


def test_build_script_caps_parallelism_to_avoid_oom() -> None:
    text = SCRIPT.read_text()
    # Native PPU kernel compiles are memory-heavy; an unbounded MAX_JOBS OOM-kills
    # the pod (exit 137). Parallelism must be capped by the cgroup memory limit
    # while staying overridable via BUILD_JOBS.
    assert "/sys/fs/cgroup/memory.max" in text
    assert 'BUILD_JOBS="${BUILD_JOBS:-' in text


def test_build_workflow_is_call_only_on_github_hosted_runner() -> None:
    import yaml

    text = WORKFLOW.read_text()
    assert set(yaml.safe_load(text)[True]) == {"workflow_call"}
    assert "workflow_dispatch:" not in text
    assert "pull_request:" not in text
    assert "push:" not in text
    # The CUDA-free wheel build runs in its pinned container on a GitHub-hosted
    # runner; scarce PPU runners are reserved for runtime validation.
    assert "runs-on: ubuntu-24.04" in text
    assert "ppu-scheduler-action" not in text
    assert "contents: read" in text


def test_build_workflow_uses_script_and_retains_artifacts() -> None:
    text = WORKFLOW.read_text()
    assert "bash scripts/ci/build_ppu_wheels.sh" in text
    import yaml

    ops = yaml.safe_load((WORKFLOW.parent / "ppu-ops-tests.yaml").read_text())["jobs"]
    for job in (yaml.safe_load(text)["jobs"]["build"], ops["ops-build"]):
        uploads = {
            step["with"]["name"]: step["with"]
            for step in job["steps"]
            if step.get("uses") == "actions/upload-artifact@v4"
        }
        assert set(uploads) == {
            "ppu-wheels-${{ github.run_id }}",
            "ppu-build-diagnostics-${{ github.run_id }}",
        }
        assert all(config["retention-days"] == 14 for config in uploads.values())
        assert all(config["overwrite"] is True for config in uploads.values())
        assert (
            uploads["ppu-wheels-${{ github.run_id }}"]["if-no-files-found"] == "error"
        )
    for job, name in (
        ("detect-ops-changes", "changes"),
        ("select-ops-tests", "selection"),
    ):
        assert any(
            step.get("with", {}).get("name")
            == f"ppu-ops-{name}-${{{{ github.run_id }}}}-${{{{ github.run_attempt }}}}"
            for step in ops[job]["steps"]
        )
    assert "if-no-files-found: error" in text
    assert "if-no-files-found: warn" in text
    assert "if: always()" in text


def test_build_workflow_uploads_manifest_with_wheels() -> None:
    assert "path: artifacts/wheels/" in WORKFLOW.read_text()
    assert "--depth 1" not in SCRIPT.read_text()


@pytest.mark.parametrize(
    "cargo_env", ["", "export CARGO_NET_GIT_FETCH_WITH_CLI=false\n"]
)
def test_cargo_git_cli_is_exported_after_environment_setup(tmp_path, cargo_env):
    text = SCRIPT.read_text()
    start = text.index('    if [[ -r "${CARGO_HOME:-$HOME/.cargo}/env" ]]')
    end = text.index("    # Follow the Rust toolchain pinned")
    assert text.index("source /usr/local/PPU_SDK/envsetup.sh") < start
    assert start < end < text.index("cargo --version")
    assert end < text.index("python tools/build_rust.py --release")
    (tmp_path / "env").write_text(cargo_env)
    preserved = {
        "GIT_CONFIG_GLOBAL": str(tmp_path / "git-config"),
        "GIT_CONFIG_NOSYSTEM": "1",
        "HTTPS_PROXY": "http://proxy.invalid:8080",
    }
    # Run the actual config snippet and verify that sourced exports reach subprocesses.
    result = subprocess.run(
        [
            "bash",
            "--noprofile",
            "--norc",
            "-euo",
            "pipefail",
            "-c",
            text[start:end]
            + "\n\"$1\" -c 'import json, os; print(json.dumps(dict(os.environ)))'\n",
            "bash",
            sys.executable,
        ],
        env={
            "PATH": os.defpath,
            "HOME": str(tmp_path),
            "CARGO_HOME": str(tmp_path),
            **preserved,
        },
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    child_env = json.loads(result.stdout)
    assert child_env.get("CARGO_NET_GIT_FETCH_WITH_CLI") == "true"
    assert {key: child_env[key] for key in preserved} == preserved


def test_rust_build_is_explicit_before_wheel_packaging() -> None:
    text = SCRIPT.read_text()
    required = text.index(
        "export VLLM_TARGET_DEVICE=empty VLLM_REQUIRE_RUST_FRONTEND=1"
    )
    rust = text.index("python tools/build_rust.py --release")
    check = text.index("required local Rust artifacts were not built")
    wheel = text.index("python -m pip wheel --verbose")
    assert required < rust < check < wheel
    assert "vllm-rust-build.log" in text


@pytest.mark.parametrize(
    "declared,present,binary,expected",
    [
        ([], {}, b"frontend", True),
        (["_rust_tool_parser"], {"_rust_tool_parser": b"bridge"}, b"frontend", True),
        (["_rust_tool_parser"], {}, b"frontend", False),
        ([], {"_rust_tool_parser": b"stale bridge"}, b"frontend", False),
        ([], {}, None, False),
        ([], {}, b"", False),
        (["_rust_tool_parser"], {"_rust_tool_parser": b""}, b"frontend", False),
    ],
)
def test_rust_artifact_gate_follows_upstream_declarations(
    tmp_path: Path,
    declared: list[str],
    present: dict[str, bytes],
    binary: bytes | None,
    expected: bool,
) -> None:
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools/build_rust.py").write_text(
        f"def rust_py_extension_module_names():\n    return {declared!r}\n"
    )
    output = tmp_path / "vllm"
    output.mkdir()
    if binary is not None:
        (output / "vllm-rs").write_bytes(binary)
    for module, contents in present.items():
        (output / f"{module}.abi3.so").write_bytes(contents)
    source = SCRIPT.read_text().split("python - <<'RUST_ARTIFACTS'\n", 1)[1]
    source = source.split("\nRUST_ARTIFACTS\n", 1)[0]
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) == expected, result.stdout + result.stderr
    if expected:
        assert "Local Rust artifact: vllm/vllm-rs" in result.stdout


def test_image_build_captures_the_pinned_image_stack_before_building():
    text = SCRIPT.read_text()
    for forbidden in (
        "SDK_URL",
        "TORCH_URL",
        "SDK_INSTALLER",
        "TORCH_ARCHIVE",
        'mv "${path}"',
        "torch_wheels",
        "download()",
    ):
        assert forbidden not in text
    capture = text.index('ppu_image_stack.py" capture')
    verify = text.index('ppu_image_stack.py" verify')
    assert capture < text.index("decision = decide(probe())")
    assert capture < text.index("python -m pip install") < verify
    assert verify < text.index("status=succeeded")


def test_ci_uses_separate_pinned_build_and_runtime_stacks():
    import yaml

    build_image = (
        "download.t-head.cn/docker_release/pytorch@sha256:"
        "a33a67832f7ec442c481421e94255518dd67e464406f8e610c15d2b594b21f80"
    )
    workflow = yaml.safe_load(WORKFLOW.read_text())
    inputs = yaml.safe_load((WORKFLOW.parent / "ppu-ops-tests.yaml").read_text())[True][
        "workflow_dispatch"
    ]["inputs"]
    assert not {"sdk_url", "torch_url"} & inputs.keys()
    assert inputs["build_image"]["default"] == build_image
    assert inputs["ops_image"]["required"] is True
    assert "default" not in inputs["ops_image"]
    build = workflow["jobs"]["build"]
    assert build_image in build["container"]["image"]
    assert build["env"]["VLLM_SAIL_BUILD_IMAGE"] == build["container"]["image"]
    pipeline = yaml.safe_load((WORKFLOW.parent / "ppu-build-and-test.yaml").read_text())
    assert "runtime_image" not in pipeline[True]["workflow_call"]["inputs"]
    for job in ("pin-unit-tests", "model-tests"):
        assert "image" not in pipeline["jobs"][job]["with"]
    for name in ("_ppu-pin-ut.yaml", "_ppu-model-tests.yaml", "ppu-debug.yaml"):
        triggers = yaml.safe_load((WORKFLOW.parent / name).read_text())[True]
        for trigger in triggers.values():
            assert "image" not in trigger["inputs"]


@pytest.fixture
def image_stack():
    import importlib.util

    path = ROOT / "scripts/ci/ppu_image_stack.py"
    assert path.is_file(), "image stack fingerprint helper is missing"
    spec = importlib.util.spec_from_file_location("image_stack", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def stack_trees(tmp_path):
    sdk = tmp_path / "sdk"
    torch = tmp_path / "torch"
    sdk.mkdir()
    torch.mkdir()
    (sdk / "hgcc").write_bytes(b"compiler")
    (torch / "_C.so").write_bytes(b"torch")
    return sdk, torch


def test_image_stack_hashes_installed_contents_and_ignores_bytecode(
    image_stack, stack_trees
):
    sdk, torch = stack_trees
    before = image_stack.snapshot(sdk, torch, "2.13.0", {"torch": "2.13.0"})
    (torch / "__pycache__").mkdir()
    (torch / "__pycache__/generated.pyc").write_bytes(b"cache")
    after = image_stack.snapshot(sdk, torch, "2.13.0", {"torch": "2.13.0"})
    image_stack.verify_unchanged(before, after)
    assert before["stack_source"] == "image"
    assert len(before["sdk_sha256"]) == len(before["torch_sha256"]) == 64
    assert before["sdk_sha256"] != before["torch_sha256"]


def test_sdk_jit_cache_is_separate_from_immutable_stack(image_stack, stack_trees):
    sdk, torch = stack_trees
    before = image_stack.snapshot(sdk, torch, "2.13.0", {"torch": "2.13.0"})
    cache = sdk / "rtccache" / "PPU0015"
    cache.mkdir(parents=True)
    (cache / "generated.bin").write_bytes(b"compiled kernel")
    after = image_stack.snapshot(sdk, torch, "2.13.0", {"torch": "2.13.0"})
    image_stack.verify_unchanged(before, after)
    # The unfiltered inventory still observes the newly generated bytes.
    assert image_stack.tree_digest(sdk) != after["sdk_sha256"]


@pytest.mark.parametrize("component", ["sdk_nested_cache", "torch_cache"])
def test_only_sdk_root_runtime_cache_is_excluded(image_stack, stack_trees, component):
    sdk, torch = stack_trees
    before = image_stack.snapshot(sdk, torch, "2.13.0", {"torch": "2.13.0"})
    path = (
        sdk / "include" / "rtccache"
        if component == "sdk_nested_cache"
        else torch / "rtccache"
    )
    path.mkdir(parents=True)
    (path / "code.so").write_bytes(b"protected code")
    after = image_stack.snapshot(sdk, torch, "2.13.0", {"torch": "2.13.0"})
    with pytest.raises(ValueError, match="image stack changed"):
        image_stack.verify_unchanged(before, after)


@pytest.mark.parametrize("component", ["sdk", "torch", "packages", "version"])
def test_image_stack_rejects_changes(image_stack, stack_trees, component):
    sdk, torch = stack_trees
    versions = {"torch": "2.13.0", "triton": "3.5.0"}
    before = image_stack.snapshot(sdk, torch, "2.13.0", versions)
    version = "2.13.0"
    if component == "sdk":
        (sdk / "hgcc").write_bytes(b"replaced")
    elif component == "torch":
        (torch / "_C.so").write_bytes(b"replaced")
    elif component == "packages":
        versions = {**versions, "triton": "3.6.0"}
    else:
        version = "2.14.0"
    after = image_stack.snapshot(sdk, torch, version, versions)
    with pytest.raises(ValueError, match="image stack changed"):
        image_stack.verify_unchanged(before, after)


def test_image_stack_constraints_protect_installed_and_absent_device_packages(
    image_stack,
):
    from packaging.requirements import Requirement

    constraints = image_stack.constraints_text(
        {"torch": "2.13.0+ppu", "triton": "3.5.0"}
    )
    requirements = {r.name: r for r in map(Requirement, constraints.splitlines())}
    assert requirements["torch"].specifier.contains("2.13.0+ppu")
    assert not requirements["torch"].specifier.contains("2.13.0")
    assert not requirements["triton"].specifier.contains("3.6.0")
    assert not requirements["flash-attn"].specifier.contains("2.8.2")
    assert "vllm" not in requirements


def test_image_stack_rejects_missing_sdk(image_stack, tmp_path):
    with pytest.raises(ValueError, match="missing or empty"):
        image_stack.tree_digest(tmp_path / "missing")


def test_image_stack_hashes_symlinked_library_directories(
    image_stack, stack_trees, tmp_path
):
    sdk, _ = stack_trees
    libraries = tmp_path / "libraries"
    libraries.mkdir()
    (libraries / "runtime.so").write_bytes(b"runtime")
    (sdk / "lib").symlink_to(libraries, target_is_directory=True)
    before = image_stack.tree_digest(sdk)
    (libraries / "runtime.so").write_bytes(b"replacement")
    assert image_stack.tree_digest(sdk) != before


def test_image_stack_handles_directory_symlink_cycles(image_stack, stack_trees):
    sdk, _ = stack_trees
    (sdk / "back").symlink_to(sdk, target_is_directory=True)
    before = image_stack.tree_digest(sdk)
    assert image_stack.tree_digest(sdk) == before
    (sdk / "hgcc").write_bytes(b"changed")
    assert image_stack.tree_digest(sdk) != before


def test_image_stack_hashes_symlinked_files(image_stack, stack_trees, tmp_path):
    sdk, _ = stack_trees
    library = tmp_path / "runtime.so"
    library.write_bytes(b"runtime")
    (sdk / "runtime.so").symlink_to(library)
    before = image_stack.tree_digest(sdk)
    library.write_bytes(b"replacement")
    assert image_stack.tree_digest(sdk) != before


def test_image_stack_rejects_empty_sdk(image_stack, tmp_path):
    with pytest.raises(ValueError, match="missing or empty"):
        image_stack.tree_digest(tmp_path)


def test_image_stack_cli_capture_verify_and_refuse_mutation(tmp_path):
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    (sdk / "hgcc").write_bytes(b"compiler")
    site = tmp_path / "site"
    torch = site / "torch"
    torch.mkdir(parents=True)
    (torch / ".ppu_compat").mkdir()
    (torch / ".ppu_compat/compatible_wrapper.h").write_text("header")
    (torch / "__init__.py").write_text('__version__ = "2.13.0"\n')
    dist = site / "torch-2.13.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Name: torch\nVersion: 2.13.0\n")
    output = tmp_path / "stack.json"
    constraints = tmp_path / "constraints.txt"
    script = ROOT / "scripts/ci/ppu_image_stack.py"

    def run(command, *extra):
        return subprocess.run(
            [
                sys.executable,
                str(script),
                command,
                "--sdk",
                str(sdk),
                "--output",
                str(output),
                *extra,
            ],
            env={**os.environ, "PYTHONPATH": str(site)},
            capture_output=True,
            text=True,
            cwd=tmp_path,
        )

    captured = run("capture", "--constraints", str(constraints))
    assert captured.returncode == 0, captured.stderr
    assert "stack_source=image" in captured.stdout
    assert "torch===2.13.0" in constraints.read_text()
    verified = run("verify")
    assert verified.returncode == 0, verified.stderr
    assert "stack_unchanged=true" in verified.stdout
    (sdk / "hgcc").write_bytes(b"changed")
    refused = run("verify")
    assert refused.returncode != 0
    assert "image stack changed: sdk_sha256" in refused.stderr


def test_torch_override_is_content_pinned(image_stack):
    assert image_stack.TORCH_OVERRIDE == {
        "torch_archive_url": "https://art-pub.eng.t-head.cn/artifactory/apackage/daily/ppu.2v2_release/latest/PPU_SDK_PYPI/pytorch/ppu_sdk_hggcrt3-pytorch2.13.0-ubuntu2404-py312.tar.gz",
        "torch_archive_sha256": "63e23fedca52f1039b2deac29aa37d7b88d8ba7413f9b5600d223ffb095ec1f7",
        "torch_wheel_sha256": "24a95ab1e6bbc9d8dd61a4347020dd1975e7bdfb06e2110bbc618874a8675760",
    }


@pytest.fixture
def torch_archive(image_stack, tmp_path, monkeypatch):
    import hashlib
    import io
    import tarfile

    archive = tmp_path / "torch.tar.gz"
    wheel_data = b"verified torch wheel"

    def create(names):
        with tarfile.open(archive, "w:gz") as target:
            for name in names:
                info = tarfile.TarInfo(name)
                info.size = len(wheel_data)
                target.addfile(info, io.BytesIO(wheel_data))
        monkeypatch.setattr(
            image_stack,
            "TORCH_OVERRIDE",
            {
                "torch_archive_url": archive.as_uri(),
                "torch_archive_sha256": hashlib.sha256(
                    archive.read_bytes()
                ).hexdigest(),
                "torch_wheel_sha256": hashlib.sha256(wheel_data).hexdigest(),
            },
            raising=False,
        )
        return archive

    create(
        [
            "package/torch-2.13.0-cp312-cp312-linux_x86_64.whl",
            "package/torchvision-0.28.0-cp312-cp312-linux_x86_64.whl",
        ]
    )
    return create


def test_prepare_installs_only_verified_torch_and_keeps_other_packages(
    image_stack, stack_trees, torch_archive, monkeypatch
):
    sdk, _ = stack_trees
    versions = {"torch": "2.13.0", "triton": "3.7.0", "torchvision": "0.28.0"}
    monkeypatch.setattr(image_stack, "package_versions", lambda: versions)
    monkeypatch.setenv("PIP_CONSTRAINT", "/not-yet-created/constraints.txt")
    commands = []

    def install(command, **kwargs):
        commands.append(command)
        assert command[:-1] == [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--force-reinstall",
        ]
        assert Path(command[-1]).name == "torch-2.13.0-cp312-cp312-linux_x86_64.whl"
        assert Path(command[-1]).read_bytes() == b"verified torch wheel"
        assert "PIP_CONSTRAINT" not in kwargs["env"]
        assert kwargs["check"] is True

    monkeypatch.setattr(subprocess, "run", install)
    before = image_stack.tree_digest(sdk)
    image_stack.install_torch(sdk)
    assert len(commands) == 1
    assert image_stack.tree_digest(sdk) == before


@pytest.mark.parametrize("failure", ["timeout", "connection", "read", "http503"])
def test_prepare_retries_transient_download_before_installing(
    image_stack, stack_trees, torch_archive, monkeypatch, failure
):
    import io
    import time
    import urllib.error

    original_open = image_stack.urllib.request.urlopen
    attempts, delays, installs, responses = [], [], [], []
    errors = {
        "timeout": TimeoutError("download timed out"),
        "connection": urllib.error.URLError(ConnectionResetError("connection reset")),
        "http503": urllib.error.HTTPError(
            "https://example.invalid", 503, "busy", {}, None
        ),
    }

    class InterruptedResponse(io.BytesIO):
        def read(self, size=-1):
            if self.tell():
                raise ConnectionResetError("download interrupted")
            return super().read(5)

    def open_archive(url, *, timeout):
        attempts.append((url, timeout))
        if len(attempts) <= 2:
            if failure != "read":
                raise errors[failure]
            response = InterruptedResponse(b"partial archive")
            responses.append(response)
            return response
        return original_open(url, timeout=timeout)

    def install(command, **kwargs):
        assert Path(command[-1]).read_bytes() == b"verified torch wheel"
        installs.append(command)

    monkeypatch.setattr(image_stack, "package_versions", lambda: {})
    monkeypatch.setattr(image_stack.urllib.request, "urlopen", open_archive)
    monkeypatch.setattr(time, "sleep", delays.append)
    monkeypatch.setattr(subprocess, "run", install)
    image_stack.install_torch(stack_trees[0])
    assert attempts == [(image_stack.TORCH_OVERRIDE["torch_archive_url"], 90)] * 3
    assert delays == [1, 2]
    assert len(installs) == 1
    assert all(response.closed for response in responses)


def test_prepare_download_stops_after_three_attempts(
    image_stack, stack_trees, monkeypatch
):
    import time

    failure = TimeoutError("download timed out")
    attempts, delays = [], []

    def open_archive(*args, **kwargs):
        attempts.append(args)
        raise failure

    monkeypatch.setattr(image_stack, "package_versions", lambda: {})
    monkeypatch.setattr(image_stack.urllib.request, "urlopen", open_archive)
    monkeypatch.setattr(time, "sleep", delays.append)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: pytest.fail("failed downloads must not reach installation"),
    )
    with pytest.raises(TimeoutError) as caught:
        image_stack.install_torch(stack_trees[0])
    assert caught.value is failure
    assert len(attempts) == 3
    assert delays == [1, 2]


def test_prepare_does_not_retry_http_not_found(image_stack, stack_trees, monkeypatch):
    import time
    import urllib.error

    failure = urllib.error.HTTPError(
        "https://example.invalid", 404, "missing", {}, None
    )
    attempts = []

    def open_archive(*args, **kwargs):
        attempts.append(args)
        raise failure

    monkeypatch.setattr(image_stack, "package_versions", lambda: {})
    monkeypatch.setattr(image_stack.urllib.request, "urlopen", open_archive)
    monkeypatch.setattr(
        time, "sleep", lambda _: pytest.fail("permanent errors must not be retried")
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: pytest.fail("failed downloads must not reach installation"),
    )
    with pytest.raises(urllib.error.HTTPError) as caught:
        image_stack.install_torch(stack_trees[0])
    assert caught.value is failure
    assert len(attempts) == 1


@pytest.mark.parametrize("damage", ["archive", "wheel", "missing", "duplicate"])
def test_prepare_rejects_invalid_torch_before_pip(
    image_stack, stack_trees, torch_archive, monkeypatch, damage
):
    import time

    monkeypatch.setattr(
        time, "sleep", lambda _: pytest.fail("validation errors must not be retried")
    )
    if damage in ("archive", "wheel"):
        image_stack.TORCH_OVERRIDE[f"torch_{damage}_sha256"] = "0" * 64
    elif damage == "missing":
        torch_archive(["package/torchvision-0.28.0-cp312-cp312-linux_x86_64.whl"])
    else:
        torch_archive(
            [
                "first/torch-2.13.0-cp312-cp312-linux_x86_64.whl",
                "second/torch-2.13.0-cp312-cp312-linux_x86_64.whl",
            ]
        )
    monkeypatch.setattr(image_stack, "package_versions", lambda: {})
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: pytest.fail("unverified packages must not be installed"),
    )
    with pytest.raises(ValueError, match="SHA256|exactly one"):
        image_stack.install_torch(stack_trees[0])


@pytest.mark.parametrize("damage", ["sdk", "packages"])
def test_prepare_rejects_changes_outside_torch(
    image_stack, stack_trees, torch_archive, monkeypatch, damage
):
    sdk, _ = stack_trees
    versions = {"torch": "2.13.0", "triton": "3.7.0"}
    monkeypatch.setattr(image_stack, "package_versions", lambda: dict(versions))

    def install(*args, **kwargs):
        if damage == "sdk":
            (sdk / "hgcc").write_bytes(b"changed")
        else:
            versions["triton"] = "different"

    monkeypatch.setattr(subprocess, "run", install)
    with pytest.raises(ValueError, match="image stack changed"):
        image_stack.install_torch(sdk)


@pytest.mark.parametrize("header", [True, False])
def test_prepare_records_replacement_provenance_and_checks_header(
    image_stack, stack_trees, torch_archive, tmp_path, monkeypatch, header, capsys
):
    from types import SimpleNamespace

    original_run = subprocess.run
    sdk, torch_root = stack_trees
    if header:
        (torch_root / ".ppu_compat").mkdir()
        (torch_root / ".ppu_compat/compatible_wrapper.h").write_text("header")
    output = tmp_path / "stack.json"
    constraints = tmp_path / "constraints.txt"
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(image_stack, "package_versions", lambda: {"torch": "2.13.0"})

    def install(*args, **kwargs):
        assert "torch" not in sys.modules
        print("pip installation progress", file=kwargs.get("stdout"))
        monkeypatch.setitem(
            sys.modules,
            "torch",
            SimpleNamespace(
                __file__=str(torch_root / "__init__.py"), __version__="2.13.0"
            ),
        )

    monkeypatch.setattr(subprocess, "run", install)
    argv = ["stack", "prepare", "--sdk", str(sdk), "--output", str(output)]
    monkeypatch.setattr(
        sys,
        "argv",
        [*argv, "--constraints", str(constraints), "--require-ppu-headers"],
    )
    if not header:
        with pytest.raises(ValueError, match="compatible_wrapper.h"):
            image_stack.main()
        assert not output.exists()
        return
    image_stack.main()
    captured = capsys.readouterr()
    from scripts.ci.ppu_ops_select import read_diagnostics

    manifest = tmp_path / "build-manifest.txt"
    text = SCRIPT.read_text()
    start = text.index("{\n    printf 'status=running")
    end = text.index('} >>"${MANIFEST}"', start) + len('} >>"${MANIFEST}"')
    original_run(
        ["bash", "-c", text[start:end]],
        env={
            **os.environ,
            "MANIFEST": str(manifest),
            "VLLM_REF": "a" * 40,
            "PYTORCH_SAIL_ARCH": "ppu_15;ppu_10",
        },
        check=True,
    )
    manifest.write_text(manifest.read_text() + captured.out)
    assert read_diagnostics(manifest)["stack_source"] == "image+torch"
    observed = json.loads(output.read_text())
    assert observed["stack_source"] == "image+torch"
    assert all(observed[k] == v for k, v in image_stack.TORCH_OVERRIDE.items())
    assert "torch===2.13.0" in constraints.read_text()
    monkeypatch.setattr(sys, "argv", ["stack", "verify", *argv[2:]])
    image_stack.main()
    (torch_root / "_C.so").write_bytes(b"modified after preparation")
    with pytest.raises(ValueError, match="torch_sha256"):
        image_stack.main()


@pytest.mark.parametrize("script", ["ppu_pin_ut.sh", "ppu_e2e_smoke.sh"])
def test_runtime_scripts_capture_same_image_stack_before_installing_wheels(script):
    text = (ROOT / "scripts/ci" / script).read_text()
    capture = text.index('ppu_image_stack.py" capture')
    install = text.index("python -m pip install")
    verify = text.index('ppu_image_stack.py" verify')
    assert capture < install < verify
    assert "--constraints" in text[capture:install]


def test_native_repack_is_verified_and_falls_back_to_full_build() -> None:
    text = SCRIPT.read_text()
    fingerprint = text.index('ppu_wheel_manifest.py" \\\n        fingerprint')
    verify = text.index('ppu_wheel_manifest.py" verify')
    extract = text.index('ppu_wheel_manifest.py" extract-native')
    prebuilt = text.index('export VLLM_SAIL_PREBUILT_EXT_DIR="${PREBUILT_EXT_DIR}"')
    package = text.index("python setup.py bdist_wheel")
    assert verify < fingerprint < extract < prebuilt < package
    assert "if ((NATIVE_REUSE == 0)); then" in text
    assert 'git init "${VLLM_SRC}"' in text
    assert "Native reuse unavailable or invalid; compiling" in text
    assert "|| return 1" in text
