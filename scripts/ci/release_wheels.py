# SPDX-License-Identifier: Apache-2.0
"""Attach original wheels once, then hand SAIL alone to the PTG template."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

MANIFEST = "release-manifest.json"
AXES = (
    "python_version",
    "os_version",
    "runtime_version",
    "torch_version",
    "sdk_version",
)
ROOT = Path(__file__).resolve().parents[2]


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def outputs(**values: str) -> None:
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
        for key, value in values.items():
            if "\n" in value or "\r" in value:
                raise ValueError("workflow outputs must be single-line values")
            stream.write(f"{key}={value}\n")


def validate_tag(tag: str) -> str:
    from packaging.version import Version

    if not re.fullmatch(r"[vV]?[0-9][A-Za-z0-9.!+_-]*", tag):
        raise ValueError("release tag must be a version")
    Version(tag)
    return tag


def plan() -> None:
    entries = json.loads(os.environ["MATRIX_JSON"])["include"]
    if len(entries) != 1:
        raise ValueError(
            "release attachments currently require exactly one matrix entry"
        )
    entry = entries[0]
    tag = os.environ.get("TAG_VERSION", "")
    if not tag and os.environ.get("GITHUB_REF_TYPE") == "tag":
        tag = os.environ["GITHUB_REF_NAME"]
    if not tag:
        for candidate in command(
            "git", "tag", "--points-at", "HEAD", "--sort=-version:refname"
        ).splitlines():
            try:
                validate_tag(candidate)
            except ValueError:
                continue
            if "rc" not in candidate.lower():
                tag = candidate
                break
    if tag:
        validate_tag(tag)
        if (
            command("git", "rev-parse", f"refs/tags/{tag}^{{commit}}")
            != os.environ["GITHUB_SHA"]
        ):
            raise ValueError(
                "target tag must point to the selected workflow source commit"
            )
    pin = (ROOT / ".github/vllm-main-verified.commit").read_text().strip()
    if not re.fullmatch("[0-9a-f]{40}", pin):
        raise ValueError("expected a full upstream commit")
    outputs(
        image=entry["image"],
        sdk_url=entry["sdk_url"],
        pccl_url=entry["pccl_url"],
        configuration=json.dumps({key: entry[key] for key in AXES}, sort_keys=True),
        tag=tag,
        vllm_commit=pin,
    )


def stage(wheels: Path, output: Path) -> None:
    pin = (ROOT / ".github/vllm-main-verified.commit").read_text().strip()
    # Reuse PPU CI's CPU-only native/Rust/metadata/hash verifier.
    original = json.loads(
        command(
            sys.executable,
            str(ROOT / "scripts/ci/ppu_wheel_manifest.py"),
            "verify",
            str(wheels),
            "--sail-commit",
            os.environ["GITHUB_SHA"],
            "--vllm-commit",
            pin,
        )
    )
    output.mkdir(parents=True, exist_ok=False)
    records = {}
    for name, record in original["wheels"].items():
        source = wheels / record["file"]
        if source.is_symlink() or not source.resolve().is_relative_to(wheels.resolve()):
            raise ValueError("wheel must be a regular file inside the build output")
        shutil.copyfile(source, output / source.name)
        records[name] = {**record, "file": source.name}
    data = {
        "schema": 1,
        "tag": os.environ["RELEASE_TAG"],
        "commits": original["commits"],
        "configuration": json.loads(os.environ["RELEASE_CONFIGURATION"]),
        "wheels": records,
    }
    target = output / MANIFEST
    target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    outputs(manifest_sha256=digest(target))


def read_manifest(
    path: Path, expected: str, tag: str, commit: str, configuration: dict
) -> dict:
    if not re.fullmatch("[0-9a-f]{64}", expected) or digest(path) != expected:
        raise ValueError("release manifest SHA256 mismatch")
    data = json.loads(path.read_text())
    if (
        data["schema"] != 1
        or data["tag"] != tag
        or data["commits"]["vllm-sail"] != commit
        or data["configuration"] != configuration
    ):
        raise ValueError("release manifest source, tag or configuration mismatch")
    if set(data["wheels"]) != {"vllm", "vllm-sail"}:
        raise ValueError("release must contain both wheels")
    for name, record in data["wheels"].items():
        filename = record["file"]
        if not re.fullmatch(r"[A-Za-z0-9_.+!-]+\.whl", filename):
            raise ValueError("invalid release wheel filename")
        if not filename.startswith(name.replace("-", "_") + "-"):
            raise ValueError("release wheel package mismatch")
        if not re.fullmatch("[0-9a-f]{64}", record["sha256"]):
            raise ValueError("invalid wheel SHA256")
    return data


def repository_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value):
        raise ValueError("invalid GitHub repository")
    return value


def gh_json(endpoint: str) -> dict:
    return json.loads(command("gh", "api", endpoint))


def publish(directory: Path) -> None:
    repository = repository_name(os.environ["GITHUB_REPOSITORY"])
    tag = validate_tag(os.environ["RELEASE_TAG"])
    commit = os.environ["GITHUB_SHA"]
    configuration = json.loads(os.environ["RELEASE_CONFIGURATION"])
    expected = os.environ["EXPECTED_MANIFEST_SHA256"]
    data = read_manifest(directory / MANIFEST, expected, tag, commit, configuration)
    files = [directory / record["file"] for record in data["wheels"].values()]
    for path, record in zip(files, data["wheels"].values(), strict=False):
        if path.is_symlink() or digest(path) != record["sha256"]:
            raise ValueError("release wheel SHA256 mismatch")
    if set(directory.glob("*.whl")) != set(files):
        raise ValueError("unexpected wheel in release assets")
    ref = gh_json(
        f"repos/{repository}/git/ref/tags/{urllib.parse.quote(tag, safe='')}"
    )["object"]
    while ref["type"] == "tag":
        ref = gh_json(f"repos/{repository}/git/tags/{ref['sha']}")["object"]
    if ref["type"] != "commit" or ref["sha"] != commit:
        raise ValueError("remote tag no longer points to the built source commit")
    endpoint = f"repos/{repository}/releases/tags/{urllib.parse.quote(tag, safe='')}"
    found = subprocess.run(["gh", "api", endpoint], capture_output=True, text=True)
    if found.returncode:
        if "404" not in found.stderr:
            raise RuntimeError(found.stderr)
        args = [
            "gh",
            "release",
            "create",
            tag,
            "--repo",
            repository,
            "--verify-tag",
            "--title",
            tag,
            "--prerelease",
            "--latest=false",
            "--notes",
            "",
        ]
        subprocess.run(args, check=True)
    release = gh_json(endpoint)
    if release["draft"]:
        raise ValueError("PTG requires a published, non-draft GitHub Release")
    files.append(directory / MANIFEST)
    existing = {asset["name"] for asset in release["assets"]}
    # Release notes, promotion and latest selection belong to the maintainer.
    # A retry may reuse a manually finalized Release, but cannot modify it.
    missing = [path.name for path in files if path.name not in existing]
    if not release["prerelease"] and missing:
        raise ValueError(
            "finalized Release is missing required assets: " + ", ".join(missing)
        )
    # Check every existing name before uploading anything. Never clobber assets.
    with tempfile.TemporaryDirectory() as temporary:
        for path in files:
            if path.name in existing:
                subprocess.run(
                    [
                        "gh",
                        "release",
                        "download",
                        tag,
                        "--repo",
                        repository,
                        "--pattern",
                        path.name,
                        "--dir",
                        temporary,
                    ],
                    check=True,
                )
                if digest(Path(temporary) / path.name) != digest(path):
                    raise ValueError(f"existing release asset differs: {path.name}")
    for path in files:
        if path.name not in existing:
            subprocess.run(
                ["gh", "release", "upload", tag, str(path), "--repo", repository],
                check=True,
            )
    outputs(
        ptg_command=shlex.join(
            [
                "python3",
                "scripts/ci/release_wheels.py",
                "fetch-ptg",
                "--repository",
                repository,
                "--tag",
                tag,
                "--commit",
                commit,
                "--manifest-sha256",
                expected,
                "--configuration",
                json.dumps(configuration, sort_keys=True),
            ]
        )
    )


def download(url: str, target: Path) -> None:
    for attempt in range(3):
        try:
            with (
                urllib.request.urlopen(url, timeout=120) as stream,
                target.open("wb") as output,
            ):
                shutil.copyfileobj(stream, output)
            return
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def fetch_ptg(args: argparse.Namespace) -> None:
    repository = repository_name(args.repository)
    tag = validate_tag(args.tag)
    configuration = json.loads(args.configuration)
    if any(os.environ[key.upper()] != configuration[key] for key in AXES):
        raise ValueError("PTG matrix differs from the wheel build configuration")
    base = f"https://github.com/{repository}/releases/download/{urllib.parse.quote(tag, safe='')}/"
    dist = Path("dist")
    if dist.is_symlink() or (dist.exists() and any(dist.iterdir())):
        raise ValueError("PTG dist must be an empty regular directory")
    with tempfile.TemporaryDirectory() as temporary:
        staging = Path(temporary)
        manifest = staging / MANIFEST
        download(base + MANIFEST, manifest)
        data = read_manifest(
            manifest, args.manifest_sha256, tag, args.commit, configuration
        )
        record = data["wheels"]["vllm-sail"]
        wheel = staging / record["file"]
        download(base + urllib.parse.quote(wheel.name, safe=""), wheel)
        if digest(wheel) != record["sha256"]:
            raise ValueError("downloaded SAIL wheel SHA256 mismatch")
        dist.mkdir(exist_ok=True)
        shutil.copyfile(wheel, dist / wheel.name)
    print(f"Staged verified SAIL wheel for the unchanged PTG publisher: {wheel.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan")
    stage_parser = commands.add_parser("stage")
    stage_parser.add_argument("--wheels", type=Path, required=True)
    stage_parser.add_argument("--output", type=Path, required=True)
    publish_parser = commands.add_parser("publish")
    publish_parser.add_argument("--directory", type=Path, required=True)
    fetch_parser = commands.add_parser("fetch-ptg")
    for name in ("repository", "tag", "commit", "manifest-sha256", "configuration"):
        fetch_parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    if args.command == "plan":
        plan()
    elif args.command == "stage":
        stage(args.wheels, args.output)
    elif args.command == "publish":
        publish(args.directory)
    else:
        fetch_ptg(args)


if __name__ == "__main__":
    main()
