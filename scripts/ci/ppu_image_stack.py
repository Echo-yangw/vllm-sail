# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

TORCH_OVERRIDE = {
    "torch_archive_url": "https://art-pub.eng.t-head.cn/artifactory/apackage/daily/ppu.2v2_release/latest/PPU_SDK_PYPI/pytorch/ppu_sdk_hggcrt3-pytorch2.13.0-ubuntu2404-py312.tar.gz",
    "torch_archive_sha256": "63e23fedca52f1039b2deac29aa37d7b88d8ba7413f9b5600d223ffb095ec1f7",
    "torch_wheel_sha256": "24a95ab1e6bbc9d8dd61a4347020dd1975e7bdfb06e2110bbc618874a8675760",
}

PROTECTED_PACKAGES = (
    "torch",
    "torchvision",
    "torchaudio",
    "triton",
    "flash-attn",
    "flash-attn-3",
    "pla",
    "deep-gemm",
    "acext",
    "tokenspeed-mla",
    "tokenspeed-triton",
)


def tree_digest(root, *, excluded=()):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"missing or empty image component: {root}")
    digest = hashlib.sha256()
    count = 0

    def visit(directory, ancestors):
        nonlocal count
        resolved = directory.resolve()
        if resolved in ancestors:
            return
        ancestors = ancestors | {resolved}
        for path in sorted(directory.iterdir()):
            relative = path.relative_to(root)
            if relative.as_posix() in excluded:
                continue
            if path.name == "__pycache__" or path.suffix in (".pyc", ".pyo"):
                continue
            if path.is_symlink():
                digest.update(f"link:{relative}:{path.readlink()}\0".encode())
            if path.is_dir():
                visit(path, ancestors)
            elif path.is_file():
                digest.update(f"file:{relative}:{path.stat().st_size}\0".encode())
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                count += 1

    visit(root, set())
    if not count:
        raise ValueError(f"missing or empty image component: {root}")
    return digest.hexdigest()


def snapshot(sdk, torch_root, torch_version, packages):
    # The SDK generates compiled kernels under this root during inference.
    # Runtime evidence inventories that mutable cache separately from SDK code.
    return {
        "stack_source": "image",
        "fingerprint_kind": "installed-tree-sha256",
        "sdk_path": str(Path(sdk).resolve()),
        "sdk_sha256": tree_digest(sdk, excluded=("rtccache",)),
        "torch_path": str(Path(torch_root).resolve()),
        "torch_sha256": tree_digest(torch_root),
        "torch_version": torch_version,
        "packages": dict(packages),
    }


def verify_unchanged(before, after):
    if before != after:
        changed = sorted(
            key
            for key in before.keys() | after.keys()
            if before.get(key) != after.get(key)
        )
        raise ValueError(f"image stack changed: {', '.join(changed)}")


def package_versions():
    versions = {}
    for name in PROTECTED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def constraints_text(versions):
    return "".join(
        f"{name}==={versions[name]}\n" if name in versions else f"{name}<0\n"
        for name in PROTECTED_PACKAGES
    )


def copy_verified(source, destination, expected):
    digest = hashlib.sha256()
    with destination.open("wb") as target:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            target.write(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(
            f"SHA256 mismatch for {destination.name}: {digest.hexdigest()}"
        )


def install_torch(sdk):
    def preserved():
        return {
            "sdk_sha256": tree_digest(sdk, excluded=("rtccache",)),
            "packages": {k: v for k, v in package_versions().items() if k != "torch"},
        }

    before = preserved()
    with tempfile.TemporaryDirectory(prefix="ppu-torch-") as temporary:
        archive = Path(temporary) / "torch.tar.gz"
        print(
            f"torch_archive_url={TORCH_OVERRIDE['torch_archive_url']}",
            file=sys.stderr,
            flush=True,
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(
                    TORCH_OVERRIDE["torch_archive_url"], timeout=90
                ) as source:
                    copy_verified(
                        source, archive, TORCH_OVERRIDE["torch_archive_sha256"]
                    )
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                if isinstance(error, urllib.error.HTTPError) and error.code not in (
                    408,
                    429,
                    500,
                    502,
                    503,
                    504,
                ):
                    raise
                if attempt == 2:
                    raise
                delay = 2**attempt
                print(
                    f"Torch download attempt {attempt + 1}/3 failed: {error}; "
                    f"retrying in {delay}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
        with tarfile.open(archive, "r:gz") as source:
            wheels = [
                member
                for member in source.getmembers()
                if member.isfile()
                and Path(member.name).name.startswith("torch-")
                and member.name.endswith(".whl")
            ]
            if len(wheels) != 1:
                raise ValueError(
                    f"expected exactly one torch wheel, found {len(wheels)}"
                )
            wheel = Path(temporary) / Path(wheels[0].name).name
            with source.extractfile(wheels[0]) as stream:
                copy_verified(stream, wheel, TORCH_OVERRIDE["torch_wheel_sha256"])
        env = dict(os.environ)
        env.pop("PIP_CONSTRAINT", None)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--force-reinstall",
                str(wheel),
            ],
            env=env,
            check=True,
            stdout=sys.stderr,
        )
    verify_unchanged(before, preserved())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "capture", "verify"))
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--constraints", type=Path)
    parser.add_argument("--require-ppu-headers", action="store_true")
    args = parser.parse_args()
    capture = args.command in ("capture", "prepare")
    if capture and args.constraints is None:
        parser.error(f"{args.command} requires --constraints")
    before = {} if capture else json.loads(args.output.read_text())
    replaced = args.command == "prepare" or before.get("stack_source") == "image+torch"
    if args.command == "prepare":
        install_torch(args.sdk)

    import torch

    if (
        args.require_ppu_headers
        and not (
            Path(torch.__file__).parent / ".ppu_compat/compatible_wrapper.h"
        ).is_file()
    ):
        source = "prepared" if replaced else "image-provided"
        raise ValueError(f"{source} Torch is missing .ppu_compat/compatible_wrapper.h")
    observed = snapshot(
        args.sdk,
        Path(torch.__file__).parent,
        str(torch.__version__),
        package_versions(),
    )
    if replaced:
        observed.update(stack_source="image+torch", **TORCH_OVERRIDE)
    if capture:
        args.output.write_text(json.dumps(observed, sort_keys=True, indent=2) + "\n")
        args.constraints.write_text(constraints_text(observed["packages"]))
        for key in (
            "stack_source",
            "fingerprint_kind",
            "sdk_sha256",
            "torch_sha256",
            "torch_version",
        ):
            print(f"{key}={observed[key]}")
        if replaced:
            for key, value in TORCH_OVERRIDE.items():
                print(f"{key}={value}")
    else:
        verify_unchanged(before, observed)
        print("stack_unchanged=true")


if __name__ == "__main__":
    main()
