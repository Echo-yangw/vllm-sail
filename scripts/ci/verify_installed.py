#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check installed vLLM/SAIL payload against the original wheel bytes, without pip reinstall."""

import argparse
import hashlib
import importlib
import json
import zipfile
from importlib.metadata import distribution
from pathlib import Path, PurePosixPath


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "wheel-manifest.json").read_text(encoding="utf-8"))
    results = {}
    for package, module in (("vllm", "vllm"), ("vllm-sail", "vllm_sail")):
        entry = manifest["wheels"][package]
        relative = PurePosixPath(entry["file"])
        assert not relative.is_absolute() and ".." not in relative.parts, (
            "unsafe wheel path"
        )
        wheel = root.joinpath(*relative.parts).resolve()
        assert wheel.is_relative_to(root), "wheel escaped root"
        assert digest(wheel) == entry["sha256"], "original wheel hash mismatch"
        installed = distribution(package)
        assert installed.version == entry["version"], (
            f"{package}: installed version mismatch"
        )
        payload = []
        with zipfile.ZipFile(wheel) as archive:
            for info in archive.infolist():
                name = PurePosixPath(info.filename)
                assert not name.is_absolute() and ".." not in name.parts
                if info.is_dir() or not info.filename.startswith(module + "/"):
                    continue
                location = Path(installed.locate_file(info.filename)).resolve()
                assert location.is_file(), (
                    f"{package}: missing installed payload: {info.filename}"
                )
                assert not any(
                    location.is_relative_to(Path(p))
                    for p in (
                        "/source",
                        "/wheels",
                        "/checks",
                        "/tmp/sail-existing-tests",
                    )
                )
                with archive.open(info) as handle:
                    expected = hashlib.sha256()
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        expected.update(chunk)
                assert digest(location) == expected.hexdigest(), (
                    f"{package}: changed installed payload: {info.filename}"
                )
                payload.append(location)
        assert payload, f"{package}: wheel has no package payload"
        imported = importlib.import_module(module)
        imported_path = Path(imported.__file__).resolve()
        assert imported_path in payload, (
            f"{package}: import is shadowed: {imported_path}"
        )
        results[package] = {
            "version": installed.version,
            "verified_files": len(payload),
            "module": str(imported_path),
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheels", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.wheels), indent=2))


if __name__ == "__main__":
    main()
