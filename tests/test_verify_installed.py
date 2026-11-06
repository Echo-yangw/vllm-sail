# SPDX-License-Identifier: Apache-2.0
"""Offline synthetic-wheel tests; no real pip, Docker, PPU or network."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    "installed_check",
    Path(__file__).resolve().parents[1] / "scripts/ci/verify_installed.py",
)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class TestInstalledWheelCheck(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.wheels = self.root / "wheels"
        self.site = self.root / "site"
        self.wheels.mkdir()
        self.site.mkdir()
        self.manifest = {"wheels": {}}
        self.dists = {}
        self.modules = {}
        for package, module in (("vllm", "vllm"), ("vllm-sail", "vllm_sail")):
            relative = f"{package}/{module}-1.0-py3-none-any.whl"
            wheel = self.wheels / relative
            wheel.parent.mkdir()
            with zipfile.ZipFile(wheel, "w") as archive:
                for filename, content in (
                    (f"{module}/__init__.py", b'version = "1.0"\n'),
                    (f"{module}/native.so", b"synthetic native payload"),
                ):
                    archive.writestr(filename, content)
                    target = self.site / filename
                    target.parent.mkdir(exist_ok=True)
                    target.write_bytes(content)
            self.manifest["wheels"][package] = {
                "file": relative,
                "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                "version": "1.0",
            }
            self.dists[package] = SimpleNamespace(
                version="1.0", locate_file=lambda path, site=self.site: site / path
            )
            self.modules[module] = SimpleNamespace(
                __file__=str(self.site / module / "__init__.py")
            )
        self.save_manifest()
        self.dist_patch = mock.patch.object(
            CHECK, "distribution", side_effect=lambda name: self.dists[name]
        )
        self.import_patch = mock.patch.object(
            CHECK.importlib,
            "import_module",
            side_effect=lambda name: self.modules[name],
        )
        self.dist_patch.start()
        self.import_patch.start()
        self.addCleanup(self.dist_patch.stop)
        self.addCleanup(self.import_patch.stop)

    def save_manifest(self):
        (self.wheels / "wheel-manifest.json").write_text(
            json.dumps(self.manifest), encoding="utf-8"
        )

    def test_original_python_and_native_payloads_pass(self):
        result = CHECK.verify(self.wheels)
        self.assertEqual(result["vllm"]["verified_files"], 2)
        self.assertEqual(result["vllm-sail"]["verified_files"], 2)

    def test_changed_native_and_python_payload_fail(self):
        for filename in ("vllm/native.so", "vllm_sail/__init__.py"):
            with self.subTest(filename=filename):
                path = self.site / filename
                original = path.read_bytes()
                path.write_bytes(b"old base image payload")
                with self.assertRaisesRegex(
                    AssertionError, "changed installed payload"
                ):
                    CHECK.verify(self.wheels)
                path.write_bytes(original)

    def test_original_wheel_hash_is_mandatory(self):
        self.manifest["wheels"]["vllm"]["sha256"] = "0" * 64
        self.save_manifest()
        with self.assertRaisesRegex(AssertionError, "original wheel hash mismatch"):
            CHECK.verify(self.wheels)

    def test_version_and_import_shadow_fail(self):
        self.dists["vllm"].version = "0.9"
        with self.assertRaisesRegex(AssertionError, "installed version mismatch"):
            CHECK.verify(self.wheels)
        self.dists["vllm"].version = "1.0"
        self.modules["vllm"].__file__ = "/source/vllm/__init__.py"
        with self.assertRaisesRegex(AssertionError, "import is shadowed"):
            CHECK.verify(self.wheels)

    def test_manifest_path_traversal_fail(self):
        self.manifest["wheels"]["vllm"]["file"] = "../elsewhere.whl"
        self.save_manifest()
        with self.assertRaisesRegex(AssertionError, "unsafe wheel path"):
            CHECK.verify(self.wheels)


if __name__ == "__main__":
    unittest.main()
