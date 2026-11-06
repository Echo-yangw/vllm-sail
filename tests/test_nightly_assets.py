# SPDX-License-Identifier: Apache-2.0
"""Offline, standard-library tests for the shared nightly assets CLI."""

from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
import io
import json
import os
import stat
import tempfile
import unittest
import urllib.error
import urllib.parse
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

TOOL = Path(__file__).resolve().parents[1] / "scripts" / "ci" / "nightly_assets.py"
SPEC = importlib.util.spec_from_file_location("nightly_assets", TOOL)
assets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assets)
UPSTREAM = "vllm-project/vllm-sail"
FORK = "example/vllm-sail"
UP_SHA = "a" * 40
VLLM_SHA = "b" * 40
CONTROL_SHA = "c" * 40
IMAGE_REPO = "ghcr.io/example/vllm-sail"
IMAGE_DIGEST = "sha256:" + "d" * 64
NOW = datetime.now(timezone.utc).isoformat()


def run_record(repo, run_id, workflow, *, created=NOW, sha=None):
    return {
        "id": run_id,
        "run_number": run_id // 10 + 1,
        "name": "Nightly regression (minimum)"
        if workflow == "nightly-tests.yaml"
        else workflow,
        "status": "completed",
        "conclusion": "success",
        "event": "schedule",
        "head_branch": "main",
        "repository": {"full_name": repo},
        "head_repository": {"full_name": repo},
        "path": f".github/workflows/{workflow}",
        "workflow_id": 11 if workflow == "ppu-accuracy-nightly.yaml" else 22,
        "head_sha": sha or (UP_SHA if repo == UPSTREAM else CONTROL_SHA),
        "created_at": created,
    }


def make_wheels(root):
    root.mkdir(parents=True)
    wheels = {}
    for package, filename in (
        ("vllm", "vllm-1.0-py3-none-any.whl"),
        ("vllm-sail", "vllm_sail-1.0-py3-none-any.whl"),
    ):
        (root / package).mkdir()
        wheel = root / package / filename
        # Wheel bytes are deliberately opaque to the promotion tool: never rebuild.
        wheel.write_bytes(f"original upstream bytes for {package}".encode())
        wheels[package] = {
            "file": f"{package}/{filename}",
            "version": "1.0",
            "sha256": assets.sha256_file(wheel),
        }
    manifest = {
        "schema": 1,
        "commits": {"vllm": VLLM_SHA, "vllm-sail": UP_SHA},
        "wheels": wheels,
    }
    assets.write_json(root / assets.MANIFEST, manifest)
    return manifest


def make_selection(root, run_id=100):
    manifest = assets.validate_wheels(root)
    return {
        "schema": 1,
        "upstream_repository": UPSTREAM,
        "upstream_workflow": "ppu-accuracy-nightly.yaml",
        "upstream_run_id": run_id,
        "upstream_run_number": run_id // 10 + 1,
        "upstream_created_at": NOW,
        "upstream_sha": UP_SHA,
        "vllm_sha": VLLM_SHA,
        "artifact_id": 77,
        "artifact_name": f"ppu-wheels-{run_id}",
        "wheel_manifest_sha256": assets.sha256_file(root / assets.MANIFEST),
        "wheel_hashes": {
            p: record["sha256"] for p, record in manifest["wheels"].items()
        },
    }


def make_candidate(source):
    return {
        "schema": 1,
        "source": source,
        "image_repository": IMAGE_REPO,
        "image_digest": IMAGE_DIGEST,
        "image": f"{IMAGE_REPO}@{IMAGE_DIGEST}",
        "control_sha": CONTROL_SHA,
        "dockerfile_sha256": "e" * 64,
    }


def make_evidence(candidate, test_id=500):
    return {
        "schema": 1,
        "status": "passed",
        "test_run_id": test_id,
        "candidate_sha256": assets.sha256_bytes(assets.json_bytes(candidate)),
        "counts": {"ut": 2, "ops": 9},
        "results": {name: "f" * 64 for name in ("ut.xml", "ops.xml", "nodes.json")},
    }


def make_baseline(candidate, evidence):
    return {
        "schema": 1,
        "quality_scope": "minimum-regression",
        "channel": "preview",
        **{
            key: candidate[key]
            for key in (
                "source",
                "image_repository",
                "image_digest",
                "image",
                "control_sha",
                "dockerfile_sha256",
            )
        },
        "test_run_id": evidence["test_run_id"],
        "evidence": evidence,
        "immutable_release_tag": f"preview-nightly-{candidate['source']['upstream_run_id']}-test-{evidence['test_run_id']}",
    }


def junit(count, *, skipped=False, summary_count=None, identities=None):
    if identities is None:
        if count == 2:
            identities = [
                ("tests.ut.test_patch_install", "test_install_is_idempotent"),
                ("tests.ut.test_patch_install", "test_patches_actually_land"),
            ]
        elif count == 9:
            identities = [
                (
                    "tests.e2e.test_native_activation",
                    f"test_silu_and_mul[shape{i}-{dtype}]",
                )
                for i in range(3)
                for dtype in ("float16", "bfloat16", "float32")
            ]
        else:
            identities = [("fixed", f"test_{i}") for i in range(count)]
    cases = "".join(
        f'<testcase classname="{classname}" name="{name}">'
        + ("<skipped message='xfail'/>" if skipped else "")
        + "</testcase>"
        for classname, name in identities
    )
    return (
        f'<testsuites tests="{summary_count if summary_count is not None else count}">'
        f'<testsuite tests="{count}">{cases}</testsuite></testsuites>'
    ).encode()


def node_report(board="ppu15"):
    collected = sorted(
        f"tests/e2e/test_native_activation.py::test_silu_and_mul[shape{i}-{dtype}]"
        for i in range(3)
        for dtype in ("float16", "bfloat16", "float32")
    )
    modules = {
        "torch",
        "triton",
        "vllm",
        "vllm_sail",
        "vllm_sail.envs",
        "vllm.platforms",
        "vllm_sail.native.extensions",
        "vllm_sail._C",
        "vllm_sail._moe_C",
        "vllm_sail._upstream_C",
        "vllm_sail._upstream_moe_C",
    }
    paths = {
        name: f"/opt/python/site-packages/{name.replace('.', '/')}.py"
        for name in modules
    }
    paths.update(
        plugin="/tmp/sail-existing-tests/tests/support/ppu_ci.py",
        conftest="/tmp/sail-existing-tests/tests/conftest.py",
    )
    return {
        "schema": 1,
        "board": board,
        "file": "tests/e2e/test_native_activation.py",
        "status": "success",
        "exitstatus": 0,
        "errors": [],
        "collected": collected,
        "applicable": collected.copy(),
        "hardware_excluded": [],
        "nodes": {
            node: {
                phase: {
                    "outcome": "passed",
                    "wasxfail": None,
                    "duration_seconds": 0.01,
                    "reason": None,
                }
                for phase in ("setup", "call", "teardown")
            }
            for node in collected
        },
        "runtime": {
            "isolated_root": "/tmp/sail-existing-tests",
            "checkout_root": "/source",
            "paths": paths,
            "runtime_env": {"VLLM_SAIL_USE_PLA": True},
            "native_providers": ["_C"],
            "capability": [8, 1],
        },
        "benchmark_seconds": {},
    }


class FakeAPI:
    """Stateful REST substitute: never makes a request or publishes remotely."""

    def __init__(self):
        self.runs = {
            (UPSTREAM, 100): run_record(UPSTREAM, 100, "ppu-accuracy-nightly.yaml"),
            (FORK, 500): run_record(FORK, 500, "nightly-tests.yaml"),
        }
        self.releases = {}
        self.tags = {}
        self.payloads = {}
        self.asset_records = {}
        self.artifacts = [
            {
                "id": 77,
                "name": "ppu-wheels-100",
                "expired": False,
                "expires_at": (
                    datetime.now(timezone.utc) + timedelta(days=1)
                ).isoformat(),
                "workflow_run": {"id": 100, "head_sha": UP_SHA},
            }
        ]
        self.artifact_zip = b""
        self.mutations = []
        self.reads = []
        self.next_release = 1
        self.next_asset = 1000

    def get(self, path):
        self.reads.append(path)
        parts = path.split("/")
        repo = "/".join(parts[2:4])
        tail = "/".join(parts[4:])
        if not tail:
            return {"full_name": repo, "default_branch": "main"}
        if tail.startswith("actions/workflows/"):
            workflow = parts[-1]
            return {
                "id": 11 if workflow == "ppu-accuracy-nightly.yaml" else 22,
                "name": "Nightly regression (minimum)"
                if workflow == "nightly-tests.yaml"
                else workflow,
                "path": f".github/workflows/{workflow}",
            }
        if tail.startswith("actions/runs/"):
            return copy.deepcopy(self.runs[(repo, int(parts[-1]))])
        if tail.startswith("releases/tags/"):
            key = (repo, parts[-1])
            if key not in self.releases or self.releases[key].get("draft") is not False:
                raise assets.APIError(404)
            return copy.deepcopy(self.releases[key])
        if tail.startswith("commits/"):
            return {"sha": parts[-1]}
        if tail.startswith("git/ref/tags/"):
            key = (repo, parts[-1])
            if key not in self.tags:
                raise assets.APIError(404)
            return {"object": {"type": "commit", "sha": self.tags[key]}}
        raise AssertionError(f"unhandled fake read: {path}")

    def optional(self, path):
        try:
            return self.get(path)
        except assets.APIError as error:
            if error.status == 404:
                return None
            raise

    def pages(self, path, key=None):
        self.reads.append(path)
        repo = "/".join(path.split("/")[2:4])
        if "/actions/workflows/" in path:
            return iter(
                copy.deepcopy(
                    [r for (owner, _), r in self.runs.items() if owner == repo]
                )
            )
        if path.endswith("/artifacts"):
            return iter(copy.deepcopy(self.artifacts))
        if path.endswith("/assets"):
            release_id = int(path.split("/")[-2])
            return iter(
                copy.deepcopy(
                    [
                        record
                        for (owner, rid, _), record in self.asset_records.items()
                        if owner == repo and rid == release_id
                    ]
                )
            )
        if path.endswith("/releases"):
            return iter(
                copy.deepcopy(
                    [
                        release
                        for (owner, _), release in self.releases.items()
                        if owner == repo
                    ]
                )
            )
        raise AssertionError(f"unhandled fake list: {path}")

    def bytes(self, path, limit=assets.MAX_JSON):
        self.reads.append(path)
        data = self.payloads[int(path.split("/")[-1])]
        assets.require(len(data) <= limit, "fake download exceeds size limit")
        return data

    def download(self, path, output):
        self.reads.append(path)
        data = (
            self.artifact_zip
            if path.endswith("/zip")
            else self.bytes(path, assets.MAX_DOWNLOAD)
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            stream.write(data)

    def asset_hash(self, repo, asset_id):
        self.reads.append(f"/repos/{repo}/releases/assets/{asset_id}")
        return assets.sha256_bytes(self.payloads[asset_id])

    def mutate(self, method, path, value):
        self.mutations.append((method, path, copy.deepcopy(value)))
        parts = path.split("/")
        repo = "/".join(parts[2:4])
        if method == "POST" and path.endswith("/releases"):
            tag = value["tag_name"]
            if (repo, tag) in self.releases:
                raise assets.APIError(422)
            record = {
                "id": self.next_release,
                **value,
                "published_at": None,
                "url": f"{assets.API}/repos/{repo}/releases/{self.next_release}",
            }
            self.next_release += 1
            self.releases[(repo, tag)] = record
            self.tags.setdefault((repo, tag), value["target_commitish"])
            return copy.deepcopy(record)
        if method == "PATCH" and "/releases/assets/" in path:
            asset_id = int(parts[-1])
            for record in self.asset_records.values():
                if record["id"] == asset_id:
                    record.update(value)
                    return copy.deepcopy(record)
        if method == "PATCH" and "/releases/" in path:
            release_id = int(parts[-1])
            for record in self.releases.values():
                if record["id"] == release_id:
                    record.update(value)
                    if not record["draft"]:
                        record["published_at"] = NOW
                    return copy.deepcopy(record)
        raise AssertionError(f"unhandled fake mutation: {path}")

    def upload(self, repo, release_id, name, path, content_type):
        self.mutations.append(("POST", f"uploads/{release_id}/{name}", {}))
        existing = [
            x
            for (owner, rid, _), x in self.asset_records.items()
            if owner == repo and rid == release_id and x["name"] == name
        ]
        assert not existing, "fake GitHub would reject same-name upload"
        asset_id = self.next_asset
        self.next_asset += 1
        record = {"id": asset_id, "name": name, "state": "uploaded"}
        self.asset_records[(repo, release_id, asset_id)] = record
        self.payloads[asset_id] = path.read_bytes()
        return copy.deepcopy(record)

    def install_baseline(self, baseline, archive=b"placeholder"):
        self.install_release("preview-baseline", baseline, archive)
        self.mutations.clear()

    def install_release(self, tag, baseline, archive):
        release = self.mutate(
            "POST",
            f"/repos/{FORK}/releases",
            {
                "tag_name": tag,
                "target_commitish": baseline["control_sha"],
                "draft": True,
                "prerelease": True,
                "name": f"Preview {tag} (minimum regression)",
            },
        )
        for name, data in (
            ("preview-baseline.json", assets.json_bytes(baseline)),
            ("preview-wheels.zip", archive),
        ):
            asset_id = self.next_asset
            self.next_asset += 1
            self.asset_records[(FORK, release["id"], asset_id)] = {
                "id": asset_id,
                "name": name,
                "state": "uploaded",
            }
            self.payloads[asset_id] = data
        self.mutate(
            "PATCH", f"/repos/{FORK}/releases/{release['id']}", {"draft": False}
        )
        return copy.deepcopy(self.releases[(FORK, tag)])

    def install_legacy_release(self, tag="ci-baseline"):
        release = self.mutate(
            "POST",
            f"/repos/{FORK}/releases",
            {
                "tag_name": tag,
                "target_commitish": "1" * 40,
                "draft": True,
                "prerelease": False,
                "name": "Stable verified baseline",
            },
        )
        for name, data in (
            ("ci-baseline.json", b'{"verified_pin":"unchanged","bom":"stable"}'),
            ("ci-wheels.zip", b"legacy stable bytes; never read or repaired"),
        ):
            asset_id = self.next_asset
            self.next_asset += 1
            self.asset_records[(FORK, release["id"], asset_id)] = {
                "id": asset_id,
                "name": name,
                "state": "uploaded",
            }
            self.payloads[asset_id] = data
        self.mutate(
            "PATCH", f"/repos/{FORK}/releases/{release['id']}", {"draft": False}
        )
        self.mutations.clear()
        return copy.deepcopy(self.releases[(FORK, tag)])


class Response(io.BytesIO):
    def __init__(self, data=b"{}", headers=None):
        super().__init__(data)
        self.headers = headers or {}


class TestNightlyAssets(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.env = mock.patch.dict(
            os.environ, {"GH_TOKEN": "unit-test-secret"}, clear=True
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        # Catch any accidental networking even outside explicit transport tests.
        self.network = mock.patch(
            "urllib.request.OpenerDirector.open",
            side_effect=AssertionError("real networking forbidden in tests"),
        )
        self.network.start()
        self.addCleanup(self.network.stop)
        self.api = FakeAPI()

    def bundle(
        self, run_id=100, results=True, *, wheel_bytes=None, upstream_sha=UP_SHA
    ):
        bundle = self.root / f"bundle-{run_id}"
        wheelhouse = bundle / "wheelhouse"
        manifest = make_wheels(wheelhouse)
        if wheel_bytes is not None:
            record = manifest["wheels"]["vllm-sail"]
            wheel = wheelhouse / record["file"]
            wheel.write_bytes(wheel_bytes)
            record["sha256"] = assets.sha256_file(wheel)
        manifest["commits"]["vllm-sail"] = upstream_sha
        assets.write_json(wheelhouse / assets.MANIFEST, manifest)
        source = make_selection(wheelhouse, run_id)
        source["upstream_sha"] = upstream_sha
        candidate = make_candidate(source)
        evidence = make_evidence(candidate)
        assets.write_json(bundle / "selection.json", source)
        assets.write_json(bundle / "candidate.json", candidate)
        if results:
            counts, hashes = self.results(bundle / "results")
            evidence["counts"] = counts
            evidence["results"] = hashes
        assets.write_json(bundle / "test-evidence.json", evidence)
        return bundle, candidate, evidence

    def results(self, root):
        root.mkdir(parents=True)
        (root / "ut.xml").write_bytes(junit(2))
        (root / "ops.xml").write_bytes(junit(9))
        assets.write_json(root / "nodes.json", node_report())
        return assets.validate_results(root)

    def nightly_args(self, **changes):
        values = dict(
            source_repository=UPSTREAM,
            workflow="ppu-accuracy-nightly.yaml",
            branch="main",
            run_id=None,
            consumer_repository=FORK,
            include_published=False,
            output=self.root / "flow",
        )
        values.update(changes)
        return argparse.Namespace(**values)

    def zip_data(self, members):
        path = self.root / "input.zip"
        with zipfile.ZipFile(path, "w") as bundle:
            for name, data in members:
                bundle.writestr(name, data)
        return path

    def test_zip_rejects_traversal_absolute_windows_and_nul(self):
        for name in (
            "../outside",
            "/outside",
            "a/../../outside",
            "C:/outside",
            "a\\..\\outside",
        ):
            with self.subTest(name=name):
                path = self.zip_data([(name, b"bad")])
                with self.assertRaises(assets.AssetError):
                    assets.safe_extract(path, self.root / "extract")
                self.assertFalse((self.root / "extract").exists())
        with self.assertRaises(assets.AssetError):
            assets.safe_relative("a\x00b")

    def test_zip_rejects_symlink_special_file_and_size_limits(self):
        for mode in (stat.S_IFLNK, stat.S_IFIFO):
            info = zipfile.ZipInfo("link")
            info.create_system = 3
            info.external_attr = (mode | 0o777) << 16
            path = self.zip_data([(info, b"../outside")])
            with self.assertRaisesRegex(assets.AssetError, "symlink or special"):
                assets.safe_extract(path, self.root / "extract")
        path = self.zip_data([("a", b"123"), ("b", b"456")])
        with self.assertRaisesRegex(assets.AssetError, "member exceeds"):
            assets.safe_extract(path, self.root / "extract", max_file=2)
        with self.assertRaisesRegex(assets.AssetError, "unpacked size"):
            assets.safe_extract(path, self.root / "extract", max_total=5)
        self.assertFalse((self.root / "extract").exists())

    def test_zip_rejects_duplicates_collisions_and_overwrite(self):
        for members in ([("A", b"1"), ("a", b"2")], [("a", b"1"), ("a/b", b"2")]):
            with self.subTest(members=members):
                with self.assertRaises(assets.AssetError):
                    assets.safe_extract(self.zip_data(members), self.root / "extract")
        destination = self.root / "extract"
        destination.mkdir()
        (destination / "a").write_bytes(b"keep")
        with self.assertRaises(assets.AssetError):
            assets.safe_extract(self.zip_data([("a", b"replace")]), destination)
        self.assertEqual((destination / "a").read_bytes(), b"keep")
        (destination / "link").symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(assets.AssetError):
            assets.safe_extract(self.zip_data([("link/escape", b"bad")]), destination)

    def test_wheel_hash_and_source_sha_fail_closed(self):
        bundle, candidate, _ = self.bundle()
        root = bundle / "wheelhouse"
        bad = copy.deepcopy(candidate["source"])
        bad["upstream_sha"] = "1" * 40
        with self.assertRaisesRegex(assets.AssetError, "source SHA"):
            assets.validate_wheels(root, bad)
        with self.assertRaisesRegex(assets.AssetError, "source SHA"):
            assets.validate_wheels(root, upstream_sha="1" * 40)
        bad = copy.deepcopy(candidate["source"])
        bad["wheel_manifest_sha256"] = "1" * 64
        with self.assertRaisesRegex(assets.AssetError, "manifest SHA256"):
            assets.validate_wheels(root, bad)
        wheel = next((root / "vllm").glob("*.whl"))
        wheel.write_bytes(b"tampered")
        with self.assertRaisesRegex(assets.AssetError, "wheel SHA256"):
            assets.validate_wheels(root, candidate["source"])

    def test_exactly_two_wheels_and_safe_manifest_paths(self):
        bundle, candidate, _ = self.bundle()
        root = bundle / "wheelhouse"
        manifest = assets.read_json(root / assets.MANIFEST)
        bad = copy.deepcopy(manifest)
        bad["wheels"]["vllm"]["file"] = "../external.whl"
        assets.write_json(root / assets.MANIFEST, bad)
        with self.assertRaises(assets.AssetError):
            assets.validate_wheels(root)
        assets.write_json(root / assets.MANIFEST, manifest)
        (root / "vllm" / "vllm-2.0-py3-none-any.whl").write_bytes(b"extra")
        with self.assertRaisesRegex(assets.AssetError, "unexpected"):
            assets.validate_wheels(root, candidate["source"])

    def test_junit_all_skip_xfail_and_counts_cannot_be_faked(self):
        path = self.root / "ut.xml"
        for data in (
            junit(2, skipped=True),
            junit(0, summary_count=2),
            junit(1, summary_count=2),
            junit(3, summary_count=2),
            b'<testsuite><testcase name="a"/><testcase name="a"/></testsuite>',
            b'<testsuite><testcase name="a"><failure/></testcase><testcase name="b"/></testsuite>',
        ):
            with self.subTest(data=data):
                path.write_bytes(data)
                with self.assertRaises(assets.AssetError):
                    assets.junit_count(path, 2)
        path.write_bytes(junit(2, summary_count=999))
        self.assertEqual(assets.junit_count(path, 2), 2)
        path.write_bytes(b'<!DOCTYPE x [<!ENTITY a "b">]><testsuite/>')
        with self.assertRaises(assets.AssetError):
            assets.junit_count(path, 2)

    def test_ops_requires_nine_and_successful_nodes(self):
        root = self.root / "results"
        self.results(root)
        (root / "ops.xml").write_bytes(junit(8, summary_count=9))
        with self.assertRaises(assets.AssetError):
            assets.validate_results(root)
        (root / "ops.xml").write_bytes(junit(9))
        for changes in (
            {"exitstatus": 1},
            {"status": "failure"},
            {"errors": ["failed"]},
            {"nodes": [{"exitstatus": 3, "errors": []}]},
            {"exitstatus": False},
        ):
            data = {
                "schema": 1,
                "status": "success",
                "exitstatus": 0,
                "errors": [],
                **changes,
            }
            assets.write_json(root / "nodes.json", data)
            with self.assertRaises(assets.AssetError):
                assets.validate_results(root)

    def test_zip_rejects_all_existing_destination_ancestors_before_writes(self):
        real = self.root / "real"
        real.mkdir()
        (real / "existing").mkdir()
        alias = self.root / "alias"
        alias.symlink_to(real, target_is_directory=True)
        archive = self.zip_data([("payload", b"bad")])
        for destination in (
            alias / "extract",
            alias / "existing" / "deep" / "extract",
            alias / "existing",
        ):
            with (
                self.subTest(destination=destination),
                self.assertRaises(assets.AssetError),
            ):
                assets.safe_extract(archive, destination)
        self.assertEqual(list((real / "existing").iterdir()), [])
        self.assertFalse((real / "extract").exists())

    def test_junit_rejects_other_encodings_entities_and_wrong_identities(self):
        path = self.root / "junit.xml"
        entity = '<!DOCTYPE testsuite [<!ENTITY a "test_patches_actually_land">]>'
        xml = entity + junit(2).decode()
        bad = [
            xml.encode(),
            xml.encode("utf-16"),
            xml.encode("utf-16-le"),
            xml.encode("utf-16-be"),
            junit(2).decode().encode("utf-32"),
            b'<?xml version="1.0" encoding="ISO-8859-1"?>' + junit(2),
            b'<?xml version="1.0" encoding="UTF-16"?>' + junit(2),
            junit(2) + b"\xff",
        ]
        for data in bad:
            with self.subTest(data=data[:40]):
                path.write_bytes(data)
                with self.assertRaises(assets.AssetError):
                    assets.junit_count(path, 2)
        root = self.root / "results"
        self.results(root)
        for filename, count in (("ut.xml", 2), ("ops.xml", 9)):
            original = (root / filename).read_bytes()
            (root / filename).write_bytes(
                junit(
                    count, identities=[("unrelated", f"test_{i}") for i in range(count)]
                )
            )
            with self.assertRaisesRegex(assets.AssetError, "identity"):
                assets.validate_results(root)
            (root / filename).write_bytes(original)
        data = junit(9).replace(b"shape2-float32", b"shape9-float32")
        (root / "ops.xml").write_bytes(data)
        with self.assertRaisesRegex(assets.AssetError, "identity"):
            assets.validate_results(root)

    def test_node_report_complete_contract_and_strict_types(self):
        root = self.root / "results"
        self.results(root)
        for board in ("ppu10", "ppu15"):
            assets.write_json(root / "nodes.json", node_report(board))
            self.assertEqual(assets.validate_results(root)[0], {"ut": 2, "ops": 9})
        node = node_report()["collected"][0]
        mutations = [
            lambda r: r.update(schema=True),
            lambda r: r.update(board="cuda"),
            lambda r: r.update(file="tests/e2e/other.py"),
            lambda r: r.update(exitstatus=False),
            lambda r: r.update(errors={}),
            lambda r: r.update(status="skipped"),
            lambda r: r.update(nodes={}),
            lambda r: r.update(nodes=[]),
            lambda r: r.update(collected=list(reversed(r["collected"]))),
            lambda r: r["collected"].append(node),
            lambda r: r.update(applicable=r["applicable"][:-1]),
            lambda r: r.update(hardware_excluded=[node]),
            lambda r: r.update(collected=[1] * 9),
            lambda r: r["nodes"].pop(node),
            lambda r: r["nodes"].update(fake=r["nodes"][node]),
            lambda r: r["nodes"][node].pop("setup"),
            lambda r: r["nodes"][node].pop("teardown"),
            lambda r: r["nodes"][node].update(extra=r["nodes"][node]["call"]),
            lambda r: r["nodes"][node]["call"].update(outcome="skipped"),
            lambda r: r["nodes"][node]["setup"].update(outcome="failed"),
            lambda r: r["nodes"][node]["teardown"].update(outcome="failed"),
            lambda r: r["nodes"][node]["call"].update(wasxfail="expected"),
            lambda r: r["nodes"][node]["call"].pop("wasxfail"),
            lambda r: r["nodes"][node]["call"].update(reason="failed"),
            lambda r: r["nodes"][node]["call"].update(duration_seconds=True),
            lambda r: r["nodes"][node]["call"].update(duration_seconds=-1),
            lambda r: r["nodes"][node]["call"].update(duration_seconds="0.1"),
            lambda r: r.update(runtime={}),
            lambda r: r["runtime"]["paths"].pop("triton"),
            lambda r: r["runtime"].update(isolated_root="/source/isolated"),
            lambda r: r["runtime"].update(checkout_root="source"),
            lambda r: r["runtime"].update(isolated_root="/tmp/../isolated"),
            lambda r: r["runtime"]["paths"].update(torch="/source/torch/__init__.py"),
            lambda r: r["runtime"]["paths"].update(
                vllm="/tmp/sail-existing-tests/vllm/__init__.py"
            ),
            lambda r: r["runtime"]["paths"].update(
                vllm_sail="/opt/../source/vllm_sail.py"
            ),
            lambda r: r["runtime"]["paths"].update(
                plugin="/source/tests/support/ppu_ci.py"
            ),
            lambda r: r["runtime"]["paths"].update(
                conftest="/tmp/other/tests/conftest.py"
            ),
            lambda r: r["runtime"]["paths"].update(torch=1),
            lambda r: r["runtime"]["runtime_env"].update(VLLM_SAIL_USE_PLA=1),
        ]
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                report = node_report()
                mutation(report)
                (root / "nodes.json").write_bytes(assets.json_bytes(report))
                with self.assertRaises(assets.AssetError):
                    assets.validate_results(root)
        for duration in (float("inf"), float("nan"), 10**400):
            report = node_report()
            report["nodes"][node]["call"]["duration_seconds"] = duration
            with self.assertRaises(assets.AssetError):
                assets.validate_activation_report(report)

    def test_bundle_requires_results_and_rejects_candidate_extra_fields(self):
        bundle, candidate, _ = self.bundle(results=False)
        with self.assertRaisesRegex(assets.AssetError, "results"):
            assets.check_bundle(bundle, 500)
        self.results(bundle / "results")
        candidate["unexpected"] = "not part of reconstructable candidate"
        assets.write_json(bundle / "candidate.json", candidate)
        assets.write_json(bundle / "test-evidence.json", make_evidence(candidate))
        with self.assertRaisesRegex(assets.AssetError, "candidate fields"):
            assets.check_bundle(bundle, 500)
        with self.assertRaisesRegex(assets.AssetError, "candidate fields"):
            assets.attest_command(
                argparse.Namespace(
                    candidate=bundle / "candidate.json",
                    wheels=bundle / "wheelhouse",
                    results=bundle / "results",
                    test_run_id=500,
                    output=bundle / "test-evidence.json",
                )
            )
        self.assertEqual(self.api.mutations, [])

    def test_each_result_hash_must_match_evidence(self):
        bundle, _, _ = self.bundle()
        for filename in ("ut.xml", "ops.xml", "nodes.json"):
            path = bundle / "results" / filename
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            with (
                self.subTest(filename=filename),
                self.assertRaisesRegex(assets.AssetError, "evidence SHA256"),
            ):
                assets.check_bundle(bundle, 500)
            path.write_bytes(original)

    def test_candidate_digest_format_and_cli_output(self):
        bundle, _, _ = self.bundle()
        metadata = bundle / "build.json"
        dockerfile = bundle / "Dockerfile.input"
        dockerfile.write_bytes(b"FROM fixed@sha256:original\n")
        args = [
            "candidate",
            "--selection",
            str(bundle / "selection.json"),
            "--wheels",
            str(bundle / "wheelhouse"),
            "--build-metadata",
            str(metadata),
            "--image-repository",
            IMAGE_REPO,
            "--control-sha",
            CONTROL_SHA,
            "--dockerfile",
            str(dockerfile),
            "--output",
            str(bundle / "candidate.json"),
        ]
        for digest in (
            "sha256:abc",
            "sha256:" + "D" * 64,
            "sha512:" + "d" * 64,
            "sha256:" + "d" * 63,
            "sha256:" + "d" * 64 + "\n",
        ):
            metadata.write_text(json.dumps({"containerimage.digest": digest}))
            with redirect_stderr(io.StringIO()):
                self.assertEqual(assets.main(args), 1)
        metadata.write_text(json.dumps({"containerimage.digest": IMAGE_DIGEST}))
        output = bundle / "github-output"
        output.write_text("existing=keep\n")
        os.environ["GITHUB_OUTPUT"] = str(output)
        self.assertEqual(assets.main(args), 0)
        self.assertEqual(
            output.read_text(), f"existing=keep\nimage={IMAGE_REPO}@{IMAGE_DIGEST}\n"
        )
        value = assets.read_json(bundle / "candidate.json")
        self.assertEqual(value["dockerfile_sha256"], assets.sha256_file(dockerfile))

    def test_attest_and_evidence_tamper_checks(self):
        bundle, _, _ = self.bundle(results=True)
        args = [
            "attest",
            "--candidate",
            str(bundle / "candidate.json"),
            "--wheels",
            str(bundle / "wheelhouse"),
            "--results",
            str(bundle / "results"),
            "--test-run-id",
            "500",
            "--output",
            str(bundle / "test-evidence.json"),
        ]
        self.assertEqual(assets.main(args), 0)
        assets.check_bundle(bundle, 500)
        with self.assertRaisesRegex(assets.AssetError, "test run mismatch"):
            assets.check_bundle(bundle, 501)
        (bundle / "results" / "ut.xml").write_bytes(junit(2, summary_count=77))
        with self.assertRaisesRegex(assets.AssetError, "evidence SHA256"):
            assets.check_bundle(bundle, 500)
        value = assets.read_json(bundle / "candidate.json")
        value["image_digest"] = "sha256:" + "2" * 64
        value["image"] = f"{IMAGE_REPO}@{value['image_digest']}"
        assets.write_json(bundle / "candidate.json", value)
        with self.assertRaisesRegex(assets.AssetError, "candidate evidence"):
            assets.check_bundle(bundle, 500)

    def test_trusted_run_rejects_pr_wrong_repository_workflow_and_branch(self):
        original = copy.deepcopy(self.api.runs[(FORK, 500)])
        for changes in (
            {"event": "pull_request"},
            {"event": "push"},
            {"head_branch": "preview"},
            {"status": "in_progress"},
            {"conclusion": "failure"},
            {"head_repository": {"full_name": UPSTREAM}},
            {"repository": {"full_name": UPSTREAM}},
            {"path": ".github/workflows/another.yaml"},
            {"workflow_id": 99},
            {"name": "Nightly tests"},
            {"name": None},
        ):
            with self.subTest(changes=changes):
                self.api.runs[(FORK, 500)] = {**original, **changes}
                with self.assertRaises(assets.AssetError):
                    assets.trusted_run(self.api, FORK, "nightly-tests.yaml", 500)
        self.api.runs[(FORK, 500)] = original
        with (
            mock.patch.object(assets, "GitHub", return_value=self.api),
            redirect_stdout(io.StringIO()) as out,
        ):
            self.assertEqual(
                assets.main(
                    [
                        "resolve-test-run",
                        "--consumer-repository",
                        FORK,
                        "--workflow",
                        "nightly-tests.yaml",
                        "--run-id",
                        "500",
                    ]
                ),
                0,
            )
        self.assertEqual(
            out.getvalue(),
            f"test_run_id=500\ncontrol_sha={CONTROL_SHA}\n"
            "ready=false\nbundle_name=nightly-tested-bundle-500-1\n",
        )

    def test_same_repository_fetch_publish_and_read_preserve_source_identity(self):
        bundle, candidate, evidence = self.bundle()
        self.api.runs[(UPSTREAM, 500)] = run_record(
            UPSTREAM, 500, "nightly-tests.yaml", sha=CONTROL_SHA
        )
        wheelhouse = bundle / "wheelhouse"
        archive = self.root / "source-wheels.zip"
        assets.pack_wheels(
            wheelhouse, archive, assets.validate_wheels(wheelhouse), candidate["source"]
        )
        self.api.artifact_zip = archive.read_bytes()
        args = self.nightly_args(consumer_repository=UPSTREAM, include_published=True)
        with redirect_stdout(io.StringIO()):
            assets.fetch_nightly(self.api, args)
        self.assertEqual(
            assets.read_json(args.output / "selection.json"), candidate["source"]
        )
        with redirect_stdout(io.StringIO()):
            assets.publish(
                self.api,
                argparse.Namespace(
                    consumer_repository=UPSTREAM, bundle=bundle, test_run_id=500
                ),
            )
            fetched = self.root / "preview-read"
            assets.fetch_preview(
                self.api,
                argparse.Namespace(consumer_repository=UPSTREAM, output=fetched),
            )
        baseline = assets.read_json(fetched / "baseline.json")
        self.assertEqual(baseline, make_baseline(candidate, evidence))
        self.assertEqual(baseline["source"]["upstream_sha"], UP_SHA)
        self.assertEqual(baseline["control_sha"], CONTROL_SHA)
        self.assertEqual(
            assets.validate_wheels(fetched / "wheelhouse", baseline["source"]),
            assets.validate_wheels(wheelhouse),
        )
        self.assertEqual({repo for repo, _ in self.api.releases}, {UPSTREAM})

    def test_minimum_regression_workflow_display_name_is_required(self):
        bundle, _, _ = self.bundle()
        get = self.api.get
        argv = [
            "resolve-test-run",
            "--consumer-repository",
            FORK,
            "--workflow",
            "nightly-tests.yaml",
            "--run-id",
            "500",
        ]
        for invalid in (None, "Nightly tests", "Nightly regression (Full)"):

            def wrong_definition(path, invalid=invalid):
                value = get(path)
                if path.endswith("/actions/workflows/nightly-tests.yaml"):
                    value["name"] = invalid
                return value

            with (
                self.subTest(invalid=invalid),
                mock.patch.object(self.api, "get", side_effect=wrong_definition),
            ):
                with self.assertRaisesRegex(assets.AssetError, "workflow name"):
                    assets.promotion_check(self.api, FORK, bundle, 500)
                with (
                    mock.patch.object(assets, "GitHub", return_value=self.api),
                    redirect_stderr(io.StringIO()),
                ):
                    self.assertEqual(assets.main(argv), 1)
        self.assertEqual(self.api.mutations, [])

    def test_resolve_test_run_current_attempt_four_outputs(self):
        self.api.runs[(FORK, 500)]["run_attempt"] = 3
        argv = [
            "resolve-test-run",
            "--consumer-repository",
            FORK,
            "--workflow",
            "nightly-tests.yaml",
            "--run-id",
            "500",
        ]
        for records, ready in (
            ([], False),
            (
                [{"id": 1, "name": "nightly-tested-bundle-500-2", "expired": False}],
                False,
            ),
            (
                [{"id": 2, "name": "nightly-tested-bundle-500-3", "expired": True}],
                False,
            ),
            (
                [{"id": 3, "name": "nightly-tested-bundle-500-3", "expired": False}],
                True,
            ),
        ):
            with self.subTest(records=records):
                self.api.artifacts = records
                with (
                    mock.patch.object(assets, "GitHub", return_value=self.api),
                    redirect_stdout(io.StringIO()) as out,
                ):
                    self.assertEqual(assets.main(argv), 0)
                self.assertEqual(
                    out.getvalue(),
                    f"test_run_id=500\ncontrol_sha={CONTROL_SHA}\n"
                    f"ready={str(ready).lower()}\nbundle_name=nightly-tested-bundle-500-3\n",
                )

    def test_manual_nightly_strict_validation_and_no_stale_fallback(self):
        for changes in (
            {"event": "pull_request"},
            {"head_branch": "pr"},
            {"conclusion": "failure"},
            {"head_repository": {"full_name": FORK}},
            {
                "created_at": (
                    datetime.now(timezone.utc) - timedelta(days=4)
                ).isoformat()
            },
        ):
            original = self.api.runs[(UPSTREAM, 100)]
            self.api.runs[(UPSTREAM, 100)] = {**original, **changes}
            with self.assertRaises(assets.AssetError):
                assets.fetch_nightly(self.api, self.nightly_args(run_id=100))
            self.api.runs[(UPSTREAM, 100)] = original
        self.api.artifacts[0]["expired"] = True
        with self.assertRaises(assets.AssetError):
            assets.fetch_nightly(self.api, self.nightly_args(run_id=100))
        with redirect_stdout(io.StringIO()) as out:
            assets.fetch_nightly(self.api, self.nightly_args())
        self.assertEqual(out.getvalue(), "available=false\n")
        self.assertFalse((self.root / "flow").exists())

    def test_expiration_race_is_unavailable_automatic_but_error_manual(self):
        for status in (404, 410):
            with mock.patch.object(
                self.api, "download", side_effect=assets.APIError(status)
            ):
                with redirect_stdout(io.StringIO()) as out:
                    assets.fetch_nightly(self.api, self.nightly_args())
                self.assertEqual(out.getvalue(), "available=false\n")
                with self.assertRaises(assets.APIError):
                    assets.fetch_nightly(self.api, self.nightly_args(run_id=100))

    def test_fetch_nightly_selects_latest_trusted_run_and_preserves_bytes(self):
        bundle, candidate, _ = self.bundle()
        archive = bundle / "archive.zip"
        manifest = assets.validate_wheels(bundle / "wheelhouse")
        assets.pack_wheels(
            bundle / "wheelhouse", archive, manifest, candidate["source"]
        )
        self.api.artifact_zip = archive.read_bytes()
        pr = run_record(UPSTREAM, 999, "ppu-accuracy-nightly.yaml")
        pr["event"] = "pull_request"
        self.api.runs[(UPSTREAM, 999)] = pr
        with redirect_stdout(io.StringIO()) as out:
            assets.fetch_nightly(self.api, self.nightly_args())
        self.assertEqual(
            out.getvalue(),
            f"available=true\nupstream_sha={UP_SHA}\nupstream_run_id=100\n",
        )
        flow = self.root / "flow"
        selection = assets.read_json(flow / "selection.json")
        self.assertEqual(selection, candidate["source"])
        for record in manifest["wheels"].values():
            self.assertEqual(
                (flow / "wheelhouse" / record["file"]).read_bytes(),
                (bundle / "wheelhouse" / record["file"]).read_bytes(),
            )

    def test_latest_promoted_or_stale_success_does_not_choose_older_artifact(self):
        _, candidate, evidence = self.bundle()
        self.api.install_baseline(make_baseline(candidate, evidence))
        with redirect_stdout(io.StringIO()) as out:
            assets.fetch_nightly(self.api, self.nightly_args())
        self.assertEqual(out.getvalue(), "available=false\n")
        self.api.releases.clear()
        self.api.runs[(UPSTREAM, 100)]["created_at"] = (
            datetime.now(timezone.utc) - timedelta(hours=49)
        ).isoformat()
        with redirect_stdout(io.StringIO()) as out:
            assets.fetch_nightly(self.api, self.nightly_args())
        self.assertEqual(out.getvalue(), "available=false\n")

    def test_include_published_retests_same_run_without_moving_fixed_preview(self):
        baseline, _ = self.previous_release(100)
        before = copy.deepcopy(
            (
                self.api.releases,
                self.api.tags,
                self.api.asset_records,
                self.api.payloads,
            )
        )
        archive = self.root / "published-upstream.zip"
        wheelhouse = self.root / "bundle-100" / "wheelhouse"
        assets.pack_wheels(
            wheelhouse, archive, assets.validate_wheels(wheelhouse), baseline["source"]
        )
        self.api.artifact_zip = archive.read_bytes()
        argv = [
            "fetch-nightly",
            "--source-repository",
            UPSTREAM,
            "--consumer-repository",
            FORK,
            "--workflow",
            "ppu-accuracy-nightly.yaml",
            "--output",
            str(self.root / "flow"),
        ]
        self.assertIs(assets.parser().parse_args(argv).include_published, False)
        self.assertIs(
            assets.parser()
            .parse_args(argv + ["--include-published"])
            .include_published,
            True,
        )
        with (
            mock.patch.object(assets, "GitHub", return_value=self.api),
            redirect_stdout(io.StringIO()) as out,
        ):
            self.assertEqual(assets.main(argv), 0)
        self.assertEqual(out.getvalue(), "available=false\n")
        with (
            mock.patch.object(assets, "GitHub", return_value=self.api),
            redirect_stdout(io.StringIO()) as out,
        ):
            self.assertEqual(assets.main(argv + ["--include-published"]), 0)
        self.assertEqual(
            out.getvalue(),
            f"available=true\nupstream_sha={UP_SHA}\nupstream_run_id=100\n",
        )
        self.assertEqual(
            assets.read_json(self.root / "flow" / "selection.json"), baseline["source"]
        )
        self.assertEqual(
            before,
            (
                self.api.releases,
                self.api.tags,
                self.api.asset_records,
                self.api.payloads,
            ),
        )
        self.assertEqual(self.api.mutations, [])
        # A distinct successful test run is useful regression evidence, not a new pointer.
        bundle = self.root / "bundle-100"
        candidate = assets.read_json(bundle / "candidate.json")
        evidence = assets.read_json(bundle / "test-evidence.json")
        evidence["test_run_id"] = 501
        assets.write_json(bundle / "test-evidence.json", evidence)
        self.api.runs[(FORK, 501)] = run_record(FORK, 501, "nightly-tests.yaml")
        for command in ("check-preview", "publish-preview"):
            with (
                mock.patch.object(assets, "GitHub", return_value=self.api),
                redirect_stdout(io.StringIO()) as out,
            ):
                self.assertEqual(
                    assets.main(
                        [
                            command,
                            "--consumer-repository",
                            FORK,
                            "--bundle",
                            str(bundle),
                            "--test-run-id",
                            "501",
                        ]
                    ),
                    0,
                )
            self.assertEqual(out.getvalue(), "promote=false\n")
        self.assertEqual(
            before,
            (
                self.api.releases,
                self.api.tags,
                self.api.asset_records,
                self.api.payloads,
            ),
        )
        self.assertEqual(self.api.mutations, [])
        self.assertNotIn((FORK, "preview-nightly-100-test-501"), self.api.releases)
        self.assertEqual(assets.validate_baseline(baseline), candidate)

    def test_include_published_keeps_trust_expiration_and_no_stale_fallback(self):
        self.previous_release(100)
        original = copy.deepcopy(self.api.runs[(UPSTREAM, 100)])
        for changes in (
            {"event": "pull_request"},
            {"head_branch": "pr"},
            {"conclusion": "failure"},
            {"status": "in_progress"},
            {"repository": {"full_name": FORK}},
            {"head_repository": {"full_name": FORK}},
            {"workflow_id": 99},
            {"path": ".github/workflows/another.yaml"},
            {
                "created_at": (
                    datetime.now(timezone.utc) - timedelta(hours=49)
                ).isoformat()
            },
        ):
            with self.subTest(changes=changes):
                self.api.runs[(UPSTREAM, 100)] = {**original, **changes}
                with self.assertRaises(assets.AssetError):
                    assets.fetch_nightly(
                        self.api, self.nightly_args(run_id=100, include_published=True)
                    )
                with redirect_stdout(io.StringIO()) as out:
                    assets.fetch_nightly(
                        self.api, self.nightly_args(include_published=True)
                    )
                self.assertEqual(out.getvalue(), "available=false\n")
        self.api.runs[(UPSTREAM, 100)] = original
        for changes in ({"expired": True}, {"expires_at": NOW}):
            artifact = copy.deepcopy(self.api.artifacts[0])
            self.api.artifacts[0].update(changes)
            with self.subTest(changes=changes), redirect_stdout(io.StringIO()) as out:
                assets.fetch_nightly(
                    self.api, self.nightly_args(include_published=True)
                )
            self.assertEqual(out.getvalue(), "available=false\n")
            with self.assertRaises(assets.AssetError):
                assets.fetch_nightly(
                    self.api, self.nightly_args(run_id=100, include_published=True)
                )
            self.api.artifacts[0] = artifact
        with self.assertRaisesRegex(assets.AssetError, "default branch"):
            assets.fetch_nightly(
                self.api, self.nightly_args(branch="other", include_published=True)
            )
        # Select the latest success, even when the previous published run has an artifact.
        self.api.runs[(UPSTREAM, 101)] = run_record(
            UPSTREAM, 101, "ppu-accuracy-nightly.yaml"
        )
        with redirect_stdout(io.StringIO()) as out:
            assets.fetch_nightly(self.api, self.nightly_args(include_published=True))
        self.assertEqual(out.getvalue(), "available=false\n")
        self.assertIn(f"/repos/{UPSTREAM}/actions/runs/101/artifacts", self.api.reads)
        self.assertFalse((self.root / "flow").exists())
        self.assertEqual(self.api.mutations, [])

    def test_include_published_48_hour_cycle_boundary_and_expiration_race(self):
        bundle, candidate, _ = self.bundle()
        archive = bundle / "archive.zip"
        assets.pack_wheels(
            bundle / "wheelhouse",
            archive,
            assets.validate_wheels(bundle / "wheelhouse"),
            candidate["source"],
        )
        self.api.artifact_zip = archive.read_bytes()
        now = datetime.now(timezone.utc)
        for age, available in (
            (timedelta(hours=48), True),
            (timedelta(hours=48, microseconds=1), False),
        ):
            with self.subTest(age=age):
                self.api.runs[(UPSTREAM, 100)]["created_at"] = (now - age).isoformat()
                with (
                    mock.patch.object(assets, "datetime", wraps=datetime) as clock,
                    redirect_stdout(io.StringIO()) as out,
                ):
                    clock.now.return_value = now
                    assets.fetch_nightly(
                        self.api,
                        self.nightly_args(
                            include_published=True,
                            output=self.root / f"cycle-{available}",
                        ),
                    )
                self.assertEqual(
                    out.getvalue().startswith("available=true\n"), available
                )
        self.api.runs[(UPSTREAM, 100)]["created_at"] = NOW
        for status in (404, 410):
            with mock.patch.object(
                self.api, "download", side_effect=assets.APIError(status)
            ):
                with redirect_stdout(io.StringIO()) as out:
                    assets.fetch_nightly(
                        self.api, self.nightly_args(include_published=True)
                    )
                self.assertEqual(out.getvalue(), "available=false\n")
                with self.assertRaises(assets.APIError):
                    assets.fetch_nightly(
                        self.api, self.nightly_args(run_id=100, include_published=True)
                    )
        self.assertEqual(self.api.mutations, [])

    def test_fetch_preview_published_baseline_only_and_hashes(self):
        args = argparse.Namespace(consumer_repository=FORK, output=self.root / "flow")
        with self.assertRaisesRegex(assets.AssetError, "no source-build fallback"):
            assets.fetch_preview(self.api, args)
        bundle, candidate, evidence = self.bundle()
        archive = bundle / "archive.zip"
        assets.pack_wheels(
            bundle / "wheelhouse",
            archive,
            assets.validate_wheels(bundle / "wheelhouse"),
            candidate["source"],
        )
        baseline = make_baseline(candidate, evidence)
        self.api.install_baseline(baseline, archive.read_bytes())
        with redirect_stdout(io.StringIO()) as out:
            assets.fetch_preview(self.api, args)
        self.assertIn(f"base_image={candidate['image']}\n", out.getvalue())
        self.assertEqual(assets.read_json(args.output / "baseline.json"), baseline)
        self.assertEqual(
            assets.read_json(args.output / "selection.json"), candidate["source"]
        )
        self.api.releases[(FORK, "preview-baseline")]["draft"] = True
        self.api.reads.clear()
        self.assertEqual(assets.load_baseline(self.api, FORK), (None, None, []))
        with self.assertRaisesRegex(assets.AssetError, "no source-build fallback"):
            assets.fetch_preview(self.api, args)
        self.assertEqual(
            self.api.reads, [f"/repos/{FORK}/releases/tags/preview-baseline"] * 2
        )
        # Even a broken tag endpoint must not make a draft a readable baseline.
        draft = copy.deepcopy(self.api.releases[(FORK, "preview-baseline")])
        with mock.patch.object(self.api, "optional", return_value=draft):
            with self.assertRaisesRegex(assets.AssetError, "not published"):
                assets.load_baseline(self.api, FORK)
        self.assertEqual(self.api.mutations, [])

    def test_baseline_rejects_image_source_and_evidence_mismatch(self):
        _, candidate, evidence = self.bundle()
        baseline = make_baseline(candidate, evidence)
        for mutate in (
            lambda b: b.update(image=f"{IMAGE_REPO}:latest"),
            lambda b: b["source"].update(upstream_sha="1" * 40),
            lambda b: b["evidence"].update(candidate_sha256="1" * 64),
            lambda b: b["source"].update(artifact_name="ppu-wheels-99"),
            lambda b: b["evidence"].update(counts={"ut": 2, "ops": 0}),
        ):
            bad = copy.deepcopy(baseline)
            mutate(bad)
            with self.assertRaises(assets.AssetError):
                assets.validate_baseline(bad)

    def test_preview_labels_round_trip_without_changing_candidate_hash(self):
        bundle, candidate, evidence = self.bundle()
        candidate_bytes = (bundle / "candidate.json").read_bytes()
        baseline = assets.make_publication_baseline(candidate, evidence)
        self.assertEqual(baseline, make_baseline(candidate, evidence))
        self.assertEqual(baseline["quality_scope"], "minimum-regression")
        self.assertEqual(baseline["channel"], "preview")
        round_trip = assets.decode_json(assets.json_bytes(baseline))
        restored = assets.validate_baseline(round_trip)
        self.assertEqual(restored, candidate)
        self.assertEqual(assets.json_bytes(restored), candidate_bytes)
        self.assertEqual(
            assets.sha256_bytes(assets.json_bytes(restored)),
            evidence["candidate_sha256"],
        )
        self.assertEqual((bundle / "candidate.json").read_bytes(), candidate_bytes)
        for field, invalid in (
            ("quality_scope", "Full"),
            ("quality_scope", "verified"),
            ("quality_scope", None),
            ("channel", "stable"),
            ("channel", None),
            ("channel", True),
        ):
            with self.subTest(field=field, invalid=invalid):
                bad = copy.deepcopy(baseline)
                bad[field] = invalid
                with self.assertRaisesRegex(
                    assets.AssetError, "quality scope or channel"
                ):
                    assets.validate_baseline(bad)
        for field in ("quality_scope", "channel"):
            bad = copy.deepcopy(baseline)
            bad.pop(field)
            with self.assertRaisesRegex(assets.AssetError, "quality scope or channel"):
                assets.validate_baseline(bad)
            bad_candidate = {**candidate, field: baseline[field]}
            with self.assertRaisesRegex(assets.AssetError, "candidate fields"):
                assets.validate_candidate(bad_candidate)
        bad = copy.deepcopy(baseline)
        bad["immutable_release_tag"] = "ci-nightly-100-test-500"
        with self.assertRaisesRegex(assets.AssetError, "immutable release identity"):
            assets.validate_baseline(bad)

    def test_preview_releases_require_prerelease_true_and_explicit_preview_title(self):
        bundle, candidate, evidence = self.bundle()
        baseline = make_baseline(candidate, evidence)
        self.api.install_baseline(baseline)
        fixed = self.api.releases[(FORK, "preview-baseline")]
        original = copy.deepcopy(fixed)
        for invalid in (None, False, 1, "true"):
            with self.subTest(prerelease=invalid):
                fixed.clear()
                fixed.update(original)
                if invalid is None:
                    fixed.pop("prerelease")
                else:
                    fixed["prerelease"] = invalid
                with self.assertRaisesRegex(assets.AssetError, "must be a prerelease"):
                    assets.load_baseline(self.api, FORK)
                with self.assertRaisesRegex(assets.AssetError, "must be a prerelease"):
                    assets.ensure_release(
                        self.api, FORK, "preview-baseline", CONTROL_SHA, immutable=False
                    )
                with self.assertRaisesRegex(assets.AssetError, "must be a prerelease"):
                    assets.publish_release(self.api, FORK, fixed)
                with self.assertRaisesRegex(assets.AssetError, "must be a prerelease"):
                    assets.promotion_check(self.api, FORK, bundle, 500, publisher=True)
                self.assertEqual(self.api.mutations, [])
        for title in (None, "preview baseline", "Stable verified baseline"):
            fixed.clear()
            fixed.update(original)
            fixed["name"] = title
            with (
                self.subTest(title=title),
                self.assertRaisesRegex(assets.AssetError, "title"),
            ):
                assets.load_baseline(self.api, FORK)
        fixed.clear()
        fixed.update(original)
        # Refuse legacy or stable tags before even reading them, let alone creating them.
        self.api.reads.clear()
        for tag, immutable in (
            ("ci-baseline", False),
            ("ci-nightly-100-test-500", True),
        ):
            with self.assertRaisesRegex(assets.AssetError, "only preview"):
                assets.ensure_release(
                    self.api, FORK, tag, CONTROL_SHA, immutable=immutable
                )
        self.assertEqual(self.api.reads, [])
        self.assertEqual(self.api.mutations, [])

    def test_legacy_ci_baseline_preserved_across_preview_cli_operations_and_recovery(
        self,
    ):
        legacy = self.api.install_legacy_release()
        legacy_unique = self.api.install_legacy_release("ci-nightly-99999-test-9000")
        legacy_ids = {legacy["id"], legacy_unique["id"]}
        legacy_assets = {
            key: copy.deepcopy(record) for key, record in self.api.asset_records.items()
        }
        legacy_payloads = copy.deepcopy(self.api.payloads)
        legacy_tags = copy.deepcopy(self.api.tags)
        self.api.reads.clear()
        fetch = [
            "fetch-preview",
            "--consumer-repository",
            FORK,
            "--output",
            str(self.root / "preview-read"),
        ]
        with (
            mock.patch.object(assets, "GitHub", return_value=self.api),
            redirect_stderr(io.StringIO()) as err,
        ):
            self.assertEqual(assets.main(fetch), 1)
        self.assertIn("published preview-baseline is required", err.getvalue())
        bundle, candidate, evidence = self.bundle()
        archive = bundle / "upstream.zip"
        assets.pack_wheels(
            bundle / "wheelhouse",
            archive,
            assets.validate_wheels(bundle / "wheelhouse"),
            candidate["source"],
        )
        self.api.artifact_zip = archive.read_bytes()
        with redirect_stdout(io.StringIO()) as out:
            assets.fetch_nightly(self.api, self.nightly_args())
        self.assertTrue(out.getvalue().startswith("available=true\n"))
        argv = [
            "--consumer-repository",
            FORK,
            "--bundle",
            str(bundle),
            "--test-run-id",
            "500",
        ]
        with (
            mock.patch.object(assets, "GitHub", return_value=self.api),
            redirect_stdout(io.StringIO()) as out,
        ):
            self.assertEqual(assets.main(["check-preview", *argv]), 0)
        self.assertTrue(out.getvalue().startswith("promote=true\n"))
        upload = self.api.upload

        def interrupt_fixed(repo, release_id, name, path, content_type):
            fixed = self.api.releases.get((FORK, "preview-baseline"))
            if fixed and fixed["id"] == release_id:
                raise assets.APIError(503)
            return upload(repo, release_id, name, path, content_type)

        with (
            mock.patch.object(assets, "GitHub", return_value=self.api),
            mock.patch.object(self.api, "upload", side_effect=interrupt_fixed),
            redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(assets.main(["publish-preview", *argv]), 1)
        self.assertTrue(self.api.releases[(FORK, "preview-baseline")]["draft"])
        # The legacy unique run is newer, but is not a preview recovery fence.
        with (
            mock.patch.object(assets, "GitHub", return_value=self.api),
            redirect_stdout(io.StringIO()) as out,
        ):
            self.assertEqual(assets.main(["publish-preview", *argv]), 0)
            self.assertEqual(assets.main(fetch), 0)
        self.assertEqual(
            assets.read_json(self.root / "preview-read" / "baseline.json"),
            make_baseline(candidate, evidence),
        )
        self.assertTrue((self.root / "preview-read" / "preview-wheels.zip").is_file())
        self.assertEqual(self.api.releases[(FORK, "ci-baseline")], legacy)
        self.assertEqual(
            self.api.releases[(FORK, "ci-nightly-99999-test-9000")], legacy_unique
        )
        self.assertEqual({key: self.api.tags[key] for key in legacy_tags}, legacy_tags)
        self.assertEqual(
            {key: self.api.asset_records[key] for key in legacy_assets}, legacy_assets
        )
        self.assertEqual(
            {key: self.api.payloads[key] for key in legacy_payloads}, legacy_payloads
        )
        for path in self.api.reads:
            self.assertNotIn("/tags/ci-", path)
            for release_id in legacy_ids:
                self.assertNotEqual(path, f"/repos/{FORK}/releases/{release_id}/assets")
            for asset_id in legacy_payloads:
                self.assertNotEqual(path, f"/repos/{FORK}/releases/assets/{asset_id}")
        for _, path, _ in self.api.mutations:
            for release_id in legacy_ids:
                self.assertNotEqual(path, f"/repos/{FORK}/releases/{release_id}")
                self.assertFalse(path.startswith(f"uploads/{release_id}/"))
            for asset_id in legacy_payloads:
                self.assertNotEqual(path, f"/repos/{FORK}/releases/assets/{asset_id}")

    def test_promotion_never_downgrades_or_duplicates(self):
        bundle, candidate, evidence = self.bundle()
        self.assertTrue(assets.promotion_check(self.api, FORK, bundle, 500)[0])
        for current_id in (100, 101, 999):
            self.api = FakeAPI()
            current = copy.deepcopy(candidate)
            current["source"]["upstream_run_id"] = current_id
            current["source"]["artifact_name"] = f"ppu-wheels-{current_id}"
            self.api.install_baseline(make_baseline(current, make_evidence(current)))
            self.assertFalse(assets.promotion_check(self.api, FORK, bundle, 500)[0])
            self.assertFalse(self.api.mutations)
        self.api = FakeAPI()
        previous = copy.deepcopy(candidate)
        previous["source"]["upstream_run_id"] = 99
        previous["source"]["artifact_name"] = "ppu-wheels-99"
        self.api.install_baseline(make_baseline(previous, make_evidence(previous)))
        self.assertTrue(assets.promotion_check(self.api, FORK, bundle, 500)[0])
        candidate["control_sha"] = "1" * 40
        assets.write_json(bundle / "candidate.json", candidate)
        changed_evidence = assets.read_json(bundle / "test-evidence.json")
        changed_evidence["candidate_sha256"] = assets.sha256_file(
            bundle / "candidate.json"
        )
        assets.write_json(bundle / "test-evidence.json", changed_evidence)
        with self.assertRaisesRegex(assets.AssetError, "control SHA"):
            assets.promotion_check(self.api, FORK, bundle, 500)

    def test_publish_orders_immutable_then_fixed_wheels_then_marker_no_delete(self):
        bundle, candidate, evidence = self.bundle(results=True)
        self.previous_release(99)
        args = argparse.Namespace(
            consumer_repository=FORK, bundle=bundle, test_run_id=500
        )
        summary = bundle / "summary.txt"
        os.environ["GITHUB_STEP_SUMMARY"] = str(summary)
        with redirect_stdout(io.StringIO()):
            assets.publish(self.api, args)
        baseline, fixed, records = assets.load_baseline(self.api, FORK)
        self.assertEqual(baseline, make_baseline(candidate, evidence))
        archived = [a["name"] for a in records if a["name"].startswith("archive-")]
        self.assertEqual(len(archived), 2)
        writes = self.api.mutations
        self.assertTrue(all(method in {"POST", "PATCH"} for method, _, _ in writes))
        fixed_uploads = [
            path
            for method, path, _ in writes
            if path.startswith(f"uploads/{fixed['id']}/")
        ]
        self.assertEqual(
            fixed_uploads,
            [
                f"uploads/{fixed['id']}/preview-wheels.zip",
                f"uploads/{fixed['id']}/preview-baseline.json",
            ],
        )
        unique = self.api.releases[(FORK, baseline["immutable_release_tag"])]
        publish_index = next(
            i
            for i, (_, path, body) in enumerate(writes)
            if path == f"/repos/{FORK}/releases/{unique['id']}"
            and body.get("draft") is False
        )
        first_fixed = next(
            i
            for i, (_, path, _) in enumerate(writes)
            if path.startswith(f"uploads/{fixed['id']}/")
        )
        self.assertLess(publish_index, first_fixed)
        self.assertEqual(unique["target_commitish"], CONTROL_SHA)
        for release in (fixed, unique):
            self.assertIs(release["prerelease"], True)
            self.assertTrue(release["name"].startswith("Preview "))
        for _method, path, body in writes:
            if path == f"/repos/{FORK}/releases" or body.get("draft") is False:
                self.assertIs(body["prerelease"], True)
                self.assertTrue(body["name"].startswith("Preview "))
        summary_text = summary.read_text(encoding="ascii")
        for phrase in (
            "not verified",
            "not Full",
            "not a production Release",
            "minimum-regression",
            "upstream pin unchanged",
            "non-atomic",
            "retained",
            "48-hour",
            "no-op",
            "Rollback",
        ):
            self.assertIn(phrase, summary_text)
        self.assertNotIn("ci-baseline", summary_text)
        self.assertNotIn("unit-test-secret", summary_text)
        self.api.mutations.clear()
        with redirect_stdout(io.StringIO()) as out:
            assets.publish(self.api, args)
        self.assertEqual(out.getvalue(), "promote=false\n")
        self.assertEqual(self.api.mutations, [])
        with zipfile.ZipFile(bundle / "preview-wheels.zip") as archive:
            for record in assets.validate_wheels(bundle / "wheelhouse")[
                "wheels"
            ].values():
                self.assertEqual(
                    archive.read(record["file"]),
                    (bundle / "wheelhouse" / record["file"]).read_bytes(),
                )

    def test_fake_tag_lookup_hides_drafts_and_duplicate_release_post_is_422(self):
        tag = "preview-nightly-100-test-500"
        path = f"/repos/{FORK}/releases"
        body = {
            "tag_name": tag,
            "target_commitish": CONTROL_SHA,
            "draft": True,
            "prerelease": True,
            "name": f"Preview {tag} (minimum regression)",
        }
        release = self.api.mutate("POST", path, body)
        for draft in (True, False):
            with self.subTest(draft=draft):
                if not draft:
                    release = assets.publish_release(self.api, FORK, release)
                if draft:
                    with self.assertRaises(assets.APIError) as error:
                        self.api.get(f"{path}/tags/{tag}")
                    self.assertEqual(error.exception.status, 404)
                    self.assertIsNone(self.api.optional(f"{path}/tags/{tag}"))
                else:
                    self.assertEqual(self.api.get(f"{path}/tags/{tag}"), release)
                self.assertEqual(list(self.api.pages(path)), [release])
                before = copy.deepcopy(
                    (self.api.releases, self.api.tags, self.api.next_release)
                )
                with self.assertRaises(assets.APIError) as error:
                    self.api.mutate("POST", path, body)
                self.assertEqual(error.exception.status, 422)
                self.assertEqual(
                    before, (self.api.releases, self.api.tags, self.api.next_release)
                )

    def test_publisher_lookup_validates_full_identity_and_immutable_target(self):
        tag = "preview-nightly-100-test-500"
        release = assets.ensure_release(
            self.api, FORK, tag, CONTROL_SHA, immutable=True
        )
        self.api.mutations.clear()
        invalid = [
            ("id", None),
            ("id", True),
            ("id", 0),
            ("id", release["id"] + 1),
            ("url", None),
            ("url", f"{assets.API}/repos/{UPSTREAM}/releases/{release['id']}"),
            ("url", release["url"] + "?other=repo"),
            ("target_commitish", None),
            ("target_commitish", "main"),
            ("target_commitish", "c" * 39),
            ("draft", None),
            ("draft", 1),
        ]
        for source in ("tag", "list"):
            for field, value in invalid:
                bad = {**release, field: value}
                with (
                    self.subTest(source=source, field=field, value=value),
                    mock.patch.object(
                        self.api,
                        "optional",
                        return_value=bad if source == "tag" else None,
                    ),
                    mock.patch.object(self.api, "pages", return_value=iter([bad])),
                ):
                    with self.assertRaises(assets.AssetError):
                        assets.publisher_lookup_release(self.api, FORK, tag)
                    self.assertEqual(self.api.mutations, [])
        with mock.patch.object(
            self.api,
            "optional",
            return_value={**release, "tag_name": "preview-baseline"},
        ):
            with self.assertRaisesRegex(assets.AssetError, "release tag mismatch"):
                assets.publisher_lookup_release(self.api, FORK, tag)
        # A full target SHA must also equal the tested commit for immutable reuse.
        self.api.releases[(FORK, tag)]["target_commitish"] = "1" * 40
        with self.assertRaisesRegex(
            assets.AssetError, "immutable release control SHA mismatch"
        ):
            assets.ensure_release(self.api, FORK, tag, CONTROL_SHA, immutable=True)
        self.assertEqual(self.api.mutations, [])

    def test_publish_retries_upload_response_loss_on_fixed_and_immutable_drafts(self):
        bundle, candidate, evidence = self.bundle()
        args = argparse.Namespace(
            consumer_repository=FORK, bundle=bundle, test_run_id=500
        )
        for tag in ("preview-baseline", "preview-nightly-100-test-500"):
            for name in ("preview-wheels.zip", "preview-baseline.json"):
                with self.subTest(tag=tag, name=name):
                    self.api = FakeAPI()
                    upload = self.api.upload
                    uploaded_ids = []

                    def lose_response(
                        repo,
                        release_id,
                        asset_name,
                        path,
                        content_type,
                        upload=upload,
                        tag=tag,
                        name=name,
                        uploaded_ids=uploaded_ids,
                    ):
                        record = upload(
                            repo, release_id, asset_name, path, content_type
                        )
                        target = self.api.releases.get((FORK, tag))
                        if target and target["id"] == release_id and asset_name == name:
                            uploaded_ids.append(record["id"])
                            raise assets.APIError(503)
                        return record

                    with mock.patch.object(
                        self.api, "upload", side_effect=lose_response
                    ):
                        with self.assertRaises(assets.APIError):
                            assets.publish(self.api, args)
                    self.assertEqual(len(uploaded_ids), 1)
                    release = copy.deepcopy(self.api.releases[(FORK, tag)])
                    self.assertTrue(release["draft"])
                    self.assertIsNone(
                        self.api.optional(f"/repos/{FORK}/releases/tags/{tag}")
                    )
                    payload = self.api.payloads[uploaded_ids[0]]
                    self.api.mutations.clear()
                    with redirect_stdout(io.StringIO()):
                        assets.publish(self.api, args)
                    self.assertEqual(
                        assets.load_baseline(self.api, FORK)[0],
                        make_baseline(candidate, evidence),
                    )
                    self.assertEqual(
                        self.api.releases[(FORK, tag)]["id"], release["id"]
                    )
                    self.assertEqual(self.api.payloads[uploaded_ids[0]], payload)
                    reused = assets.named_asset(
                        assets.release_assets(self.api, FORK, release), name
                    )
                    self.assertEqual(reused["id"], uploaded_ids[0])
                    self.assertFalse(
                        any(
                            method == "POST" and body.get("tag_name") == tag
                            for method, _, body in self.api.mutations
                        )
                    )
                    self.assertFalse(
                        any(
                            path == f"uploads/{release['id']}/{name}"
                            for _, path, _ in self.api.mutations
                        )
                    )
                    self.assertTrue(
                        all(
                            method in {"POST", "PATCH"}
                            for method, _, _ in self.api.mutations
                        )
                    )

    def test_publish_rejects_published_immutable_conflicts_without_mutation(self):
        bundle, candidate, evidence = self.bundle()
        baseline = make_baseline(candidate, evidence)
        archive = bundle / "preview-wheels.zip"
        assets.pack_wheels(
            bundle / "wheelhouse",
            archive,
            assets.validate_wheels(bundle / "wheelhouse"),
            candidate["source"],
        )
        args = argparse.Namespace(
            consumer_repository=FORK, bundle=bundle, test_run_id=500
        )
        for failure in ("wheels", "marker", "missing-asset", "target"):
            with self.subTest(failure=failure):
                self.api = FakeAPI()
                tag = baseline["immutable_release_tag"]
                release = self.api.install_release(tag, baseline, archive.read_bytes())
                records = assets.release_assets(self.api, FORK, release)
                if failure == "target":
                    self.api.releases[(FORK, tag)]["target_commitish"] = "1" * 40
                else:
                    name = (
                        "preview-baseline.json"
                        if failure == "marker"
                        else "preview-wheels.zip"
                    )
                    record = assets.named_asset(records, name)
                    if failure == "missing-asset":
                        self.api.asset_records.pop((FORK, release["id"], record["id"]))
                    else:
                        self.api.payloads[record["id"]] = b"conflicting published bytes"
                before = copy.deepcopy(
                    (
                        self.api.releases,
                        self.api.tags,
                        self.api.asset_records,
                        self.api.payloads,
                    )
                )
                self.api.mutations.clear()
                with self.assertRaises(assets.AssetError):
                    assets.publish(self.api, args)
                self.assertEqual(
                    before,
                    (
                        self.api.releases,
                        self.api.tags,
                        self.api.asset_records,
                        self.api.payloads,
                    ),
                )
                self.assertEqual(self.api.mutations, [])
                self.assertNotIn((FORK, "preview-baseline"), self.api.releases)

    def test_immutable_assets_reuse_and_conflict_fail_without_overwrite(self):
        path = self.root / "payload"
        path.write_bytes(b"unchanged")
        release = assets.ensure_release(
            self.api, FORK, "preview-nightly-100-test-500", CONTROL_SHA, immutable=True
        )
        assets.upload_asset(
            self.api,
            FORK,
            release,
            "preview-wheels.zip",
            path,
            "application/zip",
            immutable=True,
        )
        self.api.mutations.clear()
        assets.upload_asset(
            self.api,
            FORK,
            release,
            "preview-wheels.zip",
            path,
            "application/zip",
            immutable=True,
        )
        self.assertEqual(self.api.mutations, [])
        release = assets.publish_release(self.api, FORK, release)
        self.api.mutations.clear()
        path.write_bytes(b"conflicting bytes")
        with self.assertRaisesRegex(
            assets.AssetError, "immutable release asset hash conflict"
        ):
            assets.upload_asset(
                self.api,
                FORK,
                release,
                "preview-wheels.zip",
                path,
                "application/zip",
                immutable=True,
            )
        self.assertEqual(self.api.mutations, [])
        self.api.tags[(FORK, "preview-nightly-200-test-500")] = "1" * 40
        with self.assertRaisesRegex(assets.AssetError, "tag control SHA"):
            assets.ensure_release(
                self.api,
                FORK,
                "preview-nightly-200-test-500",
                CONTROL_SHA,
                immutable=True,
            )

    def test_publish_resumes_completed_immutable_release_without_duplicate_assets(self):
        bundle, candidate, _ = self.bundle()
        args = argparse.Namespace(
            consumer_repository=FORK, bundle=bundle, test_run_id=500
        )
        mutate = self.api.mutate

        def fail_before_fixed(method, path, value):
            if method == "POST" and value.get("tag_name") == "preview-baseline":
                raise assets.APIError(503)
            return mutate(method, path, value)

        with mock.patch.object(self.api, "mutate", side_effect=fail_before_fixed):
            with self.assertRaises(assets.APIError):
                assets.publish(self.api, args)
        tag = "preview-nightly-100-test-500"
        unique = self.api.releases[(FORK, tag)]
        self.assertFalse(unique["draft"])
        immutable_records = assets.release_assets(self.api, FORK, unique)
        self.assertEqual(len(immutable_records), 2)
        ids_before = {record["id"] for record in immutable_records}
        self.api.mutations.clear()
        with redirect_stdout(io.StringIO()):
            assets.publish(self.api, args)
        self.assertEqual(
            {record["id"] for record in assets.release_assets(self.api, FORK, unique)},
            ids_before,
        )
        self.assertFalse(
            any(
                path.startswith(f"uploads/{unique['id']}/")
                for _, path, _ in self.api.mutations
            )
        )
        self.assertEqual(
            assets.load_baseline(self.api, FORK)[0]["source"], candidate["source"]
        )

    def test_newer_success_without_artifact_does_not_fall_back_to_older_run(self):
        self.api.runs[(UPSTREAM, 101)] = run_record(
            UPSTREAM, 101, "ppu-accuracy-nightly.yaml"
        )
        with redirect_stdout(io.StringIO()) as out:
            assets.fetch_nightly(self.api, self.nightly_args())
        self.assertEqual(out.getvalue(), "available=false\n")
        self.assertIn(f"/repos/{UPSTREAM}/actions/runs/101/artifacts", self.api.reads)
        self.assertFalse((self.root / "flow").exists())

    def test_publish_rechecks_after_immutable_release_for_race(self):
        bundle, candidate, _ = self.bundle()
        args = argparse.Namespace(
            consumer_repository=FORK, bundle=bundle, test_run_id=500
        )
        original = assets.promotion_check
        calls = 0

        def race(*parameters, **keywords):
            nonlocal calls
            calls += 1
            if calls == 2:
                # A competing successful publisher creates both the fixed pointer
                # and a verified immutable release, not a marker-only fixture.
                self.previous_release(101)
            return original(*parameters, **keywords)

        with (
            mock.patch.object(assets, "promotion_check", side_effect=race),
            redirect_stdout(io.StringIO()) as out,
        ):
            assets.publish(self.api, args)
        self.assertEqual(out.getvalue(), "promote=false\n")
        self.assertEqual(
            assets.load_baseline(self.api, FORK)[0]["source"]["upstream_run_id"], 101
        )
        self.assertEqual(calls, 2)

    def test_uploaded_new_wheels_with_old_marker_recovers_on_all_retry_orders(self):
        previous, _ = self.previous_release(99)
        bundle, candidate, evidence = self.bundle(100, wheel_bytes=b"new wheel payload")
        args = argparse.Namespace(
            consumer_repository=FORK, bundle=bundle, test_run_id=500
        )
        upload = self.api.upload

        def interrupt_after_new_wheels(repo, release_id, name, path, content_type):
            record = upload(repo, release_id, name, path, content_type)
            fixed = self.api.releases.get((FORK, "preview-baseline"))
            if fixed and release_id == fixed["id"] and name == "preview-wheels.zip":
                raise assets.APIError(503)
            return record

        with mock.patch.object(
            self.api, "upload", side_effect=interrupt_after_new_wheels
        ):
            with self.assertRaises(assets.APIError):
                assets.publish(self.api, args)
        self.assertEqual(assets.load_baseline(self.api, FORK)[0], previous)
        interrupted = copy.deepcopy(self.api)
        for retry_id in (98, 100, 101):
            with self.subTest(retry_id=retry_id):
                self.api = copy.deepcopy(interrupted)
                retry = bundle if retry_id == 100 else self.bundle(retry_id)[0]
                with redirect_stdout(io.StringIO()):
                    assets.publish(
                        self.api,
                        argparse.Namespace(
                            consumer_repository=FORK, bundle=retry, test_run_id=500
                        ),
                    )
                expected = 99 if retry_id == 98 else retry_id
                baseline = assets.load_baseline(self.api, FORK)[0]
                self.assertEqual(baseline["source"]["upstream_run_id"], expected)
                out = self.root / f"ci-read-after-retry-{retry_id}"
                with redirect_stdout(io.StringIO()):
                    assets.fetch_preview(
                        self.api,
                        argparse.Namespace(consumer_repository=FORK, output=out),
                    )
                assets.validate_wheels(out / "wheelhouse", baseline["source"])

    def previous_release(self, run_id=99):
        bundle, candidate, evidence = self.bundle(run_id)
        baseline = make_baseline(candidate, evidence)
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted((bundle / "wheelhouse").rglob("*")):
                if path.is_file():
                    archive.writestr(
                        path.relative_to(bundle / "wheelhouse").as_posix(),
                        path.read_bytes(),
                    )
        data = stream.getvalue()
        self.api.install_baseline(baseline, data)
        unique = self.api.install_release(
            baseline["immutable_release_tag"], baseline, data
        )
        self.api.mutations.clear()
        return baseline, unique

    def test_first_fixed_draft_interruptions_recover_from_verified_immutable(self):
        bundle, candidate, evidence = self.bundle()
        args = argparse.Namespace(
            consumer_repository=FORK, bundle=bundle, test_run_id=500
        )
        for stage in ("wheels", "marker", "publication"):
            with self.subTest(stage=stage):
                self.api = FakeAPI()
                upload, mutate = self.api.upload, self.api.mutate

                def interrupt_upload(
                    repo,
                    release_id,
                    name,
                    path,
                    content_type,
                    upload=upload,
                    stage=stage,
                ):
                    fixed = self.api.releases.get((FORK, "preview-baseline"))
                    if (
                        fixed
                        and release_id == fixed["id"]
                        and stage == "wheels"
                        and name == "preview-wheels.zip"
                    ):
                        raise assets.APIError(503)
                    if (
                        fixed
                        and release_id == fixed["id"]
                        and stage == "marker"
                        and name == "preview-baseline.json"
                    ):
                        raise assets.APIError(503)
                    return upload(repo, release_id, name, path, content_type)

                def interrupt_mutate(method, path, value, mutate=mutate, stage=stage):
                    fixed = self.api.releases.get((FORK, "preview-baseline"))
                    if (
                        stage == "publication"
                        and fixed
                        and path == f"/repos/{FORK}/releases/{fixed['id']}"
                        and value.get("draft") is False
                    ):
                        raise assets.APIError(503)
                    return mutate(method, path, value)

                with (
                    mock.patch.object(self.api, "upload", side_effect=interrupt_upload),
                    mock.patch.object(self.api, "mutate", side_effect=interrupt_mutate),
                ):
                    with self.assertRaises(assets.APIError):
                        assets.publish(self.api, args)
                fixed = self.api.releases[(FORK, "preview-baseline")]
                self.assertTrue(fixed["draft"])
                self.assertIsNone(
                    self.api.optional(f"/repos/{FORK}/releases/tags/preview-baseline")
                )
                self.assertEqual(assets.load_baseline(self.api, FORK), (None, None, []))
                self.assertTrue(assets.promotion_check(self.api, FORK, bundle, 500)[0])
                unique = self.api.releases[(FORK, "preview-nightly-100-test-500")]
                ids = {r["id"] for r in assets.release_assets(self.api, FORK, unique)}
                self.api.mutations.clear()
                if stage == "publication":
                    with (
                        mock.patch.object(assets, "GitHub", return_value=self.api),
                        redirect_stdout(io.StringIO()) as out,
                    ):
                        self.assertEqual(
                            assets.main(
                                [
                                    "check-preview",
                                    "--consumer-repository",
                                    FORK,
                                    "--bundle",
                                    str(bundle),
                                    "--test-run-id",
                                    "500",
                                ]
                            ),
                            0,
                        )
                    self.assertEqual(out.getvalue(), "promote=false\n")
                else:
                    with redirect_stdout(io.StringIO()):
                        assets.publish(self.api, args)
                self.assertEqual(
                    assets.load_baseline(self.api, FORK)[0],
                    make_baseline(candidate, evidence),
                )
                self.assertEqual(
                    ids,
                    {r["id"] for r in assets.release_assets(self.api, FORK, unique)},
                )
                self.assertFalse(
                    any(
                        p.startswith(f"uploads/{unique['id']}/")
                        for _, p, _ in self.api.mutations
                    )
                )

    def test_markerless_draft_recovers_original_run_not_retry_candidate(self):
        bundle, candidate, evidence = self.bundle()
        args = argparse.Namespace(
            consumer_repository=FORK, bundle=bundle, test_run_id=500
        )
        upload = self.api.upload

        def fail_fixed(repo, release_id, name, path, content_type):
            fixed = self.api.releases.get((FORK, "preview-baseline"))
            if fixed and fixed["id"] == release_id:
                raise assets.APIError(503)
            return upload(repo, release_id, name, path, content_type)

        with mock.patch.object(self.api, "upload", side_effect=fail_fixed):
            with self.assertRaises(assets.APIError):
                assets.publish(self.api, args)
        older, old_candidate, old_evidence = self.bundle(99)
        # A completed older immutable release must not turn this retry into a downgrade.
        old_archive = older / "preview-wheels.zip"
        assets.pack_wheels(
            older / "wheelhouse",
            old_archive,
            assets.validate_wheels(older / "wheelhouse"),
            old_candidate["source"],
        )
        self.api.install_release(
            "preview-nightly-99-test-500",
            make_baseline(old_candidate, old_evidence),
            old_archive.read_bytes(),
        )
        with redirect_stdout(io.StringIO()) as out:
            assets.publish(
                self.api,
                argparse.Namespace(
                    consumer_repository=FORK, bundle=older, test_run_id=500
                ),
            )
        self.assertEqual(out.getvalue(), "promote=false\n")
        self.assertEqual(
            assets.load_baseline(self.api, FORK)[0], make_baseline(candidate, evidence)
        )

    def test_markerless_draft_newer_retry_restores_before_advancing(self):
        bundle, candidate, evidence = self.bundle()
        archive = bundle / "preview-wheels.zip"
        assets.pack_wheels(
            bundle / "wheelhouse",
            archive,
            assets.validate_wheels(bundle / "wheelhouse"),
            candidate["source"],
        )
        self.api.install_release(
            "preview-nightly-100-test-500",
            make_baseline(candidate, evidence),
            archive.read_bytes(),
        )
        assets.ensure_release(
            self.api, FORK, "preview-baseline", CONTROL_SHA, immutable=False
        )
        newer, new_candidate, new_evidence = self.bundle(101)
        with redirect_stdout(io.StringIO()):
            assets.publish(
                self.api,
                argparse.Namespace(
                    consumer_repository=FORK, bundle=newer, test_run_id=500
                ),
            )
        self.assertEqual(
            assets.load_baseline(self.api, FORK)[0],
            make_baseline(new_candidate, new_evidence),
        )
        fixed = self.api.releases[(FORK, "preview-baseline")]
        markers = [
            self.api.payloads[r["id"]]
            for r in assets.release_assets(self.api, FORK, fixed)
            if r["name"].startswith("archive-")
            and r["name"].endswith("preview-baseline.json")
        ]
        self.assertIn(assets.json_bytes(make_baseline(candidate, evidence)), markers)

    def test_archived_marker_interruption_recovers_without_downgrading(self):
        previous, _ = self.previous_release()
        bundle, candidate, evidence = self.bundle()
        args = argparse.Namespace(
            consumer_repository=FORK, bundle=bundle, test_run_id=500
        )
        old_marker = assets.named_asset(
            assets.load_baseline(self.api, FORK)[2], "preview-baseline.json"
        )
        mutate = self.api.mutate

        def interrupt_after_archive(method, path, value):
            result = mutate(method, path, value)
            if (
                path == f"/repos/{FORK}/releases/assets/{old_marker['id']}"
                and value.get("name", "").startswith("archive-")
            ):
                raise assets.APIError(503)
            return result

        with mock.patch.object(self.api, "mutate", side_effect=interrupt_after_archive):
            with self.assertRaises(assets.APIError):
                assets.publish(self.api, args)
        with self.assertRaises(assets.AssetError):
            assets.load_baseline(self.api, FORK)
        self.api.mutations.clear()
        with (
            mock.patch.object(assets, "GitHub", return_value=self.api),
            redirect_stdout(io.StringIO()) as out,
        ):
            self.assertEqual(
                assets.main(
                    [
                        "check-preview",
                        "--consumer-repository",
                        FORK,
                        "--bundle",
                        str(bundle),
                        "--test-run-id",
                        "500",
                    ]
                ),
                0,
            )
        self.assertIn("promote=true\n", out.getvalue())
        self.assertEqual(assets.load_baseline(self.api, FORK)[0], previous)
        with redirect_stdout(io.StringIO()):
            assets.publish(self.api, args)
        self.assertEqual(
            assets.load_baseline(self.api, FORK)[0], make_baseline(candidate, evidence)
        )
        self.assertEqual(
            self.api.payloads[old_marker["id"]], assets.json_bytes(previous)
        )
        self.assertTrue(
            all(method in {"PATCH", "POST"} for method, _, _ in self.api.mutations)
        )

    def test_archived_newer_marker_is_fence_even_for_older_publisher(self):
        previous, _ = self.previous_release(101)
        fixed = self.api.releases[(FORK, "preview-baseline")]
        marker = assets.named_asset(
            assets.release_assets(self.api, FORK, fixed), "preview-baseline.json"
        )
        self.api.mutate(
            "PATCH",
            f"/repos/{FORK}/releases/assets/{marker['id']}",
            {"name": "archive-interrupted-preview-baseline.json"},
        )
        bundle, _, _ = self.bundle()
        self.api.mutations.clear()
        with redirect_stdout(io.StringIO()) as out:
            assets.publish(
                self.api,
                argparse.Namespace(
                    consumer_repository=FORK, bundle=bundle, test_run_id=500
                ),
            )
        self.assertEqual(out.getvalue(), "promote=false\n")
        self.assertEqual(assets.load_baseline(self.api, FORK)[0], previous)
        self.assertNotIn((FORK, "preview-nightly-100-test-500"), self.api.releases)

    def test_recovery_failures_make_no_mutations_and_can_retry(self):
        previous, unique = self.previous_release()
        fixed = self.api.releases[(FORK, "preview-baseline")]
        marker = assets.named_asset(
            assets.release_assets(self.api, FORK, fixed), "preview-baseline.json"
        )
        self.api.mutate(
            "PATCH",
            f"/repos/{FORK}/releases/assets/{marker['id']}",
            {"name": "archive-interrupted-preview-baseline.json"},
        )
        bundle, _, _ = self.bundle()
        args = argparse.Namespace(
            consumer_repository=FORK, bundle=bundle, test_run_id=500
        )
        original = copy.deepcopy(self.api)
        for failure in (
            "missing-release",
            "draft",
            "tag",
            "marker",
            "wheels",
            "starter",
            "missing-fence",
            "prerelease-false",
            "prerelease-absent",
            "title",
        ):
            with self.subTest(failure=failure):
                self.api = copy.deepcopy(original)
                tag = previous["immutable_release_tag"]
                records = assets.release_assets(self.api, FORK, unique)
                if failure == "missing-release":
                    self.api.releases.pop((FORK, tag))
                elif failure == "draft":
                    self.api.releases[(FORK, tag)]["draft"] = True
                elif failure == "tag":
                    self.api.tags[(FORK, tag)] = "1" * 40
                elif failure == "marker":
                    remote = assets.named_asset(records, "preview-baseline.json")
                    self.api.payloads[remote["id"]] = b'{"schema":1}'
                elif failure == "wheels":
                    remote = assets.named_asset(records, "preview-wheels.zip")
                    self.api.payloads[remote["id"]] = self.zip_data(
                        [("wrong", b"bad")]
                    ).read_bytes()
                elif failure == "starter":
                    remote = assets.named_asset(records, "preview-wheels.zip")
                    for record in self.api.asset_records.values():
                        if record["id"] == remote["id"]:
                            record["state"] = "starter"
                elif failure == "prerelease-false":
                    self.api.releases[(FORK, tag)]["prerelease"] = False
                elif failure == "prerelease-absent":
                    self.api.releases[(FORK, tag)].pop("prerelease")
                elif failure == "title":
                    self.api.releases[(FORK, tag)]["name"] = "Verified stable"
                else:
                    self.api.payloads[marker["id"]] = b'{"schema":1}'
                self.api.mutations.clear()
                with self.assertRaises(assets.AssetError):
                    assets.publish(self.api, args)
                self.assertEqual(self.api.mutations, [])
        self.api = original
        with redirect_stdout(io.StringIO()):
            assets.publish(self.api, args)
        self.assertEqual(
            assets.load_baseline(self.api, FORK)[0]["source"]["upstream_run_id"], 100
        )

    def test_recovery_itself_can_be_interrupted_and_retried(self):
        previous, _ = self.previous_release(101)
        fixed = self.api.releases[(FORK, "preview-baseline")]
        marker = assets.named_asset(
            assets.release_assets(self.api, FORK, fixed), "preview-baseline.json"
        )
        self.api.mutate(
            "PATCH",
            f"/repos/{FORK}/releases/assets/{marker['id']}",
            {"name": "archive-interrupted-preview-baseline.json"},
        )
        bundle, _, _ = self.bundle()
        args = argparse.Namespace(
            consumer_repository=FORK, bundle=bundle, test_run_id=500
        )
        upload = self.api.upload

        def fail_restore(repo, release_id, name, path, content_type):
            if release_id == fixed["id"] and name == "preview-baseline.json":
                raise assets.APIError(502)
            return upload(repo, release_id, name, path, content_type)

        with mock.patch.object(self.api, "upload", side_effect=fail_restore):
            with self.assertRaises(assets.APIError):
                assets.publish(self.api, args)
        with self.assertRaises(assets.AssetError):
            assets.load_baseline(self.api, FORK)
        with redirect_stdout(io.StringIO()):
            assets.publish(self.api, args)
        self.assertEqual(assets.load_baseline(self.api, FORK)[0], previous)

    def test_502_starter_assets_recover_for_draft_immutable_and_fixed(self):
        bundle, candidate, evidence = self.bundle()
        args = argparse.Namespace(
            consumer_repository=FORK, bundle=bundle, test_run_id=500
        )
        for target_tag, name in (
            ("preview-nightly-100-test-500", "preview-wheels.zip"),
            ("preview-nightly-100-test-500", "preview-baseline.json"),
            ("preview-baseline", "preview-wheels.zip"),
            ("preview-baseline", "preview-baseline.json"),
        ):
            with self.subTest(target_tag=target_tag, name=name):
                self.api = FakeAPI()
                upload = self.api.upload
                starter_ids = []

                def starter(
                    repo,
                    release_id,
                    asset_name,
                    path,
                    content_type,
                    target_tag=target_tag,
                    name=name,
                    upload=upload,
                    starter_ids=starter_ids,
                ):
                    release = self.api.releases.get((FORK, target_tag))
                    if release and release["id"] == release_id and asset_name == name:
                        record = upload(
                            repo, release_id, asset_name, path, content_type
                        )
                        self.api.asset_records[(repo, release_id, record["id"])][
                            "state"
                        ] = "starter"
                        self.api.payloads[record["id"]] = b"partial bytes"
                        starter_ids.append(record["id"])
                        raise assets.APIError(502)
                    return upload(repo, release_id, asset_name, path, content_type)

                with mock.patch.object(self.api, "upload", side_effect=starter):
                    with self.assertRaises(assets.APIError):
                        assets.publish(self.api, args)
                self.assertEqual(len(starter_ids), 1)
                release = self.api.releases[(FORK, target_tag)]
                with self.assertRaisesRegex(assets.AssetError, "not uploaded"):
                    assets.named_asset(
                        assets.release_assets(self.api, FORK, release), name
                    )
                with redirect_stdout(io.StringIO()):
                    assets.publish(self.api, args)
                archived = self.api.asset_records[(FORK, release["id"], starter_ids[0])]
                self.assertTrue(archived["name"].startswith("archive-"))
                self.assertEqual(self.api.payloads[starter_ids[0]], b"partial bytes")
                self.assertEqual(
                    assets.load_baseline(self.api, FORK)[0],
                    make_baseline(candidate, evidence),
                )
                self.assertTrue(
                    all(
                        method in {"POST", "PATCH"}
                        for method, _, _ in self.api.mutations
                    )
                )

    def test_published_fixed_starters_after_second_promotion_are_recoverable(self):
        previous, _ = self.previous_release()
        original = copy.deepcopy(self.api)
        bundle, candidate, evidence = self.bundle()
        args = argparse.Namespace(
            consumer_repository=FORK, bundle=bundle, test_run_id=500
        )
        for name in ("preview-wheels.zip", "preview-baseline.json"):
            with self.subTest(name=name):
                self.api = copy.deepcopy(original)
                fixed = self.api.releases[(FORK, "preview-baseline")]
                upload = self.api.upload
                starter_id = None

                def leave_starter(
                    repo,
                    release_id,
                    asset_name,
                    path,
                    content_type,
                    upload=upload,
                    fixed=fixed,
                    name=name,
                ):
                    nonlocal starter_id
                    result = upload(repo, release_id, asset_name, path, content_type)
                    if release_id == fixed["id"] and asset_name == name:
                        starter_id = result["id"]
                        self.api.asset_records[(repo, release_id, starter_id)][
                            "state"
                        ] = "starter"
                        self.api.payloads[starter_id] = b"partial"
                        raise assets.APIError(502)
                    return result

                with mock.patch.object(self.api, "upload", side_effect=leave_starter):
                    with self.assertRaises(assets.APIError):
                        assets.publish(self.api, args)
                self.assertIsNotNone(starter_id)
                with self.assertRaisesRegex(assets.AssetError, "not uploaded"):
                    assets.load_baseline(self.api, FORK)
                with redirect_stdout(io.StringIO()):
                    assets.publish(self.api, args)
                self.assertEqual(
                    assets.load_baseline(self.api, FORK)[0],
                    make_baseline(candidate, evidence),
                )
                self.assertTrue(
                    self.api.asset_records[(FORK, fixed["id"], starter_id)][
                        "name"
                    ].startswith("archive-")
                )
                self.assertEqual(self.api.payloads[starter_id], b"partial")
                archived_markers = [
                    self.api.payloads[r["id"]]
                    for r in self.api.asset_records.values()
                    if r["name"].startswith("archive-")
                    and r["name"].endswith("preview-baseline.json")
                ]
                self.assertIn(assets.json_bytes(previous), archived_markers)

    def test_incomplete_fixed_without_recovery_fence_is_not_a_fresh_start(self):
        bundle, _, _ = self.bundle()
        args = argparse.Namespace(
            consumer_repository=FORK, bundle=bundle, test_run_id=500
        )
        for draft in (True, False):
            with self.subTest(draft=draft):
                self.api = FakeAPI()
                fixed = assets.ensure_release(
                    self.api, FORK, "preview-baseline", CONTROL_SHA, immutable=False
                )
                if not draft:
                    assets.publish_release(self.api, FORK, fixed)
                self.api.mutations.clear()
                with self.assertRaises(assets.AssetError):
                    assets.publish(self.api, args)
                self.assertEqual(self.api.mutations, [])
                if draft:
                    self.assertEqual(
                        assets.load_baseline(self.api, FORK), (None, None, [])
                    )
                else:
                    with self.assertRaises(assets.AssetError):
                        assets.load_baseline(self.api, FORK)

    def test_draft_conflict_repair_and_published_starter_refused(self):
        path = self.root / "payload"
        path.write_bytes(b"old")
        release = assets.ensure_release(
            self.api, FORK, "preview-nightly-100-test-500", CONTROL_SHA, immutable=True
        )
        assets.upload_asset(
            self.api,
            FORK,
            release,
            "preview-wheels.zip",
            path,
            "application/zip",
            immutable=True,
        )
        old = assets.named_asset(
            assets.release_assets(self.api, FORK, release), "preview-wheels.zip"
        )
        path.write_bytes(b"new")
        assets.upload_asset(
            self.api,
            FORK,
            release,
            "preview-wheels.zip",
            path,
            "application/zip",
            immutable=True,
        )
        self.assertEqual(self.api.payloads[old["id"]], b"old")
        release = assets.publish_release(self.api, FORK, release)
        current = assets.named_asset(
            assets.release_assets(self.api, FORK, release), "preview-wheels.zip"
        )
        self.api.asset_records[(FORK, release["id"], current["id"])]["state"] = (
            "starter"
        )
        self.api.mutations.clear()
        with self.assertRaisesRegex(
            assets.AssetError, "immutable release asset hash conflict"
        ):
            assets.upload_asset(
                self.api,
                FORK,
                release,
                "preview-wheels.zip",
                path,
                "application/zip",
                immutable=True,
            )
        self.assertEqual(self.api.mutations, [])
        with self.assertRaisesRegex(assets.AssetError, "missing assets"):
            assets.upload_asset(
                self.api,
                FORK,
                release,
                "preview-baseline.json",
                path,
                "application/json",
                immutable=True,
            )
        self.assertEqual(self.api.mutations, [])

    def test_read_only_bounded_retry_and_no_secret_errors(self):
        client = assets.GitHub()
        with (
            mock.patch.object(
                client, "_open_once", side_effect=assets.APIError(503)
            ) as op,
            mock.patch.object(assets.time, "sleep"),
        ):
            with self.assertRaises(assets.APIError):
                client.get(f"/repos/{FORK}")
            self.assertEqual(op.call_count, assets.READ_ATTEMPTS)
        with mock.patch.object(
            client, "_open_once", side_effect=assets.APIError(503)
        ) as op:
            with self.assertRaises(assets.APIError):
                client.mutate("POST", f"/repos/{FORK}/releases", {"name": "test"})
            self.assertEqual(op.call_count, 1)
        with (
            mock.patch.object(
                client,
                "_open_once",
                side_effect=urllib.error.URLError("unit-test-secret"),
            ),
            mock.patch.object(assets.time, "sleep"),
        ):
            with self.assertRaises(assets.AssetError) as error:
                client.get(f"/repos/{FORK}")
            self.assertNotIn("unit-test-secret", str(error.exception))
        with self.assertRaises(assets.AssetError):
            client._open_once("GET", "https://evil.example/repos/x/y")

    def test_redirect_strips_authorization_and_never_restores(self):
        client = assets.GitHub()
        requests = []

        def redirect(request, timeout):
            requests.append(request)
            if len(requests) == 1:
                raise urllib.error.HTTPError(
                    request.full_url,
                    302,
                    "redirect",
                    {"Location": "https://blob.example/wheels?signature=secret"},
                    None,
                )
            if len(requests) == 2:
                raise urllib.error.HTTPError(
                    request.full_url,
                    302,
                    "redirect",
                    {
                        "Location": f"https://api.github.com/repos/{FORK}/releases/assets/1"
                    },
                    None,
                )
            return Response(b"download")

        client._opener = mock.Mock()
        client._opener.open.side_effect = redirect
        self.assertEqual(client.bytes(f"/repos/{FORK}/releases/assets/1"), b"download")
        self.assertEqual(
            requests[0].get_header("Authorization"), "Bearer unit-test-secret"
        )
        self.assertIsNone(requests[1].get_header("Authorization"))
        self.assertIsNone(requests[2].get_header("Authorization"))
        client._opener.open.side_effect = urllib.error.HTTPError(
            "unused", 302, "redirect", {"Location": "http://blob.example/unsafe"}, None
        )
        with self.assertRaisesRegex(assets.AssetError, "unsafe download redirect"):
            client.bytes(f"/repos/{FORK}/releases/assets/1")
        client._opener.open.side_effect = urllib.error.HTTPError(
            "unused", 302, "redirect", {"Location": "https://blob.example/unsafe"}, None
        )
        with self.assertRaises(assets.APIError):
            client.mutate("POST", f"/repos/{FORK}/releases", {"draft": True})

    def test_publisher_lookup_second_page_drafts_support_ensure_and_recovery(self):
        bundle, candidate, evidence = self.bundle()
        baseline = make_baseline(candidate, evidence)
        archive = bundle / "preview-wheels.zip"
        assets.pack_wheels(
            bundle / "wheelhouse",
            archive,
            assets.validate_wheels(bundle / "wheelhouse"),
            candidate["source"],
        )
        for tag, immutable in (
            ("preview-nightly-100-test-500", True),
            ("preview-baseline", False),
        ):
            with self.subTest(tag=tag):
                self.api = FakeAPI()
                draft = assets.ensure_release(
                    self.api, FORK, tag, CONTROL_SHA, immutable=immutable
                )
                self.assertIsNone(
                    self.api.optional(f"/repos/{FORK}/releases/tags/{tag}")
                )
                if not immutable:
                    self.api.install_release(
                        baseline["immutable_release_tag"],
                        baseline,
                        archive.read_bytes(),
                    )
                self.api.mutations.clear()
                client = assets.GitHub()
                client._opener = mock.Mock()
                requested_pages = []

                def page_response(request, timeout, requested_pages=requested_pages):
                    parsed = urllib.parse.urlsplit(request.full_url)
                    self.assertEqual(parsed.path, f"/repos/{FORK}/releases")
                    query = urllib.parse.parse_qs(parsed.query)
                    self.assertEqual(query["per_page"], ["100"])
                    self.assertEqual(
                        request.get_header("Authorization"), "Bearer unit-test-secret"
                    )
                    page = int(query["page"][0])
                    requested_pages.append(page)
                    if page == 1:
                        records = [
                            {"id": 10000 + i, "tag_name": f"unrelated-{i}"}
                            for i in range(100)
                        ]
                    else:
                        self.assertEqual(page, 2)
                        records = list(self.api.releases.values())
                    return Response(json.dumps(records).encode())

                client._opener.open.side_effect = page_response
                fake_pages = self.api.pages

                def pages(path, key=None, client=client, fake_pages=fake_pages):
                    if path == f"/repos/{FORK}/releases":
                        self.api.reads.append(path)
                        return client.pages(path, key)
                    return fake_pages(path, key)

                with mock.patch.object(self.api, "pages", side_effect=pages):
                    found = assets.ensure_release(
                        self.api, FORK, tag, CONTROL_SHA, immutable=immutable
                    )
                    self.assertEqual(found, draft)
                    self.assertEqual(self.api.mutations, [])
                    self.assertEqual(requested_pages, [1, 2])
                    if not immutable:
                        with redirect_stdout(io.StringIO()) as out:
                            assets.publish(
                                self.api,
                                argparse.Namespace(
                                    consumer_repository=FORK,
                                    bundle=bundle,
                                    test_run_id=500,
                                ),
                            )
                        self.assertEqual(out.getvalue(), "promote=false\n")
                        self.assertEqual(
                            assets.load_baseline(self.api, FORK)[0], baseline
                        )
                        self.assertEqual(
                            self.api.releases[(FORK, tag)]["id"], draft["id"]
                        )
                        self.assertEqual(requested_pages, [1, 2, 1, 2, 1, 2])
                        self.assertFalse(
                            any(
                                method == "POST" and body.get("tag_name") == tag
                                for method, _, body in self.api.mutations
                            )
                        )

    def test_publisher_lookup_rejects_duplicate_tags_across_pages_and_truncated_scan(
        self,
    ):
        tag = "preview-nightly-100-test-500"
        draft = assets.ensure_release(self.api, FORK, tag, CONTROL_SHA, immutable=True)
        self.api.mutations.clear()
        first = [draft] + [
            {"id": 10000 + i, "tag_name": f"unrelated-{i}"} for i in range(99)
        ]
        duplicate = {
            **draft,
            "id": 9000,
            "url": f"{assets.API}/repos/{FORK}/releases/9000",
        }
        client = assets.GitHub()
        client._opener = mock.Mock()
        client._opener.open.side_effect = [
            Response(json.dumps(first).encode()),
            Response(json.dumps([duplicate]).encode()),
        ]
        with mock.patch.object(self.api, "pages", side_effect=client.pages):
            with self.assertRaisesRegex(
                assets.AssetError, "duplicate matching release tag"
            ):
                assets.ensure_release(self.api, FORK, tag, CONTROL_SHA, immutable=True)
        self.assertEqual(client._opener.open.call_count, 2)
        self.assertEqual(self.api.mutations, [])
        client._opener.open.side_effect = [Response(json.dumps(first).encode())]
        with (
            mock.patch.object(assets, "MAX_PAGES", 1),
            mock.patch.object(self.api, "pages", side_effect=client.pages),
        ):
            with self.assertRaisesRegex(assets.AssetError, "pagination limit"):
                assets.ensure_release(self.api, FORK, tag, CONTROL_SHA, immutable=True)
        self.assertEqual(self.api.mutations, [])

    def test_paginated_runs_artifacts_and_release_assets_reach_second_page(self):
        client = assets.GitHub()
        client._opener = mock.Mock()
        for key, path in (
            ("workflow_runs", f"/repos/{FORK}/actions/workflows/22/runs"),
            ("artifacts", f"/repos/{FORK}/actions/runs/500/artifacts"),
            (None, f"/repos/{FORK}/releases/1/assets"),
        ):
            first = [{"id": i} for i in range(100)]
            second = [{"id": 777}]
            data = [
                json.dumps({key: records} if key else records).encode()
                for records in (first, second)
            ]
            client._opener.open.side_effect = [Response(data[0]), Response(data[1])]
            client._opener.open.reset_mock()
            results = list(client.pages(path, key))
            self.assertEqual(results[-1]["id"], 777)
            self.assertEqual(len(results), 101)
            urls = [c.args[0].full_url for c in client._opener.open.call_args_list]
            self.assertIn("page=1", urls[0])
            self.assertIn("page=2", urls[1])
        with mock.patch.object(assets, "MAX_PAGES", 2):
            client._opener.open.side_effect = [
                Response(json.dumps(first).encode()),
                Response(json.dumps(first).encode()),
            ]
            with self.assertRaisesRegex(assets.AssetError, "pagination limit"):
                list(client.pages(f"/repos/{FORK}/releases/1/assets"))

    def test_upload_fixed_host_content_length_and_no_automatic_mutation_retry(self):
        client = assets.GitHub()
        client._opener = mock.Mock()
        client._opener.open.return_value = Response(
            b'{"id":1,"name":"preview-wheels.zip","state":"uploaded"}'
        )
        path = self.root / "zip"
        path.write_bytes(b"wheel bytes")
        client.upload(FORK, 1, "preview-wheels.zip", path, "application/zip")
        request = client._opener.open.call_args.args[0]
        self.assertTrue(request.full_url.startswith("https://uploads.github.com/"))
        self.assertEqual(request.get_header("Content-length"), str(path.stat().st_size))
        self.assertEqual(request.get_method(), "POST")

    def test_cli_parser_all_seven_contracts_and_stdlib_only_no_delete(self):
        commands = assets.parser()._subparsers._group_actions[0].choices
        self.assertEqual(
            set(commands),
            {
                "fetch-nightly",
                "fetch-preview",
                "candidate",
                "attest",
                "resolve-test-run",
                "check-preview",
                "publish-preview",
            },
        )
        help_text = assets.parser().format_help() + "".join(
            command.format_help() for command in commands.values()
        )
        for old in ("fetch-ci", "check-promotion", "publish"):
            self.assertNotIn(old, commands)
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                assets.parser().parse_args([old])
            self.assertEqual(error.exception.code, 2)
        for old in ("fetch-ci", "check-promotion", "nightly promotion", "ci-baseline"):
            self.assertNotIn(old, help_text)
        self.assertIn("minimum", help_text)
        self.assertIn("not verified", help_text)
        self.assertIn("--include-published", help_text)
        self.assertNotIn("ci-baseline", TOOL.read_text())
        tree = ast.parse(TOOL.read_text())
        forbidden_modules = {"subprocess", "shutil", "requests"}
        forbidden_calls = {"unlink", "remove", "rmdir", "rmtree", "system", "popen"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertFalse(
                    forbidden_modules.intersection(n.name for n in node.names)
                )
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, forbidden_calls)
        with self.assertRaises(assets.AssetError):
            assets.decode_json(b'{"schema":1,"schema":1}')
        with self.assertRaises(assets.AssetError):
            assets.decode_json(b'{"schema":NaN}')
        os.environ.pop("GH_TOKEN")
        os.environ["GITHUB_TOKEN"] = "not-allowed"
        with self.assertRaisesRegex(assets.AssetError, "GH_TOKEN"):
            assets.GitHub()


if __name__ == "__main__":
    unittest.main()
