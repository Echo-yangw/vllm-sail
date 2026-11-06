# SPDX-License-Identifier: Apache-2.0
"""Offline unittest coverage for the persistent-runner deployment gates."""

from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

TOOL = Path(__file__).resolve().parents[1] / "scripts" / "ci" / "deployment.py"
SPEC = importlib.util.spec_from_file_location("deployment", TOOL)
deployment = importlib.util.module_from_spec(SPEC)
with mock.patch.object(sys, "path", [str(TOOL.parent), *sys.path]):
    SPEC.loader.exec_module(deployment)

REPOSITORY = "example/vllm-sail"
HEAD_SHA = "a" * 40
BASE_SHA = "b" * 40
CONTROL_SHA = "c" * 40
VLLM_SHA = "d" * 40
IMAGE_REPOSITORY = "ghcr.io/example/vllm-sail-preview"
IMAGE_DIGEST = "sha256:" + "e" * 64
IMAGE = f"{IMAGE_REPOSITORY}@{IMAGE_DIGEST}"
BASE_IMAGE = "runtime.example/sail/pytorch@sha256:" + "f" * 64
ENV = {
    "PPU_TRANSITION_ENABLED": "true",
    "PREVIEW_REGISTRY_HOST": "ghcr.io",
    "PREVIEW_IMAGE_REPOSITORY": IMAGE_REPOSITORY,
    "RUNTIME_BASE_IMAGE": BASE_IMAGE,
    "TRUSTED_PPU_TESTS_ENABLED": "true",
    "APPROVE_HOST_EXECUTION": "true",
    "GH_ACTOR": "maintainer",
    "GH_TOKEN": "offline-secret-never-print",
}
DTYPE_VALUES = '["float16", "bfloat16", "float32"]'
SHAPE_VALUES = "[(1, 64), (7, 130), (32, 4096)]"
TEST_IDENTITY = "tests/e2e/test_native_activation.py::test_silu_and_mul"
ACTIVATION_SOURCE = """import pytest

pytestmark = pytest.mark.ppu


@pytest.fixture(scope="module")
def ppu_runtime():
    raise RuntimeError("fixture must not execute")


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize("shape", [(1, 64), (7, 130), (32, 4096)])
def test_silu_and_mul(ppu_runtime, dtype_name, shape):
    raise RuntimeError("test body must not execute")
"""


def catalog():
    return {
        "schema": 1,
        "quality_scope": "minimum-regression",
        "expected_ut": sorted(deployment.EXPECTED_UT),
        "expected_ops_file": "tests/e2e/test_native_activation.py",
        "dtypes": ["float16", "bfloat16", "float32"],
        "shapes": [[1, 64], [7, 130], [32, 4096]],
        "runner_label": "8-ppu-m890p",
        "reference_model": None,
        "exclusions": [],
    }


def candidate():
    return {
        "schema": 1,
        "source": {"upstream_sha": HEAD_SHA, "vllm_sha": VLLM_SHA},
        "image_repository": IMAGE_REPOSITORY,
        "image_digest": IMAGE_DIGEST,
        "image": IMAGE,
        "control_sha": CONTROL_SHA,
        "dockerfile_sha256": "0" * 64,
    }


def runtime():
    return {
        "schema": 1,
        "python": "3.11.10",
        "os": "Linux",
        "torch": "2.7.0+sail",
        "sdk": {"name": "SAIL SDK", "version": "observed-version"},
        "gpu": {"count": 1, "capability": [8, 1]},
        "image": IMAGE,
        "source_paths": {"torch": "/opt/runtime/torch/__init__.py"},
        "host_contract": {"kmd": "not_measured", "firmware": "not_measured"},
    }


class FakeGitHub:
    """Only GET/optional are supported; no network or state mutation exists."""

    def __init__(self):
        self.reads = []
        self.metadata = {"full_name": REPOSITORY, "default_branch": "main"}
        self.pr = {
            "number": 7,
            "state": "open",
            "draft": False,
            "base": {"ref": "main", "repo": {"full_name": REPOSITORY}, "sha": BASE_SHA},
            "head": {
                "ref": "trusted-change",
                "repo": {"full_name": REPOSITORY},
                "sha": HEAD_SHA,
            },
            "user": {"login": "author"},
            "author_association": "OWNER",
            "title": "do-not-echo-title",
            "body": "do-not-echo-body",
        }
        self.permissions = {
            "maintainer": {"permission": "maintain", "user": {"login": "maintainer"}},
            "author": {"permission": "write", "user": {"login": "author"}},
        }

    def get(self, path):
        self.reads.append(("get", path))
        if path == f"/repos/{REPOSITORY}":
            return copy.deepcopy(self.metadata)
        if path == f"/repos/{REPOSITORY}/pulls/7":
            return copy.deepcopy(self.pr)
        raise AssertionError(f"unexpected fake GET: {path}")

    def optional(self, path):
        self.reads.append(("optional", path))
        prefix = f"/repos/{REPOSITORY}/collaborators/"
        if path.startswith(prefix) and path.endswith("/permission"):
            return copy.deepcopy(
                self.permissions.get(path[len(prefix) : -len("/permission")])
            )
        raise AssertionError(f"unexpected fake optional GET: {path}")


class OfflineTestCase(unittest.TestCase):
    def setUp(self):
        # Also protects a mistakenly unmocked real GitHub client in any test.
        network = mock.patch(
            "urllib.request.OpenerDirector.open",
            side_effect=AssertionError("real API requests are forbidden"),
        )
        network.start()
        self.addCleanup(network.stop)
        environment = mock.patch.dict(os.environ, ENV, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.temp = tempfile.TemporaryDirectory(prefix="deployment-tests-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()

    def invoke(self, argv):
        with redirect_stdout(self.stdout), redirect_stderr(self.stderr):
            return deployment.main(argv)

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=True), encoding="ascii")
        return path

    def preflight_args(self, mode="nightly"):
        return argparse.Namespace(
            mode=mode,
            catalog=self.write("catalog.json", catalog()),
            output=self.root / "flow" / "plan.json",
        )

    def make_plan(self, mode="nightly"):
        args = self.preflight_args(mode)
        with mock.patch.object(deployment, "outputs"):
            plan = deployment.preflight(args)
        return args.output, plan

    def evidence_args(self):
        plan_path, _ = self.make_plan()
        return argparse.Namespace(
            plan=plan_path,
            runtime=self.write("flow/results/runtime.json", runtime()),
            candidate=self.write("flow/candidate.json", candidate()),
            output=self.root / "flow" / "deployment-evidence.json",
        )


class PreflightTests(OfflineTestCase):
    def test_all_modes_pass_without_a_token_or_an_api_client(self):
        os.environ.pop("GH_TOKEN")
        for mode in deployment.MODES:
            with (
                self.subTest(mode=mode),
                mock.patch.object(deployment, "GitHub") as github,
            ):
                args = self.preflight_args(mode)
                with redirect_stdout(self.stdout):
                    plan = deployment.preflight(args)
                github.assert_not_called()
                self.assertEqual(plan["mode"], mode)
                self.assertEqual(plan["quality_scope"], "minimum-regression")
                self.assertEqual(
                    plan["execution_profile"], "persistent-selfhosted-transition"
                )
                self.assertEqual(plan["catalog"], catalog())
                self.assertEqual(
                    plan["unverified_coverage"], list(deployment.UNVERIFIED_COVERAGE)
                )
                self.assertEqual(deployment.read_json(args.output), plan)
        self.assertIn("test_plan_digest=", self.stdout.getvalue())
        self.assertIn(f"runtime_base_image={BASE_IMAGE}", self.stdout.getvalue())
        self.assertIn(f"image_repository={IMAGE_REPOSITORY}", self.stdout.getvalue())

    def test_missing_required_variables_fail_before_writing_a_plan(self):
        for mode in deployment.MODES:
            for name in (
                "PPU_TRANSITION_ENABLED",
                "PREVIEW_REGISTRY_HOST",
                "PREVIEW_IMAGE_REPOSITORY",
                "RUNTIME_BASE_IMAGE",
            ):
                with (
                    self.subTest(mode=mode, name=name),
                    mock.patch.dict(os.environ, clear=False),
                ):
                    os.environ.pop(name)
                    args = self.preflight_args(mode)
                    with self.assertRaises(deployment.AssetError):
                        deployment.preflight(args)
                    self.assertFalse(args.output.exists())

    def test_transition_does_not_default_to_enabled(self):
        for value in ("", "false", "True", "TRUE", "1", " true", "true\n"):
            with (
                self.subTest(value=value),
                mock.patch.dict(os.environ, PPU_TRANSITION_ENABLED=value),
            ):
                with self.assertRaisesRegex(deployment.AssetError, "explicitly true"):
                    deployment.preflight(self.preflight_args())

    def test_dockerfile_requires_both_separate_manual_gates(self):
        for name in ("TRUSTED_PPU_TESTS_ENABLED", "APPROVE_HOST_EXECUTION"):
            for value in (None, "false", "True", "1", "true\n"):
                with self.subTest(name=name, value=value), mock.patch.dict(os.environ):
                    if value is None:
                        os.environ.pop(name)
                    else:
                        os.environ[name] = value
                    args = self.preflight_args("dockerfile")
                    with self.assertRaises(deployment.AssetError):
                        deployment.preflight(args)
                    self.assertFalse(args.output.exists())

    def test_nightly_and_preview_do_not_require_dockerfile_approval(self):
        os.environ.pop("TRUSTED_PPU_TESTS_ENABLED")
        os.environ.pop("APPROVE_HOST_EXECUTION")
        for mode in ("nightly", "preview"):
            with self.subTest(mode=mode), mock.patch.object(deployment, "outputs"):
                deployment.preflight(self.preflight_args(mode))

    def test_registry_rejects_scheme_slash_credentials_and_invalid_ports(self):
        for host in (
            "https://ghcr.io",
            "ghcr.io/",
            "ghcr.io/example",
            "user@ghcr.io",
            "ghcr.io:0",
            "ghcr.io:65536",
            "ghcr.io:0443",
            "-host",
            "host..io",
            "ghcr.io?token=hidden",
            "host\n",
            "registry.\u00e9xample",
        ):
            with (
                self.subTest(host=host),
                mock.patch.dict(os.environ, PREVIEW_REGISTRY_HOST=host),
            ):
                with self.assertRaises(deployment.AssetError):
                    deployment.preflight(self.preflight_args())

    def test_registry_port_and_case_normalization(self):
        with (
            mock.patch.dict(
                os.environ,
                PREVIEW_REGISTRY_HOST="REGISTRY.EXAMPLE:5000",
                PREVIEW_IMAGE_REPOSITORY="registry.example:5000/team/image",
            ),
            mock.patch.object(deployment, "outputs"),
        ):
            plan = deployment.preflight(self.preflight_args())
        self.assertEqual(plan["image_repository"], "registry.example:5000/team/image")

    def test_image_repository_rejects_tags_digests_uppercase_and_wrong_host(self):
        for repository in (
            IMAGE_REPOSITORY + ":latest",
            IMAGE,
            "ghcr.io/Example/image",
            "other.example/team/image",
            "ghcr.io",
            "ghcr.io/team/../image",
            "ghcr.io/team//image",
            "ghcr.io/team/image\n",
        ):
            with (
                self.subTest(repository=repository),
                mock.patch.dict(os.environ, PREVIEW_IMAGE_REPOSITORY=repository),
            ):
                with self.assertRaises(deployment.AssetError):
                    deployment.preflight(self.preflight_args())

    def test_runtime_base_requires_exact_sha256_digest(self):
        for base in (
            "runtime.example/team/image:latest",
            "runtime.example/team/image",
            "runtime.example/team/image@sha256:" + "a" * 63,
            "runtime.example/team/image@sha256:" + "A" * 64,
            "runtime.example/team/image@sha256:" + "g" * 64,
            BASE_IMAGE + "@sha256:" + "a" * 64,
        ):
            with (
                self.subTest(base=base),
                mock.patch.dict(os.environ, RUNTIME_BASE_IMAGE=base),
            ):
                with self.assertRaises(deployment.AssetError):
                    deployment.preflight(self.preflight_args())

    def test_circular_candidate_repository_is_rejected_even_with_a_tag(self):
        for base in (IMAGE, IMAGE_REPOSITORY + ":runtime@" + IMAGE_DIGEST):
            with (
                self.subTest(base=base),
                mock.patch.dict(os.environ, RUNTIME_BASE_IMAGE=base),
            ):
                with self.assertRaisesRegex(deployment.AssetError, "as its base"):
                    deployment.preflight(self.preflight_args())

    def test_other_repository_tag_at_digest_is_allowed(self):
        with (
            mock.patch.dict(
                os.environ, RUNTIME_BASE_IMAGE=BASE_IMAGE.replace("@", ":Pinned@")
            ),
            mock.patch.object(deployment, "outputs"),
        ):
            deployment.preflight(self.preflight_args())

    def test_cli_writes_to_github_output_without_echoing_a_secret(self):
        args = self.preflight_args("preview")
        output = self.root / "github-output"
        os.environ["GITHUB_OUTPUT"] = str(output)
        result = self.invoke(
            [
                "preflight",
                "--mode",
                "preview",
                "--catalog",
                str(args.catalog),
                "--output",
                str(args.output),
            ]
        )
        self.assertEqual(result, 0)
        self.assertEqual(self.stdout.getvalue(), "")
        self.assertIn("test_plan_digest=", output.read_text(encoding="ascii"))
        self.assertNotIn(ENV["GH_TOKEN"], args.output.read_text(encoding="ascii"))

    def test_cli_reports_missing_configuration_as_an_error_not_green(self):
        args = self.preflight_args()
        os.environ.pop("RUNTIME_BASE_IMAGE")
        self.assertEqual(
            self.invoke(
                [
                    "preflight",
                    "--mode",
                    "nightly",
                    "--catalog",
                    str(args.catalog),
                    "--output",
                    str(args.output),
                ]
            ),
            1,
        )
        self.assertIn("RUNTIME_BASE_IMAGE", self.stderr.getvalue())
        self.assertEqual(self.stdout.getvalue(), "")
        self.assertFalse(args.output.exists())
        self.assertNotIn(ENV["GH_TOKEN"], self.stderr.getvalue())


class CatalogTests(OfflineTestCase):
    def test_digest_normalizes_object_keys_and_whitespace_but_preserves_the_catalog(
        self,
    ):
        original = catalog()
        reordered = dict(reversed(list(original.items())))
        self.assertEqual(
            deployment.catalog_digest(original), deployment.catalog_digest(reordered)
        )
        expected_bytes = json.dumps(
            original, sort_keys=True, separators=(",", ":")
        ).encode("ascii")
        self.assertEqual(
            deployment.catalog_digest(original), deployment.sha256_bytes(expected_bytes)
        )
        args = self.preflight_args()
        args.catalog.write_text(json.dumps(reordered, indent=4), encoding="ascii")
        with mock.patch.object(deployment, "outputs"):
            first = deployment.preflight(args)
        args.catalog.write_text(
            json.dumps(original, separators=(",", ":")), encoding="ascii"
        )
        with mock.patch.object(deployment, "outputs"):
            second = deployment.preflight(args)
        self.assertEqual(first, second)
        self.assertEqual(first["catalog"], original)
        changed = copy.deepcopy(original)
        changed["shapes"][0][1] += 1
        with self.assertRaisesRegex(
            deployment.AssetError, "catalog shapes must be exactly"
        ):
            deployment.catalog_digest(changed)

    def test_validation_does_not_mutate_or_alias_the_original_catalog(self):
        original = catalog()
        normalized = deployment.validate_catalog(original)
        normalized["shapes"][0][0] += 1
        self.assertEqual(original, catalog())

    def test_wrong_schema_scope_runner_reference_or_exclusions_are_rejected(self):
        for field, value in (
            ("schema", True),
            ("schema", 2),
            ("quality_scope", "verified"),
            ("runner_label", "kubernetes"),
            ("reference_model", "example/model"),
            ("exclusions", ["performance"]),
            ("exclusions", None),
        ):
            with self.subTest(field=field, value=value):
                data = catalog()
                data[field] = value
                with self.assertRaises(deployment.AssetError):
                    deployment.validate_catalog(data)
        data = catalog()
        data["unknown"] = "field"
        with self.assertRaises(deployment.AssetError):
            deployment.validate_catalog(data)
        del data["unknown"]
        del data["shapes"]
        with self.assertRaises(deployment.AssetError):
            deployment.validate_catalog(data)

    def test_literal_none_reference_model_is_supported_without_rewriting_it(self):
        data = catalog()
        data["reference_model"] = "none"
        self.assertEqual(deployment.validate_catalog(data), data)

    def test_ut_requires_exact_two_distinct_existing_ids(self):
        valid = catalog()["expected_ut"]
        for tests in (
            [],
            valid[:1],
            valid + ["third"],
            [valid[0], valid[0]],
            [valid[0], "tests/ut/untrusted.py::test_other"],
            "two",
            [1, 2],
        ):
            with self.subTest(tests=tests):
                data = catalog()
                data["expected_ut"] = tests
                with self.assertRaises(deployment.AssetError):
                    deployment.validate_catalog(data)

    def test_ops_file_is_the_existing_safe_relative_path(self):
        for path in (
            "/tests/e2e/test_native_activation.py",
            "../tests/e2e/test_native_activation.py",
            "tests/e2e/test_other.py",
            3,
        ):
            with self.subTest(path=path):
                data = catalog()
                data["expected_ops_file"] = path
                with self.assertRaises(deployment.AssetError):
                    deployment.validate_catalog(data)

    def test_dtype_count_identity_and_types(self):
        for dtypes in (
            [],
            ["float16"],
            ["float16"] * 3,
            ["float16", "bfloat16", "float64"],
            ["float16", "bfloat16", 32],
        ):
            with self.subTest(dtypes=dtypes):
                data = catalog()
                data["dtypes"] = dtypes
                with self.assertRaises(deployment.AssetError):
                    deployment.validate_catalog(data)

    def test_shapes_require_three_distinct_positive_integer_lists(self):
        for shapes in (
            [],
            [[1]],
            [[1], [2], [3], [4]],
            [[1], [1], [2]],
            [[True], [2], [3]],
            [[0], [2], [3]],
            [[-1], [2], [3]],
            [[1.0], [2], [3]],
            [[], [2], [3]],
            ["shape0", "shape1", "shape2"],
        ):
            with self.subTest(shapes=shapes):
                data = catalog()
                data["shapes"] = shapes
                with self.assertRaises(deployment.AssetError):
                    deployment.validate_catalog(data)

    def test_dtype_order_is_fixed(self):
        valid = catalog()["dtypes"]
        for dtypes in (valid[::-1], [valid[1], valid[0], valid[2]], tuple(valid)):
            with self.subTest(dtypes=dtypes):
                data = catalog()
                data["dtypes"] = dtypes
                with self.assertRaises(deployment.AssetError):
                    deployment.validate_catalog(data)

    def test_shapes_are_the_fixed_ordered_regression_values(self):
        valid = catalog()["shapes"]
        for shapes in (
            [[1, 2], [3, 4], [5, 6]],
            [[1, 16], [32, 128], [64, 256]],
            [[1, 65], [7, 130], [32, 4096]],
            valid[::-1],
            [valid[1], valid[0], valid[2]],
            [[True, 64], *valid[1:]],
            [[1.0, 64], *valid[1:]],
            [tuple(shape) for shape in valid],
        ):
            with self.subTest(shapes=shapes):
                data = catalog()
                data["shapes"] = shapes
                with self.assertRaisesRegex(
                    deployment.AssetError, "catalog shapes|JSON value type"
                ):
                    deployment.validate_catalog(data)

    def test_recomputing_a_digest_cannot_authorize_changed_parameters(self):
        _, original = self.make_plan()
        for field, value in (
            ("shapes", [[1, 2], [3, 4], [5, 6]]),
            ("shapes", catalog()["shapes"][::-1]),
            ("dtypes", catalog()["dtypes"][::-1]),
        ):
            with self.subTest(field=field, value=value):
                plan = copy.deepcopy(original)
                plan["catalog"][field] = value
                raw = json.dumps(
                    plan["catalog"], sort_keys=True, separators=(",", ":")
                ).encode("ascii")
                plan["TEST_PLAN_DIGEST"] = deployment.sha256_bytes(raw)
                with self.assertRaisesRegex(
                    deployment.AssetError, f"catalog {field} must be exactly"
                ):
                    deployment.validate_plan(plan)

    def test_unicode_is_rejected_even_when_the_json_file_is_ascii_escaped(self):
        data = catalog()
        data["reference_model"] = "n\u00f6ne"
        args = self.preflight_args()
        args.catalog = self.write("escaped-catalog.json", data)
        self.assertTrue(args.catalog.read_bytes().isascii())
        with self.assertRaisesRegex(deployment.AssetError, "ASCII"):
            deployment.preflight(args)
        data = catalog()
        data["\u00e9xtra"] = "value"
        with self.assertRaisesRegex(deployment.AssetError, "ASCII"):
            deployment.validate_catalog(data)

    def test_duplicate_json_keys_and_nonfinite_values_fail_closed(self):
        args = self.preflight_args()
        for text in ('{"schema":1,"schema":1}', '{"schema":1,"shapes":NaN}'):
            with self.subTest(text=text):
                args.catalog.write_text(text, encoding="ascii")
                with self.assertRaises(deployment.AssetError):
                    deployment.preflight(args)

    def test_plan_revalidates_embedded_catalog_and_coverage(self):
        _, original = self.make_plan()
        for field, value in (
            ("TEST_PLAN_DIGEST", "0" * 64),
            ("unverified_coverage", []),
            ("mode", "verified"),
            ("execution_profile", "kubernetes"),
            ("image_repository", "ghcr.io/team/tagged:latest"),
        ):
            with self.subTest(field=field):
                plan = copy.deepcopy(original)
                plan[field] = value
                with self.assertRaises(deployment.AssetError):
                    deployment.validate_plan(plan)


class TestSourceTests(OfflineTestCase):
    def source_file(self, text=ACTIVATION_SOURCE):
        path = self.root / "tests" / "e2e" / "test_native_activation.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="ascii")
        return path

    def assert_source_rejected(self, text):
        path = self.source_file(text)
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        self.assertEqual(
            self.invoke(["validate-test-source", "--source", str(self.root)]), 1
        )
        self.assertEqual(self.stdout.getvalue(), "")
        self.assertNotIn(ENV["GH_TOKEN"], self.stderr.getvalue())
        self.assertEqual(path.read_text(encoding="ascii"), text)

    def test_cli_verifies_identity_without_imports_execution_or_writes(self):
        text = 'raise RuntimeError("module must not execute")\n' + ACTIVATION_SOURCE
        path = self.source_file(text)
        output = self.root / "github-output"
        with (
            mock.patch.dict(os.environ, {"GITHUB_OUTPUT": str(output)}, clear=True),
            mock.patch.object(deployment, "GitHub") as github,
            mock.patch.object(deployment, "write_json") as write_json,
            mock.patch.object(deployment, "outputs") as outputs,
            mock.patch(
                "importlib.util.spec_from_file_location",
                side_effect=AssertionError("source imports are forbidden"),
            ),
        ):
            self.assertEqual(deployment.validate_test_source(self.root), TEST_IDENTITY)
            self.assertEqual(
                self.invoke(["validate-test-source", "--source", str(self.root)]), 0
            )
        github.assert_not_called()
        write_json.assert_not_called()
        outputs.assert_not_called()
        self.assertEqual(
            self.stdout.getvalue(), f"test definition verified: {TEST_IDENTITY}\n"
        )
        self.assertEqual(self.stderr.getvalue(), "")
        self.assertEqual(path.read_text(encoding="ascii"), text)
        self.assertFalse(output.exists())

    def test_static_validation_can_use_a_hosted_synthetic_ast(self):
        tree = ast.parse(ACTIVATION_SOURCE)
        path = self.root / "tests" / "e2e" / "test_native_activation.py"
        with (
            mock.patch.object(
                Path, "read_text", return_value="# synthetic AST"
            ) as read,
            mock.patch.object(deployment.ast, "parse", return_value=tree) as parse,
        ):
            self.assertEqual(deployment.validate_test_source(self.root), TEST_IDENTITY)
        read.assert_called_once_with(encoding="utf-8")
        parse.assert_called_once_with("# synthetic AST", filename=str(path))
        self.assertFalse(path.exists())

    def test_missing_source_file_fails_locally(self):
        for source in (self.root, self.root / "missing-repository"):
            with (
                self.subTest(source=source),
                mock.patch.object(deployment, "GitHub") as github,
            ):
                self.assertEqual(
                    self.invoke(["validate-test-source", "--source", str(source)]), 1
                )
                github.assert_not_called()
        self.assertEqual(self.stdout.getvalue(), "")
        self.assertIn("local I/O failure", self.stderr.getvalue())
        path = self.source_file()
        self.assertEqual(
            self.invoke(["validate-test-source", "--source", str(path)]), 1
        )

    def test_missing_or_renamed_test_function_fails(self):
        for text in (
            "import pytest\n",
            ACTIVATION_SOURCE.replace("test_silu_and_mul", "test_other"),
        ):
            with self.subTest(text=text):
                self.assert_source_rejected(text)
                self.assertIn(TEST_IDENTITY, self.stderr.getvalue())

    def test_duplicate_test_functions_fail(self):
        duplicates = (
            ACTIVATION_SOURCE,
            "def test_silu_and_mul():\n    pass\n",
            "async def test_silu_and_mul():\n    pass\n",
            "def other():\n    def test_silu_and_mul():\n        pass\n",
        )
        for extra in duplicates:
            with self.subTest(extra=extra):
                self.assert_source_rejected(ACTIVATION_SOURCE + "\n" + extra)
                self.assertIn("exactly one", self.stderr.getvalue())

    def test_nested_and_async_definitions_fail(self):
        nested = "class Tests:\n" + "\n".join(
            "    " + line for line in ACTIVATION_SOURCE.splitlines()
        )
        for text in (
            nested,
            ACTIVATION_SOURCE.replace(
                "def test_silu_and_mul", "async def test_silu_and_mul"
            ),
        ):
            with self.subTest(text=text):
                self.assert_source_rejected(text)

    def test_missing_or_extra_parametrize_decorators_fail(self):
        dtype = f'@pytest.mark.parametrize("dtype_name", {DTYPE_VALUES})\n'
        shape = f'@pytest.mark.parametrize("shape", {SHAPE_VALUES})\n'
        for text in (
            ACTIVATION_SOURCE.replace(dtype, ""),
            ACTIVATION_SOURCE.replace(shape, ""),
            ACTIVATION_SOURCE.replace(
                shape, shape + '@pytest.mark.parametrize("extra", [1])\n'
            ),
            ACTIVATION_SOURCE.replace(shape, shape + shape),
        ):
            with self.subTest(text=text):
                self.assert_source_rejected(text)
                self.assertIn("exactly two", self.stderr.getvalue())

    def test_unknown_decorators_fail(self):
        shape = f'@pytest.mark.parametrize("shape", {SHAPE_VALUES})'
        for decorator in (
            "@unknown",
            "@pytest.mark.skip",
            "@pytest.mark.skip()",
            f'@mark.parametrize("shape", {SHAPE_VALUES})',
            f'@pytest.parametrize("shape", {SHAPE_VALUES})',
            f'@other.mark.parametrize("shape", {SHAPE_VALUES})',
        ):
            with self.subTest(decorator=decorator):
                self.assert_source_rejected(ACTIVATION_SOURCE.replace(shape, decorator))
        self.assert_source_rejected(
            ACTIVATION_SOURCE.replace(shape, shape + "\n@unknown")
        )

    def test_decorator_order_is_fixed(self):
        dtype = f'@pytest.mark.parametrize("dtype_name", {DTYPE_VALUES})'
        shape = f'@pytest.mark.parametrize("shape", {SHAPE_VALUES})'
        self.assert_source_rejected(
            ACTIVATION_SOURCE.replace(dtype + "\n" + shape, shape + "\n" + dtype)
        )

    def test_parameter_names_are_exact_and_distinct(self):
        for name in (
            '"dtype"',
            '"dtype_name,shape"',
            '"shape"',
            '["dtype_name"]',
            "DTYPE_NAME",
            "None",
        ):
            with self.subTest(name=name):
                self.assert_source_rejected(
                    ACTIVATION_SOURCE.replace(
                        'parametrize("dtype_name",', f"parametrize({name},"
                    )
                )
        self.assert_source_rejected(
            ACTIVATION_SOURCE.replace(
                'parametrize("shape",', 'parametrize("dtype_name",'
            )
        )

    def test_parametrize_argument_errors_fail(self):
        original = f'@pytest.mark.parametrize("dtype_name", {DTYPE_VALUES})'
        for arguments in (
            '"dtype_name"',
            f'"dtype_name", {DTYPE_VALUES}, "extra"',
            f'"dtype_name", {DTYPE_VALUES}, ids=["a", "b", "c"]',
            f'"dtype_name", {DTYPE_VALUES}, indirect=True',
            f'argnames="dtype_name", argvalues={DTYPE_VALUES}',
            f'*("dtype_name", {DTYPE_VALUES})',
            f'"dtype_name", {DTYPE_VALUES}, **OPTIONS',
        ):
            with self.subTest(arguments=arguments):
                self.assert_source_rejected(
                    ACTIVATION_SOURCE.replace(
                        original, f"@pytest.mark.parametrize({arguments})"
                    )
                )

    def test_dtype_values_and_order_are_fixed(self):
        for values in (
            "[]",
            '["float16"]',
            '["float16", "bfloat16", "float64"]',
            '["bfloat16", "float16", "float32"]',
            '["float32", "bfloat16", "float16"]',
            '["float16", "float16", "float32"]',
            '["float16", "bfloat16", 32]',
        ):
            with self.subTest(values=values):
                self.assert_source_rejected(
                    ACTIVATION_SOURCE.replace(DTYPE_VALUES, values)
                )

    def test_shape_values_and_order_are_fixed(self):
        for values in (
            "[]",
            "[(1, 64)]",
            "[(1, 2), (3, 4), (5, 6)]",
            "[(1, 65), (7, 130), (32, 4096)]",
            "[(7, 130), (1, 64), (32, 4096)]",
            "[(32, 4096), (7, 130), (1, 64)]",
            "[(1, 64), (1, 64), (32, 4096)]",
            "[(True, 64), (7, 130), (32, 4096)]",
            "[(1.0, 64), (7, 130), (32, 4096)]",
            "[[1, 64], [7, 130], [32, 4096]]",
            "[(1, 64, 1), (7, 130), (32, 4096)]",
            "[(1, 64), (7, 130), (32, 4096), (64, 4096)]",
        ):
            with self.subTest(values=values):
                self.assert_source_rejected(
                    ACTIVATION_SOURCE.replace(SHAPE_VALUES, values)
                )

    def test_values_must_be_literal_lists(self):
        for original, replacement in (
            (DTYPE_VALUES, '("float16", "bfloat16", "float32")'),
            (SHAPE_VALUES, "((1, 64), (7, 130), (32, 4096))"),
            (DTYPE_VALUES, "DTYPES"),
            (SHAPE_VALUES, "SHAPES"),
        ):
            with self.subTest(replacement=replacement):
                self.assert_source_rejected(
                    ACTIVATION_SOURCE.replace(original, replacement)
                )

    def test_pytest_param_and_code_expressions_are_never_executed(self):
        for original, values in (
            (DTYPE_VALUES, '[pytest.param("float16"), "bfloat16", "float32"]'),
            (SHAPE_VALUES, "[pytest.param((1, 64)), (7, 130), (32, 4096)]"),
            (SHAPE_VALUES, "[(1, 32 * 2), (7, 130), (32, 4096)]"),
            (SHAPE_VALUES, "[(+1, 64), (7, 130), (32, 4096)]"),
            (DTYPE_VALUES, DTYPE_VALUES + " + []"),
            (DTYPE_VALUES, '__import__("builtins").exec("raise RuntimeError()")'),
        ):
            with self.subTest(values=values):
                self.assert_source_rejected(ACTIVATION_SOURCE.replace(original, values))
                self.assertIn("literal lists", self.stderr.getvalue())

    def test_comprehensions_are_rejected(self):
        for original, values in (
            (DTYPE_VALUES, f"[dtype for dtype in {DTYPE_VALUES}]"),
            (SHAPE_VALUES, f"[shape for shape in {SHAPE_VALUES}]"),
            (SHAPE_VALUES, f"list(shape for shape in {SHAPE_VALUES})"),
        ):
            with self.subTest(values=values):
                self.assert_source_rejected(ACTIVATION_SOURCE.replace(original, values))
                self.assertIn("literal lists", self.stderr.getvalue())

    def test_invalid_syntax_fails_without_echoing_source(self):
        self.assert_source_rejected(
            ACTIVATION_SOURCE + "\n" + ENV["GH_TOKEN"] + " = (\n"
        )
        self.assertIn("invalid syntax", self.stderr.getvalue())

    def test_source_io_errors_do_not_echo_sensitive_text(self):
        with mock.patch.object(Path, "read_text", side_effect=OSError(ENV["GH_TOKEN"])):
            self.assertEqual(
                self.invoke(["validate-test-source", "--source", str(self.root)]), 1
            )
        self.assertEqual(self.stdout.getvalue(), "")
        self.assertNotIn(ENV["GH_TOKEN"], self.stderr.getvalue())

    def test_cli_requires_source_and_has_ascii_help(self):
        with redirect_stdout(self.stdout), self.assertRaises(SystemExit) as result:
            deployment.parser().parse_args(["validate-test-source", "--help"])
        self.assertEqual(result.exception.code, 0)
        self.assertIn("--source", self.stdout.getvalue())
        self.assertTrue(self.stdout.getvalue().isascii())
        with redirect_stderr(self.stderr), self.assertRaises(SystemExit) as result:
            deployment.parser().parse_args(["validate-test-source"])
        self.assertEqual(result.exception.code, 2)


class PullRequestTests(OfflineTestCase):
    def setUp(self):
        super().setUp()
        self.api = FakeGitHub()

    def verify_args(self):
        return argparse.Namespace(
            repository=REPOSITORY, pr_number=7, head_sha=HEAD_SHA, base_sha=BASE_SHA
        )

    def test_manual_same_repository_maintainer_and_writer_author_pass(self):
        record = deployment.authorize_pr(self.api, REPOSITORY, 7)
        self.assertEqual(
            record,
            {
                "schema": 1,
                "repository": REPOSITORY,
                "pr_number": 7,
                "head_sha": HEAD_SHA,
                "base_sha": BASE_SHA,
                "author": "author",
                "actor": "maintainer",
                "mode": "trusted-maintainer-manual",
            },
        )
        self.assertIn(
            ("optional", f"/repos/{REPOSITORY}/collaborators/author/permission"),
            self.api.reads,
        )
        self.assertIn(
            ("optional", f"/repos/{REPOSITORY}/collaborators/maintainer/permission"),
            self.api.reads,
        )
        self.assertNotIn("title", record)
        self.assertNotIn("body", record)

    def test_write_maintain_and_admin_are_the_only_accepted_permissions(self):
        for role in ("write", "maintain", "admin"):
            with self.subTest(role=role):
                for user in ("maintainer", "author"):
                    self.api.permissions[user]["permission"] = role
                deployment.authorize_pr(self.api, REPOSITORY, 7)

    def test_external_pr_fails_even_with_owner_association_and_writer_author(self):
        self.api.pr["head"]["repo"]["full_name"] = "external/vllm-sail"
        with self.assertRaisesRegex(deployment.AssetError, "external or fork"):
            deployment.authorize_pr(self.api, REPOSITORY, 7)
        self.api.pr["head"]["repo"] = None
        with self.assertRaises(deployment.AssetError):
            deployment.authorize_pr(self.api, REPOSITORY, 7)

    def test_internal_author_without_real_write_permission_fails(self):
        for role in ("read", "triage", "none", None):
            with self.subTest(role=role):
                self.api.permissions["author"]["permission"] = role
                with self.assertRaisesRegex(
                    deployment.AssetError, "author must have repository"
                ):
                    deployment.authorize_pr(self.api, REPOSITORY, 7)
        self.api.permissions.pop("author")
        with self.assertRaises(deployment.AssetError):
            deployment.authorize_pr(self.api, REPOSITORY, 7)

    def test_dispatch_actor_without_real_write_permission_fails(self):
        for role in ("read", "triage", "none", None):
            with self.subTest(role=role):
                self.api.permissions["maintainer"]["permission"] = role
                with self.assertRaisesRegex(
                    deployment.AssetError, "actor must have repository"
                ):
                    deployment.authorize_pr(self.api, REPOSITORY, 7)
        self.api.permissions.pop("maintainer")
        with self.assertRaises(deployment.AssetError):
            deployment.authorize_pr(self.api, REPOSITORY, 7)

    def test_permission_user_mismatch_fails(self):
        self.api.permissions["author"]["user"]["login"] = "someone-else"
        with self.assertRaisesRegex(deployment.AssetError, "user mismatch"):
            deployment.authorize_pr(self.api, REPOSITORY, 7)

    def test_default_branch_comes_from_live_repository_metadata(self):
        self.api.metadata["default_branch"] = "release"
        with self.assertRaisesRegex(deployment.AssetError, "current default branch"):
            deployment.authorize_pr(self.api, REPOSITORY, 7)
        self.api.pr["base"]["ref"] = "release"
        deployment.authorize_pr(self.api, REPOSITORY, 7)

    def test_closed_draft_wrong_base_and_wrong_pr_number_fail(self):
        for field, value in (
            ("state", "closed"),
            ("draft", True),
            ("draft", "false"),
            ("draft", None),
            ("number", 8),
            ("number", True),
        ):
            with self.subTest(field=field, value=value):
                self.api = FakeGitHub()
                self.api.pr[field] = value
                with self.assertRaises(deployment.AssetError):
                    deployment.authorize_pr(self.api, REPOSITORY, 7)
        self.api = FakeGitHub()
        self.api.pr["base"]["repo"]["full_name"] = "other/repository"
        with self.assertRaises(deployment.AssetError):
            deployment.authorize_pr(self.api, REPOSITORY, 7)

    def test_head_and_base_must_be_exact_lowercase_40_hex_commits(self):
        for side in ("head", "base"):
            for value in (
                "a" * 39,
                "a" * 64,
                "A" * 40,
                "g" * 40,
                "a" * 40 + "\n",
                None,
            ):
                with self.subTest(side=side, value=value):
                    self.api = FakeGitHub()
                    self.api.pr[side]["sha"] = value
                    with self.assertRaises(deployment.AssetError):
                        deployment.authorize_pr(self.api, REPOSITORY, 7)

    def test_missing_actor_and_invalid_repository_or_number_fail(self):
        os.environ.pop("GH_ACTOR")
        with self.assertRaises(deployment.AssetError):
            deployment.authorize_pr(self.api, REPOSITORY, 7)
        os.environ["GH_ACTOR"] = "maintainer"
        for repository, number in (("../repo", 7), (REPOSITORY, 0), (REPOSITORY, True)):
            with self.subTest(repository=repository, number=number):
                with self.assertRaises(deployment.AssetError):
                    deployment.authorize_pr(self.api, repository, number)

    def test_verify_reauthorizes_and_rechecks_live_head_and_base(self):
        deployment.authorize_pr(self.api, REPOSITORY, 7)
        self.api.reads.clear()
        result = deployment.verify_pr(self.api, self.verify_args())
        self.assertEqual(result["head_sha"], HEAD_SHA)
        self.assertEqual(len(self.api.reads), 4)
        for side in ("head", "base"):
            with self.subTest(side=side):
                self.api = FakeGitHub()
                self.api.pr[side]["sha"] = "1" * 40
                with self.assertRaisesRegex(deployment.AssetError, "changed"):
                    deployment.verify_pr(self.api, self.verify_args())

    def test_verify_rejects_revoked_permission_or_changed_pr_trust(self):
        deployment.authorize_pr(self.api, REPOSITORY, 7)
        for user in ("maintainer", "author"):
            with self.subTest(user=user):
                self.api = FakeGitHub()
                self.api.permissions[user]["permission"] = "read"
                with self.assertRaises(deployment.AssetError):
                    deployment.verify_pr(self.api, self.verify_args())
        self.api = FakeGitHub()
        self.api.pr["head"]["repo"]["full_name"] = "external/repository"
        with self.assertRaises(deployment.AssetError):
            deployment.verify_pr(self.api, self.verify_args())

    def test_cli_authorization_writes_record_and_never_echoes_body_title_or_token(self):
        output = self.root / "flow" / "pr.json"
        with mock.patch.object(deployment, "GitHub", return_value=self.api):
            result = self.invoke(
                [
                    "authorize-pr",
                    "--repository",
                    REPOSITORY,
                    "--pr-number",
                    "7",
                    "--output",
                    str(output),
                ]
            )
        self.assertEqual(result, 0)
        self.assertEqual(
            deployment.read_json(output)["mode"], "trusted-maintainer-manual"
        )
        self.assertEqual(
            self.stdout.getvalue(),
            f"pr_number=7\nhead_sha={HEAD_SHA}\nbase_sha={BASE_SHA}\n",
        )
        text = (
            self.stdout.getvalue()
            + self.stderr.getvalue()
            + output.read_text(encoding="ascii")
        )
        for secret in (ENV["GH_TOKEN"], "do-not-echo-body", "do-not-echo-title"):
            self.assertNotIn(secret, text)

    def test_verify_cli_never_emits_old_success_outputs_after_a_change(self):
        self.api.pr["base"]["sha"] = "2" * 40
        with mock.patch.object(deployment, "GitHub", return_value=self.api):
            result = self.invoke(
                [
                    "verify-pr",
                    "--repository",
                    REPOSITORY,
                    "--pr-number",
                    "7",
                    "--head-sha",
                    HEAD_SHA,
                    "--base-sha",
                    BASE_SHA,
                ]
            )
        self.assertEqual(result, 1)
        self.assertEqual(self.stdout.getvalue(), "")
        self.assertIn("changed", self.stderr.getvalue())

    def test_missing_token_fails_locally_without_an_api_call(self):
        os.environ.pop("GH_TOKEN")
        self.assertEqual(
            self.invoke(
                [
                    "authorize-pr",
                    "--repository",
                    REPOSITORY,
                    "--pr-number",
                    "7",
                    "--output",
                    str(self.root / "pr.json"),
                ]
            ),
            1,
        )
        self.assertIn("GH_TOKEN is required", self.stderr.getvalue())


class EvidenceTests(OfflineTestCase):
    def test_evidence_binds_commits_digest_and_raw_files_without_claiming_verified(
        self,
    ):
        args = self.evidence_args()
        with mock.patch.object(deployment, "outputs") as outputs:
            evidence = deployment.record_evidence(args)
        self.assertEqual(evidence["schema"], 1)
        self.assertEqual(evidence["quality_scope"], "minimum-regression")
        self.assertEqual(evidence["mode"], "nightly")
        self.assertEqual(evidence["channel"], "preview")
        self.assertEqual(evidence["PLUGIN_TEST_COMMIT"], HEAD_SHA)
        self.assertEqual(evidence["UPSTREAM_VLLM_TEST_COMMIT"], VLLM_SHA)
        self.assertEqual(evidence["runtime_image_digest"], IMAGE_DIGEST)
        self.assertEqual(evidence["control_sha"], CONTROL_SHA)
        self.assertEqual(
            evidence["TEST_PLAN_DIGEST"], deployment.catalog_digest(catalog())
        )
        self.assertEqual(evidence["runtime"], runtime())
        self.assertFalse(evidence["verified"])
        self.assertEqual(evidence["verification_state"], "unverified")
        self.assertIn("not a verified state", evidence["statement"])
        self.assertNotIn("status", evidence)
        self.assertEqual(
            evidence["unverified_coverage"], list(deployment.UNVERIFIED_COVERAGE)
        )
        for name in ("plan", "runtime", "candidate"):
            self.assertEqual(
                evidence[f"{name}_sha256"], deployment.sha256_file(getattr(args, name))
            )
        self.assertEqual(deployment.read_json(args.output), evidence)
        outputs.assert_called_once_with(
            plugin_test_commit=HEAD_SHA,
            upstream_vllm_test_commit=VLLM_SHA,
            runtime_image_digest=IMAGE_DIGEST,
            test_plan_digest=deployment.catalog_digest(catalog()),
            control_sha=CONTROL_SHA,
            mode="nightly",
        )

    def test_runtime_image_mismatch_missing_image_and_env_mismatch_fail(self):
        for image in (
            None,
            IMAGE_REPOSITORY + "@sha256:" + "0" * 64,
            IMAGE + ":latest",
        ):
            with self.subTest(image=image):
                args = self.evidence_args()
                data = runtime()
                data["image"] = image
                args.runtime = self.write("bad-runtime.json", data)
                with self.assertRaisesRegex(
                    deployment.AssetError, "runtime image differs"
                ):
                    deployment.record_evidence(args)
                self.assertFalse(args.output.exists())
        args = self.evidence_args()
        os.environ["TEST_IMAGE"] = "ghcr.io/other/image@" + IMAGE_DIGEST
        with self.assertRaisesRegex(deployment.AssetError, "TEST_IMAGE differs"):
            deployment.record_evidence(args)

    def test_runtime_requires_one_observed_device_not_zero_skip_or_multiple(self):
        for count in (None, 0, 2, True, "1", 1.0):
            with self.subTest(count=count):
                args = self.evidence_args()
                data = runtime()
                data["gpu"]["count"] = count
                args.runtime = self.write("no-device.json", data)
                with self.assertRaisesRegex(deployment.AssetError, "exactly one"):
                    deployment.record_evidence(args)
                self.assertFalse(args.output.exists())

    def test_explicit_device_count_cannot_contradict_the_gpu_observation(self):
        args = self.evidence_args()
        data = runtime()
        data["device_count"] = 0
        args.runtime = self.write("conflicting-runtime.json", data)
        with self.assertRaises(deployment.AssetError):
            deployment.record_evidence(args)
        data["device_count"] = 1
        args.runtime = self.write("counted-runtime.json", data)
        with mock.patch.object(deployment, "outputs"):
            deployment.record_evidence(args)

    def test_runtime_requires_schema_and_nonempty_python_and_torch(self):
        for field, value in (
            ("schema", True),
            ("schema", 2),
            ("python", ""),
            ("torch", ""),
            ("python", "   "),
            ("torch", "   "),
            ("python", None),
            ("torch", 2),
            ("torch", "2.7\u00e9"),
        ):
            with self.subTest(field=field, value=value):
                args = self.evidence_args()
                data = runtime()
                data[field] = value
                args.runtime = self.write("bad-runtime.json", data)
                with self.assertRaises(deployment.AssetError):
                    deployment.record_evidence(args)

    def test_host_kmd_and_firmware_are_explicitly_not_measured(self):
        for contract in (
            None,
            {},
            {"kmd": "not_measured"},
            {"kmd": "passed", "firmware": "not_measured"},
            {"kmd": "not_measured", "firmware": "passed"},
        ):
            with self.subTest(contract=contract):
                args = self.evidence_args()
                data = runtime()
                data["host_contract"] = contract
                args.runtime = self.write("bad-host.json", data)
                with self.assertRaises(deployment.AssetError):
                    deployment.record_evidence(args)

    def test_candidate_metadata_and_plan_repository_must_match(self):
        for field, value in (
            ("schema", True),
            ("image", IMAGE_REPOSITORY + ":latest"),
            ("image_digest", "sha256:" + "0" * 64),
            ("image_repository", "ghcr.io/other/image"),
            ("control_sha", "1" * 64),
        ):
            with self.subTest(field=field):
                args = self.evidence_args()
                data = candidate()
                data[field] = value
                args.candidate = self.write("bad-candidate.json", data)
                with self.assertRaises(deployment.AssetError):
                    deployment.record_evidence(args)
        args = self.evidence_args()
        data = candidate()
        data["image_repository"] = "ghcr.io/other/image"
        data["image"] = data["image_repository"] + "@" + IMAGE_DIGEST
        args.candidate = self.write("wrong-repository.json", data)
        with self.assertRaisesRegex(deployment.AssetError, "differs from the plan"):
            deployment.record_evidence(args)

    def test_candidate_source_commits_must_be_full_and_present(self):
        for field in ("upstream_sha", "vllm_sha"):
            for value in (None, "1" * 39, "1" * 64, "A" * 40):
                with self.subTest(field=field, value=value):
                    args = self.evidence_args()
                    data = candidate()
                    data["source"][field] = value
                    args.candidate = self.write("bad-source.json", data)
                    with self.assertRaises(deployment.AssetError):
                        deployment.record_evidence(args)

    def test_tampered_plan_is_never_recorded_as_success(self):
        args = self.evidence_args()
        data = deployment.read_json(args.plan)
        data["catalog"]["shapes"][0][0] += 1
        args.plan = self.write("tampered-plan.json", data)
        with self.assertRaisesRegex(
            deployment.AssetError, "catalog shapes must be exactly"
        ):
            deployment.record_evidence(args)
        self.assertFalse(args.output.exists())

    def test_evidence_cli_is_offline_and_does_not_echo_the_token(self):
        args = self.evidence_args()
        os.environ["TEST_IMAGE"] = IMAGE
        with mock.patch.object(deployment, "GitHub") as github:
            result = self.invoke(
                [
                    "record-evidence",
                    "--plan",
                    str(args.plan),
                    "--runtime",
                    str(args.runtime),
                    "--candidate",
                    str(args.candidate),
                    "--output",
                    str(args.output),
                ]
            )
        self.assertEqual(result, 0)
        github.assert_not_called()
        self.assertIn(f"runtime_image_digest={IMAGE_DIGEST}", self.stdout.getvalue())
        self.assertNotIn(
            ENV["GH_TOKEN"],
            self.stdout.getvalue()
            + self.stderr.getvalue()
            + args.output.read_text(encoding="ascii"),
        )

    def test_files_and_cli_help_are_ascii(self):
        self.assertTrue(TOOL.read_bytes().isascii())
        self.assertTrue(Path(__file__).read_bytes().isascii())
        with redirect_stdout(self.stdout), self.assertRaises(SystemExit) as result:
            deployment.parser().parse_args(["--help"])
        self.assertEqual(result.exception.code, 0)
        self.assertTrue(self.stdout.getvalue().isascii())
        for name in (
            "preflight",
            "authorize-pr",
            "verify-pr",
            "record-evidence",
            "validate-test-source",
        ):
            self.assertIn(name, self.stdout.getvalue())

    def test_python_311_syntax_stdlib_only_and_no_external_commands_or_mutations(self):
        for path in (TOOL, Path(__file__)):
            tree = ast.parse(path.read_text(encoding="ascii"), feature_version=(3, 11))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module.split(".")[0]]
                else:
                    continue
                self.assertTrue(
                    set(modules) <= sys.stdlib_module_names | {"nightly_assets"}
                )
        tree = ast.parse(TOOL.read_text(encoding="ascii"), feature_version=(3, 11))
        forbidden = {
            "system",
            "popen",
            "spawn",
            "mutate",
            "upload",
            "unlink",
            "rmdir",
            "remove",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, forbidden)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, {"eval", "exec", "__import__"})

    def test_generic_io_errors_do_not_echo_sensitive_exception_text(self):
        args = self.evidence_args()
        with mock.patch.object(
            deployment, "read_json", side_effect=OSError(ENV["GH_TOKEN"])
        ):
            result = self.invoke(
                [
                    "record-evidence",
                    "--plan",
                    str(args.plan),
                    "--runtime",
                    str(args.runtime),
                    "--candidate",
                    str(args.candidate),
                    "--output",
                    str(args.output),
                ]
            )
        self.assertEqual(result, 1)
        self.assertNotIn(ENV["GH_TOKEN"], self.stderr.getvalue())
        self.assertEqual(self.stdout.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
