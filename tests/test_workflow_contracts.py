# SPDX-License-Identifier: Apache-2.0
"""Offline contract checks for the persistent-runner transition workflows."""

import ast
import fnmatch
import json
import os
import re
import unittest
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_ROOT = Path(os.environ.get("CI_WORKFLOW_ROOT", str(DEFAULT_ROOT)))
SOURCE_ROOT = DEFAULT_ROOT


def workflow(name):
    return (WORKFLOW_ROOT / ".github/workflows" / name).read_text(encoding="ascii")


def jobs(text):
    sections = re.split(
        r"^  ([a-z][a-z0-9-]*):\s*$", text.split("\njobs:\n", 1)[1], flags=re.M
    )
    return {
        sections[index]: sections[index + 1] for index in range(1, len(sections), 2)
    }


class TestWorkflowContracts(unittest.TestCase):
    def test_every_external_action_is_commit_pinned(self):
        for name in (
            "nightly-tests.yaml",
            "dockerfile-ci.yaml",
            "promote-nightly.yaml",
        ):
            for action in re.findall(
                r"^\s*(?:-\s*)?uses:\s*([^\s#]+)", workflow(name), flags=re.M
            ):
                self.assertRegex(
                    action, r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+@[0-9a-f]{40}$"
                )

    def test_existing_runner_labels_are_used_without_kubernetes(self):
        for name in (
            "nightly-tests.yaml",
            "dockerfile-ci.yaml",
            "promote-nightly.yaml",
        ):
            text = workflow(name)
            for label in re.findall(r"runs-on:\s*(\[[^\n]+\])", text):
                self.assertEqual(label, "[self-hosted, Linux, X64, 8-ppu-m890p]")
            self.assertNotIn("kubectl", text)
            self.assertNotIn("ppu-pr-ephemeral", text)
            self.assertNotIn("runs-on: [self-hosted, Linux, X64, build-runner]", text)

    def test_automatic_pr_only_runs_hosted_checks(self):
        definition = jobs(workflow("dockerfile-ci.yaml"))
        self.assertIn("if: github.event_name == 'pull_request'", definition["static"])
        self.assertIn("runs-on: ubuntu-latest", definition["static"])
        self.assertIn("github.event_name == 'workflow_dispatch'", definition["resolve"])
        self.assertIn("github.event.repository.default_branch", definition["resolve"])
        self.assertIn("needs: resolve", definition["build-smoke"])
        self.assertIn("environment: dockerfile-ci", definition["build-smoke"])
        self.assertIn("verify-pr", definition["build-smoke"])
        self.assertNotIn("pull_request_target:", workflow("dockerfile-ci.yaml"))
        self.assertIn("process.env.DEVICE_RESULT === 'success'", definition["report"])
        self.assertIn("pr.base.sha === process.env.BASE_SHA", definition["report"])

    def test_nightly_retests_and_missing_packages_fail(self):
        text = workflow("nightly-tests.yaml")
        self.assertIn("name: Nightly regression (minimum)", text)
        self.assertIn("--include-published", text)
        self.assertIn("if: steps.fetch.outputs.available != 'true'", text)
        self.assertIn("validate-test-source --source upstream", text)
        self.assertIn("record-evidence", text)
        self.assertIn("control/.github/vllm-main-verified.commit", text)
        self.assertNotIn("git push", text)

    def test_only_preview_distribution_is_called(self):
        text = workflow("promote-nightly.yaml")
        self.assertIn("workflows: ['Nightly regression (minimum)']", text)
        self.assertIn("check-preview", text)
        self.assertIn("publish-preview", text)
        self.assertIn("distribution-observations.json", text)
        self.assertIn("cmp ", text)
        self.assertNotIn("nightly_assets.py publish \\", text)
        self.assertNotIn("nightly_assets.py check-promotion", text)
        self.assertNotIn(":ci-${", text)

    def test_catalog_describes_only_the_frozen_minimum_cases(self):
        value = json.loads(
            (SOURCE_ROOT / ".github/ci/tests/catalog.json").read_text(encoding="ascii")
        )
        self.assertEqual(value["quality_scope"], "minimum-regression")
        self.assertEqual(value["shapes"], [[1, 64], [7, 130], [32, 4096]])
        self.assertEqual(value["dtypes"], ["float16", "bfloat16", "float32"])
        self.assertEqual(len(value["expected_ut"]), 2)
        self.assertIsNone(value["reference_model"])

    def test_cpu_merge_gate_covers_all_control_suites_without_ppu(self):
        text = workflow("ut.yaml")
        gate = jobs(text)["ci-control-tests"]
        self.assertIn("runs-on: ubuntu-latest", gate)
        self.assertIn('python-version: "3.12"', gate)
        self.assertIn("cache-mode: none", gate)
        for name in (
            "nightly_assets",
            "deployment",
            "verify_installed",
            "workflow_contracts",
        ):
            self.assertIn(f"unittest discover -s tests -p 'test_{name}.py'", gate)
        self.assertIn("validate-test-source --source .", gate)
        for name in ("run_existing_tests", "ppu_smoke"):
            self.assertIn(f"bash -n scripts/ci/{name}.sh", gate)
        for token in ("pip install", "secrets.", "self-hosted", "docker run"):
            self.assertNotIn(token, gate)
        self.assertNotIn("    if:", gate)
        self.assertIn("  pull_request:", text)
        self.assertIn("branches: [main, master]", text)
        self.assertNotIn("paths:", text.split("\njobs:\n", 1)[0])

    def test_dockerfile_path_filter_covers_preview_assets_and_contract_inputs(self):
        text = workflow("dockerfile-ci.yaml")
        paths = text.split("    paths:\n", 1)[1].split("  workflow_dispatch:", 1)[0]
        patterns = re.findall(r"^      - '([^']+)'$", paths, re.M)
        files = (
            ".github/workflows/nightly-tests.yaml",
            ".github/workflows/dockerfile-ci.yaml",
            ".github/workflows/promote-nightly.yaml",
            ".github/workflows/ut.yaml",
            ".github/workflows/release.yaml",
            ".github/ci/tests/catalog.json",
            ".github/vllm-main-verified.commit",
            "docker/Dockerfile.ci",
            "scripts/ci/nightly_assets.py",
            "scripts/ci/deployment.py",
            "scripts/ci/verify_installed.py",
            "scripts/ci/runtime_probe.py",
            "scripts/ci/run_existing_tests.sh",
            "scripts/ci/ppu_smoke.sh",
            "tests/test_nightly_assets.py",
            "tests/test_deployment.py",
            "tests/test_verify_installed.py",
            "tests/test_workflow_contracts.py",
            "tests/ut/test_patch_install.py",
            "tests/e2e/test_native_activation.py",
            "tests/support/ppu_ci.py",
            "tests/conftest.py",
            "pyproject.toml",
        )
        for name in files:
            with self.subTest(path=name):
                self.assertTrue((SOURCE_ROOT / name).is_file())
                self.assertTrue(any(fnmatch.fnmatchcase(name, p) for p in patterns))

    def test_preview_cache_is_disabled_and_publishers_are_serialized(self):
        for name in (
            "nightly-tests.yaml",
            "dockerfile-ci.yaml",
            "promote-nightly.yaml",
        ):
            header = workflow(name).split("\njobs:\n", 1)[0]
            self.assertIn("\ncache-mode: none\n", header)
            if name != "dockerfile-ci.yaml":
                self.assertIn("cancel-in-progress: false", header)
                self.assertIn("queue: max", header)
        self.assertIn(
            "group: distribute-nightly-previews", workflow("promote-nightly.yaml")
        )

    def test_pr_read_permissions_are_explicit_at_all_authorization_stages(self):
        definition = jobs(workflow("dockerfile-ci.yaml"))
        for name in ("resolve", "build-smoke", "report"):
            permissions = definition[name].split("    permissions:\n", 1)[1]
            permissions = re.split(r"^    \S", permissions, maxsplit=1, flags=re.M)[0]
            self.assertIn("      pull-requests: read", permissions)
        device = definition["build-smoke"]
        self.assertNotIn("statuses: write", device)
        self.assertNotIn("contents: write", device)
        self.assertNotIn("REGISTRY_TOKEN", device)
        self.assertNotIn("secrets: inherit", workflow("dockerfile-ci.yaml"))

    def test_preview_tags_never_enter_the_release_push_route(self):
        text = workflow("release.yaml")
        push = text.split("  push:\n", 1)[1].split("  workflow_dispatch:", 1)[0]
        patterns = re.findall(r"^      - '([^']+)'$", push, re.M)
        self.assertEqual(patterns, ["**", "!preview-*"])
        for tag, expected in (
            ("v1.2.3", True),
            ("1.2.3rc1", True),
            ("preview-baseline", False),
            ("preview-nightly-100-test-200", False),
        ):
            matched = False
            for pattern in patterns:
                if fnmatch.fnmatchcase(tag, pattern.removeprefix("!")):
                    matched = not pattern.startswith("!")
            self.assertEqual(matched, expected, tag)

    def test_nightly_consumes_the_existing_producer_without_changing_its_gate(self):
        text = workflow("nightly-tests.yaml")
        self.assertIn("--source-repository vllm-project/vllm-sail", text)
        self.assertIn("--workflow ppu-accuracy-nightly.yaml --branch main", text)
        self.assertEqual(text.count("repository: vllm-project/vllm-sail"), 2)
        self.assertNotIn("workflow_run:", text)
        producer = workflow("ppu-accuracy-nightly.yaml")
        self.assertIn("uses: ./.github/workflows/ppu-build-and-test.yaml", producer)
        self.assertIn("build_run_id: ${{ github.run_id }}", producer)
        build = workflow("_ppu-build.yaml")
        self.assertIn("name: ppu-wheels-${{ github.run_id }}", build)
        self.assertIn("path: artifacts/wheels/", build)
        self.assertNotIn("nightly-tests.yaml", workflow("ppu-build-and-test.yaml"))
        self.assertNotIn("preview-baseline", workflow("ppu-pr-ci.yaml"))

    def test_frozen_catalog_matches_current_source_test_definitions(self):
        catalog = json.loads(
            (SOURCE_ROOT / ".github/ci/tests/catalog.json").read_text()
        )
        for identity in catalog["expected_ut"]:
            relative, name = identity.split("::")
            tree = ast.parse((SOURCE_ROOT / relative).read_text())
            self.assertEqual(
                sum(
                    isinstance(node, ast.FunctionDef) and node.name == name
                    for node in tree.body
                ),
                1,
            )
        tree = ast.parse((SOURCE_ROOT / catalog["expected_ops_file"]).read_text())
        test = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "test_silu_and_mul"
        )
        parameters = {
            ast.literal_eval(d.args[0]): ast.literal_eval(d.args[1])
            for d in test.decorator_list
        }
        self.assertEqual(parameters["dtype_name"], catalog["dtypes"])
        self.assertEqual(
            [list(shape) for shape in parameters["shape"]], catalog["shapes"]
        )
        config = json.loads((SOURCE_ROOT / "scripts/ci/ppu_ops_tests.json").read_text())
        activation = next(
            entry for entry in config["test_catalog"] if entry["id"] == "activation"
        )
        self.assertEqual(activation["test"], catalog["expected_ops_file"])
        self.assertIn("ppu15", activation["boards"])
        self.assertEqual(activation["board_overrides"], {})

    def test_preview_and_existing_runtime_use_the_same_host_lock(self):
        lock = "/tmp/vllm-sail-ppu-ci.lock"
        existing = (SOURCE_ROOT / "scripts/ci/ppu_runtime_container.sh").read_text()
        self.assertIn(lock, existing)
        for name in ("run_existing_tests.sh", "ppu_smoke.sh"):
            text = (SOURCE_ROOT / "scripts/ci" / name).read_text()
            self.assertIn(lock, text)
            self.assertIn("flock -w 3600 9", text)
            self.assertIn("9>&-", text)


if __name__ == "__main__":
    unittest.main()
