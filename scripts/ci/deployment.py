#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed gates for the persistent self-hosted PPU transition (Python 3.11+).

Only minimum regression and preview distribution are described here. This helper
runs no external commands, installs no dependencies, and changes no GitHub state.
Authorize on a hosted job after default-branch manual dispatch and protected
reviewer approval; verify again immediately before executing trusted PR code.
Workflow/environment protection is an independent required control, not something
this read-only helper can install or attest. Runtime observations are not proof
that the regression tests passed; nightly_assets.py must attest actual results.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import os
import re
import sys
from pathlib import Path

from nightly_assets import (
    ACTIVATION_FILE,
    DIGEST,
    HASH,
    UT_IDENTITIES,
    AssetError,
    GitHub,
    hex_value,
    outputs,
    positive_int,
    read_json,
    require,
    safe_relative,
    schema_one,
    sha256_bytes,
    sha256_file,
    write_json,
)
from nightly_assets import (
    repository as validate_repository,
)

QUALITY_SCOPE = "minimum-regression"
EXECUTION_PROFILE = "persistent-selfhosted-transition"
RUNNER_LABEL = "8-ppu-m890p"
MODES = ("nightly", "preview", "dockerfile")
UNVERIFIED_COVERAGE = (
    "model_e2e",
    "accuracy",
    "performance",
    "multi_device",
    "kubernetes_isolation",
)
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
LOGIN = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}(?:\[bot\])?\Z")
HOST = re.compile(
    r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?"
    r"(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)*(?::[0-9]+)?\Z"
)
IMAGE_PATH = re.compile(r"[a-z0-9]+(?:[._-]+[a-z0-9]+)*\Z")
CATALOG_FIELDS = {
    "schema",
    "quality_scope",
    "expected_ut",
    "expected_ops_file",
    "dtypes",
    "shapes",
    "runner_label",
    "reference_model",
    "exclusions",
}
EXPECTED_UT = {
    f"{module.replace('.', '/')}.py::{name}" for module, name in UT_IDENTITIES
}
# Expanding this contract requires updating the upstream test and execution harness.
REGRESSION_DTYPES = ("float16", "bfloat16", "float32")
REGRESSION_SHAPES = ((1, 64), (7, 130), (32, 4096))
ACTIVATION_TEST = "test_silu_and_mul"


def ascii_text(value, name: str) -> str:
    require(
        isinstance(value, str)
        and bool(value.strip())
        and value.isascii()
        and all(32 <= ord(char) < 127 for char in value),
        f"{name} must be nonempty printable ASCII",
    )
    return value


def ascii_json(value):
    """Reject escaped Unicode too; ensure_ascii serialization alone is not enough."""
    if isinstance(value, str):
        require(
            value.isascii() and all(32 <= ord(char) < 127 for char in value),
            "JSON strings must be printable ASCII",
        )
    elif isinstance(value, dict):
        for key, item in value.items():
            require(isinstance(key, str), "JSON keys must be strings")
            ascii_json(key)
            ascii_json(item)
    elif isinstance(value, list):
        for item in value:
            ascii_json(item)
    else:
        require(
            value is None or type(value) in (bool, int, float),
            "invalid JSON value type",
        )


def required_env(name: str) -> str:
    return ascii_text(os.environ.get(name), name)


def enabled(name: str):
    require(os.environ.get(name) == "true", f"{name} must be explicitly true")


def registry_host(value, name: str) -> str:
    host = ascii_text(value, name).lower()
    require(
        len(host) <= 253 and HOST.fullmatch(host),
        f"{name} must be a registry host without a scheme or slash",
    )
    if ":" in host:
        port = host.rsplit(":", 1)[1]
        require(
            port == str(int(port)) and 1 <= int(port) <= 65535,
            f"{name} has an invalid port",
        )
    return host


def image_repository(value, name: str) -> str:
    value = ascii_text(value, name)
    parts = value.split("/")
    require(
        len(value) <= 256 and value == value.lower() and len(parts) >= 2,
        f"{name} must be a lowercase untagged registry/repository",
    )
    registry_host(parts[0], name)
    require(
        all(IMAGE_PATH.fullmatch(part) for part in parts[1:]),
        f"{name} must be a lowercase untagged registry/repository",
    )
    return value


def immutable_image(value, name: str) -> tuple[str, str]:
    """Return the repository (without a tag) and the pinned sha256 digest."""
    value = ascii_text(value, name)
    require(value.count("@") == 1, f"{name} must be pinned with @sha256")
    reference, digest = value.split("@")
    hex_value(digest, DIGEST, f"{name} digest")
    # tag@digest is immutable too. Strip the tag before the circular-base check.
    if ":" in reference.rsplit("/", 1)[-1]:
        reference, tag = reference.rsplit(":", 1)
        require(
            re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", tag),
            f"{name} has an invalid tag",
        )
    return image_repository(reference, name), digest


def validate_catalog(value: dict) -> dict:
    schema_one(value, "catalog")
    ascii_json(value)
    require(set(value) == CATALOG_FIELDS, "unexpected or missing catalog fields")
    require(
        value["quality_scope"] == QUALITY_SCOPE,
        "catalog quality_scope must be minimum-regression",
    )
    tests = value["expected_ut"]
    require(
        isinstance(tests, list)
        and len(tests) == 2
        and all(isinstance(test, str) for test in tests)
        and set(tests) == EXPECTED_UT,
        "catalog must select the two existing UT ids",
    )
    ops = value["expected_ops_file"]
    ascii_text(ops, "catalog expected_ops_file")
    safe_relative(ops)
    require(
        ops == ACTIVATION_FILE,
        "catalog must select the existing native activation file",
    )
    require(
        value["dtypes"] == list(REGRESSION_DTYPES),
        "catalog dtypes must be exactly [float16, bfloat16, float32] in order",
    )
    shapes = value["shapes"]
    require(
        isinstance(shapes, list)
        and shapes == [list(shape) for shape in REGRESSION_SHAPES]
        and all(type(dimension) is int for shape in shapes for dimension in shape),
        "catalog shapes must be exactly [[1, 64], [7, 130], [32, 4096]] in order",
    )
    require(
        value["runner_label"] == RUNNER_LABEL,
        "catalog runner label must be 8-ppu-m890p",
    )
    require(
        value["reference_model"] is None or value["reference_model"] == "none",
        "catalog must not claim reference-model coverage",
    )
    require(
        type(value["exclusions"]) is list and value["exclusions"] == [],
        "catalog exclusions must be an empty list",
    )
    # Preserve list order and all supplied values; only JSON object keys normalize.
    return copy.deepcopy(value)


def validate_test_source(source: Path) -> str:
    """Check only the fixed test's AST; never import or execute source code."""
    identity = f"{ACTIVATION_FILE}::{ACTIVATION_TEST}"
    path = Path(source) / ACTIVATION_FILE
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as error:
        raise AssetError("native activation test source has invalid syntax") from error
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.name == ACTIVATION_TEST
    ]
    require(
        len(functions) == 1
        and isinstance(functions[0], ast.FunctionDef)
        and any(node is functions[0] for node in tree.body),
        f"{identity} must have exactly one top-level synchronous definition",
    )
    decorators = functions[0].decorator_list
    require(
        len(decorators) == 2,
        f"{identity} must have exactly two pytest parametrize decorators",
    )
    for decorator, name, expected in zip(
        decorators,
        ("dtype_name", "shape"),
        (REGRESSION_DTYPES, REGRESSION_SHAPES),
        strict=False,
    ):
        require(
            isinstance(decorator, ast.Call)
            and len(decorator.args) == 2
            and not decorator.keywords,
            "parametrize decorators must have exactly two positional arguments",
        )
        function = decorator.func
        require(
            isinstance(function, ast.Attribute)
            and function.attr == "parametrize"
            and isinstance(function.value, ast.Attribute)
            and function.value.attr == "mark"
            and isinstance(function.value.value, ast.Name)
            and function.value.value.id == "pytest",
            "only pytest.mark.parametrize decorators are allowed",
        )
        parameter, values_node = decorator.args
        require(
            isinstance(parameter, ast.Constant) and parameter.value == name,
            "parametrize names must be dtype_name then shape",
        )
        require(
            isinstance(values_node, ast.List)
            and all(
                isinstance(node, ast.List | ast.Tuple | ast.Constant | ast.Load)
                for node in ast.walk(values_node)
            ),
            "parametrize values must be literal lists without expressions or calls",
        )
        try:
            values = ast.literal_eval(values_node)
        except (ValueError, TypeError) as error:
            raise AssetError("parametrize values must be safe literals") from error
        require(
            values == list(expected),
            f"parametrize {name} values must match the fixed regression list in order",
        )
        if name == "shape":
            require(
                all(type(dimension) is int for shape in values for dimension in shape),
                "parametrize shape dimensions must be integers",
            )
    return identity


def catalog_digest(catalog: dict) -> str:
    normalized = validate_catalog(catalog)
    return sha256_bytes(
        json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    )


def validate_plan(plan: dict) -> dict:
    schema_one(plan, "plan")
    ascii_json(plan)
    require(
        plan.get("quality_scope") == QUALITY_SCOPE
        and plan.get("execution_profile") == EXECUTION_PROFILE,
        "plan must describe minimum regression on persistent transition runners",
    )
    require(plan.get("mode") in MODES, "invalid plan mode")
    require(isinstance(plan.get("catalog"), dict), "plan catalog is required")
    catalog = validate_catalog(plan["catalog"])
    digest = hex_value(plan.get("TEST_PLAN_DIGEST"), HASH, "TEST_PLAN_DIGEST")
    require(digest == catalog_digest(catalog), "plan catalog digest mismatch")
    require(
        plan.get("unverified_coverage") == list(UNVERIFIED_COVERAGE),
        "plan must explicitly list all unverified coverage",
    )
    repository = image_repository(plan.get("image_repository"), "plan image_repository")
    base_repository, _ = immutable_image(
        plan.get("runtime_base_image"), "plan runtime_base_image"
    )
    require(
        repository != base_repository,
        "runtime base must not use the candidate image repository",
    )
    return plan


def preflight(args):
    require(args.mode in MODES, "invalid preflight mode")
    enabled("PPU_TRANSITION_ENABLED")
    if args.mode == "dockerfile":
        enabled("TRUSTED_PPU_TESTS_ENABLED")
        enabled("APPROVE_HOST_EXECUTION")
    host = registry_host(required_env("PREVIEW_REGISTRY_HOST"), "PREVIEW_REGISTRY_HOST")
    repository = image_repository(
        required_env("PREVIEW_IMAGE_REPOSITORY"), "PREVIEW_IMAGE_REPOSITORY"
    )
    require(
        repository.split("/", 1)[0] == host,
        "PREVIEW_IMAGE_REPOSITORY must belong to PREVIEW_REGISTRY_HOST",
    )
    base = required_env("RUNTIME_BASE_IMAGE")
    base_repository, _ = immutable_image(base, "RUNTIME_BASE_IMAGE")
    require(
        base_repository != repository,
        "RUNTIME_BASE_IMAGE must not use PREVIEW_IMAGE_REPOSITORY as its base",
    )
    catalog = validate_catalog(read_json(args.catalog))
    digest = catalog_digest(catalog)
    plan = {
        "schema": 1,
        "quality_scope": QUALITY_SCOPE,
        "execution_profile": EXECUTION_PROFILE,
        "mode": args.mode,
        "TEST_PLAN_DIGEST": digest,
        "catalog": catalog,
        "runtime_base_image": base,
        "image_repository": repository,
        "unverified_coverage": list(UNVERIFIED_COVERAGE),
    }
    write_json(args.output, plan)
    outputs(
        test_plan_digest=digest, runtime_base_image=base, image_repository=repository
    )
    return plan


def login(value, name: str) -> str:
    value = ascii_text(value, name)
    require(LOGIN.fullmatch(value), f"invalid {name}")
    return value


def object_value(value, name: str) -> dict:
    require(isinstance(value, dict), f"{name} must be an object")
    return value


def repository_matches(value, expected: str) -> bool:
    return isinstance(value, str) and value.lower() == expected.lower()


def require_writer(api, repository: str, user: str, role: str):
    permission = api.optional(f"/repos/{repository}/collaborators/{user}/permission")
    require(
        isinstance(permission, dict)
        and permission.get("permission") in {"write", "maintain", "admin"},
        f"PR {role} must have repository write, maintain, or admin permission",
    )
    returned_user = object_value(permission.get("user"), "permission user")
    require(
        isinstance(returned_user.get("login"), str)
        and returned_user["login"].lower() == user.lower(),
        "permission response user mismatch",
    )


def authorize_pr(api, repository: str, pr_number: int) -> dict:
    repository = validate_repository(repository)
    positive_int(pr_number, "PR number")
    actor = login(required_env("GH_ACTOR"), "GH_ACTOR")
    metadata = object_value(api.get(f"/repos/{repository}"), "repository metadata")
    require(
        repository_matches(metadata.get("full_name"), repository),
        "repository metadata mismatch",
    )
    default_branch = ascii_text(metadata.get("default_branch"), "default branch")
    require_writer(api, repository, actor, "actor")
    pr = object_value(api.get(f"/repos/{repository}/pulls/{pr_number}"), "pull request")
    require(
        type(pr.get("number")) is int and pr["number"] == pr_number,
        "PR number mismatch",
    )
    require(
        pr.get("state") == "open" and pr.get("draft") is False,
        "PR must be open and non-draft",
    )
    base = object_value(pr.get("base"), "PR base")
    head = object_value(pr.get("head"), "PR head")
    base_repo = object_value(base.get("repo"), "PR base repository")
    head_repo = object_value(head.get("repo"), "PR head repository")
    require(
        base.get("ref") == default_branch
        and repository_matches(base_repo.get("full_name"), repository),
        "PR base must be this repository's current default branch",
    )
    require(
        repository_matches(head_repo.get("full_name"), repository),
        "external or fork PRs cannot execute on persistent internal runners",
    )
    author = login(object_value(pr.get("user"), "PR user").get("login"), "PR author")
    # Same-repository heads and author_association are not permission evidence.
    require_writer(api, repository, author, "author")
    head_sha = hex_value(head.get("sha"), COMMIT, "PR head SHA")
    base_sha = hex_value(base.get("sha"), COMMIT, "PR base SHA")
    return {
        "schema": 1,
        "repository": repository,
        "pr_number": pr_number,
        "head_sha": head_sha,
        "base_sha": base_sha,
        "author": author,
        "actor": actor,
        "mode": "trusted-maintainer-manual",
    }


def verify_pr(api, args) -> dict:
    head_sha = hex_value(args.head_sha, COMMIT, "expected PR head SHA")
    base_sha = hex_value(args.base_sha, COMMIT, "expected PR base SHA")
    # A previous authorization file is deliberately never an input to this gate.
    current = authorize_pr(api, args.repository, args.pr_number)
    require(
        current["head_sha"] == head_sha and current["base_sha"] == base_sha,
        "PR head or base changed; fresh authorization and tests are required",
    )
    return current


def pr_outputs(record):
    outputs(
        pr_number=record["pr_number"],
        head_sha=record["head_sha"],
        base_sha=record["base_sha"],
    )


def record_evidence(args):
    plan = validate_plan(read_json(args.plan))
    runtime = read_json(args.runtime)
    candidate = read_json(args.candidate)
    schema_one(runtime, "runtime")
    schema_one(candidate, "candidate")
    ascii_json(runtime)
    ascii_json(candidate)
    ascii_text(runtime.get("python"), "runtime python")
    ascii_text(runtime.get("torch"), "runtime torch")
    gpu = object_value(runtime.get("gpu"), "runtime gpu")
    require(
        type(gpu.get("count")) is int and gpu["count"] == 1,
        "runtime must observe exactly one executed device",
    )
    # Accept a probe's explicit device_count too, but never allow conflicting data.
    if "device_count" in runtime:
        require(
            type(runtime["device_count"]) is int and runtime["device_count"] == 1,
            "runtime device_count must be exactly one",
        )
    host_contract = object_value(runtime.get("host_contract"), "runtime host_contract")
    require(
        host_contract.get("kmd") == "not_measured"
        and host_contract.get("firmware") == "not_measured",
        "runtime host_contract must explicitly mark kmd and firmware not_measured",
    )
    image = ascii_text(candidate.get("image"), "candidate image")
    repository, digest = immutable_image(image, "candidate image")
    require(
        candidate.get("image_repository") == repository
        and candidate.get("image_digest") == digest,
        "candidate image metadata mismatch",
    )
    require(
        repository == plan["image_repository"],
        "candidate repository differs from the plan",
    )
    require(
        runtime.get("image") == image, "runtime image differs from the candidate image"
    )
    # The non-secret TEST_IMAGE passed to the probe may also exist in this process.
    if "TEST_IMAGE" in os.environ:
        require(
            os.environ["TEST_IMAGE"] == image,
            "TEST_IMAGE differs from the candidate image",
        )
    source = object_value(candidate.get("source"), "candidate source")
    plugin_sha = hex_value(source.get("upstream_sha"), COMMIT, "plugin test commit")
    upstream_sha = hex_value(
        source.get("vllm_sha"), COMMIT, "upstream vLLM test commit"
    )
    control_sha = hex_value(candidate.get("control_sha"), COMMIT, "control SHA")
    evidence = {
        "schema": 1,
        "quality_scope": QUALITY_SCOPE,
        "execution_profile": EXECUTION_PROFILE,
        "mode": plan["mode"],
        "channel": "preview",
        "verification_state": "unverified",
        "verified": False,
        "statement": "Runtime observations only; not a verified state or a regression pass attestation.",
        "PLUGIN_TEST_COMMIT": plugin_sha,
        "UPSTREAM_VLLM_TEST_COMMIT": upstream_sha,
        "runtime_image_digest": digest,
        "image": image,
        "TEST_PLAN_DIGEST": plan["TEST_PLAN_DIGEST"],
        "control_sha": control_sha,
        "unverified_coverage": list(UNVERIFIED_COVERAGE),
        "runtime": runtime,
        "plan_sha256": sha256_file(args.plan),
        "runtime_sha256": sha256_file(args.runtime),
        "candidate_sha256": sha256_file(args.candidate),
    }
    write_json(args.output, evidence)
    outputs(
        plugin_test_commit=plugin_sha,
        upstream_vllm_test_commit=upstream_sha,
        runtime_image_digest=digest,
        test_plan_digest=plan["TEST_PLAN_DIGEST"],
        control_sha=control_sha,
        mode=plan["mode"],
    )
    return evidence


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    source = commands.add_parser(
        "validate-test-source", help="Statically validate the fixed test definition"
    )
    source.add_argument("--source", type=Path, required=True)
    gate = commands.add_parser(
        "preflight", help="Validate explicit transition configuration offline"
    )
    gate.add_argument("--mode", choices=MODES, required=True)
    gate.add_argument("--catalog", type=Path, required=True)
    gate.add_argument("--output", type=Path, required=True)
    for name in ("authorize-pr", "verify-pr"):
        command = commands.add_parser(
            name, help="Read current repository and PR permissions"
        )
        command.add_argument("--repository", required=True)
        command.add_argument("--pr-number", type=int, required=True)
        if name == "authorize-pr":
            command.add_argument("--output", type=Path, required=True)
        else:
            command.add_argument("--head-sha", required=True)
            command.add_argument("--base-sha", required=True)
    evidence = commands.add_parser(
        "record-evidence", help="Bind unverified runtime observations to a plan"
    )
    for name in ("plan", "runtime", "candidate", "output"):
        evidence.add_argument(f"--{name}", type=Path, required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "validate-test-source":
            print(f"test definition verified: {validate_test_source(args.source)}")
        elif args.command == "preflight":
            preflight(args)
        elif args.command == "record-evidence":
            record_evidence(args)
        else:
            api = (
                GitHub()
            )  # Credentials are read only from GH_TOKEN, never from CLI input.
            if args.command == "authorize-pr":
                record = authorize_pr(api, args.repository, args.pr_number)
                write_json(args.output, record)
            else:
                record = verify_pr(api, args)
            pr_outputs(record)
        return 0
    except AssetError as error:
        print(f"deployment: {error}", file=sys.stderr)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, RecursionError):
        # Do not echo file contents, environment values, API bodies, or credentials.
        print("deployment: invalid data or local I/O failure", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
