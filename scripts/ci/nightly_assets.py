#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Wheel-only nightly preview distribution, using Python 3.11+ and GitHub REST.

Minimum regression evidence only: not verified, Full validation or a production
Release. The upstream pin, stable BOM and CI baseline remain unchanged.
The fixed preview release is NOT atomic: wheels are uploaded first and JSON is
the commit marker. Readers fail closed on mismatches. Serialize publishers with
workflow concurrency: GitHub releases do not provide a transactional CAS.
No commands, deletion APIs, token arguments, or third-party modules are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath

API = "https://api.github.com"
UPLOADS = "https://uploads.github.com"
MANIFEST = "wheel-manifest.json"
PACKAGES = {"vllm", "vllm-sail"}
TEST_WORKFLOW = "nightly-tests.yaml"
TEST_WORKFLOW_NAME = "Nightly regression (minimum)"
MAX_DOWNLOAD = 8 * 1024**3
MAX_UNPACKED = 16 * 1024**3
MAX_JSON = 2 * 1024**2
MAX_XML = 16 * 1024**2
MAX_PAGES = 100
READ_ATTEMPTS = 3
SHA = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
HASH = re.compile(r"[0-9a-f]{64}\Z")
DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
CANDIDATE_FIELDS = {
    "schema",
    "source",
    "image_repository",
    "image_digest",
    "image",
    "control_sha",
    "dockerfile_sha256",
}
ACTIVATION_FILE = "tests/e2e/test_native_activation.py"
ACTIVATION_NAMES = {
    f"test_silu_and_mul[shape{shape}-{dtype}]"
    for shape in range(3)
    for dtype in ("float16", "bfloat16", "float32")
}
ACTIVATION_NODES = {f"{ACTIVATION_FILE}::{name}" for name in ACTIVATION_NAMES}
UT_IDENTITIES = {
    ("tests.ut.test_patch_install", name)
    for name in ("test_patches_actually_land", "test_install_is_idempotent")
}
RUNTIME_MODULES = {
    "torch",
    "triton",
    "vllm",
    "vllm_sail",
    "vllm_sail.envs",
    "vllm.platforms",
    "vllm_sail.native.extensions",
} | {f"vllm_sail.{name}" for name in ("_C", "_moe_C", "_upstream_C", "_upstream_moe_C")}


class AssetError(Exception):
    """A fail-closed validation or transport error (never includes credentials)."""


class APIError(AssetError):
    def __init__(self, status: int):
        self.status = status
        super().__init__(f"GitHub request failed (HTTP {status})")


def require(condition, message: str):
    if not condition:
        raise AssetError(message)


def positive_int(value, name: str) -> int:
    require(type(value) is int and value > 0, f"invalid {name}")
    return value


def hex_value(value, pattern, name: str) -> str:
    require(isinstance(value, str) and pattern.fullmatch(value), f"invalid {name}")
    return value


def repository(value: str) -> str:
    require(
        isinstance(value, str)
        and re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value)
        and all(part not in {".", ".."} for part in value.split("/")),
        "invalid repository",
    )
    return value


def workflow_name(value: str) -> str:
    require(
        isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]+\.ya?ml", value),
        "workflow must be a YAML filename",
    )
    return value


def timestamp(value) -> datetime:
    require(isinstance(value, str), "missing timestamp")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise AssetError("invalid timestamp") from None
    require(result.tzinfo is not None, "timestamp requires timezone")
    return result.astimezone(timezone.utc)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_stream(stream, limit=MAX_DOWNLOAD) -> str:
    digest = hashlib.sha256()
    size = 0
    while chunk := stream.read(1024 * 1024):
        size += len(chunk)
        require(size <= limit, "file exceeds size limit")
        digest.update(chunk)
    return digest.hexdigest()


def regular_file(path: Path) -> Path:
    require(not path.is_symlink() and path.is_file(), "missing or unsafe file")
    return path


def sha256_file(path: Path) -> str:
    with regular_file(path).open("rb") as stream:
        return sha256_stream(stream)


def json_bytes(value: dict) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def decode_json(data: bytes) -> dict:
    require(len(data) <= MAX_JSON, "JSON exceeds size limit")

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def constant(_value):
        raise AssetError("non-finite JSON number")

    try:
        value = json.loads(data, object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeError, ValueError):
        raise AssetError("invalid JSON") from None
    require(isinstance(value, dict), "JSON must be an object")
    return value


def read_json(path: Path) -> dict:
    with regular_file(path).open("rb") as stream:
        return decode_json(stream.read(MAX_JSON + 1))


def schema_one(value: dict, name: str):
    require(
        type(value.get("schema")) is int and value["schema"] == 1,
        f"unsupported {name} schema",
    )


def write_json(path: Path, value: dict):
    schema_one(value, "output")
    require(not path.is_symlink(), "unsafe JSON output")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(json_bytes(value))


def outputs(**values):
    lines = []
    for key, value in values.items():
        if type(value) is bool:
            value = str(value).lower()
        value = str(value)
        require(
            re.fullmatch(r"[a-z_]+", key) and "\n" not in value and "\r" not in value,
            "unsafe output",
        )
        lines.append(f"{key}={value}\n")
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with Path(output).open("a", encoding="utf-8") as stream:
            stream.writelines(lines)
    else:
        sys.stdout.writelines(lines)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class GitHub:
    def __init__(self):
        token = os.environ.get("GH_TOKEN", "")
        require(
            bool(token) and "\n" not in token and "\r" not in token,
            "GH_TOKEN is required",
        )
        self._token = token
        self._opener = urllib.request.build_opener(NoRedirect())

    def _open_once(
        self,
        method,
        url,
        data=None,
        content_type="application/json",
        accept="application/vnd.github+json",
    ):
        original = urllib.parse.urlsplit(url)
        require(
            original.scheme == "https"
            and original.hostname in {"api.github.com", "uploads.github.com"}
            and original.port is None
            and not original.username
            and not original.password,
            "invalid GitHub endpoint",
        )
        origin = (original.scheme, original.netloc)
        authorized = True
        for _ in range(6):
            parsed = urllib.parse.urlsplit(url)
            require(
                parsed.scheme == "https"
                and parsed.hostname
                and not parsed.username
                and not parsed.password
                and parsed.port in (None, 443),
                "unsafe download redirect",
            )
            headers = {
                "Accept": accept,
                "User-Agent": "nightly-assets/1",
                "X-GitHub-Api-Version": "2022-11-28",
            }
            if authorized:
                headers["Authorization"] = f"Bearer {self._token}"
            if data is not None:
                headers["Content-Type"] = content_type
                if hasattr(data, "fileno"):
                    length = os.fstat(data.fileno()).st_size - data.tell()
                    require(0 <= length <= MAX_DOWNLOAD, "upload exceeds size limit")
                    headers["Content-Length"] = str(length)
            request = urllib.request.Request(
                url, data=data, headers=headers, method=method
            )
            try:
                return self._opener.open(request, timeout=60)
            except urllib.error.HTTPError as error:
                status = error.code
                location = error.headers.get("Location")
                error.close()
                if method == "GET" and status in {301, 302, 303, 307, 308}:
                    require(bool(location), "redirect without location")
                    url = urllib.parse.urljoin(url, location)
                    next_url = urllib.parse.urlsplit(url)
                    # Once stripped, credentials are never restored, even on a bounce.
                    authorized = (
                        authorized and (next_url.scheme, next_url.netloc) == origin
                    )
                    continue
                raise APIError(status) from None
        raise AssetError("too many download redirects")

    def _open(self, method, url, **kwargs):
        for attempt in range(READ_ATTEMPTS if method == "GET" else 1):
            try:
                return self._open_once(method, url, **kwargs)
            except APIError as error:
                if method != "GET" or error.status not in {429, 500, 502, 503, 504}:
                    raise
                if attempt + 1 == READ_ATTEMPTS:
                    raise
            except (OSError, urllib.error.URLError):
                if method != "GET" or attempt + 1 == READ_ATTEMPTS:
                    raise AssetError("GitHub transport failed") from None
            time.sleep(2**attempt)
        raise AssetError("GitHub read retries exhausted")

    @staticmethod
    def _url(path):
        require(
            path.startswith("/repos/")
            and not path.startswith("//")
            and "#" not in path,
            "invalid API path",
        )
        return API + path

    def get(self, path):
        with self._open("GET", self._url(path)) as response:
            return decode_json(response.read(MAX_JSON + 1))

    def optional(self, path):
        try:
            return self.get(path)
        except APIError as error:
            if error.status == 404:
                return None
            raise

    def pages(self, path, key=None):
        # List endpoints may be JSON arrays; do not trust the first embedded page.
        for page in range(1, MAX_PAGES + 1):
            separator = "&" if "?" in path else "?"
            url = self._url(f"{path}{separator}per_page=100&page={page}")
            with self._open("GET", url) as response:
                raw = response.read(MAX_JSON + 1)
            require(len(raw) <= MAX_JSON, "API page exceeds size limit")
            if key:
                items = decode_json(raw).get(key)
            else:
                try:
                    items = json.loads(raw)
                except (UnicodeError, ValueError):
                    raise AssetError("invalid API page") from None
            require(
                isinstance(items, list) and all(isinstance(x, dict) for x in items),
                "invalid API list",
            )
            yield from items
            if len(items) < 100:
                return
        raise AssetError("API pagination limit reached")

    def bytes(self, path, limit=MAX_JSON):
        with self._open(
            "GET", self._url(path), accept="application/octet-stream"
        ) as response:
            value = response.read(limit + 1)
        require(len(value) <= limit, "download exceeds size limit")
        return value

    def download(self, path, output: Path):
        require(
            not output.exists() and not output.is_symlink(),
            "download would overwrite file",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with self._open(
            "GET", self._url(path), accept="application/octet-stream"
        ) as response:
            length = response.headers.get("Content-Length")
            if length is not None:
                require(
                    length.isdecimal() and int(length) <= MAX_DOWNLOAD,
                    "download exceeds size limit",
                )
            size = 0
            with output.open("xb") as stream:
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    require(size <= MAX_DOWNLOAD, "download exceeds size limit")
                    stream.write(chunk)
            if length is not None:
                require(size == int(length), "truncated download")

    def asset_hash(self, repo, asset_id):
        path = f"/repos/{repo}/releases/assets/{positive_int(asset_id, 'asset id')}"
        with self._open(
            "GET", self._url(path), accept="application/octet-stream"
        ) as response:
            return sha256_stream(response)

    def mutate(self, method, path, value):
        require(method in {"POST", "PATCH"}, "unsupported mutation")
        with self._open(method, self._url(path), data=json_bytes(value)) as response:
            return decode_json(response.read(MAX_JSON + 1))

    def upload(self, repo, release_id, name, path: Path, content_type):
        # Ignore server-supplied upload_url; uploads stay on the fixed GitHub host.
        url = (
            f"{UPLOADS}/repos/{repository(repo)}/releases/"
            f"{positive_int(release_id, 'release id')}/assets?"
            + urllib.parse.urlencode({"name": name})
        )
        with regular_file(path).open("rb") as stream:
            # A file object streams the upload without loading multi-GB wheels.
            with self._open(
                "POST", url, data=stream, content_type=content_type
            ) as response:
                return decode_json(response.read(MAX_JSON + 1))


def safe_relative(name: str) -> PurePosixPath:
    require(
        isinstance(name, str)
        and name
        and "\\" not in name
        and ":" not in name
        and "\x00" not in name,
        "unsafe archive path",
    )
    path = PurePosixPath(name)
    require(
        not path.is_absolute()
        and all(p not in {"", ".", ".."} for p in name.split("/")),
        "unsafe archive path",
    )
    return path


def safe_extract(
    archive: Path, destination: Path, *, max_file=MAX_DOWNLOAD, max_total=MAX_UNPACKED
):
    """Preflight every member before writing anything; never use extractall."""
    regular_file(archive)
    require(archive.stat().st_size <= MAX_DOWNLOAD, "ZIP exceeds size limit")
    destination = (
        destination.absolute()
    )  # Preserve links for inspection, do not resolve them.
    for parent in (destination, *destination.parents):
        require(
            not parent.is_symlink() and (not parent.exists() or parent.is_dir()),
            "unsafe extraction directory ancestor",
        )
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        require(0 < len(members) <= 128, "invalid ZIP member count")
        planned = []
        seen = {}
        total = 0
        for info in members:
            original = info.orig_filename
            require(original == info.filename, "NUL in ZIP member")
            name = original[:-1] if info.is_dir() else original
            path = safe_relative(name)
            mode = info.external_attr >> 16
            kind = stat.S_IFMT(mode)
            require(
                kind in {0, stat.S_IFREG, stat.S_IFDIR}, "ZIP symlink or special file"
            )
            require(kind != stat.S_IFDIR or info.is_dir(), "invalid ZIP directory")
            require(not (info.flag_bits & 1), "encrypted ZIP is not supported")
            key = path.as_posix().casefold()
            require(key not in seen, "duplicate ZIP path")
            seen[key] = info.is_dir()
            require(info.file_size <= max_file, "ZIP member exceeds size limit")
            total += info.file_size
            require(total <= max_total, "ZIP exceeds unpacked size limit")
            target = destination.joinpath(*path.parts)
            require(
                not target.exists() and not target.is_symlink(),
                "ZIP would overwrite existing path",
            )
            for parent in target.parents:
                if parent == destination.parent:
                    break
                require(
                    not parent.is_symlink()
                    and (not parent.exists() or parent.is_dir()),
                    "unsafe ZIP parent",
                )
            planned.append((info, target))
        for key in seen:
            for parent in PurePosixPath(key).parents:
                if str(parent) != ".":
                    require(seen.get(str(parent), True), "ZIP file/directory collision")
        destination.mkdir(parents=True, exist_ok=True)
        for info, target in planned:
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            # O_EXCL rejects a concurrent file creation; O_NOFOLLOW rejects links.
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            with (
                bundle.open(info) as source,
                os.fdopen(os.open(target, flags, 0o600), "wb") as out,
            ):
                size = 0
                while chunk := source.read(1024 * 1024):
                    size += len(chunk)
                    require(
                        size <= info.file_size and size <= max_file,
                        "ZIP member exceeds declared size",
                    )
                    out.write(chunk)
                require(size == info.file_size, "truncated ZIP member")


def validate_selection(source: dict):
    schema_one(source, "selection")
    repository(source.get("upstream_repository"))
    workflow_name(source.get("upstream_workflow"))
    run_id = positive_int(source.get("upstream_run_id"), "upstream run id")
    positive_int(source.get("upstream_run_number"), "upstream run number")
    timestamp(source.get("upstream_created_at"))
    hex_value(source.get("upstream_sha"), SHA, "upstream SHA")
    hex_value(source.get("vllm_sha"), SHA, "vLLM SHA")
    positive_int(source.get("artifact_id"), "artifact id")
    require(
        source.get("artifact_name") == f"ppu-wheels-{run_id}",
        "artifact identity mismatch",
    )
    hex_value(source.get("wheel_manifest_sha256"), HASH, "manifest SHA256")
    hashes = source.get("wheel_hashes")
    require(
        isinstance(hashes, dict) and set(hashes) == PACKAGES, "invalid wheel hashes"
    )
    for value in hashes.values():
        hex_value(value, HASH, "wheel SHA256")


def validate_wheels(
    directory: Path, selection: dict | None = None, upstream_sha: str | None = None
) -> dict:
    require(directory.is_dir() and not directory.is_symlink(), "missing wheelhouse")
    manifest = read_json(directory / MANIFEST)
    schema_one(manifest, "wheel manifest")
    commits = manifest.get("commits")
    wheels = manifest.get("wheels")
    require(
        isinstance(commits, dict) and set(commits) == PACKAGES, "invalid wheel commits"
    )
    require(
        isinstance(wheels, dict) and set(wheels) == PACKAGES,
        "expected exactly two wheels",
    )
    for value in commits.values():
        hex_value(value, SHA, "wheel source commit")
    if upstream_sha is not None:
        require(commits["vllm-sail"] == upstream_sha, "vllm-sail source SHA mismatch")
    expected = {MANIFEST}
    for package, record in wheels.items():
        require(isinstance(record, dict), "invalid wheel record")
        path = safe_relative(record.get("file"))
        prefix = "vllm-" if package == "vllm" else "vllm_sail-"
        require(
            len(path.parts) == 2
            and path.parts[0] == package
            and path.name.startswith(prefix)
            and path.name.endswith(".whl"),
            "invalid wheel path or package",
        )
        require(
            isinstance(record.get("version"), str) and record["version"],
            "missing wheel version",
        )
        expected.add(path.as_posix())
        digest = hex_value(record.get("sha256"), HASH, "wheel SHA256")
        require(
            sha256_file(directory.joinpath(*path.parts)) == digest,
            "wheel SHA256 mismatch",
        )
    actual = set()
    for path in directory.rglob("*"):
        require(not path.is_symlink(), "symlink in wheelhouse")
        if path.is_dir():
            require(
                path.relative_to(directory).as_posix() in PACKAGES,
                "unexpected wheel directory",
            )
        else:
            require(path.is_file(), "special file in wheelhouse")
            actual.add(path.relative_to(directory).as_posix())
    require(
        actual == expected and len(expected) == 3,
        "unexpected or missing wheelhouse files",
    )
    if selection is not None:
        validate_selection(selection)
        require(
            commits
            == {"vllm": selection["vllm_sha"], "vllm-sail": selection["upstream_sha"]},
            "selection source SHA mismatch",
        )
        require(
            sha256_file(directory / MANIFEST) == selection["wheel_manifest_sha256"],
            "manifest SHA256 mismatch",
        )
        require(
            {p: wheels[p]["sha256"] for p in PACKAGES} == selection["wheel_hashes"],
            "selection wheel hashes mismatch",
        )
    return manifest


def validate_candidate(candidate: dict):
    schema_one(candidate, "candidate")
    require(
        set(candidate) == CANDIDATE_FIELDS, "unexpected or missing candidate fields"
    )
    require(isinstance(candidate.get("source"), dict), "missing candidate source")
    validate_selection(candidate["source"])
    repo = candidate.get("image_repository")
    require(
        isinstance(repo, str)
        and len(repo) <= 256
        and re.fullmatch(
            r"[a-z0-9][a-z0-9.-]*(?::[0-9]+)?(?:/[a-z0-9]+(?:[._-]+[a-z0-9]+)*)+", repo
        ),
        "invalid image repository (use an untagged repository)",
    )
    digest = hex_value(candidate.get("image_digest"), DIGEST, "image digest")
    require(
        candidate.get("image") == f"{repo}@{digest}", "image digest reference mismatch"
    )
    hex_value(candidate.get("control_sha"), SHA, "control SHA")
    hex_value(candidate.get("dockerfile_sha256"), HASH, "Dockerfile SHA256")


def junit_count(
    path: Path, expected: int, expected_identities: set | None = None
) -> int:
    with regular_file(path).open("rb") as stream:
        data = stream.read(MAX_XML + 1)
    require(len(data) <= MAX_XML, "unsafe or oversized JUnit XML")
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeError:
        raise AssetError("JUnit must be UTF-8") from None
    require(
        "\x00" not in text
        and "<!DOCTYPE" not in text.upper()
        and "<!ENTITY" not in text.upper(),
        "unsafe JUnit XML",
    )
    declarations = re.findall(r"<\?xml\s+.*?\?>", text, re.DOTALL | re.IGNORECASE)
    for declaration in declarations:
        encoding = re.search(
            r"\bencoding\s*=\s*(['\"])(.*?)\1", declaration, re.IGNORECASE
        )
        require(
            encoding is None or encoding[2].lower() == "utf-8", "JUnit must be UTF-8"
        )
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        raise AssetError("invalid JUnit XML") from None
    require(root.tag in {"testsuite", "testsuites"}, "invalid JUnit root")
    cases = list(root.iter("testcase"))
    require(len(cases) == expected, "JUnit testcase count mismatch")
    identities = set()
    for case in cases:
        name = case.get("name", "")
        require(bool(name), "JUnit testcase missing name")
        identity = (case.get("classname", ""), name)
        require(identity not in identities, "duplicate JUnit testcase")
        identities.add(identity)
        require(
            case.get("status", "").lower() in {"", "run", "passed", "success"},
            "JUnit testcase did not pass",
        )
    if expected_identities is not None:
        require(identities == expected_identities, "JUnit testcase identity mismatch")
    for element in root.iter():
        require(
            "}" not in element.tag and ":" not in element.tag,
            "namespaced JUnit is not supported",
        )
        require(
            element.tag not in {"failure", "error", "skipped"},
            "JUnit contains failure, error or skip",
        )
        for key in ("failures", "errors", "skipped", "skip", "disabled"):
            if key in element.attrib:
                require(
                    element.attrib[key].isdigit() and int(element.attrib[key]) == 0,
                    "JUnit reports failure, error or skip",
                )
    return len(cases)


def report_path(value, name):
    require(
        type(value) is str and value and "\x00" not in value and "\\" not in value,
        f"invalid {name} path",
    )
    path = PurePosixPath(value)
    require(
        path.is_absolute() and path.as_posix() == value and ".." not in path.parts,
        f"invalid {name} path",
    )
    return path


def validate_activation_report(report: dict):
    """Offline copy of ppu_ci.validate_node_report, pinned to the activation entry.

    Both supported boards run all nine cases; no hardware exclusions are valid.
    Paths describe the isolated test process, not files that must exist on this CPU host.
    """
    schema_one(report, "ops nodes")
    require(
        type(report.get("board")) is str
        and report["board"] in {"ppu10", "ppu15"}
        and report.get("file") == ACTIVATION_FILE,
        "ops nodes identity mismatch",
    )
    require(
        report.get("status") == "success"
        and type(report.get("exitstatus")) is int
        and report["exitstatus"] == 0
        and type(report.get("errors")) is list
        and report["errors"] == [],
        "ops nodes did not succeed",
    )
    for key in ("collected", "applicable", "hardware_excluded"):
        values = report.get(key)
        require(
            type(values) is list
            and all(type(v) is str for v in values)
            and values == sorted(set(values)),
            "invalid ops node collection",
        )
    require(
        set(report["collected"]) == ACTIVATION_NODES
        and report["collected"] == report["applicable"]
        and report["hardware_excluded"] == [],
        "ops node collection mismatch",
    )
    nodes = report.get("nodes")
    require(
        type(nodes) is dict and set(nodes) == ACTIVATION_NODES,
        "ops node phases missing or unexpected",
    )
    for phases in nodes.values():
        require(
            type(phases) is dict and set(phases) == {"setup", "call", "teardown"},
            "missing ops node execution phase",
        )
        for phase in phases.values():
            require(
                type(phase) is dict
                and phase.get("outcome") == "passed"
                and "wasxfail" in phase
                and phase["wasxfail"] is None
                and phase.get("reason") is None,
                "ops node did not normally pass",
            )
            duration = phase.get("duration_seconds")
            require(type(duration) in {int, float}, "invalid ops node duration")
            try:
                valid_duration = math.isfinite(duration) and duration >= 0
            except OverflowError:
                valid_duration = False
            require(valid_duration, "invalid ops node duration")
    runtime = report.get("runtime")
    require(
        type(runtime) is dict and type(runtime.get("paths")) is dict,
        "missing installed runtime evidence",
    )
    paths = runtime["paths"]
    require(
        RUNTIME_MODULES | {"plugin", "conftest"} <= paths.keys(),
        "missing installed runtime paths",
    )
    isolated = report_path(runtime.get("isolated_root"), "isolated root")
    checkout = report_path(runtime.get("checkout_root"), "checkout root")
    require(not isolated.is_relative_to(checkout), "invalid isolated root")
    for name in RUNTIME_MODULES:
        path = report_path(paths[name], name)
        require(
            not any(path.is_relative_to(root) for root in (isolated, checkout)),
            "runtime package is source-shadowed",
        )
    for name, suffix in (
        ("plugin", "tests/support/ppu_ci.py"),
        ("conftest", "tests/conftest.py"),
    ):
        require(paths[name] == str(isolated / suffix), "test helper is source-shadowed")
    env = runtime.get("runtime_env")
    require(
        type(env) is dict
        and set(env) == {"VLLM_SAIL_USE_PLA"}
        and env["VLLM_SAIL_USE_PLA"] is True,
        "invalid PLA runtime evidence",
    )


def validate_results(directory: Path) -> tuple[dict, dict]:
    require(
        directory.is_dir() and not directory.is_symlink(), "missing or unsafe results"
    )
    counts = {
        "ut": junit_count(directory / "ut.xml", 2, UT_IDENTITIES),
        "ops": junit_count(
            directory / "ops.xml",
            9,
            {("tests.e2e.test_native_activation", name) for name in ACTIVATION_NAMES},
        ),
    }
    validate_activation_report(read_json(directory / "nodes.json"))
    hashes = {
        name: sha256_file(directory / name)
        for name in ("ut.xml", "ops.xml", "nodes.json")
    }
    return counts, hashes


def validate_evidence(evidence: dict, candidate_hash: str, test_run_id: int):
    schema_one(evidence, "evidence")
    require(evidence.get("status") == "passed", "evidence did not pass")
    positive_int(evidence.get("test_run_id"), "evidence test run id")
    require(evidence["test_run_id"] == test_run_id, "evidence test run mismatch")
    require(
        evidence.get("candidate_sha256") == candidate_hash,
        "candidate evidence SHA256 mismatch",
    )
    require(
        evidence.get("counts") == {"ut": 2, "ops": 9}
        and all(type(v) is int for v in evidence["counts"].values()),
        "invalid evidence counts",
    )
    hashes = evidence.get("results")
    require(
        isinstance(hashes, dict) and set(hashes) == {"ut.xml", "ops.xml", "nodes.json"},
        "missing result hashes",
    )
    for value in hashes.values():
        hex_value(value, HASH, "result SHA256")


def validate_baseline(baseline: dict) -> dict:
    schema_one(baseline, "baseline")
    require(
        baseline.get("quality_scope") == "minimum-regression"
        and baseline.get("channel") == "preview",
        "invalid preview quality scope or channel",
    )
    # Distribution labels are not candidate inputs; reconstruct the original hash.
    candidate = {
        "schema": 1,
        **{
            key: baseline.get(key)
            for key in (
                "source",
                "image_repository",
                "image_digest",
                "image",
                "control_sha",
                "dockerfile_sha256",
            )
        },
    }
    validate_candidate(candidate)
    test_id = positive_int(baseline.get("test_run_id"), "baseline test run id")
    require(isinstance(baseline.get("evidence"), dict), "missing baseline evidence")
    validate_evidence(
        baseline["evidence"], sha256_bytes(json_bytes(candidate)), test_id
    )
    require(
        baseline.get("immutable_release_tag")
        == f"preview-nightly-{candidate['source']['upstream_run_id']}-test-{test_id}",
        "invalid immutable release identity",
    )
    return candidate


def release_assets(api, repo, release):
    release_id = positive_int(release.get("id"), "release id")
    return list(api.pages(f"/repos/{repo}/releases/{release_id}/assets"))


def asset_record(assets, name, *, optional=False):
    matches = [asset for asset in assets if asset.get("name") == name]
    require(len(matches) <= 1, "duplicate release asset")
    if not matches and optional:
        return None
    require(len(matches) == 1, "required release asset is missing")
    asset = matches[0]
    positive_int(asset.get("id"), "asset id")
    require(
        asset.get("state") in {"uploaded", "starter"}, "invalid release asset state"
    )
    return asset


def named_asset(assets, name, *, optional=False):
    asset = asset_record(assets, name, optional=optional)
    if asset is not None:
        require(asset.get("state") == "uploaded", "release asset is not uploaded")
    return asset


def validate_preview_release(release):
    require(
        re.fullmatch(
            r"preview-(?:baseline|nightly-[1-9][0-9]*-test-[1-9][0-9]*)",
            str(release.get("tag_name", "")),
        ),
        "invalid preview release tag",
    )
    require(release.get("prerelease") is True, "preview release must be a prerelease")
    require(
        isinstance(release.get("name"), str) and release["name"].startswith("Preview "),
        "preview release title must explicitly identify Preview",
    )


def validate_publisher_release(release, repo, tag):
    """Bind a release to its full REST identity, not a guessed tag commit."""
    require(isinstance(release, dict), "invalid release record")
    release_id = positive_int(release.get("id"), "release id")
    url = release.get("url")
    require(
        isinstance(url, str)
        and url.casefold()
        == f"{API}/repos/{repository(repo)}/releases/{release_id}".casefold(),
        "release repository or id mismatch",
    )
    require(release.get("tag_name") == tag, "release tag mismatch")
    hex_value(release.get("target_commitish"), SHA, "release target_commitish")
    require(type(release.get("draft")) is bool, "invalid release draft state")
    validate_preview_release(release)
    if release["draft"] is False:
        timestamp(release.get("published_at"))


def publisher_lookup_release(api, repo, tag):
    """Look up published releases or drafts visible to a writer-authorized token.

    The tag endpoint never returns drafts. Scan every list page after a 404 and
    require a unique match before using it; reader baseline lookup stays separate.
    """
    repository(repo)
    release = api.optional(f"/repos/{repo}/releases/tags/{tag}")
    if release is None:
        matches = [
            record
            for record in api.pages(f"/repos/{repo}/releases")
            if record.get("tag_name") == tag
        ]
        require(len(matches) <= 1, "duplicate matching release tag")
        if not matches:
            return None
        release = matches[0]
    validate_publisher_release(release, repo, tag)
    return release


def load_baseline(api, repo):
    repository(repo)
    release = api.optional(f"/repos/{repo}/releases/tags/preview-baseline")
    if release is None:
        return None, None, []
    validate_preview_release(release)
    require(
        release.get("tag_name") == "preview-baseline"
        and release.get("draft") is False
        and release.get("published_at"),
        "preview-baseline is not published",
    )
    timestamp(release["published_at"])
    assets = release_assets(api, repo, release)
    named_asset(assets, "preview-wheels.zip")
    marker = named_asset(assets, "preview-baseline.json")
    baseline = decode_json(api.bytes(f"/repos/{repo}/releases/assets/{marker['id']}"))
    validate_baseline(baseline)
    return baseline, release, assets


def recover_baseline(api, repo, bundle, candidate, evidence):
    """Publisher-only recovery; never reinterpret an incomplete fixed release as absent.

    An archived marker is a non-downgrade fence. Verify its published immutable
    release and wheel bytes BEFORE any repair. On a markerless first draft, use
    the highest published immutable identity, never the retry's incoming run.
    This repair is itself non-atomic and safely repeatable by serialized publishers.
    """
    fixed = publisher_lookup_release(api, repo, "preview-baseline")
    if fixed is None:
        return None, None, []
    records = release_assets(api, repo, fixed)
    marker = asset_record(records, "preview-baseline.json", optional=True)
    asset_record(records, "preview-wheels.zip", optional=True)
    # Uploaded state alone is not completeness: a new wheel ZIP may coexist with
    # the previous JSON marker after interruption. Verify the marker's immutable
    # release and restore its exact pair below. Healthy byte-identical assets are
    # reused by upload_asset without mutation.
    fences = []
    if marker and marker["state"] == "uploaded":
        fences.append(
            decode_json(api.bytes(f"/repos/{repo}/releases/assets/{marker['id']}"))
        )
    else:
        for record in records:
            name = record.get("name", "")
            if (
                isinstance(name, str)
                and name.startswith("archive-")
                and name.endswith("-preview-baseline.json")
            ):
                positive_int(record.get("id"), "archived marker id")
                require(
                    record.get("state") in {"uploaded", "starter"},
                    "invalid archived marker state",
                )
                if record["state"] == "uploaded":
                    fences.append(
                        decode_json(
                            api.bytes(f"/repos/{repo}/releases/assets/{record['id']}")
                        )
                    )
    for fence in fences:
        validate_baseline(fence)
        require(
            fence["source"]["upstream_repository"].casefold()
            == candidate["source"]["upstream_repository"].casefold()
            and fence["source"]["upstream_workflow"]
            == candidate["source"]["upstream_workflow"],
            "recovery upstream identity mismatch",
        )
    if fences:
        restore = max(
            fences, key=lambda b: (b["source"]["upstream_run_id"], b["test_run_id"])
        )
        require(
            all(
                b == restore
                for b in fences
                if b["source"]["upstream_run_id"]
                == restore["source"]["upstream_run_id"]
                and b["test_run_id"] == restore["test_run_id"]
            ),
            "conflicting recovery markers",
        )
    else:
        require(
            fixed["draft"], "missing recovery marker; refusing to reset baseline fence"
        )
        # A first draft may have no marker at all. The caller's bundle is NOT a
        # fence: a later retry may carry an older or newer run. Find the highest
        # published immutable identity, then verify it below without falling back.
        releases = []
        for release in api.pages(f"/repos/{repo}/releases"):
            identity = re.fullmatch(
                r"preview-nightly-([1-9][0-9]*)-test-([1-9][0-9]*)",
                str(release.get("tag_name", "")),
            )
            if (
                identity
                and release.get("draft") is False
                and release.get("published_at")
            ):
                releases.append(((int(identity[1]), int(identity[2])), release))
        require(bool(releases), "recovery requires a published immutable release")
        identity, latest = max(releases, key=lambda item: item[0])
        validate_publisher_release(latest, repo, latest["tag_name"])
        latest_marker = named_asset(
            release_assets(api, repo, latest), "preview-baseline.json"
        )
        restore = decode_json(
            api.bytes(f"/repos/{repo}/releases/assets/{latest_marker['id']}")
        )
        validate_baseline(restore)
        require(
            restore["immutable_release_tag"] == latest["tag_name"]
            and (restore["source"]["upstream_run_id"], restore["test_run_id"])
            == identity,
            "draft recovery immutable identity mismatch",
        )
    validate_baseline(restore)
    require(
        restore["source"]["upstream_repository"].casefold()
        == candidate["source"]["upstream_repository"].casefold()
        and restore["source"]["upstream_workflow"]
        == candidate["source"]["upstream_workflow"],
        "recovery upstream identity mismatch",
    )
    tag = restore["immutable_release_tag"]
    unique = publisher_lookup_release(api, repo, tag)
    require(
        unique is not None
        and unique.get("tag_name") == tag
        and unique.get("draft") is False
        and unique.get("published_at")
        and unique.get("target_commitish") == restore["control_sha"],
        "recovery requires a published immutable release",
    )
    validate_preview_release(unique)
    timestamp(unique["published_at"])
    require(
        api.optional(f"/repos/{repo}/git/ref/tags/{tag}") is not None,
        "recovery immutable tag is missing",
    )
    verify_tag(api, repo, tag, restore["control_sha"])
    immutable_records = release_assets(api, repo, unique)
    immutable_marker = named_asset(immutable_records, "preview-baseline.json")
    remote = decode_json(
        api.bytes(f"/repos/{repo}/releases/assets/{immutable_marker['id']}")
    )
    validate_baseline(remote)
    require(remote == restore, "recovery immutable marker mismatch")
    immutable_wheels = named_asset(immutable_records, "preview-wheels.zip")
    work = bundle / f"baseline-recovery-{time.time_ns()}"
    archive = work / "preview-wheels.zip"
    api.download(f"/repos/{repo}/releases/assets/{immutable_wheels['id']}", archive)
    safe_extract(archive, work / "wheelhouse")
    validate_wheels(work / "wheelhouse", restore["source"])
    require(
        sha256_file(archive) == api.asset_hash(repo, immutable_wheels["id"]),
        "recovery immutable wheels changed",
    )
    restored_marker = work / "preview-baseline.json"
    write_json(restored_marker, restore)
    upload_asset(
        api,
        repo,
        fixed,
        "preview-wheels.zip",
        archive,
        "application/zip",
        immutable=False,
    )
    upload_asset(
        api,
        repo,
        fixed,
        "preview-baseline.json",
        restored_marker,
        "application/json",
        immutable=False,
    )
    publish_release(api, repo, fixed)
    installed, release, records = load_baseline(api, repo)
    require(
        installed == restore
        and api.asset_hash(repo, named_asset(records, "preview-wheels.zip")["id"])
        == sha256_file(archive),
        "baseline recovery verification failed",
    )
    return installed, release, records


def trusted_run(api, repo, workflow, run_id, branch=None):
    repository(repo)
    workflow_name(workflow)
    positive_int(run_id, "run id")
    info = api.get(f"/repos/{repo}")
    require(
        str(info.get("full_name", "")).casefold() == repo.casefold(),
        "repository mismatch",
    )
    default = info.get("default_branch")
    require(isinstance(default, str) and default, "missing default branch")
    if branch is not None:
        require(branch == default, "branch is not the repository default")
    definition = api.get(f"/repos/{repo}/actions/workflows/{workflow}")
    path = f".github/workflows/{workflow}"
    require(definition.get("path") == path, "workflow path mismatch")
    if workflow == TEST_WORKFLOW:
        require(
            definition.get("name") == TEST_WORKFLOW_NAME, "test workflow name mismatch"
        )
    workflow_id = positive_int(definition.get("id"), "workflow id")
    run = api.get(f"/repos/{repo}/actions/runs/{run_id}")
    if workflow == TEST_WORKFLOW:
        require(run.get("name") == TEST_WORKFLOW_NAME, "test run name mismatch")
    require(type(run.get("id")) is int and run["id"] == run_id, "run identity mismatch")
    require(
        run.get("status") == "completed" and run.get("conclusion") == "success",
        "run is not completed success",
    )
    require(
        run.get("event") in {"schedule", "workflow_dispatch"}, "untrusted run event"
    )
    require(run.get("head_branch") == default, "run is not on default branch")
    for key in ("repository", "head_repository"):
        record = run.get(key)
        require(
            isinstance(record, dict)
            and str(record.get("full_name", "")).casefold() == repo.casefold(),
            "run repository mismatch",
        )
    require(
        type(run.get("workflow_id")) is int
        and run["workflow_id"] == workflow_id
        and run.get("path") == path,
        "run workflow mismatch",
    )
    hex_value(run.get("head_sha"), SHA, "run head SHA")
    timestamp(run.get("created_at"))
    positive_int(run.get("run_number"), "run number")
    return run


def fetch_nightly(api, args):
    repo = repository(args.source_repository)
    consumer = repository(args.consumer_repository)
    workflow = workflow_name(args.workflow)
    baseline, _, _ = load_baseline(api, consumer)
    current = baseline["source"] if baseline else None
    if current:
        require(
            current["upstream_repository"].casefold() == repo.casefold()
            and current["upstream_workflow"] == workflow,
            "baseline upstream identity mismatch",
        )
    now = datetime.now(timezone.utc)
    if args.run_id is not None:
        run = trusted_run(api, repo, workflow, args.run_id, args.branch)
    else:
        info = api.get(f"/repos/{repo}")
        require(
            str(info.get("full_name", "")).casefold() == repo.casefold()
            and info.get("default_branch") == args.branch,
            "source default branch mismatch",
        )
        definition = api.get(f"/repos/{repo}/actions/workflows/{workflow}")
        require(
            definition.get("path") == f".github/workflows/{workflow}",
            "workflow path mismatch",
        )
        workflow_id = positive_int(definition.get("id"), "workflow id")
        path = (
            f"/repos/{repo}/actions/workflows/{workflow_id}/runs?"
            + urllib.parse.urlencode({"branch": args.branch, "status": "success"})
        )
        runs = []
        for item in api.pages(path, "workflow_runs"):
            if (
                item.get("status") == "completed"
                and item.get("conclusion") == "success"
                and item.get("event") in {"schedule", "workflow_dispatch"}
                and item.get("head_branch") == args.branch
                and item.get("workflow_id") == workflow_id
                and item.get("path") == definition["path"]
                and isinstance(item.get("head_repository"), dict)
                and str(item["head_repository"].get("full_name", "")).casefold()
                == repo.casefold()
                and isinstance(item.get("repository"), dict)
                and str(item["repository"].get("full_name", "")).casefold()
                == repo.casefold()
            ):
                positive_int(item.get("id"), "run id")
                runs.append(item)
        if not runs:
            outputs(available=False)
            return
        latest = max(runs, key=lambda r: (timestamp(r.get("created_at")), r["id"]))
        run = trusted_run(api, repo, workflow, latest["id"], args.branch)
    created = timestamp(run["created_at"])
    fresh = now - timedelta(hours=48) <= created <= now + timedelta(minutes=5)
    if not fresh:
        require(
            args.run_id is None, "manual run is outside the 48-hour freshness window"
        )
        outputs(available=False)
        return
    # Daily minimum regression may retest the latest already-distributed run.
    # Only dedup is bypassed: trust, freshness and expiration checks still apply.
    if (
        not args.include_published
        and current
        and (
            run["id"] <= current["upstream_run_id"]
            or created < timestamp(current["upstream_created_at"])
        )
    ):
        outputs(available=False)
        return
    artifacts = list(
        api.pages(f"/repos/{repo}/actions/runs/{run['id']}/artifacts", "artifacts")
    )
    matches = [
        a
        for a in artifacts
        if a.get("name") == f"ppu-wheels-{run['id']}"
        and a.get("expired") is False
        and (not a.get("expires_at") or timestamp(a["expires_at"]) > now)
    ]
    require(len(matches) <= 1, "duplicate upstream wheel artifact")
    if not matches:
        require(args.run_id is None, "manual run has no unexpired wheel artifact")
        outputs(available=False)
        return
    artifact = matches[0]
    artifact_id = positive_int(artifact.get("id"), "artifact id")
    associated = artifact.get("workflow_run")
    if associated is not None:
        require(
            isinstance(associated, dict)
            and associated.get("id") == run["id"]
            and associated.get("head_sha") == run["head_sha"],
            "artifact run identity mismatch",
        )
    archive = args.output / "upstream-wheels.zip"
    try:
        api.download(f"/repos/{repo}/actions/artifacts/{artifact_id}/zip", archive)
    except APIError as error:
        if error.status in {404, 410} and args.run_id is None:
            outputs(available=False)
            return
        raise
    safe_extract(archive, args.output / "wheelhouse")
    manifest = validate_wheels(args.output / "wheelhouse", upstream_sha=run["head_sha"])
    source = {
        "schema": 1,
        "upstream_repository": repo,
        "upstream_workflow": workflow,
        "upstream_run_id": run["id"],
        "upstream_run_number": run["run_number"],
        "upstream_created_at": run["created_at"],
        "upstream_sha": run["head_sha"],
        "vllm_sha": manifest["commits"]["vllm"],
        "artifact_id": artifact_id,
        "artifact_name": artifact["name"],
        "wheel_manifest_sha256": sha256_file(args.output / "wheelhouse" / MANIFEST),
        "wheel_hashes": {p: manifest["wheels"][p]["sha256"] for p in sorted(PACKAGES)},
    }
    validate_selection(source)
    write_json(args.output / "selection.json", source)
    outputs(
        available=True, upstream_sha=source["upstream_sha"], upstream_run_id=run["id"]
    )


def fetch_preview(api, args):
    repo = repository(args.consumer_repository)
    baseline, _, assets = load_baseline(api, repo)
    require(
        baseline is not None,
        "published preview-baseline is required; no source-build fallback",
    )
    asset = named_asset(assets, "preview-wheels.zip")
    archive = args.output / "preview-wheels.zip"
    api.download(f"/repos/{repo}/releases/assets/{asset['id']}", archive)
    safe_extract(archive, args.output / "wheelhouse")
    validate_wheels(args.output / "wheelhouse", baseline["source"])
    write_json(args.output / "selection.json", baseline["source"])
    write_json(args.output / "baseline.json", baseline)
    outputs(
        upstream_sha=baseline["source"]["upstream_sha"],
        upstream_run_id=baseline["source"]["upstream_run_id"],
        base_image=baseline["image"],
    )


def candidate_command(args):
    source = read_json(args.selection)
    validate_wheels(args.wheels, source)
    metadata = read_json(args.build_metadata)
    digest = hex_value(
        metadata.get("containerimage.digest"), DIGEST, "build image digest"
    )
    candidate = {
        "schema": 1,
        "source": source,
        "image_repository": args.image_repository,
        "image_digest": digest,
        "image": f"{args.image_repository}@{digest}",
        "control_sha": args.control_sha,
        "dockerfile_sha256": sha256_file(args.dockerfile),
    }
    validate_candidate(candidate)
    write_json(args.output, candidate)
    outputs(image=candidate["image"])


def attest_command(args):
    candidate = read_json(args.candidate)
    validate_candidate(candidate)
    validate_wheels(args.wheels, candidate["source"])
    positive_int(args.test_run_id, "test run id")
    counts, hashes = validate_results(args.results)
    evidence = {
        "schema": 1,
        "status": "passed",
        "test_run_id": args.test_run_id,
        "candidate_sha256": sha256_file(args.candidate),
        "results": hashes,
        "counts": counts,
    }
    validate_evidence(evidence, sha256_file(args.candidate), args.test_run_id)
    write_json(args.output, evidence)


def check_bundle(bundle: Path, test_run_id: int):
    positive_int(test_run_id, "test run id")
    candidate = read_json(bundle / "candidate.json")
    validate_candidate(candidate)
    # Candidate CLI output is canonical, so the baseline can reconstruct its hash.
    require(
        sha256_file(bundle / "candidate.json") == sha256_bytes(json_bytes(candidate)),
        "candidate must use canonical CLI serialization",
    )
    manifest = validate_wheels(bundle / "wheelhouse", candidate["source"])
    if (bundle / "selection.json").exists():
        require(
            read_json(bundle / "selection.json") == candidate["source"],
            "bundle selection mismatch",
        )
    evidence = read_json(bundle / "test-evidence.json")
    validate_evidence(evidence, sha256_file(bundle / "candidate.json"), test_run_id)
    counts, hashes = validate_results(bundle / "results")
    require(
        evidence["counts"] == counts and evidence["results"] == hashes,
        "test result evidence SHA256 mismatch",
    )
    return candidate, evidence, manifest


def promotion_check(api, repo, bundle: Path, test_run_id: int, *, publisher=False):
    repository(repo)
    candidate, evidence, manifest = check_bundle(bundle, test_run_id)
    run = trusted_run(api, repo, TEST_WORKFLOW, test_run_id)
    require(
        candidate["control_sha"] == run["head_sha"],
        "candidate control SHA differs from test run",
    )
    baseline, _, _ = (
        recover_baseline(api, repo, bundle, candidate, evidence)
        if publisher
        else load_baseline(api, repo)
    )
    if baseline:
        current = baseline["source"]
        incoming = candidate["source"]
        require(
            current["upstream_repository"].casefold()
            == incoming["upstream_repository"].casefold()
            and current["upstream_workflow"] == incoming["upstream_workflow"],
            "promotion upstream identity mismatch",
        )
        promote = incoming["upstream_run_id"] > current["upstream_run_id"]
    else:
        promote = True
    return promote, candidate, evidence, manifest


def promotion_outputs(promote, candidate):
    values = {"promote": promote}
    if promote:
        values.update(
            {
                key: candidate[key]
                for key in ("image", "image_repository", "image_digest")
            }
        )
        values["upstream_run_id"] = candidate["source"]["upstream_run_id"]
    outputs(**values)


def pack_wheels(wheelhouse: Path, archive: Path, manifest: dict, source: dict):
    validate_wheels(wheelhouse, source)
    require(
        not archive.exists() and not archive.is_symlink(),
        "wheel ZIP would overwrite file",
    )
    with zipfile.ZipFile(
        archive, "x", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as bundle:
        for name in sorted(
            [MANIFEST, *(record["file"] for record in manifest["wheels"].values())]
        ):
            # Deterministic metadata makes immutable upload retries byte-identical.
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            info.compress_type = zipfile.ZIP_STORED
            digest = hashlib.sha256()
            with (
                regular_file(wheelhouse / name).open("rb") as src,
                bundle.open(info, "w", force_zip64=True) as out,
            ):
                while chunk := src.read(1024 * 1024):
                    digest.update(chunk)
                    out.write(chunk)
            expected = (
                source["wheel_manifest_sha256"]
                if name == MANIFEST
                else next(
                    record["sha256"]
                    for record in manifest["wheels"].values()
                    if record["file"] == name
                )
            )
            require(digest.hexdigest() == expected, "wheel bytes changed while packing")


def verify_tag(api, repo, tag, control_sha):
    ref = api.optional(f"/repos/{repo}/git/ref/tags/{tag}")
    if ref is None:
        return
    obj = ref.get("object")
    for _ in range(5):
        require(isinstance(obj, dict), "invalid release tag")
        if obj.get("type") == "commit":
            require(obj.get("sha") == control_sha, "release tag control SHA mismatch")
            return
        require(obj.get("type") == "tag", "release tag does not point to a commit")
        tag_sha = hex_value(obj.get("sha"), SHA, "tag SHA")
        obj = api.get(f"/repos/{repo}/git/tags/{tag_sha}").get("object")
    raise AssetError("release tag nesting limit reached")


def ensure_release(api, repo, tag, control_sha, *, immutable):
    expected = (
        re.fullmatch(r"preview-nightly-[1-9][0-9]*-test-[1-9][0-9]*", tag)
        if immutable
        else tag == "preview-baseline"
    )
    require(expected, "only preview release tags may be used")
    repository(repo)
    hex_value(control_sha, SHA, "control SHA")
    commit = api.get(f"/repos/{repo}/commits/{control_sha}")
    require(
        commit.get("sha") == control_sha, "control SHA is not an existing fork commit"
    )
    if immutable:
        verify_tag(api, repo, tag, control_sha)
    release = publisher_lookup_release(api, repo, tag)
    if release is None:
        release = api.mutate(
            "POST",
            f"/repos/{repo}/releases",
            {
                "tag_name": tag,
                "target_commitish": control_sha,
                "name": f"Preview {tag} (minimum regression)",
                "draft": True,
                "prerelease": True,
                "body": (
                    "Preview wheels and image digest with minimum regression evidence. "
                    "Not verified; not Full validation or a production Release. Upstream pin unchanged. "
                    + (
                        "Immutable Preview release."
                        if immutable
                        else "Non-atomic preview baseline: wheels first, preview-baseline.json last "
                        "as commit marker. Archive assets are retained. Restore from an immutable "
                        "preview-nightly release with publishers paused."
                    )
                ),
            },
        )
    validate_publisher_release(release, repo, tag)
    if immutable:
        require(
            release.get("target_commitish") == control_sha,
            "immutable release control SHA mismatch",
        )
        verify_tag(api, repo, tag, control_sha)
    return release


def upload_asset(api, repo, release, name, path, content_type, *, immutable):
    assets = release_assets(api, repo, release)
    existing = asset_record(assets, name, optional=True)
    local_hash = sha256_file(path)
    if existing:
        if (
            existing["state"] == "uploaded"
            and api.asset_hash(repo, existing["id"]) == local_hash
        ):
            return
        require(
            not immutable or release.get("draft") is True,
            "immutable release asset hash conflict",
        )
        # A 502 can leave a starter. Only draft immutable releases may be repaired.
        # No DELETE: retain previous bytes under a unique, human-readable name.
        archive_name = f"archive-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{time.time_ns()}-{name}"
        api.mutate(
            "PATCH",
            f"/repos/{repo}/releases/assets/{existing['id']}",
            {"name": archive_name},
        )
    elif immutable:
        require(
            release.get("draft") is True,
            "published immutable release has missing assets",
        )
    asset = api.upload(repo, release["id"], name, path, content_type)
    positive_int(asset.get("id"), "uploaded asset id")
    require(
        asset.get("name") == name and asset.get("state") == "uploaded",
        "upload did not complete",
    )
    require(
        api.asset_hash(repo, asset["id"]) == local_hash,
        "uploaded asset SHA256 mismatch",
    )


def publish_release(api, repo, release):
    validate_preview_release(release)
    if release.get("draft") is True:
        release = api.mutate(
            "PATCH",
            f"/repos/{repo}/releases/{release['id']}",
            {
                "draft": False,
                "prerelease": True,
                "name": release["name"],
            },
        )
    validate_preview_release(release)
    require(
        release.get("draft") is False and release.get("published_at"),
        "release publication failed",
    )
    return release


def step_summary(repo, baseline):
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    source = baseline["source"]
    tag = baseline["immutable_release_tag"]
    text = (
        "## Nightly Preview published (minimum regression)\n\n"
        "Preview channel; quality_scope: minimum-regression. This is **not verified**, "
        "not Full validation and not a production Release. Evidence covers only "
        "2 unit tests and all 9 activation cases, with the candidate Dockerfile hash "
        "and image digest recorded. Policy: **upstream pin unchanged**; stable verified "
        "pin, BOM and CI baseline remain unchanged.\n\n"
        f"- Source run: https://github.com/{source['upstream_repository']}/actions/runs/{source['upstream_run_id']}\n"
        f"- Source SHA: `{source['upstream_sha']}`; vLLM SHA: `{source['vllm_sha']}`\n"
        f"- Image digest: `{baseline['image']}`\n"
        f"- Tested run: https://github.com/{repo}/actions/runs/{baseline['test_run_id']}\n"
        f"- Immutable Preview release: https://github.com/{repo}/releases/tag/{tag}\n"
        f"- Fixed Preview release: https://github.com/{repo}/releases/tag/preview-baseline\n\n"
        "The fixed release update is **non-atomic** (not atomic): wheels first, JSON last "
        "as commit marker; readers reject any transient hash mismatch. All archive "
        "assets are **retained**, never deleted. A single publisher must be serialized.\n\n"
        "Daily regression may retest an already-distributed upstream run within the "
        "48-hour freshness window. A same-run or older-run preview is a publication "
        "no-op: the fixed pointer does not move and never downgrades.\n\n"
        "Rollback (Preview only): use a prior immutable preview-nightly release's "
        "preview-wheels.zip, source/evidence and image digest. With publishers paused, "
        "archive (rename, never delete) the fixed assets, restore its wheel ZIP first "
        "and preview-baseline.json last. Rollback is an explicit operator action.\n"
    )
    with Path(path).open("a", encoding="ascii") as stream:
        stream.write(text)


def make_publication_baseline(candidate, evidence):
    baseline = {
        "schema": 1,
        "quality_scope": "minimum-regression",
        "channel": "preview",
        **{
            key: candidate[key]
            for key in (
                "source",
                "image",
                "image_repository",
                "image_digest",
                "control_sha",
                "dockerfile_sha256",
            )
        },
        "test_run_id": evidence["test_run_id"],
        "evidence": evidence,
        "immutable_release_tag": f"preview-nightly-{candidate['source']['upstream_run_id']}-test-{evidence['test_run_id']}",
    }
    validate_baseline(baseline)
    return baseline


def publish(api, args):
    repo = repository(args.consumer_repository)
    promote, candidate, evidence, manifest = promotion_check(
        api, repo, args.bundle, args.test_run_id, publisher=True
    )
    if not promote:
        outputs(promote=False)
        return
    baseline = make_publication_baseline(candidate, evidence)
    tag = baseline["immutable_release_tag"]
    archive = args.bundle / "preview-wheels.zip"
    if not archive.exists():
        pack_wheels(args.bundle / "wheelhouse", archive, manifest, candidate["source"])
    else:
        # Resume only a byte-identical deterministic archive, without overwriting it.
        check = args.bundle / f"preview-wheels-verify-{time.time_ns()}.zip"
        pack_wheels(args.bundle / "wheelhouse", check, manifest, candidate["source"])
        require(
            sha256_file(archive) == sha256_file(check),
            "existing wheel ZIP differs from original wheels",
        )
    marker = args.bundle / "preview-baseline.json"
    write_json(marker, baseline)
    unique = ensure_release(api, repo, tag, candidate["control_sha"], immutable=True)
    for name, path, content_type in (
        ("preview-wheels.zip", archive, "application/zip"),
        ("preview-baseline.json", marker, "application/json"),
    ):
        upload_asset(api, repo, unique, name, path, content_type, immutable=True)
    publish_release(api, repo, unique)
    # Re-read after completing the immutable release, before touching fixed assets.
    promote, _, _, _ = promotion_check(
        api, repo, args.bundle, args.test_run_id, publisher=True
    )
    if not promote:
        outputs(promote=False)
        return
    fixed = ensure_release(
        api, repo, "preview-baseline", candidate["control_sha"], immutable=False
    )
    upload_asset(
        api,
        repo,
        fixed,
        "preview-wheels.zip",
        archive,
        "application/zip",
        immutable=False,
    )
    # Check again immediately before the commit marker; REST has no atomic CAS.
    # A brand-new draft has no readable baseline yet.
    if fixed.get("draft") is not True:
        # During our own wheel->marker transition, inspect only the previous
        # marker fence. Do not run publisher recovery and undo the pending ZIP.
        promote, _, _, _ = promotion_check(
            api, repo, args.bundle, args.test_run_id, publisher=False
        )
        require(
            promote,
            "baseline advanced during publication; publishers must be serialized",
        )
    upload_asset(
        api,
        repo,
        fixed,
        "preview-baseline.json",
        marker,
        "application/json",
        immutable=False,
    )
    publish_release(api, repo, fixed)
    installed, _, assets = load_baseline(api, repo)
    require(installed == baseline, "baseline changed during publication")
    wheels_asset = named_asset(assets, "preview-wheels.zip")
    require(
        api.asset_hash(repo, wheels_asset["id"]) == sha256_file(archive),
        "baseline wheel ZIP changed during publication",
    )
    step_summary(repo, baseline)
    promotion_outputs(True, candidate)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    nightly = commands.add_parser("fetch-nightly")
    nightly.add_argument("--source-repository", required=True)
    nightly.add_argument("--workflow", required=True)
    nightly.add_argument("--branch", default="main")
    nightly.add_argument("--run-id", type=int)
    nightly.add_argument(
        "--include-published",
        action="store_true",
        help="Retest already-distributed Preview runs; keep trust and 48-hour checks",
    )
    nightly.add_argument("--consumer-repository", required=True)
    nightly.add_argument("--output", type=Path, required=True)
    preview = commands.add_parser(
        "fetch-preview", help="Fetch minimum-regression Preview wheels"
    )
    preview.add_argument("--consumer-repository", required=True)
    preview.add_argument("--output", type=Path, required=True)
    build = commands.add_parser("candidate")
    build.add_argument("--selection", type=Path, required=True)
    build.add_argument("--wheels", type=Path, required=True)
    build.add_argument("--build-metadata", type=Path, required=True)
    build.add_argument("--image-repository", required=True)
    build.add_argument("--control-sha", required=True)
    build.add_argument("--dockerfile", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    attest = commands.add_parser("attest")
    attest.add_argument("--candidate", type=Path, required=True)
    attest.add_argument("--wheels", type=Path, required=True)
    attest.add_argument("--results", type=Path, required=True)
    attest.add_argument("--test-run-id", type=int, required=True)
    attest.add_argument("--output", type=Path, required=True)
    resolve = commands.add_parser("resolve-test-run")
    resolve.add_argument("--consumer-repository", required=True)
    resolve.add_argument("--workflow", required=True)
    resolve.add_argument("--run-id", type=int, required=True)
    for name in ("check-preview", "publish-preview"):
        command = commands.add_parser(name)
        command.add_argument("--consumer-repository", required=True)
        command.add_argument("--bundle", type=Path, required=True)
        command.add_argument("--test-run-id", type=int, required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "candidate":
            candidate_command(args)
        elif args.command == "attest":
            attest_command(args)
        else:
            api = GitHub()
            if args.command == "fetch-nightly":
                fetch_nightly(api, args)
            elif args.command == "fetch-preview":
                fetch_preview(api, args)
            elif args.command == "resolve-test-run":
                require(
                    args.workflow == TEST_WORKFLOW,
                    "only nightly-tests.yaml is trusted for minimum-regression Preview",
                )
                run = trusted_run(
                    api, args.consumer_repository, args.workflow, args.run_id
                )
                attempt = positive_int(run.get("run_attempt", 1), "test run attempt")
                bundle_name = f"nightly-tested-bundle-{run['id']}-{attempt}"
                artifacts = list(
                    api.pages(
                        f"/repos/{args.consumer_repository}/actions/runs/{run['id']}/artifacts",
                        "artifacts",
                    )
                )
                bundles = [
                    a
                    for a in artifacts
                    if a.get("name") == bundle_name and a.get("expired") is False
                ]
                require(len(bundles) <= 1, "duplicate tested bundle")
                outputs(
                    test_run_id=run["id"],
                    control_sha=run["head_sha"],
                    ready=bool(bundles),
                    bundle_name=bundle_name,
                )
            elif args.command == "check-preview":
                # This gate runs inside the serialized publisher workflow, before publish.
                # Recover here too, otherwise a draft/missing marker blocks workflow retries.
                promote, candidate, _, _ = promotion_check(
                    api,
                    args.consumer_repository,
                    args.bundle,
                    args.test_run_id,
                    publisher=True,
                )
                promotion_outputs(promote, candidate)
            elif args.command == "publish-preview":
                publish(api, args)
        return 0
    except AssetError as error:
        print(f"nightly-assets: {error}", file=sys.stderr)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, RuntimeError):
        # Do not print URLs, response bodies, environment, or tracebacks with secrets.
        print("nightly-assets: invalid input or I/O failure", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
