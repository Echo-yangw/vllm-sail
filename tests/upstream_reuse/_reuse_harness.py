# SPDX-License-Identifier: Apache-2.0
"""Harness that loads whitelisted upstream vLLM tests into the vllm-sail runtime.

vllm-sail pins an upstream vLLM release and disguises its PPU platform as CUDA
(``PPUPlatform._enum = PlatformEnum.CUDA``, so ``current_platform.is_cuda()`` is
True). Upstream tests written against the CUDA platform -- but that assert
scheduling / config / sampling / tokenization *logic* rather than concrete
kernel numerics -- therefore run unchanged inside the vllm-sail runtime. This
module is the mechanism that makes that reuse explicit, auditable and CI-gated.

Design rules (shared with the pinned-source contract tests):

* Importing this module never imports torch or vLLM. It only reads the
  committed ``REUSE_MANIFEST.toml`` and shells out. Collection stays CPU-only,
  so ``--collect-only`` reports 0 errors on a bare host.
* The upstream checkout is located from ``VLLM_SOURCE_ROOT`` (the same variable
  the rest of the suite uses). Without it, the whole directory is skipped.
* An upstream test file is executed as an **isolated pytest subprocess** rooted
  at the upstream checkout. This is deliberate: it lets upstream's own
  ``conftest.py`` fixtures and markers resolve, and it confines an upstream
  ``ImportError`` (missing optional dependency, drifted module) to that one
  parameterized case instead of corrupting our collection. The subprocess
  inherits the ambient environment, which -- in CI -- has vllm-sail installed,
  so ``current_platform`` is the CUDA-disguised PPU platform while the upstream
  test body believes it is talking to CUDA.
* Results are read back from a JUnit XML report so pass/fail/skip/error counts
  and the exact failing node names are available for the expected-xfail logic in
  ``test_reuse_adapted.py``.

Everything here is Python 3.9-compatible: annotations are strings (PEP 563 via
``from __future__ import annotations``) and no runtime ``X | Y`` union or
``zip(strict=)`` is used.
"""

from __future__ import annotations

import importlib.util
import os
import re
import socket
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urlparse

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on 3.9/3.10
    import tomli as tomllib

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
MANIFEST_PATH = HERE / "REUSE_MANIFEST.toml"

MANIFEST_SCHEMA = 1
VALID_MODES = ("direct", "adapted")
VALID_CHIPS = ("cpu", "810e", "890p")
#: Subprocess wall-clock ceiling for one upstream test file.
RUN_TIMEOUT_SECONDS = 1800


class ReuseEntry(NamedTuple):
    """One whitelisted upstream test file and how to run it."""

    id: str
    path: str
    mode: str
    chips: tuple[str, ...]
    category: str
    reason: str
    min_tests: int
    markers: str
    keywords: str
    deselect: tuple[str, ...]
    env: dict[str, str]
    xfail: tuple[str, ...]
    skip_reason: str

    @property
    def cpu_runnable(self) -> bool:
        """True when the entry can execute on a device-less CPU runner."""
        return "cpu" in self.chips

    @property
    def device_only(self) -> bool:
        """True when the entry needs a real PPU (excluded from the CPU lane)."""
        return not self.cpu_runnable


class RunResult(NamedTuple):
    """Parsed outcome of one upstream pytest subprocess."""

    returncode: int
    collected: int
    passed: int
    failures: int
    errors: int
    skipped: int
    failed_names: tuple[str, ...]
    tail: str
    skipped_by_manifest: bool = False
    environment_gap: str = ""


#: Verbatim marker pytest prints when a ``conftest.py`` it needs raises on
#: import.  Exit status 4 is pytest's "usage error", and a run that dies here
#: never collected a single test.
CONFTEST_IMPORT_MARKER = "ImportError while loading conftest"

#: pytest's exit status for a usage error (bad CLI expression, conftest that
#: fails to import, an unusable plugin, ...).
PYTEST_USAGE_ERROR = 4

#: How many lines of the conftest traceback to quote in a skip reason: enough to
#: name the missing module, short enough to stay readable in ``-rs`` output.
CONFTEST_REASON_LINES = 8

_EXCEPTION_LINE = re.compile(r"^[A-Za-z_][\w.]*(Error|Exception|Interrupt)\b")


def conftest_import_error(result: RunResult) -> str | None:
    """Verbatim reason when the ambient runtime cannot load upstream's conftest.

    ``None`` for every other outcome.  This is deliberately narrow: it only
    matches pytest's own *conftest import* failure on a run that collected
    nothing, which means the subprocess died before executing a line of any
    upstream test.  Upstream's ``tests/conftest.py`` imports test-only
    dependencies (``tblib`` at the pinned commit) that are not part of the
    vllm-sail runtime, so a container without them cannot run this lane at all.

    What this must never absorb:

    * a bad ``-m`` / ``-k`` / ``--deselect`` expression from our own manifest --
      also exit 4, but pytest reports it as ``Wrong expression passed to '-m'``
      and there is no conftest marker, so it stays a hard failure;
    * an ``ImportError`` inside one upstream *test module* -- that surfaces as a
      collection error with a JUnit ``<error>`` entry and a nonzero ``errors``,
      not as exit 4 with nothing collected.

    Both remain real regressions and keep failing the suite.
    """
    if result.returncode != PYTEST_USAGE_ERROR or result.collected:
        return None
    lines = [line.strip() for line in result.tail.splitlines() if line.strip()]
    start = next(
        (i for i, line in enumerate(lines) if CONFTEST_IMPORT_MARKER in line), None
    )
    if start is None:
        return None
    block = lines[start : start + CONFTEST_REASON_LINES]
    cause = next(
        (line for line in reversed(lines[start:]) if _EXCEPTION_LINE.match(line)), ""
    )
    if cause and cause not in block:
        block.append(cause)
    return " ".join(block)


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------
def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def load_manifest(path: Path = MANIFEST_PATH) -> list[ReuseEntry]:
    """Parse ``REUSE_MANIFEST.toml`` into validated :class:`ReuseEntry` rows."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    meta = data.get("meta", {})
    schema = meta.get("schema")
    if schema != MANIFEST_SCHEMA:
        raise AssertionError(f"unsupported REUSE_MANIFEST schema: {schema!r}")
    raw_entries = data.get("entry", [])
    if not isinstance(raw_entries, list) or not raw_entries:
        raise AssertionError("REUSE_MANIFEST.toml declares no [[entry]] rows")

    entries: list[ReuseEntry] = []
    seen: set[str] = set()
    for raw in raw_entries:
        entry_id = raw["id"]
        if entry_id in seen:
            raise AssertionError(f"duplicate manifest id: {entry_id!r}")
        seen.add(entry_id)
        mode = raw["mode"]
        if mode not in VALID_MODES:
            raise AssertionError(f"{entry_id}: invalid mode {mode!r}")
        chips = _as_str_tuple(raw.get("chips"))
        for chip in chips:
            if chip not in VALID_CHIPS:
                raise AssertionError(f"{entry_id}: invalid chip {chip!r}")
        if not chips:
            raise AssertionError(f"{entry_id}: declares no chips")
        env_raw = raw.get("env", {}) or {}
        entries.append(
            ReuseEntry(
                id=entry_id,
                path=raw["path"],
                mode=mode,
                chips=chips,
                category=raw.get("category", ""),
                reason=raw.get("reason", ""),
                min_tests=int(raw.get("min_tests", 1)),
                markers=raw.get("markers", ""),
                keywords=raw.get("keywords", ""),
                deselect=_as_str_tuple(raw.get("deselect")),
                env={str(key): str(val) for key, val in env_raw.items()},
                xfail=_as_str_tuple(raw.get("xfail")),
                skip_reason=raw.get("skip_reason", ""),
            )
        )
    return entries


def entries_for_mode(mode: str, path: Path = MANIFEST_PATH) -> list[ReuseEntry]:
    """Manifest rows whose ``mode`` equals ``mode`` (order preserved)."""
    return [entry for entry in load_manifest(path) if entry.mode == mode]


def entry_marks(entry: ReuseEntry) -> list[pytest.MarkDecorator]:
    """Markers that route one entry through the shared device/chip gates.

    Every reused upstream test is an ``upstream_source`` check (it needs the
    selected checkout). Entries that cannot run on a device-less host are also
    marked ``ppu`` -- so the PR CPU lane's ``-m "upstream_source and not ppu"``
    excludes them and ``tests/conftest.py`` auto-skips them without a real
    device. Chip-exclusive entries additionally carry ``cap80`` / ``cap89``.
    """
    marks = [pytest.mark.upstream_source]
    if entry.device_only:
        marks.append(pytest.mark.ppu)
        has_810e = "810e" in entry.chips
        has_890p = "890p" in entry.chips
        if has_810e and not has_890p:
            marks.append(pytest.mark.cap80)
        elif has_890p and not has_810e:
            marks.append(pytest.mark.cap89)
    return marks


def param_for_entry(entry: ReuseEntry) -> pytest.param:
    """A ``pytest.param`` carrying the entry, its id and its routing marks."""
    return pytest.param(entry, id=entry.id, marks=entry_marks(entry))


# ---------------------------------------------------------------------------
# Environment probes
# ---------------------------------------------------------------------------
def resolve_source_root(raw: str | None) -> Path | None:
    """Return the upstream checkout for ``raw``, or ``None`` when unusable.

    Unlike ``tests.support.source.source_root`` this never raises or calls
    ``pytest.skip`` -- the caller decides. ``None`` means "no upstream source",
    which the directory conftest turns into a whole-directory skip.
    """
    if not raw:
        return None
    root = Path(raw).expanduser().resolve()
    if not (root / "vllm" / "__init__.py").is_file():
        return None
    return root


def runtime_available() -> tuple[bool, str]:
    """Whether an installed vLLM + torch runtime can actually execute tests.

    Upstream test modules ``import vllm`` / ``import torch`` at module scope, so
    without both installed the reuse subprocess can only fail on import. We
    detect that up front and skip gracefully instead -- this is what keeps a
    bare host (no vLLM) at "all skipped, 0 errors".
    """
    if importlib.util.find_spec("torch") is None:
        return False, "torch is not installed in this environment"
    if importlib.util.find_spec("vllm") is None:
        return False, "vllm is not installed in this environment"
    return True, ""


# ---------------------------------------------------------------------------
# Hugging Face hub reachability
# ---------------------------------------------------------------------------
#: Where upstream resolves a checkpoint when a test names a repo id.  Overridable
#: through ``HF_ENDPOINT``, which :func:`hub_endpoint` honours.
DEFAULT_HUB_ENDPOINT = "https://huggingface.co"

#: Connect budget for the one probe per session.  Long enough to survive a slow
#: proxy handshake, short enough that an air-gapped runner does not spend minutes
#: finding out it is air-gapped.
HUB_PROBE_TIMEOUT_SECONDS = 5.0

#: Variables that disable outgoing hub traffic outright, no probe needed.
HUB_OFFLINE_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")

#: A ``"org/name"`` literal -- how an upstream test names a checkpoint.
_HUB_REPO_ID = re.compile(r"""["']([A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*)["']""")

#: First path segments that are repository directories, not hub organisations.
_NON_REPO_ORGS = frozenset(
    {
        ".",
        "..",
        "assets",
        "benchmarks",
        "docs",
        "examples",
        "requirements",
        "tests",
        "tools",
        "vllm",
    }
)

#: Second path segments that are file names, not checkpoint revisions.
_NON_REPO_SUFFIXES = (
    ".bin",
    ".cfg",
    ".csv",
    ".gguf",
    ".in",
    ".jinja",
    ".json",
    ".md",
    ".png",
    ".pt",
    ".py",
    ".safetensors",
    ".sh",
    ".so",
    ".toml",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
)

#: One probe per (host, port, timeout); twenty entries share the answer.
_HUB_PROBE_CACHE: dict[tuple[str, int, float], tuple[bool, str]] = {}


def _env_flag(name: str) -> bool:
    """Truthy the way ``huggingface_hub`` reads its offline switches."""
    return os.environ.get(name, "").strip().lower() in {"1", "on", "true", "yes"}


def hub_endpoint() -> tuple[str, int]:
    """Host and port this runtime would dial to reach the hub."""
    raw = os.environ.get("HF_ENDPOINT") or DEFAULT_HUB_ENDPOINT
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    host = parsed.hostname or "huggingface.co"
    if parsed.port:
        return host, parsed.port
    return host, 443 if parsed.scheme == "https" else 80


def hub_reachable(timeout: float = HUB_PROBE_TIMEOUT_SECONDS) -> tuple[bool, str]:
    """Whether this runtime can dial the hub, plus the verbatim reason it cannot.

    Probed once per session and cached, so an air-gapped runner pays the connect
    timeout once instead of once per entry.  An explicit offline switch
    short-circuits without touching the network at all.
    """
    for name in HUB_OFFLINE_VARS:
        if _env_flag(name):
            return False, f"{name} is set, so outgoing hub traffic is disabled"
    host, port = hub_endpoint()
    key = (host, port, timeout)
    cached = _HUB_PROBE_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        with socket.create_connection((host, port), timeout=timeout):
            outcome: tuple[bool, str] = (True, "")
    except OSError as exc:
        outcome = (
            False,
            f"cannot reach {host}:{port} from this runtime "
            f"({type(exc).__name__}: {exc})",
        )
    _HUB_PROBE_CACHE[key] = outcome
    return outcome


def hub_repo_ids(source: str) -> tuple[str, ...]:
    """Checkpoint ids an upstream module names, in first-seen order.

    A heuristic on purpose.  The question is only "could this file make the hub
    resolve something", and the two ways to be wrong cost very differently: a
    false positive skips one entry on a runner that is air-gapped anyway, while
    a false negative burns ``RUN_TIMEOUT_SECONDS`` and still ends in a visible
    failure.  Repository directories (``tests/models``) and file names
    (``config.json``) are dropped so ordinary import strings do not count.
    """
    found: list[str] = []
    for match in _HUB_REPO_ID.finditer(source):
        repo_id = match.group(1)
        org, name = repo_id.split("/", 1)
        if org in _NON_REPO_ORGS or name.lower().endswith(_NON_REPO_SUFFIXES):
            continue
        if repo_id not in found:
            found.append(repo_id)
    return tuple(found)


def hub_cache_dir() -> Path:
    """The local hub cache, honouring the variables ``huggingface_hub`` reads."""
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        raw = os.environ.get(name)
        if raw:
            return Path(raw).expanduser()
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def cached_repo_ids(cache: Path | None = None) -> frozenset[str]:
    """Repo ids already on disk, from the ``models--<org>--<name>`` layout."""
    root = cache if cache is not None else hub_cache_dir()
    if not root.is_dir():
        return frozenset()
    found: set[str] = set()
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith("models--"):
            continue
        parts = child.name[len("models--") :].split("--")
        if len(parts) >= 2 and parts[0] and parts[1]:
            found.add("/".join(parts[:2]))
    return frozenset(found)


def offline_hub_gap(entry: ReuseEntry, root: Path) -> str | None:
    """Reason when an entry needs hub access this runtime does not have.

    ``None`` whenever the entry has a chance of running -- which is always the
    case in CI, where the probe succeeds and this guard is inert, so no coverage
    is given up on a runner that has a network.

    On an air-gapped runner the alternative to skipping is not a clean failure
    but a hang.  Upstream resolves a checkpoint through
    ``vllm/transformers_utils/repo_utils.py::file_or_path_exists``, which retries
    ``list_repo_files`` from inside ``socket.create_connection``; with no route to
    the hub that blocks long past the entry's budget, and ``RUN_TIMEOUT_SECONDS``
    eventually reports a timeout that says nothing about the cause.  A real
    ZW-890P run of ``tests/config/test_multimodal_config.py`` was aborted at
    100 s (``kill -ABRT`` under ``-X faulthandler``) with this stack::

        socket.py:837                        create_connection
        huggingface_hub/hf_api.py:3851       list_repo_tree
        vllm/transformers_utils/repo_utils.py:162  lookup_files
        vllm/transformers_utils/config.py:869      get_config
        vllm/config/model.py:635             ModelConfig.__post_init__
        tests/models/utils.py:308            build_model_context
        tests/config/test_multimodal_config.py:35  test_supports_multimodal_inputs

    What this must never absorb: an entry whose checkpoints *are* cached (it runs
    offline and must be executed), an entry that names no checkpoint (its
    failures are real), and a missing upstream file (that is pin drift, which
    :meth:`UpstreamRunner.run` reports as an error rather than an excuse).
    """
    reachable, probe = hub_reachable()
    if reachable:
        return None
    target = root / entry.path
    if not target.is_file():
        return None
    wanted = hub_repo_ids(target.read_text(encoding="utf-8"))
    if not wanted:
        return None
    have = cached_repo_ids()
    missing = tuple(repo for repo in wanted if repo not in have)
    if not missing:
        return None
    return (
        f"upstream resolves {', '.join(missing)} from the Hugging Face hub and "
        f"{probe}; the local cache ({hub_cache_dir()}) has none of them, so this "
        "entry cannot execute here. It runs for real wherever the hub is "
        "reachable, which is what CI does."
    )


# ---------------------------------------------------------------------------
# JUnit XML parsing
# ---------------------------------------------------------------------------
def _parse_junit(path: Path) -> tuple[int, int, int, int, list[str]]:
    """Return (collected, failures, errors, skipped, failed_names) from XML."""
    collected = failures = errors = skipped = 0
    failed_names: list[str] = []
    if not path.is_file():
        return collected, failures, errors, skipped, failed_names
    root = ET.parse(path).getroot()  # noqa: S314 - trusted local CI artifact
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    for suite in suites:
        collected += int(suite.get("tests", "0") or 0)
        skipped += int(suite.get("skipped", "0") or 0)
        for case in suite.iter("testcase"):
            classname = case.get("classname", "")
            name = case.get("name", "")
            node = f"{classname}::{name}" if classname else name
            has_failure = case.find("failure") is not None
            has_error = case.find("error") is not None
            if has_failure:
                failures += 1
                failed_names.append(node)
            elif has_error:
                errors += 1
                failed_names.append(node)
    return collected, failures, errors, skipped, failed_names


# ---------------------------------------------------------------------------
# Upstream test runner
# ---------------------------------------------------------------------------
class UpstreamRunner:
    """Executes whitelisted upstream test files inside the ambient runtime."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def target(self, entry: ReuseEntry) -> Path:
        return self.root / entry.path

    def build_command(self, entry: ReuseEntry, junit_path: Path) -> list[str]:
        """Assemble the pytest argv for one entry (no execution)."""
        command = [
            sys.executable,
            "-m",
            "pytest",
            entry.path,
            "-q",
            "-rN",
            "--no-header",
            "-p",
            "no:cacheprovider",
            "--continue-on-collection-errors",
            f"--junitxml={junit_path}",
        ]
        if entry.markers:
            command += ["-m", entry.markers]
        if entry.keywords:
            command += ["-k", entry.keywords]
        for node in entry.deselect:
            command += ["--deselect", node]
        return command

    def build_env(self, entry: ReuseEntry) -> dict[str, str]:
        """Ambient env + upstream root on PYTHONPATH + per-entry overrides."""
        env = dict(os.environ)
        existing = env.get("PYTHONPATH", "")
        parts = [str(self.root)]
        if existing:
            parts.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(parts)
        env.update(entry.env)
        return env

    def run(self, entry: ReuseEntry) -> RunResult:
        """Run one upstream test file and parse its JUnit report."""
        if entry.skip_reason:
            return RunResult(
                returncode=0,
                collected=0,
                passed=0,
                failures=0,
                errors=0,
                skipped=0,
                failed_names=(),
                tail=entry.skip_reason,
                skipped_by_manifest=True,
            )
        gap = offline_hub_gap(entry, self.root)
        if gap is not None:
            # Decided before the subprocess starts: an entry that can only hang
            # is not worth ``RUN_TIMEOUT_SECONDS`` of wall clock to prove it.
            return RunResult(
                returncode=0,
                collected=0,
                passed=0,
                failures=0,
                errors=0,
                skipped=0,
                failed_names=(),
                tail=gap,
                environment_gap=gap,
            )
        target = self.target(entry)
        if not target.is_file():
            return RunResult(
                returncode=-1,
                collected=0,
                passed=0,
                failures=0,
                errors=1,
                skipped=0,
                failed_names=(f"MISSING::{entry.path}",),
                tail=f"upstream test file not found at pinned commit: {entry.path}",
            )
        with tempfile.TemporaryDirectory(prefix="vllm_sail_reuse_") as workdir:
            junit_path = Path(workdir) / "report.xml"
            command = self.build_command(entry, junit_path)
            try:
                completed = subprocess.run(
                    command,
                    cwd=str(self.root),
                    env=self.build_env(entry),
                    capture_output=True,
                    text=True,
                    timeout=RUN_TIMEOUT_SECONDS,
                    check=False,
                )
                returncode = completed.returncode
                output = completed.stdout + "\n" + completed.stderr
            except subprocess.TimeoutExpired as exc:
                returncode = -1
                output = f"upstream reuse run timed out after {RUN_TIMEOUT_SECONDS}s\n"
                output += (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            collected, failures, errors, skipped, failed_names = _parse_junit(
                junit_path
            )
        # Validate returncode beyond JUnit XML.  pytest exit codes:
        #   0 = all passed, 1 = some failures, 5 = no tests collected,
        #   2 = interrupted / collection error, 3 = internal error,
        #   4 = usage error (handled by conftest_import_error).
        # Codes 2/3 indicate a broken run even if the XML looks clean.
        if returncode in (2, 3) and errors == 0:
            errors = 1
            if not failed_names:
                failed_names.append(f"PYTEST_EXIT_{returncode}::{entry.path}")
        # Code 5 means nothing was collected at all.  Force collected to 0
        # so the min_tests guard in consumers catches it reliably.
        if returncode == 5:
            collected = 0

        passed = max(collected - failures - errors - skipped, 0)
        tail = "\n".join(line for line in output.splitlines() if line.strip())[-2000:]
        return RunResult(
            returncode=returncode,
            collected=collected,
            passed=passed,
            failures=failures,
            errors=errors,
            skipped=skipped,
            failed_names=tuple(failed_names),
            tail=tail,
        )
