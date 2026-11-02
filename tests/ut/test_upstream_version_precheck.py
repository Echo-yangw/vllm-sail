# SPDX-License-Identifier: Apache-2.0
"""Upstream version pre-check: block breaking vLLM upgrades at the interface.

vllm-sail pins an upstream vLLM release line and hangs ~250 runtime patches off
its internal symbols. When upstream moves one of those symbols -- renames a
parameter, deletes a method, relocates a module -- the affected patch either
silently stops applying or crashes at import time, and the failure surfaces far
from its cause. ``interface_lock.toml`` is the snapshot that turns that drift
into a fast, CPU-only, PR-gating signal.

This module owns the shared upstream-AST helpers (symbol resolution and the
signature fingerprint) that :mod:`tests.ut.test_upstream_interface_compat` and
:mod:`tests.ut.test_patch_manifest_signature` also import, so the three files
agree on exactly what "resolved" and "same signature" mean.

Design rules, mirroring ``tests/support/source.py``:

* Everything is static. The helpers ``ast.parse`` upstream files; they never
  ``import vllm`` or ``torch``, so collection and the CPU-only tests stay free of
  device dependencies.
* Tests that read the upstream checkout take the ``upstream_source_root``
  fixture and carry the ``upstream_source`` marker: they skip when
  ``VLLM_SOURCE_ROOT`` is unset and *fail early* (never silently skip) under
  ``--require-upstream-source``, per the established convention.
* The purely structural tests over the committed lock and manifest run
  everywhere; they need no upstream checkout.

Regenerate the snapshot after a reviewed upstream bump::

    VLLM_SOURCE_ROOT=/path/to/pinned/vllm \
        python tests/ut/test_upstream_version_precheck.py --update-lock
"""

from __future__ import annotations

import ast
import hashlib
import io
import sys
import tokenize
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on 3.10
    import tomli as tomllib

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
LOCK_PATH = HERE / "interface_lock.toml"
MANIFEST_PATH = REPO_ROOT / "vllm_sail" / "patch" / "manifest.md"
PATCH_ROOT = REPO_ROOT / "vllm_sail" / "patch"
#: Directories under the patch package that hold no shipped patches.
SKIP_DIRS = frozenset({"template", "__pycache__"})

LOCK_SCHEMA = 1
SUPPORTED_RANGE = ">=0.32.0,<0.33.0"

#: AST node-type groups. These are plain tuples of type objects (never PEP 604
#: ``X | Y`` unions) so ``isinstance`` stays valid on Python 3.9, where a runtime
#: type union raises ``TypeError``. Naming them also keeps ruff's UP038 -- which
#: would rewrite a literal tuple into that 3.9-hostile union -- from firing.
FUNC_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
DEF_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
BLOCK_NODES = (ast.If, ast.Try, ast.For, ast.While, ast.With)

#: Curated upstream symbols vllm-sail structurally depends on but does not
#: necessarily patch. These are the load-bearing contracts (platform identity,
#: attention backend base + enum, MoE/linear kernel registration, torch-op
#: registration) whose drift would break whole subsystems at once.
KEY_INTERFACES: tuple[tuple[str, str], ...] = (
    ("vllm.platforms.interface.PlatformEnum", "enum"),
    ("vllm.platforms.interface.Platform", "class"),
    ("vllm.platforms.cuda.NvmlCudaPlatform", "class"),
    ("vllm.model_executor.custom_op.CustomOp", "class"),
    ("vllm.v1.attention.backend.AttentionBackend", "class"),
    ("vllm.v1.attention.backend.AttentionMetadataBuilder", "class"),
    ("vllm.v1.attention.backends.registry.AttentionBackendEnum", "enum"),
    ("vllm.v1.attention.backends.registry.register_backend", "function"),
    ("vllm.model_executor.kernels.linear.register_linear_kernel", "function"),
    ("vllm.utils.torch_utils.direct_register_custom_op", "function"),
)


# ---------------------------------------------------------------------------
# Manifest parsing
# ---------------------------------------------------------------------------
class ManifestEntry(NamedTuple):
    """One row of ``vllm_sail/patch/manifest.md``.

    ``kind`` is the *patch* kind recorded by the framework -- ``module``,
    ``class`` or ``value`` -- not the AST node type the target resolves to.
    """

    target: str
    kind: str
    reason: str
    affected_versions: str
    remove_when: str


MANIFEST_KINDS = frozenset({"module", "class", "value"})


def parse_manifest(path: Path = MANIFEST_PATH) -> list[ManifestEntry]:
    """Read the generated patch manifest without importing anything.

    The manifest is a fixed five-column table produced by
    ``tools/patch_manifest.py``; cells never contain a literal ``|`` (the
    generator escapes it), so a plain split is exact.
    """
    entries: list[ManifestEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| vllm"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 5:
            raise AssertionError(f"manifest row does not have five cells: {line!r}")
        entries.append(ManifestEntry(*cells))
    return entries


# ---------------------------------------------------------------------------
# Static upstream symbol resolution
# ---------------------------------------------------------------------------
class ResolvedSymbol(NamedTuple):
    """Where a dotted upstream symbol was found (or not)."""

    file: Path | None
    """Upstream ``.py`` file backing the target module, if any."""
    symbol_path: tuple[str, ...]
    """Trailing attribute path inside the module (empty for a bare module)."""
    node: ast.AST | None
    """The resolved AST node, or ``None`` when the symbol is absent."""

    @property
    def present(self) -> bool:
        return self.node is not None

    @property
    def module_found(self) -> bool:
        return self.file is not None


def _names_in_body(body: Sequence[ast.stmt]) -> dict[str, ast.AST]:
    """Map names a body binds, including conditional and imported names.

    Module-level ``if``/``try``/``for``/``with`` blocks are walked so a symbol
    defined only under a platform guard still counts as present, and imports are
    included because patches frequently target a name a module re-exports.
    """
    found: dict[str, ast.AST] = {}
    for node in body:
        if isinstance(node, DEF_NODES):
            found.setdefault(node.name, node)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    found.setdefault(target.id, node)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            found.setdefault(node.target.id, node)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                found.setdefault((alias.asname or alias.name).split(".")[0], node)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                found.setdefault(alias.asname or alias.name, node)
        elif isinstance(node, BLOCK_NODES):
            for field in ("body", "orelse", "finalbody", "handlers"):
                nested = _names_in_body(getattr(node, field, None) or [])
                for name, child in nested.items():
                    found.setdefault(name, child)
    return found


_parse_cache: dict[Path, dict[str, ast.AST]] = {}


def _module_names(path: Path) -> dict[str, ast.AST]:
    cached = _parse_cache.get(path)
    if cached is None:
        cached = _names_in_body(ast.parse(path.read_text(encoding="utf-8")).body)
        _parse_cache[path] = cached
    return cached


def module_file(root: Path, dotted: str) -> tuple[Path | None, list[str]]:
    """Split a dotted target into an upstream ``.py`` file plus attribute path.

    Uses the longest module prefix that maps to a real file or package, so
    ``vllm.config.compilation.Cfg.method`` resolves the module
    ``vllm/config/compilation.py`` and leaves ``("Cfg", "method")``.
    """
    parts = dotted.split(".")
    for depth in range(len(parts), 0, -1):
        base = root.joinpath(*parts[:depth])
        as_file = base.with_suffix(".py")
        if as_file.is_file():
            return as_file, parts[depth:]
        as_pkg = base / "__init__.py"
        if as_pkg.is_file():
            return as_pkg, parts[depth:]
    return None, []


def resolve_symbol(root: Path, dotted: str) -> ResolvedSymbol:
    """Locate ``dotted`` in the upstream tree at ``root`` without importing."""
    file, rest = module_file(root, dotted)
    if file is None:
        return ResolvedSymbol(None, tuple(rest), None)
    if not rest:
        return ResolvedSymbol(file, (), None)
    node: ast.AST | None = _module_names(file).get(rest[0])
    for part in rest[1:]:
        if node is None:
            break
        body = getattr(node, "body", None)
        node = _names_in_body(body).get(part) if body is not None else None
    return ResolvedSymbol(file, tuple(rest), node)


# ---------------------------------------------------------------------------
# Signature fingerprint
# ---------------------------------------------------------------------------
def _unparse(node: ast.AST) -> str:
    """Normalize Python 3.10's extra space in zero-argument lambdas."""
    text = ast.unparse(node)
    edits: dict[int, list[tuple[int, int]]] = {}
    previous = None
    for current in tokenize.generate_tokens(io.StringIO(text).readline):
        if (
            previous is not None
            and previous.type == tokenize.NAME
            and previous.string == "lambda"
            and current.string == ":"
            and previous.end[0] == current.start[0]
        ):
            edits.setdefault(current.start[0] - 1, []).append(
                (previous.end[1], current.start[1])
            )
        previous = current
    lines = text.splitlines(keepends=True)
    for line, spans in edits.items():
        for start, end in reversed(spans):
            lines[line] = lines[line][:start] + lines[line][end:]
    return "".join(lines)


def _function_signature(node: ast.AST) -> str:
    """Normalized ``lambda``-style signature text plus return annotation."""
    assert isinstance(node, FUNC_NODES)
    lam = ast.Lambda(args=node.args, body=ast.Constant(value=None))
    ast.fix_missing_locations(lam)
    text = _unparse(lam)
    returns = getattr(node, "returns", None)
    if returns is not None:
        text += " -> " + _unparse(returns)
    return text


def signature_text(node: ast.AST | None) -> str:
    """Return the interface-relevant source shape of ``node`` ("" if absent).

    Only the *signature* is captured -- parameter names/order/defaults/
    annotations for callables, and the member signatures for a class -- never
    statement bodies. Body drift is the job of ``tools/check_patch_drift.py``;
    this file is about interface compatibility.
    """
    if node is None:
        return ""
    if isinstance(node, FUNC_NODES):
        return _function_signature(node)
    if isinstance(node, ast.ClassDef):
        bases = ",".join(_unparse(base) for base in node.bases)
        keywords = ",".join(f"{kw.arg}={_unparse(kw.value)}" for kw in node.keywords)
        members = []
        for name, member in sorted(_names_in_body(node.body).items()):
            if isinstance(member, FUNC_NODES):
                members.append(f"{name}{_function_signature(member)}")
            else:
                members.append(f"{name}={_unparse(member)}")
        return f"class({bases}|{keywords}){{" + ";".join(members) + "}"
    return _unparse(node)


def fingerprint(node: ast.AST | None) -> str:
    """Stable ``sha256:`` digest of :func:`signature_text`, or "" if absent."""
    text = signature_text(node)
    if not text:
        return ""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Lock build / load / diff
# ---------------------------------------------------------------------------
def _toml_str(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def build_lock(root: Path) -> str:
    """Render ``interface_lock.toml`` from the upstream checkout at ``root``.

    Deterministic: the same source always yields byte-identical output, so the
    ``--update-lock`` refresh and the freshness assertion agree.
    """
    entries = parse_manifest()
    lines: list[str] = [
        "# interface_lock.toml -- upstream interface contract snapshot.",
        "#",
        "# AUTO-GENERATED; do not edit by hand. It records the signature",
        "# fingerprint of every patched upstream symbol plus a curated set of",
        "# load-bearing key interfaces, as seen in the pinned vLLM release.",
        "# Regenerate after a reviewed upstream bump:",
        "#   VLLM_SOURCE_ROOT=/path/to/pinned/vllm \\",
        "#       python tests/ut/test_upstream_version_precheck.py --update-lock",
        "",
        "[meta]",
        f"schema = {LOCK_SCHEMA}",
        f"vllm_version_range = {_toml_str(SUPPORTED_RANGE)}",
        f"key_interface_count = {len(KEY_INTERFACES)}",
        f"target_count = {len(entries)}",
        "",
    ]
    for symbol, kind in KEY_INTERFACES:
        resolved = resolve_symbol(root, symbol)
        lines += [
            "[[key_interface]]",
            f"symbol = {_toml_str(symbol)}",
            f"kind = {_toml_str(kind)}",
            f"present = {'true' if resolved.present else 'false'}",
            f"fingerprint = {_toml_str(fingerprint(resolved.node))}",
            "",
        ]
    for entry in entries:
        resolved = resolve_symbol(root, entry.target)
        relative = (
            resolved.file.relative_to(root).as_posix()
            if resolved.file is not None
            else ""
        )
        lines += [
            "[[patch_target]]",
            f"name = {_toml_str(entry.target)}",
            f"kind = {_toml_str(entry.kind)}",
            f"module = {_toml_str(relative)}",
            f"symbol = {_toml_str('.'.join(resolved.symbol_path))}",
            f"present = {'true' if resolved.present else 'false'}",
            f"fingerprint = {_toml_str(fingerprint(resolved.node))}",
            "",
        ]
    return "\n".join(lines)


def load_lock(path: Path = LOCK_PATH) -> dict[str, Any]:
    """Parse the committed lock, asserting the minimum structural contract."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    meta = data.get("meta", {})
    assert meta.get("schema") == LOCK_SCHEMA, f"unsupported lock schema: {meta}"
    assert isinstance(data.get("key_interface"), list)
    assert isinstance(data.get("patch_target"), list)
    return data


class DriftReport(NamedTuple):
    """Four-category interface change list between the lock and ``root``."""

    added: tuple[str, ...]
    """Locked-absent symbol now exists upstream (an additive patch may go)."""
    removed: tuple[str, ...]
    """Locked-present symbol vanished upstream (breaking)."""
    changed: tuple[str, ...]
    """Locked-present symbol's signature fingerprint differs (breaking)."""
    moved: tuple[str, ...]
    """The module file backing a locked symbol no longer exists (breaking)."""

    @property
    def breaking(self) -> tuple[str, ...]:
        return self.removed + self.changed + self.moved


def _diff_entries(
    locked: list[dict[str, Any]],
    name_key: str,
    root: Path,
    report: dict[str, list[str]],
) -> None:
    for entry in locked:
        name = entry[name_key]
        module = entry.get("module", "")
        resolved = resolve_symbol(root, name)
        if module and resolved.file is None:
            report["moved"].append(f"{name} (module {module} no longer resolves)")
            continue
        was_present = bool(entry.get("present"))
        if was_present and not resolved.present:
            report["removed"].append(name)
        elif was_present and resolved.present:
            if fingerprint(resolved.node) != entry.get("fingerprint", ""):
                report["changed"].append(name)
        elif not was_present and resolved.present:
            report["added"].append(name)


def diff_against_lock(lock: dict[str, Any], root: Path) -> DriftReport:
    """Classify every locked symbol against the upstream checkout at ``root``."""
    report: dict[str, list[str]] = {
        "added": [],
        "removed": [],
        "changed": [],
        "moved": [],
    }
    _diff_entries(lock.get("key_interface", []), "symbol", root, report)
    _diff_entries(lock.get("patch_target", []), "name", root, report)
    return DriftReport(
        added=tuple(sorted(report["added"])),
        removed=tuple(sorted(report["removed"])),
        changed=tuple(sorted(report["changed"])),
        moved=tuple(sorted(report["moved"])),
    )


def write_lock(root: Path, path: Path = LOCK_PATH) -> int:
    """Regenerate ``path`` from ``root``; return the number of locked targets."""
    text = build_lock(root)
    path.write_text(text, encoding="utf-8")
    return len(parse_manifest()) + len(KEY_INTERFACES)


def _resolve_root_from_env() -> Path:
    import os

    raw = os.environ.get("VLLM_SOURCE_ROOT")
    if not raw:
        raise SystemExit(
            "--update-lock requires VLLM_SOURCE_ROOT to point at the pinned "
            "upstream vLLM checkout."
        )
    root = Path(raw).expanduser().resolve()
    if not (root / "vllm" / "__init__.py").is_file():
        raise SystemExit(f"VLLM_SOURCE_ROOT must contain vllm/__init__.py: {root}")
    return root


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def lock() -> dict[str, Any]:
    """The committed ``interface_lock.toml`` parsed once per module."""
    return load_lock()


@pytest.fixture(scope="module")
def manifest() -> list[ManifestEntry]:
    """The committed patch manifest parsed once per module."""
    return parse_manifest()


def _locked_targets(lock: dict[str, Any]) -> Iterator[dict[str, Any]]:
    yield from lock.get("key_interface", [])
    yield from lock.get("patch_target", [])


# ---------------------------------------------------------------------------
# CPU-only structural tests (run everywhere; no upstream checkout needed)
# ---------------------------------------------------------------------------
def test_lock_file_is_well_formed(lock: dict[str, Any]) -> None:
    """The committed snapshot must be self-consistent and fully populated."""
    meta = lock["meta"]
    assert meta["vllm_version_range"] == SUPPORTED_RANGE
    assert meta["key_interface_count"] == len(lock["key_interface"])
    assert meta["target_count"] == len(lock["patch_target"])
    assert lock["key_interface"], "lock must cover the curated key interfaces"
    assert lock["patch_target"], "lock must cover the patch targets"

    for entry in lock["key_interface"]:
        assert entry["symbol"].startswith("vllm.")
        assert isinstance(entry["present"], bool)
        digest = entry["fingerprint"]
        if entry["present"]:
            assert digest.startswith("sha256:") and len(digest) == len("sha256:") + 64
        else:
            assert digest == ""

    for entry in lock["patch_target"]:
        assert entry["name"].startswith("vllm.")
        assert entry["kind"] in MANIFEST_KINDS
        assert isinstance(entry["present"], bool)
        if entry["present"]:
            assert entry["module"].endswith(".py")
            assert entry["fingerprint"].startswith("sha256:")
        else:
            # An additive patch: absent upstream, so it has no fingerprint and
            # its module still had to resolve when the lock was generated.
            assert entry["fingerprint"] == ""


def test_lock_covers_every_manifest_target(
    lock: dict[str, Any], manifest: list[ManifestEntry]
) -> None:
    """The snapshot and the manifest must describe exactly the same targets.

    A patch added without refreshing the lock would escape the pre-check; a
    stale lock entry would chase a symbol nothing patches any more.
    """
    locked = {entry["name"] for entry in lock["patch_target"]}
    declared = {entry.target for entry in manifest}
    assert locked == declared, (
        "interface_lock.toml is out of sync with manifest.md; regenerate with "
        f"--update-lock. only-in-lock={sorted(locked - declared)[:5]} "
        f"only-in-manifest={sorted(declared - locked)[:5]}"
    )
    # The manifest is generated sorted by target; the lock preserves that order.
    assert [entry["name"] for entry in lock["patch_target"]] == sorted(
        entry["name"] for entry in lock["patch_target"]
    )


def test_key_interfaces_are_a_known_curated_set(lock: dict[str, Any]) -> None:
    """Guard against silently dropping a load-bearing contract from the lock."""
    locked_symbols = [entry["symbol"] for entry in lock["key_interface"]]
    assert locked_symbols == [symbol for symbol, _ in KEY_INTERFACES]


# ---------------------------------------------------------------------------
# Upstream-dependent tests (skip without VLLM_SOURCE_ROOT; fail early under
# --require-upstream-source via the upstream_source_root fixture)
# ---------------------------------------------------------------------------
@pytest.mark.upstream_source
def test_key_interfaces_resolve(
    upstream_source_root: Path, lock: dict[str, Any]
) -> None:
    """Every curated key interface must exist in the pinned upstream."""
    missing = [
        entry["symbol"]
        for entry in lock["key_interface"]
        if not resolve_symbol(upstream_source_root, entry["symbol"]).present
    ]
    assert not missing, f"key upstream interfaces vanished: {missing}"


@pytest.mark.upstream_source
def test_no_breaking_interface_drift(
    upstream_source_root: Path, lock: dict[str, Any]
) -> None:
    """The pinned upstream must still match the locked signatures.

    ADDED (an additive patch upstream finally provides) is reported but not a
    failure -- it is a signal a patch may now be deletable. REMOVED / CHANGED /
    MOVED are breaking and must be resolved before the upgrade lands.
    """
    report = diff_against_lock(lock, upstream_source_root)
    if report.added:
        # Surfaced in the -s capture and in the failure text below if any.
        print(
            f"\ninterface_lock: {len(report.added)} additive target(s) now "
            f"exist upstream (patch may be removable): {list(report.added)[:10]}"
        )
    assert not report.breaking, (
        "upstream interface drifted from interface_lock.toml. Review each "
        "affected patch, then regenerate the lock with --update-lock.\n"
        f"  REMOVED ({len(report.removed)}): {list(report.removed)[:20]}\n"
        f"  CHANGED ({len(report.changed)}): {list(report.changed)[:20]}\n"
        f"  MOVED   ({len(report.moved)}): {list(report.moved)[:20]}"
    )


@pytest.mark.upstream_source
def test_lock_is_current(upstream_source_root: Path) -> None:
    """The committed snapshot must equal a fresh regeneration.

    This is what makes ``--update-lock`` auditable: a stale lock fails here even
    if no *breaking* drift happened yet, so the reviewable artifact never lags
    the pinned source silently.
    """
    expected = build_lock(upstream_source_root)
    actual = LOCK_PATH.read_text(encoding="utf-8")
    if expected != actual:
        # Point at the first differing line to keep the failure actionable.
        want_lines = expected.splitlines()
        got_lines = actual.splitlines()
        for index in range(min(len(want_lines), len(got_lines))):
            want, got = want_lines[index], got_lines[index]
            if want != got:
                raise AssertionError(
                    "interface_lock.toml is stale; regenerate with --update-lock. "
                    f"First difference at line {index + 1}:\n"
                    f"  regenerated: {want}\n  committed:   {got}"
                )
        raise AssertionError(
            "interface_lock.toml is stale (line count differs); regenerate with "
            "--update-lock."
        )


if __name__ == "__main__":  # pragma: no cover - manual maintenance entry point
    if "--update-lock" in sys.argv:
        count = write_lock(_resolve_root_from_env())
        print(f"wrote {LOCK_PATH} ({count} locked symbols)")
    else:
        print(__doc__)
        raise SystemExit(2)
