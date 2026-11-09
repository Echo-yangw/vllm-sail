# SPDX-License-Identifier: Apache-2.0
"""Small static upstream helpers: no upstream imports or signature snapshots."""

from __future__ import annotations

import ast
import inspect
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[2]
PATCH_ROOT = ROOT / "vllm_sail/patch"
SKIP_DIRS = frozenset({"template", "__pycache__"})
SUPPORTED_RANGE = ">=0.32.0,<0.33.0"
FUNC_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
DEF_NODES = (*FUNC_NODES, ast.ClassDef)
BLOCK_NODES = (ast.If, ast.Try, ast.ExceptHandler, ast.For, ast.While, ast.With)


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


def _module_names(path: Path) -> dict[str, ast.AST]:
    return _names_in_body(ast.parse(path.read_text(encoding="utf-8")).body)


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


def resolve_symbol(
    root: Path, dotted: str, _seen: frozenset[str] = frozenset()
) -> ResolvedSymbol:
    """Locate ``dotted`` in the upstream tree at ``root`` without importing."""
    if dotted in _seen:
        return ResolvedSymbol(None, (), None)
    seen = _seen | {dotted}
    file, rest = module_file(root, dotted)
    if file is None:
        return ResolvedSymbol(None, tuple(rest), None)
    if not rest:
        return ResolvedSymbol(file, (), None)
    node: ast.AST | None = _module_names(file).get(rest[0])
    if isinstance(node, ast.ImportFrom) and node.module:
        module_parts = file.relative_to(root).with_suffix("").parts
        imported_module = node.module
        if node.level:
            package = module_parts[:-1]
            imported_module = ".".join(
                (*package[: len(package) - node.level + 1], node.module)
            )
        for alias in node.names:
            if (alias.asname or alias.name) == rest[0]:
                imported = resolve_symbol(
                    root,
                    f"{imported_module}.{alias.name}"
                    + ("." + ".".join(rest[1:]) if len(rest) > 1 else ""),
                    seen,
                )
                if imported.present or imported_module.startswith("vllm."):
                    if not imported.present:
                        # Some optional source imports have an explicit local
                        # fallback (FlashMLA's unavailable stubs). A dangling
                        # re-export without such a binding remains absent.
                        tree = ast.parse(file.read_text())
                        parents = {
                            child: parent
                            for parent in ast.walk(tree)
                            for child in ast.iter_child_nodes(parent)
                        }
                        current = next(
                            child
                            for child in ast.walk(tree)
                            if isinstance(child, ast.ImportFrom)
                            and child.lineno == node.lineno
                        )
                        while current in parents:
                            parent = parents[current]
                            if isinstance(parent, ast.If) and len(rest) == 1:
                                alternative = (
                                    parent.orelse
                                    if current in parent.body
                                    else parent.body
                                )
                                fallback = _names_in_body(alternative).get(rest[0])
                                if isinstance(
                                    fallback, (ast.Assign, ast.AnnAssign, *DEF_NODES)
                                ):
                                    return ResolvedSymbol(file, tuple(rest), fallback)
                            current = parent
                    return imported
    for index, part in enumerate(rest[1:], 1):
        if node is None:
            break
        body = getattr(node, "body", None)
        child = _names_in_body(body).get(part) if body is not None else None
        if child is None and isinstance(node, ast.ClassDef):
            if part in {"__init__", "__eq__", "__hash__"} and dataclass_method(
                node, part
            ):
                child = node
            else:
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        prefix = ".".join(dotted.split(".")[: -len(rest)])
                        inherited = resolve_symbol(
                            root, f"{prefix}.{base.id}.{'.'.join(rest[index:])}", seen
                        )
                        if inherited.present:
                            return inherited
        node = child
    return ResolvedSymbol(file, tuple(rest), node)


def dataclass_method(node: ast.ClassDef, name: str) -> bool:
    """Account for generated dataclass methods using their explicit options."""
    for decorator in node.decorator_list:
        func = decorator.func if isinstance(decorator, ast.Call) else decorator
        if not isinstance(func, ast.Name) or func.id != "dataclass":
            continue
        options = (
            {
                kw.arg: kw.value.value
                for kw in decorator.keywords
                if isinstance(kw.value, ast.Constant)
            }
            if isinstance(decorator, ast.Call)
            else {}
        )
        return {
            "__init__": options.get("init", True),
            "__eq__": options.get("eq", True),
            "__hash__": options.get("unsafe_hash", False)
            or (options.get("frozen", False) and options.get("eq", True)),
        }.get(name, False)
    return False


def callable_signature(node: ast.AST) -> inspect.Signature:
    """Read binding rules without evaluating annotations or default expressions."""
    if not isinstance(node, FUNC_NODES):
        raise TypeError(f"expected function, got {type(node).__name__}")
    args = node.args
    parameters = []
    positional = args.posonlyargs + args.args
    required = len(positional) - len(args.defaults)
    for index, arg in enumerate(positional):
        kind = (
            inspect.Parameter.POSITIONAL_ONLY
            if index < len(args.posonlyargs)
            else inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        parameters.append(
            inspect.Parameter(
                arg.arg,
                kind,
                default=inspect.Parameter.empty if index < required else object(),
            )
        )
    if args.vararg:
        parameters.append(
            inspect.Parameter(args.vararg.arg, inspect.Parameter.VAR_POSITIONAL)
        )
    for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=False):
        parameters.append(
            inspect.Parameter(
                arg.arg,
                inspect.Parameter.KEYWORD_ONLY,
                default=inspect.Parameter.empty if default is None else object(),
            )
        )
    if args.kwarg:
        parameters.append(
            inspect.Parameter(args.kwarg.arg, inspect.Parameter.VAR_KEYWORD)
        )
    return inspect.Signature(parameters)


def bound_value_problems(
    before: inspect.BoundArguments, after: inspect.BoundArguments
) -> list[str]:
    """Keep positional meaning checks identical for generic and real calls."""
    return [
        f"argument {name!r} changes meaning"
        for name, value in before.arguments.items()
        if name in after.arguments
        and before.signature.parameters[name].kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        and after.arguments[name] is not value
    ]


def compatibility_problems(replacement: ast.AST, upstream: ast.AST) -> list[str]:
    """Check representative calls, including positional meaning and required args.

    This checks the declared callable surface, not arbitrary Python callers or
    forwarding inside a wrapper. Feature tests exercise the latter separately.
    """
    rep = callable_signature(replacement)
    up = callable_signature(upstream)
    positional = [
        p
        for p in up.parameters.values()
        if p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    values = {p.name: object() for p in up.parameters.values()}
    problems = []
    for prefix in range(len(positional) + 1):
        # All supplied, and optional arguments omitted: both matter when a
        # replacement has accidentally made an optional upstream input required.
        for optional in (False, True):
            args = [values[p.name] for p in positional[:prefix]]
            kwargs = {
                p.name: values[p.name]
                for p in up.parameters.values()
                if p.kind
                in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
                and p not in positional[:prefix]
                and (optional or p.default is inspect.Parameter.empty)
            }
            variants = [(args, kwargs)]
            if any(
                p.kind == inspect.Parameter.VAR_POSITIONAL
                for p in up.parameters.values()
            ):
                variants.append((args + [object()] * (len(rep.parameters) + 2), kwargs))
            if any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in up.parameters.values()
            ):
                variants.append(
                    (args, {**kwargs, "__contract_extra_keyword": object()})
                )
            for call_args, call_kwargs in variants:
                try:
                    original = up.bind(*call_args, **call_kwargs)
                except TypeError:
                    continue  # Not a call accepted by upstream.
                try:
                    bound = rep.bind(*call_args, **call_kwargs)
                except TypeError as exc:
                    problems.append(
                        f"{len(call_args)} positional, keywords {sorted(call_kwargs)}: {exc}"
                    )
                    continue
                problems.extend(bound_value_problems(original, bound))
    return sorted(set(problems))
