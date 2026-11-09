# SPDX-License-Identifier: Apache-2.0
"""Bounded static discovery of patch declarations, never a runtime inventory.

Understands strings, named constants/import aliases, and finite literal loops.
It visits every conditional branch. It does not execute imports, defaults,
installers or arbitrary expressions. Unresolved sites are returned with their
location so the gate must cover them explicitly or fail.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, NamedTuple

from tests.support.upstream import (
    DEF_NODES,
    PATCH_ROOT,
    ROOT,
    SKIP_DIRS,
    resolve_symbol,
)


class Hook(NamedTuple):
    target: str
    additive: bool
    node: ast.AST | str | None
    path: Path
    line: int
    kind: str = "replacement"
    conditional_alias: bool = False


class Unresolved(NamedTuple):
    path: Path
    line: int
    expression: str
    scope: str


class Unknown(ValueError):
    pass


def _value(node: ast.AST, env: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env or env[node.id] is None:
            raise Unknown(node.id)
        return env[node.id]
    if isinstance(node, ast.Attribute):
        value = _value(node.value, env)
        if node.attr == "__name__":
            return value.name if isinstance(value, DEF_NODES) else value
        if isinstance(value, str):
            return f"{value}.{node.attr}"
    if isinstance(node, ast.JoinedStr):
        if any(
            isinstance(part, ast.FormattedValue)
            and (part.conversion != -1 or part.format_spec is not None)
            for part in node.values
        ):
            raise Unknown("formatted targets need an explicit source contract")
        return "".join(
            str(_value(part.value, env))
            if isinstance(part, ast.FormattedValue)
            else str(_value(part, env))
            for part in node.values
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _value(node.left, env) + _value(node.right, env)
    if isinstance(node, ast.Subscript):
        return _value(node.value, env)[_value(node.slice, env)]
    if isinstance(node, ast.Tuple | ast.List):
        return tuple(_maybe_value(item, env) for item in node.elts)
    if isinstance(node, ast.Dict):
        if any(key is None for key in node.keys):
            raise Unknown("dictionary unpack needs an explicit source contract")
        return {
            _value(key, env): _maybe_value(value, env)
            for key, value in zip(node.keys, node.values, strict=False)
            if key is not None
        }
    if isinstance(node, ast.Call):
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in {"staticmethod", "classmethod"}
            and node.func.id not in env
            and len(node.args) == 1
            and not node.keywords
        ):
            # Built-in method descriptors keep the authored callable body.
            return _value(node.args[0], env)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) == 2
        ):
            owner, name = (_value(arg, env) for arg in node.args)
            if isinstance(owner, str) and isinstance(name, str):
                return f"{owner}.{name}"
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "items"
            and not node.args
            and not node.keywords
        ):
            mapping = _value(node.func.value, env)
            if not isinstance(mapping, dict):
                raise Unknown(ast.unparse(node))
            return tuple(mapping.items())
        if isinstance(node.func, ast.Name) and node.func.id in {
            "bind_body",
            "_with_target_globals",
        }:
            return _value(node.args[0], env)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "eager_break_during_capture"
        ):
            return _value(node.args[0], env)
        if isinstance(node.func, ast.Name) and isinstance(
            env.get(node.func.id), ast.FunctionDef
        ):
            factory = env[node.func.id]
            if isinstance(factory.body[-1], ast.Return) and isinstance(
                factory.body[-1].value, ast.Name
            ):
                name = factory.body[-1].value.id
                nested = [
                    child
                    for child in factory.body
                    if isinstance(child, ast.FunctionDef) and child.name == name
                ]
                if len(nested) == 1:
                    return nested[0]
        # A decorated assignment still denotes the same authored function body.
        if (
            isinstance(node.func, ast.Call)
            and isinstance(node.func.func, ast.Name)
            and node.func.func.id == "patch"
        ):
            return _value(node.args[0], env)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "tuple"
            and len(node.args) == 1
        ):
            gen = node.args[0]
            if isinstance(gen, ast.GeneratorExp) and len(gen.generators) == 1:
                loop = gen.generators[0]
                if not loop.ifs and isinstance(loop.target, ast.Name):
                    return tuple(
                        _value(gen.elt, {**env, loop.target.id: item})
                        for item in _value(loop.iter, env)
                    )
    raise Unknown(ast.unparse(node))


def _maybe_value(node: ast.AST, env: dict[str, Any]) -> Any:
    try:
        return _value(node, env)
    except (Unknown, KeyError, TypeError):
        return None


def _assign(target: ast.AST, value: Any, env: dict[str, Any]) -> None:
    if isinstance(target, ast.Name):
        env[target.id] = value
    elif isinstance(target, ast.Tuple | ast.List):
        for child, item in zip(
            target.elts, value or [None] * len(target.elts), strict=False
        ):
            _assign(child, item, env)


def scan_patch_file(path: Path) -> tuple[list[Hook], list[Unresolved]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    hooks, unresolved = [], []
    seen = set()

    def inspect_call(call, env, scope, replacement=None):
        if not isinstance(call, ast.Call):
            return
        reference = _maybe_value(call.func, env)
        name = call.func.id if isinstance(call.func, ast.Name) else None
        if isinstance(reference, str) and reference in {
            f"vllm_sail.patch{module}.{kind}"
            for module in ("", ".utils")
            for kind in ("patch", "patch_value", "PatchRecord")
        }:
            name = reference.rsplit(".", 1)[-1]
        if name == "PatchRecord":
            fields = {kw.arg: _maybe_value(kw.value, env) for kw in call.keywords}
            target, kind, missing = (
                fields.get(name) for name in ("target", "kind", "was_missing")
            )
            if (
                isinstance(target, str)
                and kind == "value"
                and isinstance(missing, bool)
            ):
                hooks.append(Hook(target, missing, None, path, call.lineno, "value"))
            else:
                unresolved.append(
                    Unresolved(path, call.lineno, ast.unparse(call), scope)
                )
            return
        if name not in {"patch", "patch_value"}:
            return
        if replacement is None:
            outer = parents.get(call)
            if isinstance(outer, ast.Call) and outer.func is call and outer.args:
                replacement = _maybe_value(outer.args[0], env)
        module = _maybe_value(call.args[0], env) if call.args else None
        attr = (
            _maybe_value(call.args[1], env)
            if len(call.args) > 1
            else getattr(replacement, "name", None)
        )
        additive = False
        for kw in call.keywords:
            if kw.arg == "allow_missing":
                additive = _maybe_value(kw.value, env)
        if (
            not isinstance(module, str)
            or not isinstance(attr, str)
            or not isinstance(additive, bool)
        ):
            unresolved.append(Unresolved(path, call.lineno, ast.unparse(call), scope))
            return
        key = (call.lineno, module, attr, additive, id(replacement))
        if key not in seen:
            seen.add(key)
            parent = parents.get(call)
            conditional_alias = False
            while parent is not None:
                if isinstance(parent, ast.If):
                    conditional_alias |= any(
                        isinstance(part, ast.Call)
                        and isinstance(part.func, ast.Name)
                        and part.func.id == "getattr"
                        and len(part.args) == 3
                        and _maybe_value(part.args[1], env) == attr
                        and isinstance(part.args[2], ast.Constant)
                        and part.args[2].value is None
                        for part in ast.walk(parent.test)
                    )
                parent = parents.get(parent)
            hooks.append(
                Hook(
                    f"{module}.{attr}",
                    additive,
                    replacement if isinstance(replacement, (*DEF_NODES, str)) else None,
                    path,
                    call.lineno,
                    "value" if name == "patch_value" else "replacement",
                    conditional_alias,
                )
            )

    def statements(body, env, scope=""):
        env = env.copy()
        for node in body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    env[alias.asname or alias.name.split(".")[0]] = (
                        alias.name if alias.asname else alias.name.split(".")[0]
                    )
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                for alias in node.names:
                    env[alias.asname or alias.name] = f"{node.module}.{alias.name}"
            elif isinstance(node, ast.Assign | ast.AnnAssign):
                value = _maybe_value(node.value, env) if node.value else None
                for target in (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                ):
                    _assign(target, value, env)
            if isinstance(node, DEF_NODES):
                env[node.name] = node
                for decorator in node.decorator_list:
                    inspect_call(decorator, env, scope, node)
                nested = env.copy()
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    for arg in (
                        node.args.posonlyargs + node.args.args + node.args.kwonlyargs
                    ):
                        nested[arg.arg] = None
                    for arg in (node.args.vararg, node.args.kwarg):
                        if arg is not None:
                            nested[arg.arg] = None
                statements(node.body, nested, node.name)
            elif isinstance(node, ast.For):
                values = _maybe_value(node.iter, env)
                for value in values if values is not None else [None]:
                    nested = env.copy()
                    _assign(node.target, value, nested)
                    statements(node.body, nested, scope)
                statements(node.orelse, env, scope)
            elif isinstance(node, ast.If | ast.Try | ast.With):
                for part in ("body", "orelse", "finalbody"):
                    statements(getattr(node, part, []), env, scope)
                for handler in getattr(node, "handlers", []):
                    statements(handler.body, env, scope)
            else:
                # Do not walk a function/loop twice with the wrong bindings.
                for child in ast.walk(node):
                    inspect_call(child, env, scope)

    statements(tree.body, {})
    providers = {
        hook.target: hook.node for hook in hooks if isinstance(hook.node, DEF_NODES)
    }
    for index, hook in enumerate(hooks):
        body = hook.node
        if isinstance(body, str):
            body = providers.get(body) or (
                resolve_symbol(ROOT, body).node
                if body.startswith("vllm_sail.")
                else None
            )
        if body is None and hook.conditional_alias:
            candidates = [
                other.node
                for other in hooks
                if not other.conditional_alias
                and other.target.rsplit(".", 1)[-1] == hook.target.rsplit(".", 1)[-1]
                and isinstance(other.node, DEF_NODES)
            ]
            if len(candidates) == 1:
                body = candidates[0]
        hooks[index] = hook._replace(node=body)
    return hooks, sorted(set(unresolved))


def scan_patches(
    root: Path = ROOT / "vllm_sail",
) -> tuple[list[Hook], list[Unresolved]]:
    hooks, unresolved = [], []
    for path in sorted(root.rglob("*.py")):
        if SKIP_DIRS & set(path.parts) or path == PATCH_ROOT / "utils.py":
            continue
        found, missing = scan_patch_file(path)
        hooks.extend(found)
        unresolved.extend(missing)
    return hooks, unresolved


def registry_alias_hooks(providers: list[Hook] | None = None) -> list[Hook]:
    """Read the oracle alias declaration actually consumed by its installer."""
    path = ROOT / "vllm_sail/registry/moe_backends/__init__.py"
    tree = ast.parse(path.read_text())
    inventory = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_ORACLE_ALIAS_CONSUMERS"
            for target in node.targets
        )
    )
    providers = scan_patches()[0] if providers is None else providers
    hooks = []
    for backend, aliases in inventory.items():
        for name, consumers in aliases.items():
            provider = f"vllm.model_executor.layers.fused_moe.oracle.{backend}.{name}"
            body = next(hook.node for hook in providers if hook.target == provider)
            for consumer in consumers:
                hooks.append(Hook(f"{consumer}.{name}", False, body, path, 0))
    return hooks


def dispatcher_hooks(path: Path) -> list[Hook]:
    """Read the two existing bind_body installers' actual invocation arguments.

    This is a feature-specific source contract, not an interpreter for installer
    bodies. Their forwarding, platform choice and aliases have behavior tests.
    """
    tree = ast.parse(path.read_text())
    env = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                env[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        elif isinstance(node, ast.FunctionDef):
            env[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                _assign(target, _maybe_value(node.value, env), env)
    found = []
    for node in tree.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not isinstance(call.func, ast.Name) or call.func.id != "_install":
            continue
        module = _value(call.args[0], env)
        if path.name == "runtime.py":
            owner, name, body = (_value(arg, env) for arg in call.args[1:])
            attribute = (
                name
                if owner == module
                else owner.removeprefix(module + ".") + "." + name
            )
        else:
            name, body = (_value(arg, env) for arg in call.args[1:])
            attribute = next(
                (
                    _value(kw.value, env)
                    for kw in call.keywords
                    if kw.arg == "target_name"
                ),
                name,
            )
        if not isinstance(body, ast.FunctionDef):
            # Imported pilot bodies are resolved from their real definition.
            if not isinstance(body, str) or not body.startswith(
                "vllm_sail.patch._upstream.runtime."
            ):
                raise Unknown(f"unresolved dispatcher body at {path}:{call.lineno}")
            from tests.support.source import definition

            body = definition(
                PATCH_ROOT / "_upstream/runtime.py", body.rsplit(".", 1)[-1]
            )
        found.append(Hook(f"{module}.{attribute}", False, body, path, call.lineno))
        consumers = env["_CONSUMERS"]
        consumers = (
            consumers.get(name, ())
            if isinstance(consumers, dict)
            else (consumers if owner == module else ())
        )
        for consumer in consumers:
            found.append(Hook(f"{consumer}.{name}", False, body, path, call.lineno))
    if not found:
        raise Unknown(f"no dispatcher invocations in {path}")
    return found
