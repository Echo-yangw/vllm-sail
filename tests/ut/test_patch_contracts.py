# SPDX-License-Identifier: Apache-2.0
"""Independent contracts replacing the derived lock/Markdown checks."""

from __future__ import annotations

import ast
import inspect

import pytest

from tests.support.patch_inventory import (
    dispatcher_hooks,
    registry_alias_hooks,
    scan_patch_file,
    scan_patches,
)
from tests.support.upstream import (
    FUNC_NODES,
    PATCH_ROOT,
    ROOT,
    bound_value_problems,
    callable_signature,
    compatibility_problems,
    dataclass_method,
    module_file,
    resolve_symbol,
)


def _feature_calls(root):
    names = {
        "_get_config_dtype_str",
        "moe_kernel_quantize_input",
        "Qwen3MoeSparseMoeBlock",
        "Qwen3MoeDecoderLayer",
        "decoder_layer_type",
    }
    calls = []
    for path in sorted((root / "vllm").rglob("*.py")):
        for call in ast.walk(ast.parse(path.read_text())):
            if isinstance(call, ast.Call):
                name = (
                    call.func.id
                    if isinstance(call.func, ast.Name)
                    else getattr(call.func, "attr", None)
                )
                if name in names:
                    calls.append((path, name, call))
    return calls


def actual_call_problems(root, hook, original, calls):
    """Narrow contracts for existing adapters whose extra PPU args alter order.

    Quantization helpers and Qwen layer factories are called with keywords by
    pinned upstream. Verify those real calls, rather than promising every call
    admitted by the old signature. Starred/unresolved forms fail closed.
    """
    leaf = hook.target.rsplit(".", 1)[-1]
    constructor = leaf == "__init__"
    if constructor:
        leaf = hook.target.split(".")[-2]
    upstream = callable_signature(original)
    replacement = callable_signature(hook.node)
    problems, checked = [], 0
    for path, name, call in calls:
        if constructor and path != root / "vllm/model_executor/models/qwen3_moe.py":
            continue
        if name != leaf and not (
            constructor
            and leaf == "Qwen3MoeDecoderLayer"
            and name == "decoder_layer_type"
        ):
            continue
        checked += 1
        location = f"{path.relative_to(root)}:{call.lineno}"
        if any(isinstance(arg, ast.Starred) for arg in call.args) or any(
            kw.arg is None for kw in call.keywords
        ):
            problems.append(
                f"{location}: unresolved expanded call needs a feature contract"
            )
            continue
        args = ([object()] if constructor else []) + [object() for _ in call.args]
        kwargs = {kw.arg: object() for kw in call.keywords}
        try:
            before = upstream.bind(*args, **kwargs)
            after = replacement.bind(*args, **kwargs)
        except TypeError as exc:
            problems.append(f"{location}: {exc}")
            continue
        problems.extend(
            f"{location}: {problem}" for problem in bound_value_problems(before, after)
        )
    return problems if checked else [f"no source calls found for {hook.target}"]


def imported_consumer_calls(root):
    """Index direct calls through module imports and local exported names.

    This deliberately handles source-visible imports and attribute chains only.
    Missing consumers and unresolved argument expansion require a separate
    feature contract instead of silently accepting an opaque callable.
    """
    calls = {}
    for path in sorted((root / "vllm").rglob("*.py")):
        parts = path.relative_to(root).with_suffix("").parts
        module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        package = (
            module.split(".")
            if path.name == "__init__.py"
            else module.split(".")[:-1]
        )
        tree = ast.parse(path.read_text())
        imports = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                owner = node.module or ""
                if node.level:
                    owner = ".".join(
                        [*package[: len(package) - node.level + 1], *owner.split(".")]
                    ).rstrip(".")
                for alias in node.names:
                    imports.setdefault(alias.asname or alias.name, set()).add(
                        f"{owner}.{alias.name}"
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name.split(".")[0]
                    imports.setdefault(name, set()).add(
                        alias.name if alias.asname else name
                    )
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call):
                continue
            attributes = []
            value = call.func
            while isinstance(value, ast.Attribute):
                attributes.insert(0, value.attr)
                value = value.value
            if not isinstance(value, ast.Name):
                continue
            # Keep every possible import binding: a nested scope must never
            # overwrite a module-level alias and hide one of its call shapes.
            owners = imports.get(value.id, {f"{module}.{value.id}"})
            targets = {".".join([owner, *attributes]) for owner in owners}
            if not attributes:
                # Rebinding an imported name patches this module's export too.
                targets.add(f"{module}.{value.id}")
            for target in targets:
                calls.setdefault(target, []).append((path, call))
    return calls


def consumer_call_problems(root, hook, calls):
    replacement = callable_signature(hook.node)
    found = calls.get(hook.target, [])
    if not found:
        return [f"no source calls found for opaque callable {hook.target}"]
    # A forwarding (*args, **kwargs) wrapper accepts every expansion. Anything
    # narrower must prove concrete bindings or supply a separate feature test.
    kinds = {parameter.kind for parameter in replacement.parameters.values()}
    forwards_all = kinds == {
        inspect.Parameter.VAR_POSITIONAL,
        inspect.Parameter.VAR_KEYWORD,
    }
    problems = []
    for path, call in found:
        location = f"{path.relative_to(root)}:{call.lineno}"
        expanded = any(isinstance(arg, ast.Starred) for arg in call.args) or any(
            keyword.arg is None for keyword in call.keywords
        )
        if expanded:
            if not forwards_all:
                problems.append(f"{location}: expanded call needs a feature contract")
            continue
        try:
            replacement.bind(
                *(object() for _ in call.args),
                **{keyword.arg: object() for keyword in call.keywords},
            )
        except TypeError as exc:
            problems.append(f"{location}: {exc}")
    return problems


def target_problems(root, hooks):
    failures = []
    calls = None
    consumers = None
    registry = ROOT / "vllm_sail/registry/moe_backends/__init__.py"
    alias_bodies = {
        hook.target: hook.node
        for hook in hooks
        if hook.path == registry and isinstance(hook.node, FUNC_NODES)
    }
    for hook in hooks:
        if hook.path == registry and hook.node is None:
            hook = hook._replace(node=alias_bodies.get(hook.target))
        resolved = resolve_symbol(root, hook.target)
        if not resolved.module_found:
            failures.append(f"{hook.target}: missing upstream module")
        elif hook.additive:
            if len(resolved.symbol_path) > 1:
                owner = hook.target.rsplit(".", 1)[0]
                if not resolve_symbol(root, owner).present:
                    failures.append(f"{hook.target}: missing additive owner {owner}")
        elif not resolved.present and not hook.conditional_alias:
            failures.append(f"{hook.target}: missing required target")
        elif isinstance(resolved.node, FUNC_NODES):
            if hook.node is None:
                # The Autotuner rebuild retains the original fn. The real
                # installer is exercised in test_fla_autotune_cache.py.
                if hook.path != PATCH_ROOT / "performance/fla/autotune_cache.py":
                    failures.append(
                        f"{hook.target}: unresolved callable replacement at {hook.path}:{hook.line}"
                    )
                continue
            problems = compatibility_problems(hook.node, resolved.node)
            # Existing feature-specific adapters deliberately add PPU inputs.
            # This small set expresses external call dependencies, not an
            # inventory of all patches or a signature snapshot.
            if hook.target.endswith(
                (
                    "._get_config_dtype_str",
                    ".moe_kernel_quantize_input",
                    ".Qwen3MoeSparseMoeBlock.__init__",
                    ".Qwen3MoeDecoderLayer.__init__",
                )
            ):
                if calls is None:
                    calls = _feature_calls(root)
                problems = actual_call_problems(root, hook, resolved.node, calls)
            failures.extend(f"{hook.target}: {problem}" for problem in problems)
        elif resolved.present and isinstance(hook.node, FUNC_NODES):
            if (
                isinstance(resolved.node, ast.ClassDef)
                and hook.path
                == PATCH_ROOT / "enhancement/attention/kv_cache_interface.py"
                and dataclass_method(resolved.node, hook.target.rsplit(".", 1)[-1])
            ):
                # The generated dataclass API has its own source dependency
                # checks and real constructor/replace/merge behavior tests.
                continue
            if consumers is None:
                consumers = imported_consumer_calls(root)
            failures.extend(
                f"{hook.target}: {problem}"
                for problem in consumer_call_problems(root, hook, consumers)
            )
    return failures


def _function(text):
    return ast.parse(text).body[0]


@pytest.mark.parametrize(
    "upstream,replacement,compatible",
    [
        ("def f(a: Unknown = dangerous()): ...", "def f(a=1): ...", True),
        ("def f(a): ... # comment", "def f(a, extra=None): ...", True),
        ("def f(a, *, new=1): ...", "def f(a, **kwargs): ...", True),
        ("def f(a, *, used=1): ...", "def f(a): ...", False),
        ("def f(a, b): ...", "def f(b, a, **kwargs): ...", False),
        ("def f(a): ...", "def f(a, required): ...", False),
        ("def f(a): ...", "def f(a, /, **kwargs): ...", False),
        ("def f(a, /): ...", "def f(*, a, **kwargs): ...", False),
        ("def f(a, *args): ...", "def f(a, **kwargs): ...", False),
        ("def f(a, **kwargs): ...", "def f(a, *args): ...", False),
    ],
)
def test_binding_contract(upstream, replacement, compatible):
    assert (
        not compatibility_problems(_function(replacement), _function(upstream))
    ) is compatible


@pytest.mark.parametrize("fault", [None, "missing-target", "missing-keyword"])
def test_discovery_to_source_gate_rejects_real_contract_faults(tmp_path, fault):
    upstream = tmp_path / "vllm/example.py"
    upstream.parent.mkdir()
    upstream.write_text("def run(a, *, used=None): ...\nclass Owner: pass\n")
    plugin = tmp_path / "plugin.py"
    name = "missing" if fault == "missing-target" else "run"
    signature = "a" if fault == "missing-keyword" else "a, *, used=None"
    plugin.write_text(
        f'@patch("vllm.example", "{name}", reason="test")\ndef replacement({signature}): ...\n'
    )
    hooks, unresolved = scan_patch_file(plugin)
    assert not unresolved
    problems = target_problems(tmp_path, hooks)
    assert bool(problems) is (fault is not None), problems
    if fault:
        assert name in " ".join(problems)


def test_discovery_covers_supported_forms_and_reports_unknowns(tmp_path):
    plugin = tmp_path / "plugin.py"
    plugin.write_text("""
from vllm import example as provider
from vllm_sail.patch.utils import patch as install
@patch("vllm.example")
def inferred(a): ...
@patch(provider.__name__, "Owner.method")
def method(self, a): ...
@install("vllm.example", "aliased")
def alias(a): ...
if condition:
    @patch("vllm.example", "Added", allow_missing=True)
    class Added: pass
for name in ("one", "two"):
    patch_value("vllm.example", name, object(), allow_missing=True)
patch(dynamic_module(), "unsupported")(method)
""")
    hooks, unresolved = scan_patch_file(plugin)
    assert {hook.target for hook in hooks} == {
        "vllm.example.inferred",
        "vllm.example.Owner.method",
        "vllm.example.aliased",
        "vllm.example.Added",
        "vllm.example.one",
        "vllm.example.two",
    }
    assert len(unresolved) == 1 and "dynamic_module()" in unresolved[0].expression


def test_discovery_reports_dictionary_unpack_instead_of_partial_success(tmp_path):
    plugin = tmp_path / "plugin.py"
    plugin.write_text("""
TARGETS = {"vllm.example": ("known",), **dynamic_targets()}
for module, names in TARGETS.items():
    for name in names:
        patch(module, name)(replacement)
""")
    hooks, unresolved = scan_patch_file(plugin)
    assert not hooks
    assert len(unresolved) == 1


def test_runtime_items_loop_does_not_hide_a_known_patch(tmp_path):
    plugin = tmp_path / "plugin.py"
    plugin.write_text("""
from vllm.models.example import model as provider
@patch("vllm.example", "run")
def replacement(a):
    mapping = provider.RUNTIME_TABLE
    for suffix, info in mapping.items():
        if suffix in a:
            return info
""")
    hooks, unresolved = scan_patch_file(plugin)
    assert [hook.target for hook in hooks] == ["vllm.example.run"]
    assert not unresolved


def test_runtime_items_installation_stays_unresolved(tmp_path):
    plugin = tmp_path / "plugin.py"
    plugin.write_text("""
from vllm.models.example import model as provider
def install():
    mapping = provider.RUNTIME_TABLE
    for name, replacement in mapping.items():
        patch("vllm.example", name)(replacement)
""")
    hooks, unresolved = scan_patch_file(plugin)
    assert not hooks
    assert len(unresolved) == 1
    assert unresolved[0].scope == "install"
    assert "vllm.example" in unresolved[0].expression


@pytest.mark.parametrize("present", [True, False])
def test_discovery_to_gate_resolves_reexport_providers(tmp_path, present):
    package = tmp_path / "vllm"
    package.mkdir()
    (package / "provider.py").write_text("def run(a): ..." if present else "# removed")
    (package / "consumer.py").write_text("from vllm.provider import run")
    plugin = tmp_path / "plugin.py"
    plugin.write_text('@patch("vllm.consumer", "run")\ndef replacement(a): ...')
    hooks, unresolved = scan_patch_file(plugin)
    assert not unresolved
    assert bool(target_problems(tmp_path, hooks)) is (not present)


def test_conditional_import_with_explicit_fallback_remains_a_real_target(tmp_path):
    package = tmp_path / "vllm"
    package.mkdir()
    (package / "consumer.py").write_text("""
if available:
    from vllm.optional import run
else:
    run = unavailable
""")
    assert resolve_symbol(tmp_path, "vllm.consumer.run").present


@pytest.mark.parametrize("guarded", [False, True])
@pytest.mark.parametrize("valid", [False, True])
@pytest.mark.parametrize("import_style", ["direct", "alias", "module", "local"])
def test_fallback_callable_checks_real_consumer_arguments(
    tmp_path, valid, import_style, guarded
):
    package = tmp_path / "vllm"
    package.mkdir()
    provider = package / "provider.py"
    provider.write_text("""
def unavailable(*args, **kwargs):
    raise RuntimeError("optional library unavailable")
if available:
    from vllm.optional import run
else:
    run = unavailable
""")
    consumer = {
        "direct": "from vllm.provider import run\nrun(1, used=2)\n",
        "alias": "from vllm.provider import run as invoke\ninvoke(1, used=2)\n",
        "module": "from vllm import provider as ops\nops.run(1, used=2)\n",
        "local": "run(1, used=2)\n",
    }[import_style]
    if import_style == "local":
        provider.write_text(provider.read_text() + consumer)
    else:
        (package / "consumer.py").write_text(consumer)
    plugin = tmp_path / "plugin.py"
    parameters = "*args, **kwargs" if valid else ""
    declaration = (
        '@patch("vllm.provider", "run")\n'
        f"def replacement({parameters}): pass\n"
    )
    if guarded:
        declaration = (
            'if getattr(consumer, "run", None) is original:\n'
            + "".join("    " + line for line in declaration.splitlines(keepends=True))
        )
    plugin.write_text(declaration)
    hooks, unresolved = scan_patch_file(plugin)
    assert not unresolved
    assert isinstance(resolve_symbol(tmp_path, "vllm.provider.run").node, ast.Assign)
    problems = target_problems(tmp_path, hooks)
    assert bool(problems) is (not valid), problems
    if not valid:
        assert "vllm.provider.run" in " ".join(problems)


@pytest.mark.parametrize("valid", [False, True])
def test_local_import_cannot_hide_another_scope_consumer(tmp_path, valid):
    package = tmp_path / "vllm"
    package.mkdir()
    (package / "provider.py").write_text("run = external_library\n")
    (package / "other.py").write_text("def run(a): pass\n")
    (package / "consumer.py").write_text("""
from vllm.provider import run as invoke
from vllm.provider import run
def use():
    invoke(1, used=2)
def unrelated():
    from vllm.other import run as invoke
    invoke(3)
run(1)
""")
    plugin = tmp_path / "plugin.py"
    parameters = "a, *, used=None" if valid else "a"
    plugin.write_text(
        '@patch("vllm.provider", "run")\n'
        f"def replacement({parameters}): pass\n"
    )
    hooks, _ = scan_patch_file(plugin)
    problems = target_problems(tmp_path, hooks)
    assert bool(problems) is (not valid), problems


def test_guarded_alias_missing_from_new_upstream_is_skipped(tmp_path):
    package = tmp_path / "vllm"
    package.mkdir()
    (package / "consumer.py").write_text("# the legacy import was removed\n")
    plugin = tmp_path / "plugin.py"
    plugin.write_text("""
if getattr(consumer, "run", None) is original:
    @patch("vllm.consumer", "run")
    def replacement(a): pass
""")
    hooks, unresolved = scan_patch_file(plugin)
    assert not unresolved and len(hooks) == 1
    assert hooks[0].conditional_alias
    assert not target_problems(tmp_path, hooks)


@pytest.mark.parametrize("guard_matches", [False, True])
@pytest.mark.parametrize("present", [False, True])
@pytest.mark.parametrize("valid", [False, True])
def test_loop_alias_guard_resolves_the_replacement_name(
    tmp_path, guard_matches, present, valid
):
    package = tmp_path / "vllm"
    package.mkdir()
    (package / "provider.py").write_text("def run(a): pass\n")
    parameters = "a" if valid else "a, *, used=None"
    (package / "consumer.py").write_text(
        f"def run({parameters}): pass\n" if present else "# legacy alias removed\n"
    )
    guard_name = "name" if guard_matches else '"unrelated"'
    plugin = tmp_path / "plugin.py"
    plugin.write_text(f"""
@patch("vllm.provider", "run")
def run(a): pass
def rebind():
    for replacement in (run,):
        name = replacement.__name__
        for consumer in ("vllm.consumer",):
            module = sys.modules.get(consumer)
            if module is not None and getattr(module, {guard_name}, None) is original:
                patch(consumer, name)(replacement)
""")
    hooks, unresolved = scan_patch_file(plugin)
    assert not unresolved
    assert {hook.target for hook in hooks} == {
        "vllm.provider.run",
        "vllm.consumer.run",
    }
    problems = target_problems(tmp_path, hooks)
    assert bool(problems) is (not valid if present else not guard_matches), problems


@pytest.mark.parametrize("descriptor", ["staticmethod", "classmethod"])
@pytest.mark.parametrize("valid", [False, True])
def test_descriptor_replacement_keeps_its_callable_contract(tmp_path, descriptor, valid):
    package = tmp_path / "vllm"
    package.mkdir()
    receiver = "cls, " if descriptor == "classmethod" else ""
    (package / "provider.py").write_text(
        f"class Owner:\n    @{descriptor}\n"
        f"    def run({receiver}a, *, used=None): pass\n"
    )
    parameters = f"{receiver}a" + (", *, used=None" if valid else "")
    plugin = tmp_path / "plugin.py"
    plugin.write_text(
        f"def replacement({parameters}): pass\n"
        f'patch("vllm.provider", "Owner.run")({descriptor}(replacement))\n'
    )
    hooks, unresolved = scan_patch_file(plugin)
    assert not unresolved
    problems = target_problems(tmp_path, hooks)
    assert bool(problems) is (not valid), problems
    if not valid:
        assert any("used" in problem for problem in problems)


@pytest.mark.parametrize("descriptor", ["staticmethod", "classmethod"])
@pytest.mark.parametrize("binding", ["assignment", "parameter", "vararg", "kwarg"])
def test_shadowed_descriptor_requires_a_callable_contract(tmp_path, descriptor, binding):
    package = tmp_path / "vllm"
    package.mkdir()
    (package / "provider.py").write_text("class Owner:\n    def run(a): pass\n")
    plugin = tmp_path / "plugin.py"
    declaration = (
        "def replacement(a): pass\n"
        f'patch("vllm.provider", "Owner.run")({descriptor}(replacement))\n'
    )
    if binding == "assignment":
        declaration = f"{descriptor} = external_decorator\n" + declaration
    else:
        prefix = {"parameter": "", "vararg": "*", "kwarg": "**"}[binding]
        declaration = f"def install({prefix}{descriptor}):\n" + "".join(
            "    " + line for line in declaration.splitlines(keepends=True)
        )
    plugin.write_text(declaration)
    hooks, unresolved = scan_patch_file(plugin)
    assert not unresolved
    assert any(
        "unresolved callable replacement" in problem
        for problem in target_problems(tmp_path, hooks)
    )


def test_unresolved_callable_without_consumers_requires_a_feature_contract(tmp_path):
    package = tmp_path / "vllm"
    package.mkdir()
    (package / "provider.py").write_text("run = external_library\n")
    plugin = tmp_path / "plugin.py"
    plugin.write_text('@patch("vllm.provider", "run")\ndef replacement(): pass\n')
    hooks, _ = scan_patch_file(plugin)
    assert "no source calls found" in " ".join(target_problems(tmp_path, hooks))


@pytest.mark.upstream_source
def test_flashmla_signature_fault_is_rejected_with_optional_source_missing(
    upstream_source_root,
):
    path = PATCH_ROOT / "enhancement/attention/flashmla_ops.py"
    hooks, unresolved = scan_patch_file(path)
    assert not unresolved
    target = "vllm.v1.attention.ops.flashmla.flash_mla_with_kvcache"
    provider = next(hook for hook in hooks if hook.target == target)
    assert not target_problems(upstream_source_root, [provider])
    broken = provider._replace(node=_function("def flash_mla_with_kvcache(): pass"))
    problems = target_problems(upstream_source_root, [broken])
    assert problems and any(target in problem for problem in problems)


def test_unconditional_dangling_import_cannot_be_repaired_by_later_assignment(tmp_path):
    package = tmp_path / "vllm"
    package.mkdir()
    (package / "consumer.py").write_text(
        "from vllm.missing import run\nrun = wrapper(run)\n"
    )
    assert not resolve_symbol(tmp_path, "vllm.consumer.run").present


def test_gate_checks_each_conditional_replacement_for_the_same_target(tmp_path):
    package = tmp_path / "vllm"
    package.mkdir()
    (package / "example.py").write_text("def run(a, *, used=None): ...")
    plugin = tmp_path / "plugin.py"
    plugin.write_text("""
if is_ppu:
    @patch("vllm.example", "run")
    def bad(a): ...
else:
    @patch("vllm.example", "run")
    def good(a, *, used=None): ...
""")
    hooks, unresolved = scan_patch_file(plugin)
    assert not unresolved and len(hooks) == 2
    assert target_problems(tmp_path, hooks)


@pytest.mark.parametrize("variation", ["replacement", "additive"])
def test_gate_preserves_distinct_variants_at_one_literal_loop_site(tmp_path, variation):
    package = tmp_path / "vllm"
    package.mkdir()
    (package / "example.py").write_text("def run(a, *, used=None): ...")
    plugin = tmp_path / "plugin.py"
    declarations = "def good(a, *, used=None): ...\ndef bad(a): ...\n"
    if variation == "replacement":
        declarations += """
for flag, body in ((True, good), (False, bad)):
    if is_ppu == flag:
        patch("vllm.example", "run")(body)
"""
    else:
        declarations += """
for additive in (True, False):
    patch("vllm.example", "missing", allow_missing=additive)(good)
"""
    plugin.write_text(declarations)
    hooks, unresolved = scan_patch_file(plugin)
    assert not unresolved and len(hooks) == 2
    assert target_problems(tmp_path, hooks)


def test_every_unresolved_installation_has_a_feature_contract():
    _, unresolved = scan_patches()
    # Only these two dispatcher mechanisms and dataclass-generated methods need
    # feature-specific source handling. A new opaque installation fails here.
    expected = {
        ("patch/enhancement/runtime.py", "_install", "module.__name__", "target"),
        ("patch/enhancement/runtime.py", "_install", "name", "attribute"),
        (
            "patch/enhancement/triton_moe.py",
            "_install",
            "module.__name__",
            "target_name",
        ),
        ("patch/enhancement/triton_moe.py", "_install", "consumer_name", "name"),
        (
            "registry/moe_backends/__init__.py",
            "_rebind_loaded_oracle_aliases",
            "consumer_name",
            "alias_name",
        ),
        *(
            (
                "patch/enhancement/attention/kv_cache_interface.py",
                "_extend_existing_subclass",
                "_MODULE",
                f"f'{{cls.__name__}}.{method}'",
            )
            for method in ("__init__", "__eq__", "__hash__")
        ),
    }
    for site in unresolved:
        call = ast.parse(site.expression, mode="eval").body
        key = (
            site.path.relative_to(ROOT / "vllm_sail").as_posix(),
            site.scope,
            *(ast.unparse(arg) for arg in call.args[:2]),
        )
        assert key in expected, (
            f"uncovered installation at {site.path}:{site.line}: {site.expression}"
        )


@pytest.mark.upstream_source
def test_declared_targets_and_calls_fit_pinned_source(upstream_source_root):
    hooks, _ = scan_patches()
    for name in ("runtime.py", "triton_moe.py"):
        hooks.extend(dispatcher_hooks(PATCH_ROOT / "enhancement" / name))
    hooks.extend(registry_alias_hooks(hooks))
    problems = target_problems(upstream_source_root, hooks)
    assert not problems, "\n".join(problems)


@pytest.mark.upstream_source
def test_patch_upstream_import_modules_resolve(upstream_source_root):
    imported = set()
    for path in (ROOT / "vllm_sail").rglob("*.py"):
        if "template" in path.parts or path.name.startswith("."):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.ImportFrom)
                and not node.level
                and node.module
                and node.module.startswith("vllm.")
            ):
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(
                    alias.name for alias in node.names if alias.name.startswith("vllm.")
                )
    assert imported
    missing = [
        name
        for name in sorted(imported)
        if module_file(upstream_source_root, name)[0] is None
    ]
    assert not missing, f"missing upstream import modules: {missing}"
