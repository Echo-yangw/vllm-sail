# SPDX-License-Identifier: Apache-2.0
"""Keep speculative sampling consumers on the platform-specific dispatcher."""

from __future__ import annotations

import ast
import importlib.util
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tests.support.source import ROOT, function

PROVIDER = "vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils"
CONSUMERS = (
    "vllm.v1.worker.gpu.spec_decode.rejection_sampler",
    "vllm.v1.watermarking.spec_decode",
)
RUNTIME = "vllm_sail/patch/enhancement/runtime.py"


def _declared_consumers():
    tree = ast.parse((ROOT / RUNTIME).read_text())
    declarations = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_CONSUMERS"
            for target in node.targets
        )
    ]
    assert len(declarations) == 1
    return ast.literal_eval(declarations[0].value)


def test_borrowed_bodies_keep_distinct_provider_globals_and_metadata():
    from vllm_sail.patch._upstream import runtime as bodies
    from vllm_sail.patch.bodies import bind_body

    runner, sampler = ModuleType("runner"), ModuleType("sampler")
    for body, provider in (
        (bodies.may_reinitialize_input_batch, runner),
        (bodies.rejection_sample, sampler),
    ):
        bound = bind_body(body, provider)
        assert bound.__globals__ is provider.__dict__
        assert bound.__code__ is body.__code__
        assert bound.__defaults__ is body.__defaults__
        assert bound.__kwdefaults__ is body.__kwdefaults__
        assert bound.__annotations__ == body.__annotations__
        assert bound.__doc__ == body.__doc__
    assert runner.__dict__ is not sampler.__dict__


@pytest.fixture
def installer(modules, monkeypatch, patch_utils_module):
    provider = modules(PROVIDER)
    loaded = sys.modules
    patches = []
    platform = SimpleNamespace(ppu=True)

    def original(*args, **kwargs):
        return "upstream", args, kwargs

    def replacement(*args, **kwargs):
        return "ppu", args, kwargs

    provider.rejection_sample = original
    # The body is a collaborator of the installer seam. Numerical/geometry
    # tests exercise the real borrowed bodies; here distinguish dispatch paths.
    from vllm_sail.patch._upstream import runtime as bodies

    monkeypatch.setattr(bodies, "rejection_sample", replacement)
    modules(
        "vllm.platforms", current_platform=SimpleNamespace(is_ppu=lambda: platform.ppu)
    )
    modules(
        "vllm.v1.worker.gpu_model_runner",
        GPUModelRunner=type(
            "GPUModelRunner", (), {"may_reinitialize_input_batch": original}
        ),
    )
    for name in CONSUMERS:
        monkeypatch.delitem(sys.modules, name, raising=False)

    def install(source=None):
        path = ROOT / RUNTIME if source is None else source
        spec = importlib.util.spec_from_file_location("_runtime_installer", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        patches.extend(
            tuple(record.target.rsplit(".", 1))
            for record in patch_utils_module.PATCH_REGISTRY
            if record.target.endswith(".rejection_sample")
        )
        return module

    return SimpleNamespace(
        provider=provider,
        modules=loaded,
        patches=patches,
        platform=platform,
        original=original,
        replacement=replacement,
        install=install,
        add_consumer=lambda name, consumer: monkeypatch.setitem(loaded, name, consumer),
    )


@pytest.mark.parametrize("consumer_name", CONSUMERS)
@pytest.mark.parametrize("is_ppu", [True, False])
def test_preloaded_consumer_uses_the_same_platform_dispatcher(
    installer, consumer_name, is_ppu
):
    consumer = ModuleType(consumer_name)
    consumer.rejection_sample = installer.original
    installer.add_consumer(consumer_name, consumer)
    installer.platform.ppu = is_ppu
    installer.install()

    assert consumer.rejection_sample is installer.provider.rejection_sample
    token, contexts = object(), object()
    assert consumer.rejection_sample(token, contexts=contexts) == (
        "ppu" if is_ppu else "upstream",
        (token,),
        {"contexts": contexts},
    )


@pytest.mark.parametrize("consumer_name", CONSUMERS)
def test_consumer_with_a_different_binding_is_preserved(installer, consumer_name):
    consumer = ModuleType(consumer_name)

    def unrelated():
        return "unrelated"

    consumer.rejection_sample = unrelated
    installer.add_consumer(consumer_name, consumer)
    installer.install()

    assert consumer.rejection_sample is unrelated
    assert installer.patches == [(PROVIDER, "rejection_sample")]


def test_cold_install_does_not_load_consumers(installer):
    installer.install()

    assert all(name not in installer.modules for name in CONSUMERS)
    assert installer.provider.rejection_sample() == ("ppu", (), {})
    assert installer.patches == [(PROVIDER, "rejection_sample")]


@pytest.mark.parametrize("fault", ["consumer-rebind", "platform-branch"])
def test_installer_checks_detect_controlled_faults(installer, tmp_path, fault):
    """Inject faults into a temporary installer, then use the same observations."""
    tree = ast.parse((ROOT / RUNTIME).read_text())
    install = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_install"
    )
    if fault == "consumer-rebind":
        install.body = [node for node in install.body if not isinstance(node, ast.If)]
    else:
        dispatch = next(
            node for node in install.body if isinstance(node, ast.FunctionDef)
        )
        branch = next(node for node in dispatch.body if isinstance(node, ast.If))
        branch.test = ast.Constant(True)
    source = tmp_path / "runtime.py"
    source.write_text(ast.unparse(ast.fix_missing_locations(tree)))
    consumer = ModuleType(CONSUMERS[0])
    consumer.rejection_sample = installer.original
    installer.add_consumer(consumer.__name__, consumer)
    installer.platform.ppu = False
    installer.install(source)
    if fault == "consumer-rebind":
        with pytest.raises(AssertionError):
            assert consumer.rejection_sample is installer.provider.rejection_sample
    else:
        with pytest.raises(AssertionError):
            assert consumer.rejection_sample() == ("upstream", (), {})


@pytest.mark.upstream_source
def test_consumer_inventory_matches_all_module_scope_imports(upstream_source_root):
    discovered = set()
    for path in (upstream_source_root / "vllm").rglob("*.py"):
        relative = path.relative_to(upstream_source_root).with_suffix("")
        parts = list(relative.parts)
        is_package = parts[-1] == "__init__"
        if is_package:
            parts.pop()
        module_name = ".".join(parts)
        package = module_name if is_package else module_name.rpartition(".")[0]
        pending = list(ast.parse(path.read_text()).body)
        while pending:
            node = pending.pop()
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                continue
            if isinstance(node, ast.ImportFrom):
                imported = node.module or ""
                if node.level:
                    imported = importlib.util.resolve_name(
                        "." * node.level + imported, package
                    )
                if imported == PROVIDER:
                    for alias in node.names:
                        if alias.name in {"rejection_sample", "*"}:
                            discovered.add(
                                (module_name, alias.asname or "rejection_sample")
                            )
            pending.extend(ast.iter_child_nodes(node))

    assert discovered == {(name, "rejection_sample") for name in _declared_consumers()}


@pytest.mark.upstream_source
@pytest.mark.parametrize("is_ppu", [True, False])
def test_real_watermark_wrapper_preserves_sampling_inputs(
    installer, upstream_source_root, is_ppu
):
    consumer = ModuleType("vllm.v1.watermarking.spec_decode")
    consumer.rejection_sample = installer.original
    installer.add_consumer(consumer.__name__, consumer)
    installer.platform.ppu = is_ppu
    installer.install()
    watermarker = SimpleNamespace(context_width=3)

    def resolve_key(actual):
        assert actual is watermarker
        return 123

    wrapper = function(
        upstream_source_root / "vllm/v1/watermarking/spec_decode.py",
        "watermarked_rejection_sample",
        dict(
            rejection_sample=consumer.rejection_sample,
            _resolve_watermark_key=resolve_key,
        ),
    )
    args = [SimpleNamespace(shape=(2, 32)), *(object() for _ in range(9)), 7]
    contexts = SimpleNamespace(shape=(2, 3))
    watermarking, skip = object(), object()
    result = wrapper(
        *args,
        contexts,
        watermarking,
        watermarker,
        watermarking_skip_mask=skip,
        use_fp64=True,
    )

    assert result == (
        "ppu" if is_ppu else "upstream",
        tuple(args),
        dict(
            use_fp64=True,
            contexts=contexts,
            watermarking=watermarking,
            watermarking_skip_mask=skip,
            watermark_key=123,
        ),
    )
