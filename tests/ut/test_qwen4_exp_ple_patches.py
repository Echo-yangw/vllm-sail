# SPDX-License-Identifier: Apache-2.0
"""Tests for the Qwen4Exp PLE compressed-tensors leaf patch module.

Covers ``patch/enhancement/qwen4_exp_ple.py``. Same two-tier design as
``test_quant_moe_patches.py``: it runs on a bare CPU runner with no vLLM
(import + metadata shape + AST checks) and exercises the dispatch against stub
upstreams, so the branch a ``CompressedTensorsConfig`` or ``Mxfp4Config``
checkpoint takes is asserted rather than merely installed.

The stub classes are rebuilt per test: ``install()`` patches them in place, and
the framework's double-patch guard rejects a class that already carries a
marker.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from vllm_sail.patch.utils import PATCH_MARKER, PATCH_REGISTRY

ENHANCEMENT_DIR = Path(__file__).parents[2] / "vllm_sail" / "patch" / "enhancement"
MODULE_NAME = "qwen4_exp_ple"
REQUIRED_METADATA = ("reason", "affected_versions", "remove_when")

_NGRAM = "vllm.models.qwen4_exp.common.ngram_embedding"
_CT = "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors"
_CT_UTILS = "vllm.model_executor.layers.quantization.compressed_tensors.utils"
_MXFP4 = "vllm.model_executor.layers.quantization.mxfp4"
_QUANT_UTILS = "vllm.model_executor.layers.quantization.utils.quant_utils"

# The prefix the caller builds (``f"{prefix}.ngram_embedding"``) and the shard
# names a real fp8-channel checkpoint lists in ``quantization_config.ignore``.
PREFIX = "ple.ple_embedding.ngram_embedding"
SPLIT_NGRAM_PARTS = 128

_CREATED: list[str] = []


@pytest.fixture(autouse=True)
def _cleanup_registry() -> Iterator[None]:
    """PATCH_REGISTRY is process-global; drop records these tests add."""
    before = len(PATCH_REGISTRY)
    yield
    del PATCH_REGISTRY[before:]
    for name in _CREATED:
        sys.modules.pop(name, None)
    _CREATED.clear()


def _leaf() -> types.ModuleType:
    """Load a fresh leaf-module instance (install state must not leak)."""
    module_name = f"_qwen4_exp_ple_test_{id(object())}"
    spec = importlib.util.spec_from_file_location(
        module_name, ENHANCEMENT_DIR / f"{MODULE_NAME}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    _CREATED.append(module_name)
    spec.loader.exec_module(module)
    return module


def _register_chain(
    monkeypatch: pytest.MonkeyPatch, dotted: str, module: types.ModuleType
) -> None:
    """Put ``module``, plus stub parents for every prefix, into ``sys.modules``."""
    parts = dotted.split(".")
    chain = [".".join(parts[: i + 1]) for i in range(len(parts))]
    modules: dict[str, types.ModuleType] = {}
    for name in chain[:-1]:
        existing = sys.modules.get(name)
        stub = (
            existing
            if isinstance(existing, types.ModuleType)
            else types.ModuleType(name)
        )
        modules[name] = stub
        monkeypatch.setitem(sys.modules, name, stub)
    modules[dotted] = module
    monkeypatch.setitem(sys.modules, dotted, module)
    for parent_name, child_name in zip(chain, chain[1:], strict=False):
        setattr(
            modules[parent_name], child_name.rpartition(".")[2], modules[child_name]
        )


# ---------------------------------------------------------------------------
# Stub upstream
# ---------------------------------------------------------------------------


def _build_stubs() -> types.SimpleNamespace:
    """Fresh stand-ins for the five upstream modules ``install()`` imports."""

    class Qwen4ExpPLEUnquantizedEmbeddingMethod:
        """The bf16 PLE table."""

    class Qwen4ExpPLEFp8EmbeddingMethod:
        """The FP8 PLE table."""

    class Fp8Config:
        """A config family upstream's whitelist does handle."""

    class CompressedTensorsConfig:
        """The llm-compressor config the PPU quantization pipeline emits."""

        def __init__(self, ignore: list[str] | None = None) -> None:
            self.ignore = list(ignore or [])
            self.packed_modules_mapping: dict[str, list[str]] = {}

    class Mxfp4Config:
        """The MXFP4-expert / FP8-channelwise-dense config the pipeline emits.

        ``ignored_layers`` is what SAIL's ``enhancement/mxfp4.py`` populates
        from the checkpoint's ``ignore`` key. Upstream's own ``from_config``
        returns ``cls()`` and drops the key, so ``None`` is also the shape a
        checkpoint that genuinely quantizes the PLE table arrives in.
        """

        def __init__(self, ignored_layers: list[str] | None = None) -> None:
            self.ignored_layers = (
                None if ignored_layers is None else list(ignored_layers)
            )
            self.packed_modules_mapping: dict[str, list[str]] = {}

    class SomeOtherConfig:
        """A family neither upstream nor the patch knows about."""

    def from_quant_config(
        quant_config: Any, prefix: str, embedding_dtype: str | None = None
    ) -> Any:
        """Mirror of upstream's fail-closed dispatcher.

        The order is what the patch must preserve: a declared FP8 PLE dtype
        wins first, ``None`` next, then the whitelisted families, and anything
        else raises.
        """
        if embedding_dtype == "float8_e4m3fn":
            return Qwen4ExpPLEFp8EmbeddingMethod()
        if quant_config is None:
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        if not isinstance(quant_config, Fp8Config):
            raise NotImplementedError(
                "Qwen4Exp PLE embedding does not support quantization config "
                f"{type(quant_config).__name__}"
            )
        return Qwen4ExpPLEFp8EmbeddingMethod()

    # One descriptor object, reused below: the framework records the exact
    # object it replaced, and the test asserts that identity.
    descriptor = staticmethod(from_quant_config)

    class Qwen4ExpPLEEmbeddingMethod:
        from_quant_config = descriptor

    # Which exclusion helper the patch consulted, so each family stays pinned
    # to its own matcher rather than whichever one happens to be imported.
    helper_calls: list[str] = []

    def should_ignore_layer(layer_name, ignore, fused_mapping) -> bool:
        """Enough of upstream's helper for a prefix naming no fused module.

        No defaults: the patch must pass ``ignore`` and ``fused_mapping``
        explicitly, and a stub stricter than upstream pins that call shape.
        """
        helper_calls.append("should_ignore_layer")
        return layer_name is not None and layer_name in set(ignore)

    def is_layer_skipped(prefix, ignored_layers, fused_mapping) -> bool:
        """Enough of upstream's MXFP4-side helper for the same prefix.

        Equally strict: ``fused_mapping`` has an upstream default the stub
        omits, so the patch is pinned to passing the config's own mapping.
        """
        helper_calls.append("is_layer_skipped")
        return prefix is not None and prefix in set(ignored_layers)

    return types.SimpleNamespace(
        unquantized=Qwen4ExpPLEUnquantizedEmbeddingMethod,
        fp8=Qwen4ExpPLEFp8EmbeddingMethod,
        fp8_config=Fp8Config,
        ct_config=CompressedTensorsConfig,
        mxfp4_config=Mxfp4Config,
        other_config=SomeOtherConfig,
        embedding_method=Qwen4ExpPLEEmbeddingMethod,
        upstream_from_quant_config=descriptor,
        should_ignore_layer=should_ignore_layer,
        is_layer_skipped=is_layer_skipped,
        helper_calls=helper_calls,
    )


def _install(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    """Stub the upstream modules, install the patch, return the stubs."""
    stubs = _build_stubs()

    ngram = types.ModuleType(_NGRAM)
    ngram.Qwen4ExpPLEEmbeddingMethod = stubs.embedding_method
    ngram.Qwen4ExpPLEUnquantizedEmbeddingMethod = stubs.unquantized
    ngram.Qwen4ExpPLEFp8EmbeddingMethod = stubs.fp8
    _register_chain(monkeypatch, _NGRAM, ngram)

    ct = types.ModuleType(_CT)
    ct.CompressedTensorsConfig = stubs.ct_config
    _register_chain(monkeypatch, _CT, ct)

    ct_utils = types.ModuleType(_CT_UTILS)
    ct_utils.should_ignore_layer = stubs.should_ignore_layer
    _register_chain(monkeypatch, _CT_UTILS, ct_utils)

    mxfp4 = types.ModuleType(_MXFP4)
    mxfp4.Mxfp4Config = stubs.mxfp4_config
    _register_chain(monkeypatch, _MXFP4, mxfp4)

    quant_utils = types.ModuleType(_QUANT_UTILS)
    quant_utils.is_layer_skipped = stubs.is_layer_skipped
    _register_chain(monkeypatch, _QUANT_UTILS, quant_utils)

    stubs.module = _leaf()
    stubs.module.install()
    stubs.dispatch = stubs.embedding_method.from_quant_config
    return stubs


def _ignored_shard_names() -> list[str]:
    """The PLE entries of a real fp8-channel ``quantization_config.ignore``."""
    return [f"{PREFIX}.shard_{i}" for i in range(SPLIT_NGRAM_PARTS)] + [
        "ple.conv1d",
        "ple.key_proj",
        "ple.norm_conv",
        "ple.norm_key",
        "ple.norm_query",
        "lm_head",
    ]


# ---------------------------------------------------------------------------
# Bare-CPU import and metadata shape
# ---------------------------------------------------------------------------


def test_module_imports_on_bare_cpu() -> None:
    module = _leaf()
    if "vllm" in sys.modules:
        pytest.skip("bare-CPU contract test; real vLLM is loaded (mode B)")
    assert callable(module.install)
    assert module.__doc__


def test_metadata_shape() -> None:
    metadata = _leaf().METADATA
    assert len(metadata) == 1
    target, reason, affected, remove_when = metadata[0]
    assert target == f"{_NGRAM}.Qwen4ExpPLEEmbeddingMethod.from_quant_config"
    for value in (reason, affected, remove_when):
        assert isinstance(value, str) and value.strip(), target
    assert ">=" in affected and "<" in affected, target
    assert remove_when.strip().lower() != "todo", target
    assert "CompressedTensorsConfig" in reason
    assert "Mxfp4Config" in reason


def test_no_top_level_vllm_import() -> None:
    tree = ast.parse((ENHANCEMENT_DIR / f"{MODULE_NAME}.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for imported in names:
            assert imported != "vllm" and not imported.startswith("vllm."), (
                f"{MODULE_NAME} imports {imported!r} at module level"
            )


def _callee_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def test_every_patch_call_carries_metadata() -> None:
    tree = ast.parse((ENHANCEMENT_DIR / f"{MODULE_NAME}.py").read_text())
    watched = {"patch", "patch_value", "PatchRecord"}
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _callee_name(node) in watched
    ]
    decorated = [
        decorator
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        for decorator in node.decorator_list
        if isinstance(decorator, ast.Call) and _callee_name(decorator) == "patch"
    ]
    assert len(decorated) == len(_leaf().METADATA)
    for call in calls:
        keywords = {kw.arg for kw in call.keywords}
        missing = set(REQUIRED_METADATA) - keywords
        assert not missing, f"patch call at line {call.lineno} lacks {missing}"


def test_wired_into_the_enhancement_package() -> None:
    """``install()`` must run as part of ``vllm_sail.patch.install()``.

    A leaf-style module nobody calls is a silent no-op, so the two lines in the
    parent ``__init__`` are load-bearing and deleting them must fail here.
    """
    text = (ENHANCEMENT_DIR / "__init__.py").read_text()
    assert f"import {MODULE_NAME}" in text, "leaf module is not imported"
    assert f"{MODULE_NAME}.install()" in text, "leaf module is never installed"


# ---------------------------------------------------------------------------
# Behaviour against stubs
# ---------------------------------------------------------------------------


def test_install_registers_the_class_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = len(PATCH_REGISTRY)
    stubs = _install(monkeypatch)

    records = PATCH_REGISTRY[before:]
    assert len(records) == 1
    record = records[0]
    assert record.target == stubs.module.METADATA[0][0]
    assert record.kind == "class"
    assert record.was_missing is False
    assert (record.reason, record.affected_versions, record.remove_when) == (
        stubs.module.METADATA[0][1:]
    )

    markers = getattr(stubs.dispatch, PATCH_MARKER, None)
    assert isinstance(markers, dict) and record.target in markers
    assert markers[record.target] is stubs.upstream_from_quant_config

    stubs.module.install()  # idempotent: the double-patch guard must not fire
    assert len(PATCH_REGISTRY[before:]) == 1


def test_compressed_tensors_with_ignored_shards_loads_unquantized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reported failure: every PLE shard is in ``ignore``, loading died.

    Regression guard for ``NotImplementedError: Qwen4Exp PLE embedding does not
    support quantization config CompressedTensorsConfig``.
    """
    stubs = _install(monkeypatch)
    config = stubs.ct_config(ignore=_ignored_shard_names())
    assert isinstance(stubs.dispatch(config, PREFIX), stubs.unquantized)


def test_compressed_tensors_naming_the_table_directly_is_honoured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ignore list naming the table itself takes the should_ignore_layer path."""
    stubs = _install(monkeypatch)
    config = stubs.ct_config(ignore=[PREFIX, "lm_head"])
    assert isinstance(stubs.dispatch(config, PREFIX), stubs.unquantized)
    assert stubs.helper_calls == ["should_ignore_layer"]


def test_compressed_tensors_that_quantizes_the_table_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The branch stays narrow: a quantized PLE table must still fail loudly.

    Otherwise the patch would silently load bf16 weights over FP8 ones.
    """
    stubs = _install(monkeypatch)
    config = stubs.ct_config(ignore=["lm_head"])
    with pytest.raises(NotImplementedError, match="CompressedTensorsConfig"):
        stubs.dispatch(config, PREFIX)


def test_mxfp4_with_ignored_shards_loads_unquantized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second reported failure: an MXFP4 checkpoint excluding the table.

    Regression guard for ``NotImplementedError: Qwen4Exp PLE embedding does not
    support quantization config Mxfp4Config``. The exclusion list arrives under
    a different attribute name than the compressed-tensors one.
    """
    stubs = _install(monkeypatch)
    config = stubs.mxfp4_config(ignored_layers=_ignored_shard_names())
    assert isinstance(stubs.dispatch(config, PREFIX), stubs.unquantized)


def test_mxfp4_naming_the_table_directly_is_honoured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """That path runs through ``is_layer_skipped``, not ``should_ignore_layer``."""
    stubs = _install(monkeypatch)
    config = stubs.mxfp4_config(ignored_layers=[PREFIX, "lm_head"])
    assert isinstance(stubs.dispatch(config, PREFIX), stubs.unquantized)
    assert stubs.helper_calls == ["is_layer_skipped"]


def test_mxfp4_that_quantizes_the_table_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ignored_layers=None`` means the table is quantized, not excluded.

    It is both the shape upstream's ``Mxfp4Config.from_config`` leaves behind
    when it discards the checkpoint's ``ignore`` key, and the shape of a
    checkpoint that genuinely quantizes the PLE table — qwen4-max declares
    ``ple_quant_scheme: fp8_pertensor`` and excludes nothing. Either way the
    patch must not guess that the table is in ``params_dtype``.
    """
    stubs = _install(monkeypatch)
    configs = (
        stubs.mxfp4_config(),
        stubs.mxfp4_config(ignored_layers=["lm_head"]),
    )
    for config in configs:
        with pytest.raises(NotImplementedError, match="Mxfp4Config"):
            stubs.dispatch(config, PREFIX)


def test_declared_fp8_embedding_dtype_keeps_upstream_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Upstream resolves ``ple_embedding_dtype`` before any quant config.

    The patch delegates first, so an explicitly declared FP8 table stays FP8
    even when the ignore list would otherwise rescue it.
    """
    stubs = _install(monkeypatch)
    for config in (
        stubs.ct_config(ignore=_ignored_shard_names()),
        stubs.mxfp4_config(ignored_layers=_ignored_shard_names()),
    ):
        assert isinstance(stubs.dispatch(config, PREFIX, "float8_e4m3fn"), stubs.fp8)


@pytest.mark.parametrize("use_fp8_config", [False, True])
def test_whitelisted_families_delegate_unchanged(
    monkeypatch: pytest.MonkeyPatch, use_fp8_config: bool
) -> None:
    stubs = _install(monkeypatch)
    quant_config = stubs.fp8_config() if use_fp8_config else None
    expected = stubs.fp8 if use_fp8_config else stubs.unquantized
    assert isinstance(stubs.dispatch(quant_config, PREFIX), expected)


def test_unknown_families_are_not_whitelisted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The patch adds one branch; the raise must not become a catch-all."""
    stubs = _install(monkeypatch)
    with pytest.raises(NotImplementedError, match="SomeOtherConfig"):
        stubs.dispatch(stubs.other_config(), PREFIX)


def test_rescue_is_keyed_on_family_not_on_duck_typing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A foreign config carrying a matching ``ignore`` list is still rejected.

    Without the family gate the rescue would read whichever attribute happened
    to exist, silently broadening it to every quantization config that ever
    grows an exclusion list.
    """
    stubs = _install(monkeypatch)
    config = stubs.other_config()
    config.ignore = _ignored_shard_names()
    config.ignored_layers = _ignored_shard_names()
    config.packed_modules_mapping = {}
    with pytest.raises(NotImplementedError, match="SomeOtherConfig"):
        stubs.dispatch(config, PREFIX)
