# SPDX-License-Identifier: Apache-2.0
"""Tests for the MXFP4 patch module (``patch/enhancement/mxfp4.py``).

Same two-tier design as ``test_residual_patches.py``, both runnable on a bare
CPU runner with no vLLM:

* **Bare-CPU import + metadata shape.** The leaf module defines its
  replacements without importing vLLM at module scope; its mandatory
  ``reason`` / ``affected_versions`` / ``remove_when`` metadata is asserted
  from ``METADATA`` and by AST-scanning every ``patch`` call.
* **Behaviour against stubs.** ``install()`` is exercised against throwaway
  stand-ins for the two upstream modules, asserting each patch lands and
  behaves as the fork intended.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from tests.support.source import assert_accepts_upstream_keywords, function
from vllm_sail.patch.utils import PATCH_REGISTRY

MODULE_PATH = (
    Path(__file__).parents[2] / "vllm_sail" / "patch" / "enhancement" / "mxfp4.py"
)
REQUIRED_METADATA = ("reason", "affected_versions", "remove_when")
N_TARGETS = 10


@pytest.fixture()
def mxfp4_module() -> Iterator[types.ModuleType]:
    """A fresh leaf-module instance per test (install state must not leak)."""
    name = f"_mxfp4_leaf_test_{id(object())}"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    registry_before = len(PATCH_REGISTRY)
    try:
        yield module
    finally:
        del PATCH_REGISTRY[registry_before:]
        sys.modules.pop(name, None)


class _Platform:
    ppu = True
    sm80 = False

    @classmethod
    def is_ppu(cls) -> bool:
        return cls.ppu

    @classmethod
    def is_device_capability(cls, cap: tuple[int, int]) -> bool:
        return cls.sm80 and cap == (8, 0)


def _register_chain(
    monkeypatch: pytest.MonkeyPatch, leaf_name: str, leaf_module: types.ModuleType
) -> None:
    parts = leaf_name.split(".")
    chain = [".".join(parts[: i + 1]) for i in range(len(parts))]
    modules: dict[str, types.ModuleType] = {}
    for name in chain[:-1]:
        existing = sys.modules.get(name)
        module = (
            existing
            if isinstance(existing, types.ModuleType)
            else types.ModuleType(name)
        )
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
    modules[leaf_name] = leaf_module
    monkeypatch.setitem(sys.modules, leaf_name, leaf_module)
    for parent_name, child_name in zip(chain, chain[1:], strict=False):
        parent = modules[parent_name]
        attribute = child_name.rpartition(".")[2]
        if hasattr(parent, attribute):
            monkeypatch.setattr(parent, attribute, modules[child_name])
        else:
            setattr(parent, attribute, modules[child_name])


class _StubState:
    """Everything the behaviour tests want to observe."""

    def __init__(self) -> None:
        self.select_calls: list[dict[str, Any]] = []
        self.upstream_method_init: list[Any] = []
        self.upstream_gptoss_init: list[Any] = []
        self.upstream_get_quant_method: list[Any] = []
        self.upstream_make_quant_config: list[dict[str, Any]] = []
        self.upstream_convert: list[Any] = []
        self.upstream_round_up: list[Any] = []
        self.ocp_kwargs: dict[str, Any] | None = None
        self.preprocessed: list[Any] = []


def _build_stub_vllm(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_StubState, dict[str, types.ModuleType]]:
    pytest.importorskip("torch")
    state = _StubState()

    platforms = types.ModuleType("vllm.platforms")
    platforms.current_platform = _Platform  # type: ignore[attr-defined]
    _register_chain(monkeypatch, "vllm.platforms", platforms)

    class LinearBase:
        pass

    linear = types.ModuleType("vllm.model_executor.layers.linear")
    linear.LinearBase = LinearBase  # type: ignore[attr-defined]
    _register_chain(monkeypatch, "vllm.model_executor.layers.linear", linear)

    class RoutedExperts:
        def __init__(self) -> None:
            self.moe_config = object()

    class UnquantizedFusedMoEMethod:
        def __init__(self, moe_config) -> None:
            self.moe_config = moe_config

    fused_moe = types.ModuleType("vllm.model_executor.layers.fused_moe")
    fused_moe.RoutedExperts = RoutedExperts  # type: ignore[attr-defined]
    fused_moe.UnquantizedFusedMoEMethod = UnquantizedFusedMoEMethod  # type: ignore[attr-defined]
    _register_chain(monkeypatch, "vllm.model_executor.layers.fused_moe", fused_moe)

    def ocp_mx_moe_quant_config(**kwargs):
        state.ocp_kwargs = kwargs
        return ("ocp_mx", kwargs)

    fm_config = types.ModuleType("vllm.model_executor.layers.fused_moe.config")
    fm_config.ocp_mx_moe_quant_config = ocp_mx_moe_quant_config  # type: ignore[attr-defined]
    _register_chain(
        monkeypatch, "vllm.model_executor.layers.fused_moe.config", fm_config
    )

    class Mxfp4MoeBackend:
        DEEPGEMM_MXFP4 = "DEEPGEMM_MXFP4"
        PPU_DEEPGEMM_MXFP4 = "PPU_DEEPGEMM_MXFP4"
        BATCHED_PPU_DEEPGEMM_MXFP4 = "BATCHED_PPU_DEEPGEMM_MXFP4"

    kMxfp4Dynamic = object()

    def select_mxfp4_moe_backend(moe, activation_key=None):
        state.select_calls.append({"moe": moe, "activation_key": activation_key})
        return (Mxfp4MoeBackend.PPU_DEEPGEMM_MXFP4, object())

    def make_mxfp4_moe_quant_config(**kwargs):
        state.upstream_make_quant_config.append(kwargs)
        return ("upstream-quant-config", kwargs)

    def convert_weight(*args, activation=None):
        state.upstream_convert.append(args)
        return ("upstream-convert",)

    def round_up_sizes(backend, hidden_size, intermediate_size, activation=None):
        state.upstream_round_up.append((backend, hidden_size, intermediate_size))
        return hidden_size, intermediate_size

    oracle = types.ModuleType("vllm.model_executor.layers.fused_moe.oracle.mxfp4")
    oracle.Mxfp4MoeBackend = Mxfp4MoeBackend  # type: ignore[attr-defined]
    oracle.kMxfp4Dynamic = kMxfp4Dynamic  # type: ignore[attr-defined]
    oracle.select_mxfp4_moe_backend = select_mxfp4_moe_backend  # type: ignore[attr-defined]
    oracle.make_mxfp4_moe_quant_config = make_mxfp4_moe_quant_config  # type: ignore[attr-defined]
    oracle.convert_weight_to_mxfp4_moe_kernel_format = convert_weight  # type: ignore[attr-defined]
    oracle.convert_gpt_oss_weight_to_mxfp4_moe_kernel_format = convert_weight  # type: ignore[attr-defined]
    oracle.mxfp4_round_up_hidden_size_and_intermediate_size = round_up_sizes  # type: ignore[attr-defined]
    _register_chain(
        monkeypatch, "vllm.model_executor.layers.fused_moe.oracle.mxfp4", oracle
    )

    def is_layer_skipped(
        prefix, ignored_layers, fused_mapping=None, *, match_mode="exact"
    ):
        if match_mode == "substring":
            return any(needle in prefix for needle in ignored_layers)
        return prefix in ignored_layers

    quant_utils = types.ModuleType(
        "vllm.model_executor.layers.quantization.utils.quant_utils"
    )
    quant_utils.is_layer_skipped = is_layer_skipped  # type: ignore[attr-defined]
    _register_chain(
        monkeypatch,
        "vllm.model_executor.layers.quantization.utils.quant_utils",
        quant_utils,
    )

    class FusedMoEMethodBase:
        def __init__(self, moe) -> None:
            self.moe = moe

    class Mxfp4Config:
        packed_modules_mapping = {}

        def __init__(self, ignored_layers=None):
            self.ignored_layers = ignored_layers

        @classmethod
        def from_config(cls, config):
            return cls()

        def get_quant_method(self, layer, prefix):
            state.upstream_get_quant_method.append((layer, prefix))
            return "upstream-method"

        def apply_vllm_mapper(self, mapper):
            pass

    class GptOssMxfp4MoEMethod(FusedMoEMethodBase):
        def __init__(self, moe):
            super().__init__(moe)
            state.upstream_gptoss_init.append(moe)
            self.mxfp4_backend = "upstream-backend"
            self.experts_cls = None

    class Mxfp4MoEMethod(FusedMoEMethodBase):
        def __init__(self, moe):
            super().__init__(moe)
            state.upstream_method_init.append(moe)
            self.mxfp4_backend = "upstream-backend"
            self.experts_cls = None

        def get_fused_moe_quant_config(self, layer):
            return "upstream-fm-quant-config"

    def _use_k3_situ_aiter(moe):
        return False

    quant_mxfp4 = types.ModuleType("vllm.model_executor.layers.quantization.mxfp4")
    quant_mxfp4.Mxfp4Config = Mxfp4Config  # type: ignore[attr-defined]
    quant_mxfp4.GptOssMxfp4MoEMethod = GptOssMxfp4MoEMethod  # type: ignore[attr-defined]
    quant_mxfp4.Mxfp4MoEMethod = Mxfp4MoEMethod  # type: ignore[attr-defined]
    quant_mxfp4._use_k3_situ_aiter = _use_k3_situ_aiter  # type: ignore[attr-defined]
    _register_chain(
        monkeypatch, "vllm.model_executor.layers.quantization.mxfp4", quant_mxfp4
    )

    def round_up(x: int, multiple: int) -> int:
        return ((x + multiple - 1) // multiple) * multiple

    math_utils = types.ModuleType("vllm.utils.math_utils")
    math_utils.round_up = round_up  # type: ignore[attr-defined]
    _register_chain(monkeypatch, "vllm.utils.math_utils", math_utils)

    return state, {
        "oracle": oracle,
        "quant_mxfp4": quant_mxfp4,
        "fused_moe": fused_moe,
    }


# ---------------------------------------------------------------------------
# Bare-CPU import and metadata shape
# ---------------------------------------------------------------------------


def test_module_imports_on_bare_cpu(mxfp4_module) -> None:
    if "vllm" in sys.modules:
        pytest.skip("bare-CPU contract test; real vLLM is loaded (mode B)")
    assert callable(mxfp4_module.install)
    assert mxfp4_module.__doc__


def test_metadata_shape(mxfp4_module) -> None:
    metadata = mxfp4_module.METADATA
    assert len(metadata) == N_TARGETS
    for target, reason, affected, remove_when in metadata:
        assert target.startswith("vllm."), target
        for value in (reason, affected, remove_when):
            assert isinstance(value, str) and value.strip(), target
        assert ">=" in affected and "<" in affected, target
        assert remove_when.strip().lower() != "todo", target


def test_no_top_level_vllm_import(mxfp4_module) -> None:
    tree = ast.parse(MODULE_PATH.read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for imported in names:
            assert imported != "vllm" and not imported.startswith("vllm."), (
                f"mxfp4 imports {imported!r} at module level; the leaf must "
                "stay importable without vLLM."
            )


def _callee_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def test_every_patch_call_carries_metadata(mxfp4_module) -> None:
    tree = ast.parse(MODULE_PATH.read_text())
    watched = {"patch", "patch_value", "PatchRecord"}
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _callee_name(node) in watched
    ]
    assert len(calls) == N_TARGETS
    for call in calls:
        keywords = {kw.arg for kw in call.keywords}
        missing = set(REQUIRED_METADATA) - keywords
        assert not missing, f"patch call at line {call.lineno} lacks {missing}"


# ---------------------------------------------------------------------------
# Behaviour against stubs
# ---------------------------------------------------------------------------


def test_install_lands_every_target_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, mxfp4_module
) -> None:
    _build_stub_vllm(monkeypatch)
    registry_before = len(PATCH_REGISTRY)
    mxfp4_module.install()
    targets = {target for target, *_ in mxfp4_module.METADATA}
    records = [r for r in PATCH_REGISTRY[registry_before:] if r.target in targets]
    assert len(records) == N_TARGETS
    assert all(record.was_missing is False for record in records)
    mxfp4_module.install()  # the guard, not the double-patch error
    assert len(PATCH_REGISTRY) == registry_before + N_TARGETS


def test_config_init_and_from_config(
    monkeypatch: pytest.MonkeyPatch, mxfp4_module
) -> None:
    _, modules = _build_stub_vllm(monkeypatch)
    mxfp4_module.install()
    Mxfp4Config = modules["quant_mxfp4"].Mxfp4Config

    config = Mxfp4Config()
    assert config.ignored_layers is None
    assert config.fp8_channelwise_layers == []
    config = Mxfp4Config(
        ignored_layers=["lm_head"], fp8_channelwise_layers=["model.dense"]
    )
    assert config.ignored_layers == ["lm_head"]
    assert config.fp8_channelwise_layers == ["model.dense"]

    _Platform.ppu = True
    config = Mxfp4Config.from_config(
        {"ignore": ["mtp"], "fp8_channelwise_layers": ["gate"]}
    )
    assert config.ignored_layers == ["mtp"]
    assert config.fp8_channelwise_layers == ["gate"]

    _Platform.ppu = False
    config = Mxfp4Config.from_config(
        {"ignore": ["mtp"], "fp8_channelwise_layers": ["gate"]}
    )
    assert config.ignored_layers == ["mtp"]
    assert config.fp8_channelwise_layers == []
    _Platform.ppu = True


def test_get_quant_method_ignores_experts_and_channelwise_linear(
    monkeypatch: pytest.MonkeyPatch, mxfp4_module
) -> None:
    state, modules = _build_stub_vllm(monkeypatch)
    mxfp4_module.install()
    Mxfp4Config = modules["quant_mxfp4"].Mxfp4Config
    RoutedExperts = modules["fused_moe"].RoutedExperts
    LinearBase = sys.modules["vllm.model_executor.layers.linear"].LinearBase

    config = Mxfp4Config(ignored_layers=["model.mtp.experts"])

    experts = RoutedExperts()
    method = config.get_quant_method(experts, "model.mtp.experts")
    assert type(method).__name__ == "UnquantizedFusedMoEMethod"
    assert method.moe_config is experts.moe_config

    other = RoutedExperts()
    assert config.get_quant_method(other, "model.layers.0.experts") == "upstream-method"
    assert state.upstream_get_quant_method[-1] == (other, "model.layers.0.experts")

    _Platform.ppu = True
    config.fp8_channelwise_layers = ["dense"]
    dense = object.__new__(LinearBase)
    dense_prefix = "model.dense"

    # The CompressedTensors scheme imports stay lazy; stub them.
    import sys as _sys

    ct = types.ModuleType("compressed_tensors.quantization")

    class QuantizationArgs:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class QuantizationStrategy:
        CHANNEL = "channel"

    class QuantizationType:
        FLOAT = "float"

    ct.QuantizationArgs = QuantizationArgs
    ct.QuantizationStrategy = QuantizationStrategy
    ct.QuantizationType = QuantizationType
    monkeypatch.setitem(_sys.modules, "compressed_tensors.quantization", ct)
    monkeypatch.setitem(
        _sys.modules, "compressed_tensors", types.ModuleType("compressed_tensors")
    )

    ct_pkg = types.ModuleType(
        "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors"
    )

    class CompressedTensorsLinearMethod:
        def __init__(self, quant_config):
            self.quant_config = quant_config

    ct_pkg.CompressedTensorsLinearMethod = CompressedTensorsLinearMethod
    _register_chain(
        monkeypatch,
        "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors",
        ct_pkg,
    )
    schemes = types.ModuleType(
        "vllm.model_executor.layers.quantization.compressed_tensors.schemes."
        "compressed_tensors_w8a8_fp8"
    )

    class CompressedTensorsW8A8Fp8:
        def __init__(self, weight_quant, is_static_input_scheme):
            self.weight_quant = weight_quant
            self.is_static_input_scheme = is_static_input_scheme

    schemes.CompressedTensorsW8A8Fp8 = CompressedTensorsW8A8Fp8
    _register_chain(
        monkeypatch,
        "vllm.model_executor.layers.quantization.compressed_tensors.schemes."
        "compressed_tensors_w8a8_fp8",
        schemes,
    )

    method = config.get_quant_method(dense, dense_prefix)
    assert isinstance(method, CompressedTensorsLinearMethod)
    assert method.quant_config is config
    assert isinstance(dense.scheme, CompressedTensorsW8A8Fp8)

    _Platform.ppu = False
    assert config.get_quant_method(dense, dense_prefix) == "upstream-method"
    _Platform.ppu = True


def test_apply_vllm_mapper_remaps_channelwise_on_ppu(
    monkeypatch: pytest.MonkeyPatch, mxfp4_module
) -> None:
    _, modules = _build_stub_vllm(monkeypatch)
    mxfp4_module.install()
    Mxfp4Config = modules["quant_mxfp4"].Mxfp4Config

    class Mapper:
        @staticmethod
        def apply_list(layers):
            return [f"mapped.{name}" for name in layers]

    config = Mxfp4Config(fp8_channelwise_layers=["a", "b"])
    _Platform.ppu = True
    config.apply_vllm_mapper(Mapper())
    assert config.fp8_channelwise_layers == ["mapped.a", "mapped.b"]

    _Platform.ppu = False
    config.apply_vllm_mapper(Mapper())
    assert config.fp8_channelwise_layers == ["mapped.a", "mapped.b"]
    _Platform.ppu = True


def test_apply_vllm_mapper_remaps_ignored_layers_on_every_platform(
    monkeypatch: pytest.MonkeyPatch, mxfp4_module
) -> None:
    """Upstream's ``Mxfp4Config`` inherits the no-op ``apply_vllm_mapper``.

    Left alone, ``ignored_layers`` keeps checkpoint naming
    (``model.language_model.*``) while the module tree uses vLLM naming
    (``language_model.model.*``), so ``is_layer_skipped`` never matches and
    neither the ignored ``RoutedExperts`` routing nor the Qwen4Exp PLE rescue
    in ``enhancement/qwen4_exp_ple.py`` can fire. Unlike
    ``fp8_channelwise_layers``, the list is not a PPU extension, so it is
    rewritten on every platform.
    """
    _, modules = _build_stub_vllm(monkeypatch)
    mxfp4_module.install()
    Mxfp4Config = modules["quant_mxfp4"].Mxfp4Config

    class Mapper:
        @staticmethod
        def apply_list(layers):
            return [
                name.replace("model.language_model.", "language_model.model.")
                for name in layers
            ]

    ignored = ["model.language_model.layers.1.mlp.experts", "lm_head"]
    expected = ["language_model.model.layers.1.mlp.experts", "lm_head"]
    for ppu in (True, False):
        _Platform.ppu = ppu
        config = Mxfp4Config(ignored_layers=list(ignored))
        config.apply_vllm_mapper(Mapper())
        assert config.ignored_layers == expected
    _Platform.ppu = True


def test_apply_vllm_mapper_tolerates_an_absent_ignore_list(
    monkeypatch: pytest.MonkeyPatch, mxfp4_module
) -> None:
    """Upstream's ``from_config`` returns ``cls()``, so ``None`` is the norm."""
    _, modules = _build_stub_vllm(monkeypatch)
    mxfp4_module.install()
    Mxfp4Config = modules["quant_mxfp4"].Mxfp4Config

    class Mapper:
        @staticmethod
        def apply_list(layers):
            raise AssertionError("called with no list to map")

    config = Mxfp4Config()
    config.apply_vllm_mapper(Mapper())
    assert config.ignored_layers is None


def test_method_initialization_preserves_upstream_factory(
    monkeypatch: pytest.MonkeyPatch, mxfp4_module
) -> None:
    state, modules = _build_stub_vllm(monkeypatch)
    cls = modules["quant_mxfp4"].Mxfp4MoEMethod
    upstream_init = cls.__init__
    mxfp4_module.install()
    # vLLM 0.30 owns construction and calls the separately registered
    # DeepSeek V4 backend selector. The quantization patch must not replace it.
    assert cls.__init__ is upstream_init
    moe = types.SimpleNamespace(max_capture_size=8, moe_backend="auto")
    cls(moe)
    assert state.upstream_method_init == [moe]

    oracle = modules["oracle"]
    _Platform.ppu, _Platform.sm80 = True, False
    modules["quant_mxfp4"].GptOssMxfp4MoEMethod(moe)
    assert state.upstream_gptoss_init == [moe]
    assert state.select_calls == [{"moe": moe, "activation_key": oracle.kMxfp4Dynamic}]


def test_get_fused_moe_quant_config_threads_swiglu_params(
    monkeypatch: pytest.MonkeyPatch, mxfp4_module
) -> None:
    state, modules = _build_stub_vllm(monkeypatch)
    mxfp4_module.install()
    oracle = modules["oracle"]
    Mxfp4MoEMethod = modules["quant_mxfp4"].Mxfp4MoEMethod

    method = Mxfp4MoEMethod(
        types.SimpleNamespace(max_capture_size=8, moe_backend="auto")
    )
    layer = types.SimpleNamespace(
        w13_weight_scale="w1s",
        w2_weight_scale="w2s",
        w13_bias="b1",
        w2_bias="b2",
        swiglu_alpha=1.7,
        swiglu_beta=0.7,
        swiglu_limit=7.0,
    )

    method.mxfp4_backend = oracle.Mxfp4MoeBackend.PPU_DEEPGEMM_MXFP4
    result = method.get_fused_moe_quant_config(layer)
    # Chained through the patched make_mxfp4_moe_quant_config into ocp_mx.
    assert result[0] == "ocp_mx"
    assert state.ocp_kwargs is not None
    assert state.ocp_kwargs["gemm1_alpha"] == 1.7
    assert state.ocp_kwargs["gemm1_beta"] == 0.7
    assert state.ocp_kwargs["gemm1_clamp_limit"] == 7.0
    assert state.ocp_kwargs["w1_scale"] == "w1s"

    method.mxfp4_backend = "SOME_OTHER_BACKEND"
    assert method.get_fused_moe_quant_config(layer) == "upstream-fm-quant-config"


def test_oracle_make_quant_config_ppu_branch(
    monkeypatch: pytest.MonkeyPatch, mxfp4_module
) -> None:
    state, modules = _build_stub_vllm(monkeypatch)
    mxfp4_module.install()
    oracle = modules["oracle"]

    result = oracle.make_mxfp4_moe_quant_config(
        mxfp4_backend=oracle.Mxfp4MoeBackend.PPU_DEEPGEMM_MXFP4,
        w1_scale="w1s",
        w2_scale="w2s",
        gemm1_alpha=1.0,
        swiglu_limit=7.0,
    )
    assert result[0] == "ocp_mx"
    assert state.ocp_kwargs is not None
    assert state.ocp_kwargs["block_shape"] == [1, 16]
    assert state.ocp_kwargs["quant_dtype"] == "mxfp4"
    assert state.ocp_kwargs["gemm1_clamp_limit"] == 7.0

    result = oracle.make_mxfp4_moe_quant_config(
        mxfp4_backend="DEEPGEMM_MXFP4", w1_scale="w1s", w2_scale="w2s"
    )
    assert result[0] == "upstream-quant-config"


def test_oracle_convert_weights_preprocesses_scales(
    monkeypatch: pytest.MonkeyPatch, mxfp4_module
) -> None:
    torch = pytest.importorskip("torch")

    state, modules = _build_stub_vllm(monkeypatch)

    def preprocess_mxfp4_scales(scale):
        state.preprocessed.append(scale)
        return scale + 1

    deep_gemm = types.ModuleType("deep_gemm")
    deep_gemm.preprocess_mxfp4_scales = preprocess_mxfp4_scales  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "deep_gemm", deep_gemm)

    mxfp4_module.install()
    oracle = modules["oracle"]

    w1s = torch.zeros(2)
    w2s = torch.ones(2)
    result = oracle.convert_weight_to_mxfp4_moe_kernel_format(
        oracle.Mxfp4MoeBackend.BATCHED_PPU_DEEPGEMM_MXFP4,
        object(),
        "w1",
        "w2",
        w1s,
        w2s,
        torch.ones(2),
        torch.ones(2),
    )
    assert len(state.preprocessed) == 2
    assert torch.equal(result[2], w1s + 1)
    assert result[4].dtype == torch.float32
    assert state.upstream_convert == []

    oracle.convert_weight_to_mxfp4_moe_kernel_format(
        "DEEPGEMM_MXFP4", object(), "w1", "w2", w1s, w2s
    )
    assert len(state.upstream_convert) == 1


def test_oracle_round_up_sizes_32_on_ppu_backends(
    monkeypatch: pytest.MonkeyPatch, mxfp4_module
) -> None:
    state, modules = _build_stub_vllm(monkeypatch)
    mxfp4_module.install()
    oracle = modules["oracle"]

    backend = oracle.Mxfp4MoeBackend.PPU_DEEPGEMM_MXFP4
    assert oracle.mxfp4_round_up_hidden_size_and_intermediate_size(
        backend, 2881, 2881
    ) == (2912, 2912)
    assert state.upstream_round_up == []

    assert oracle.mxfp4_round_up_hidden_size_and_intermediate_size(
        "DEEPGEMM_MXFP4", 2881, 2881
    ) == (2881, 2881)
    assert state.upstream_round_up == [("DEEPGEMM_MXFP4", 2881, 2881)]


@pytest.mark.parametrize(
    "ppu,sm80,backend,expected",
    [
        (True, False, "auto", "mxfp4"),
        (True, False, "ppu_deep_gemm", "mxfp4"),
        (True, True, "auto", None),
        (True, False, "marlin", None),
        (True, False, "ppu_deep_gemm_w4a16", None),
        (False, False, "auto", "upstream"),
    ],
)
def test_deepseek_v4_mxfp4_selector_preserves_activation_mode(
    monkeypatch, ppu, sm80, backend, expected
):
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(
        is_ppu=lambda: ppu,
        is_device_capability=lambda cap: sm80 and cap == (8, 0),
    )
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
    namespace = {
        "oracle": SimpleNamespace(
            kMxfp4Dynamic="mxfp4",
            select_mxfp4_moe_backend=lambda config, activation_key: activation_key,
        ),
        "_upstream_select_deepseek_v4": lambda config: "upstream",
    }
    fn = function(
        "vllm_sail/registry/moe_backends/mxfp4.py",
        "select_deepseek_v4_mxfp4_moe_backend",
        namespace,
    )
    assert fn(SimpleNamespace(moe_backend=backend)) == expected


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "local,local_name,upstream,upstream_name",
    [
        (
            "mxfp4",
            "install._convert_weight",
            "model_executor/layers/fused_moe/oracle/mxfp4",
            "convert_weight_to_mxfp4_moe_kernel_format",
        )
    ],
)
def test_replacements_accept_upstream_keywords(
    upstream_source_root, local, local_name, upstream, upstream_name
):
    assert_accepts_upstream_keywords(
        local, local_name, upstream_source_root, upstream, upstream_name
    )
