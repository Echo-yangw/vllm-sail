# SPDX-License-Identifier: Apache-2.0
"""Tests for the two residual-coverage leaf patch modules.

Covers ``patch/enhancement/compressed_tensors_ppu.py`` and
``patch/enhancement/fused_moe_ppu.py`` (the item-2 uncovered-file sweep).
Same two-tier design as ``test_residual_patches.py``, runnable on a bare CPU
runner with no vLLM: bare-CPU import + metadata shape + AST checks, and a few
behaviour tests against stub upstreams.
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

from vllm_sail.patch.utils import PATCH_REGISTRY

ENHANCEMENT_DIR = Path(__file__).parents[2] / "vllm_sail" / "patch" / "enhancement"
MODULE_NAMES = ("compressed_tensors_ppu", "fused_moe_ppu")
REQUIRED_METADATA = ("reason", "affected_versions", "remove_when")


@pytest.fixture()
def leaf_module(request) -> Iterator[types.ModuleType]:
    name = f"_item2_leaf_test_{request.param}_{id(object())}"
    path = ENHANCEMENT_DIR / f"{request.param}.py"
    spec = importlib.util.spec_from_file_location(name, path)
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


parametrize_modules = pytest.mark.parametrize(
    "leaf_module", MODULE_NAMES, indirect=True
)

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


class _Platform:
    ppu = True

    @classmethod
    def is_ppu(cls) -> bool:
        return cls.ppu


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


def _stub_platform(monkeypatch: pytest.MonkeyPatch) -> type[_Platform]:
    platforms = types.ModuleType("vllm.platforms")
    platforms.current_platform = _Platform  # type: ignore[attr-defined]
    _register_chain(monkeypatch, "vllm.platforms", platforms)
    return _Platform


# ---------------------------------------------------------------------------
# Bare-CPU import and metadata shape
# ---------------------------------------------------------------------------


@parametrize_modules
def test_module_imports_on_bare_cpu(leaf_module) -> None:
    if "vllm" in sys.modules:
        pytest.skip("bare-CPU contract test; real vLLM is loaded (mode B)")
    assert callable(leaf_module.install)
    assert leaf_module.__doc__


@parametrize_modules
def test_metadata_shape(leaf_module) -> None:
    metadata = leaf_module.METADATA
    assert metadata
    for target, reason, affected, remove_when in metadata:
        assert target.startswith("vllm."), target
        for value in (reason, affected, remove_when):
            assert isinstance(value, str) and value.strip(), target
        assert ">=" in affected and "<" in affected, target
        assert remove_when.strip().lower() != "todo", target


@parametrize_modules
def test_no_top_level_vllm_import(leaf_module) -> None:
    tree = ast.parse(Path(leaf_module.__file__).read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for imported in names:
            assert imported != "vllm" and not imported.startswith("vllm."), (
                f"{leaf_module.__name__} imports {imported!r} at module level"
            )


def _callee_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


@parametrize_modules
def test_every_patch_call_carries_metadata(leaf_module) -> None:
    tree = ast.parse(Path(leaf_module.__file__).read_text())
    watched = {"patch", "patch_value", "PatchRecord"}
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _callee_name(node) in watched
    ]
    # The provider decorators correspond one-to-one with METADATA. A separate
    # patch call may rebind multiple preloaded aliases of a provider.
    decorated_calls = [
        decorator
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        for decorator in node.decorator_list
        if isinstance(decorator, ast.Call) and _callee_name(decorator) == "patch"
    ]
    assert len(decorated_calls) == len(leaf_module.METADATA)
    for call in calls:
        keywords = {kw.arg for kw in call.keywords}
        missing = set(REQUIRED_METADATA) - keywords
        assert not missing, f"patch call at line {call.lineno} lacks {missing}"


# ---------------------------------------------------------------------------
# Behaviour against stubs: fused_moe_ppu
# ---------------------------------------------------------------------------


def _build_fused_moe_stubs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {"upstream_dtype_str": [], "quantized": []}
    platform = _stub_platform(monkeypatch)
    state["platform"] = platform

    def upstream_dtype_str(dtype, **kwargs):
        state["upstream_dtype_str"].append(kwargs)
        if kwargs.get("use_fp8_w8a8"):
            return "fp8_w8a8"
        if kwargs.get("use_int8_w8a16"):
            return "int8_w8a16"
        return None

    def upstream_int8_config(*args, **kwargs):
        return types.SimpleNamespace(gemm1_clamp_limit=None, args=args, kwargs=kwargs)

    config = types.ModuleType("vllm.model_executor.layers.fused_moe.config")
    config._get_config_dtype_str = upstream_dtype_str  # type: ignore[attr-defined]
    config.int8_w8a8_moe_quant_config = upstream_int8_config  # type: ignore[attr-defined]
    _register_chain(monkeypatch, "vllm.model_executor.layers.fused_moe.config", config)

    def upstream_make_int8(*args, **kwargs):
        return "upstream-make-int8"

    int8_oracle = types.ModuleType("vllm.model_executor.layers.fused_moe.oracle.int8")
    int8_oracle.make_int8_moe_quant_config = upstream_make_int8  # type: ignore[attr-defined]
    _register_chain(
        monkeypatch, "vllm.model_executor.layers.fused_moe.oracle.int8", int8_oracle
    )

    def upstream_quantize_input(A, A_scale, quant_dtype, per_act_token_quant, **kwargs):
        return ("upstream-quantized", quant_dtype)

    utils = types.ModuleType("vllm.model_executor.layers.fused_moe.utils")
    utils.moe_kernel_quantize_input = upstream_quantize_input  # type: ignore[attr-defined]
    _register_chain(monkeypatch, "vllm.model_executor.layers.fused_moe.utils", utils)

    class TritonExperts:
        def moe_sum(self, input, output, topk_ids=None, expert_map=None):
            state.setdefault("upstream_moe_sum", []).append(
                (input, output, topk_ids, expert_map)
            )

    triton_moe = types.ModuleType(
        "vllm.model_executor.layers.fused_moe.experts.triton_moe"
    )
    triton_moe.TritonExperts = TritonExperts  # type: ignore[attr-defined]
    _register_chain(
        monkeypatch,
        "vllm.model_executor.layers.fused_moe.experts.triton_moe",
        triton_moe,
    )

    fused_moe_mod = types.ModuleType("vllm.model_executor.layers.fused_moe.fused_moe")
    _register_chain(
        monkeypatch,
        "vllm.model_executor.layers.fused_moe.fused_moe",
        fused_moe_mod,
    )
    state["fused_moe_mod"] = fused_moe_mod

    downcast = types.ModuleType(
        "vllm_sail.model_executor.layers.quantization.utils.mxfp4_utils"
    )
    downcast.downcast_to_mxfp4 = lambda A, axis: ("mxfp4-downcast", axis)  # type: ignore[attr-defined]
    _register_chain(
        monkeypatch,
        "vllm_sail.model_executor.layers.quantization.utils.mxfp4_utils",
        downcast,
    )

    state["config"] = config
    state["int8_oracle"] = int8_oracle
    state["utils"] = utils
    state["triton_moe"] = triton_moe
    return state


def test_dtype_str_gains_int8_w8a8(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _leaf("fused_moe_ppu")
    state = _build_fused_moe_stubs(monkeypatch)
    module.install()

    dtype_str = state["config"]._get_config_dtype_str
    assert dtype_str("torch.dtype", use_int8_w8a8=True) == "int8_w8a8"
    assert dtype_str("torch.dtype", use_fp8_w8a8=True, use_int8_w8a8=True) == (
        "fp8_w8a8"
    )
    assert dtype_str("torch.dtype", use_int8_w8a16=True) == "int8_w8a16"
    module.install()  # idempotent


def test_int8_quant_config_accepts_clamp_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _leaf("fused_moe_ppu")
    state = _build_fused_moe_stubs(monkeypatch)
    module.install()

    config = state["config"].int8_w8a8_moe_quant_config("w1", "w2", None, None)
    assert config.gemm1_clamp_limit is None
    config = state["config"].int8_w8a8_moe_quant_config(
        "w1", "w2", None, None, gemm1_clamp_limit=7.0
    )
    assert config.gemm1_clamp_limit == 7.0


def test_make_int8_moe_quant_config_ppu_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _leaf("fused_moe_ppu")
    state = _build_fused_moe_stubs(monkeypatch)
    module.install()

    _Platform.ppu = True
    config = state["int8_oracle"].make_int8_moe_quant_config(
        "PPU_DEEPGEMM", "w1s", "w2s", a1_scale="a1", a2_scale="a2", swiglu_limit=7.0
    )
    assert config.gemm1_clamp_limit == 7.0
    assert config.kwargs["per_act_token_quant"] is False

    _Platform.ppu = False
    assert (
        state["int8_oracle"].make_int8_moe_quant_config("TRITON", "w1s", "w2s")
        == "upstream-make-int8"
    )
    _Platform.ppu = True


def test_moe_kernel_quantize_input_ppu_mxfp4(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _leaf("fused_moe_ppu")
    state = _build_fused_moe_stubs(monkeypatch)
    module.install()

    quantize = state["utils"].moe_kernel_quantize_input
    _Platform.ppu = True
    assert quantize("A", None, "mxfp4", False) == ("mxfp4-downcast", 1)
    assert quantize("A", None, "fp8", False)[0] == "upstream-quantized"
    _Platform.ppu = False
    assert quantize("A", None, "mxfp4", False)[0] == "upstream-quantized"
    _Platform.ppu = True


@pytest.mark.parametrize("import_before_install", [True, False])
@pytest.mark.parametrize(
    "consumer_suffix",
    [
        "experts.fused_batched_moe",
        "experts.nvfp4_emulation_moe",
        "experts.triton_moe",
        "fused_moe",
        "prepare_finalize.batched",
        "prepare_finalize.deepep_ht",
        "prepare_finalize.deepep_ll",
        "prepare_finalize.deepep_v2",
        "prepare_finalize.flashinfer_nvlink_one_sided",
        "prepare_finalize.flashinfer_nvlink_two_sided",
        "prepare_finalize.naive_dp_ep",
        "prepare_finalize.nixl_ep",
        "prepare_finalize.no_dp_ep",
    ],
)
def test_mxfp4_quantize_consumer_import_order(
    monkeypatch, consumer_suffix, import_before_install
) -> None:
    module = _leaf("fused_moe_ppu")
    state = _build_fused_moe_stubs(monkeypatch)
    monkeypatch.setattr(_Platform, "ppu", True)
    name = "vllm.model_executor.layers.fused_moe." + consumer_suffix
    # Use only our stubs: other consumers may be real modules in PPU-host UTs.
    consumer = (
        state["triton_moe"]
        if consumer_suffix == "experts.triton_moe"
        else types.ModuleType(name)
    )
    _register_chain(monkeypatch, name, consumer)

    def import_consumer():
        # Reproduce upstream's by-value import and a call through its globals.
        exec(
            "from vllm.model_executor.layers.fused_moe.utils "
            "import moe_kernel_quantize_input\n"
            "def prepare(dtype):\n"
            "    return moe_kernel_quantize_input('A', None, dtype, False)\n",
            consumer.__dict__,
        )

    if import_before_install:
        import_consumer()
    module.install()
    if not import_before_install:
        import_consumer()
    assert consumer.prepare("mxfp4") == ("mxfp4-downcast", 1)
    assert (
        consumer.moe_kernel_quantize_input is state["utils"].moe_kernel_quantize_input
    )
    assert consumer.prepare("fp8") == ("upstream-quantized", "fp8")
    monkeypatch.setattr(_Platform, "ppu", False)
    assert consumer.prepare("mxfp4") == ("upstream-quantized", "mxfp4")
    before = len(PATCH_REGISTRY)
    module.install()
    assert len(PATCH_REGISTRY) == before


def test_mxfp4_quantize_aliases_preserve_other_overrides(monkeypatch) -> None:
    module = _leaf("fused_moe_ppu")
    _build_fused_moe_stubs(monkeypatch)
    name = "vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep"
    consumer = types.ModuleType(name)

    def override(*args, **kwargs):
        return "another implementation"

    consumer.moe_kernel_quantize_input = override
    _register_chain(monkeypatch, name, consumer)
    module.install()
    assert consumer.moe_kernel_quantize_input is override


@pytest.mark.upstream_source
def test_mxfp4_quantize_alias_inventory_matches_vllm_source(
    upstream_source_root,
) -> None:
    root = upstream_source_root / "vllm"
    module = _leaf("fused_moe_ppu")
    discovered = set()
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                not isinstance(node, ast.ImportFrom)
                or node.module != module._UTILS_MODULE
            ):
                continue
            for alias in node.names:
                if alias.name == "moe_kernel_quantize_input":
                    assert alias.asname in (None, alias.name), str(path)
                    discovered.add(
                        ".".join(
                            ("vllm", *path.relative_to(root).with_suffix("").parts)
                        )
                    )
    assert discovered == set(module._QUANTIZE_INPUT_CONSUMERS)


def test_triton_moe_sum_prefers_triton_reduce_on_ppu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _leaf("fused_moe_ppu")
    state = _build_fused_moe_stubs(monkeypatch)
    module.install()

    seen = []
    kernels = types.ModuleType(
        "vllm_sail.model_executor.layers.fused_moe.triton_kernels"
    )
    kernels.moe_sum_reduce_triton = lambda i, o: seen.append((i, o))
    _register_chain(monkeypatch, kernels.__name__, kernels)

    experts = state["triton_moe"].TritonExperts()
    big_in = types.SimpleNamespace(shape=(1025, 4))
    out = object()
    _Platform.ppu = True
    experts.moe_sum(big_in, out)
    assert seen == [(big_in, out)]
    assert "upstream_moe_sum" not in state

    small_in = types.SimpleNamespace(shape=(16, 4))
    experts.moe_sum(small_in, out)
    assert state["upstream_moe_sum"] == [(small_in, out, None, None)]

    topk_ids, expert_map = object(), object()
    seen.clear()
    experts.moe_sum(big_in, out, topk_ids, expert_map)
    assert seen == []
    assert state["upstream_moe_sum"][-1] == (big_in, out, topk_ids, expert_map)

    seen.clear()
    _Platform.ppu = False
    experts.moe_sum(big_in, out)
    assert seen == []
    assert state["upstream_moe_sum"][-1] == (big_in, out, None, None)
    _Platform.ppu = True


# ---------------------------------------------------------------------------
# Behaviour against stubs: compressed_tensors_ppu
# ---------------------------------------------------------------------------


def _build_ct_stubs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {}
    platform = _stub_platform(monkeypatch)
    state["platform"] = platform

    class CompressedTensorsConfig:
        packed_modules_mapping = {}

        def __init__(self, ignore=None, config=None):
            self.ignore = ignore
            self.config = config

        @classmethod
        def from_config(cls, config):
            return cls(config=config)

        def get_scheme(self, layer, layer_name=None):
            return "upstream-scheme"

    class CompressedTensorsKVCacheMethod:
        def create_weights(self, layer, *args, **kwargs):
            torch = pytest.importorskip("torch")

            layer.k_scale = torch.nn.Parameter(torch.ones(2))
            layer.k_zero_point = torch.nn.Parameter(torch.zeros(2))
            layer.v_zero_point = torch.nn.Parameter(torch.zeros(2))
            layer.q_zero_point = torch.nn.Parameter(torch.zeros(2))

    ct = types.ModuleType(
        "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors"
    )
    ct.CompressedTensorsConfig = CompressedTensorsConfig  # type: ignore[attr-defined]
    ct.CompressedTensorsKVCacheMethod = CompressedTensorsKVCacheMethod  # type: ignore[attr-defined]
    _register_chain(
        monkeypatch,
        "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors",
        ct,
    )

    class CompressedTensorsMoEMethod:
        @staticmethod
        def get_moe_method(quant_config, layer, layer_name):
            return state["upstream_moe_method"]

    ct_moe = types.ModuleType(
        "vllm.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors_moe.compressed_tensors_moe"
    )
    ct_moe.CompressedTensorsMoEMethod = CompressedTensorsMoEMethod  # type: ignore[attr-defined]
    _register_chain(
        monkeypatch,
        "vllm.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors_moe.compressed_tensors_moe",
        ct_moe,
    )

    class W4A4Mxfp4MoEMethod:
        def __init__(self, moe_config):
            self.moe_config = moe_config

    w4a4 = types.ModuleType(
        "vllm.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors_moe.compressed_tensors_moe_w4a4_mxfp4"
    )
    w4a4.CompressedTensorsW4A4Mxfp4MoEMethod = W4A4Mxfp4MoEMethod  # type: ignore[attr-defined]
    _register_chain(
        monkeypatch,
        "vllm.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors_moe.compressed_tensors_moe_w4a4_mxfp4",
        w4a4,
    )
    state["w4a4_cls"] = W4A4Mxfp4MoEMethod

    class Mxfp4MoEMethod:
        def __init__(self, moe_config):
            self.moe_config = moe_config

    quant_mxfp4 = types.ModuleType("vllm.model_executor.layers.quantization.mxfp4")
    quant_mxfp4.Mxfp4MoEMethod = Mxfp4MoEMethod  # type: ignore[attr-defined]
    _register_chain(
        monkeypatch, "vllm.model_executor.layers.quantization.mxfp4", quant_mxfp4
    )
    state["mxfp4_method_cls"] = Mxfp4MoEMethod

    class CompressedTensorsW8A8Int8MoEMethod:
        def get_fused_moe_quant_config(self, layer):
            return types.SimpleNamespace(gemm1_clamp_limit=None)

    ct_int8 = types.ModuleType(
        "vllm.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors_moe.compressed_tensors_moe_w8a8_int8"
    )
    ct_int8.CompressedTensorsW8A8Int8MoEMethod = (  # type: ignore[attr-defined]
        CompressedTensorsW8A8Int8MoEMethod
    )
    _register_chain(
        monkeypatch,
        "vllm.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors_moe.compressed_tensors_moe_w8a8_int8",
        ct_int8,
    )

    class CompressedTensorsW8A8Fp8MoEMethod:
        def get_fused_moe_quant_config(self, layer):
            return types.SimpleNamespace(gemm1_alpha=None, gemm1_beta=None)

    ct_fp8_name = ct_int8.__name__.replace("w8a8_int8", "w8a8_fp8")
    ct_fp8 = types.ModuleType(ct_fp8_name)
    ct_fp8.CompressedTensorsW8A8Fp8MoEMethod = CompressedTensorsW8A8Fp8MoEMethod
    _register_chain(monkeypatch, ct_fp8_name, ct_fp8)
    state["ct_fp8"] = ct_fp8

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

    state["ct"] = ct
    state["ct_moe"] = ct_moe
    state["ct_int8"] = ct_int8
    return state


def test_ct_config_gains_fp8_channelwise_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _leaf("compressed_tensors_ppu")
    state = _build_ct_stubs(monkeypatch)
    module.install()
    Config = state["ct"].CompressedTensorsConfig

    config = Config(ignore=["lm_head"])
    assert config.ignore == ["lm_head"]
    assert config.fp8_channelwise_layers is None

    config = Config(fp8_channelwise_layers=["dense"])
    assert config.fp8_channelwise_layers == ["dense"]

    config = Config.from_config({"fp8_channelwise_layers": ["a"], "ignore": []})
    assert config.fp8_channelwise_layers == ["a"]
    module.install()  # idempotent


def test_ct_get_scheme_channelwise_override(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _leaf("compressed_tensors_ppu")
    state = _build_ct_stubs(monkeypatch)

    class QuantizationArgs:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    ct_quant = types.ModuleType("compressed_tensors.quantization")
    ct_quant.QuantizationArgs = QuantizationArgs  # type: ignore[attr-defined]
    ct_quant.QuantizationStrategy = types.SimpleNamespace(CHANNEL="channel")  # type: ignore[attr-defined]
    ct_quant.QuantizationType = types.SimpleNamespace(FLOAT="float")  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules, "compressed_tensors", types.ModuleType("compressed_tensors")
    )
    monkeypatch.setitem(sys.modules, "compressed_tensors.quantization", ct_quant)

    class CompressedTensorsW8A8Fp8:
        def __init__(self, weight_quant, is_static_input_scheme):
            self.weight_quant = weight_quant

    scheme_mod = types.ModuleType(
        "vllm.model_executor.layers.quantization.compressed_tensors.schemes."
        "compressed_tensors_w8a8_fp8"
    )
    scheme_mod.CompressedTensorsW8A8Fp8 = CompressedTensorsW8A8Fp8  # type: ignore[attr-defined]
    _register_chain(
        monkeypatch,
        "vllm.model_executor.layers.quantization.compressed_tensors.schemes."
        "compressed_tensors_w8a8_fp8",
        scheme_mod,
    )

    module.install()
    Config = state["ct"].CompressedTensorsConfig
    config = Config(fp8_channelwise_layers=["dense"])

    _Platform.ppu = True
    scheme = config.get_scheme(object(), "model.dense.proj")
    assert isinstance(scheme, CompressedTensorsW8A8Fp8)
    assert config.get_scheme(object(), "model.other") == "upstream-scheme"

    _Platform.ppu = False
    assert config.get_scheme(object(), "model.dense.proj") == "upstream-scheme"
    _Platform.ppu = True


def test_ct_moe_redirect_and_int8_swiglu(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _leaf("compressed_tensors_ppu")
    state = _build_ct_stubs(monkeypatch)
    module.install()

    layer = types.SimpleNamespace(moe_config="moe-cfg")
    state["upstream_moe_method"] = state["w4a4_cls"]("moe-cfg")
    _Platform.ppu = True
    method = state["ct_moe"].CompressedTensorsMoEMethod.get_moe_method(
        None, layer, "experts"
    )
    assert isinstance(method, state["mxfp4_method_cls"])
    assert method.moe_config == "moe-cfg"

    state["upstream_moe_method"] = object()
    assert (
        state["ct_moe"].CompressedTensorsMoEMethod.get_moe_method(
            None, layer, "experts"
        )
        is state["upstream_moe_method"]
    )

    _Platform.ppu = False
    state["upstream_moe_method"] = state["w4a4_cls"]("moe-cfg")
    assert (
        state["ct_moe"].CompressedTensorsMoEMethod.get_moe_method(
            None, layer, "experts"
        )
        is state["upstream_moe_method"]
    )
    _Platform.ppu = True

    int8_method = state["ct_int8"].CompressedTensorsW8A8Int8MoEMethod()
    cfg = int8_method.get_fused_moe_quant_config(
        types.SimpleNamespace(swiglu_limit=7.0)
    )
    assert cfg.gemm1_clamp_limit == 7.0
    _Platform.ppu = False
    cfg = int8_method.get_fused_moe_quant_config(
        types.SimpleNamespace(swiglu_limit=7.0)
    )
    assert cfg.gemm1_clamp_limit is None
    _Platform.ppu = True


def test_ct_kv_cache_params_not_trainable(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _leaf("compressed_tensors_ppu")
    state = _build_ct_stubs(monkeypatch)
    module.install()

    layer = types.SimpleNamespace()
    state["ct"].CompressedTensorsKVCacheMethod().create_weights(layer)
    for name in ("k_scale", "k_zero_point", "v_zero_point", "q_zero_point"):
        assert getattr(layer, name).requires_grad is False, name


def _leaf(name: str) -> types.ModuleType:
    """Load a fresh leaf-module instance (install state must not leak)."""
    module_name = f"_item2_leaf_test_{name}_{id(object())}"
    path = ENHANCEMENT_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    _CREATED.append(module_name)
    spec.loader.exec_module(module)
    return module


def test_ct_fp8_and_int8_preserve_swiglu_parameters(monkeypatch):
    state = _build_ct_stubs(monkeypatch)
    module = _leaf("compressed_tensors_ppu")
    module.install()
    layer = types.SimpleNamespace(swiglu_alpha=1.7, swiglu_beta=0.3, swiglu_limit=6.0)
    for provider, cls_name in (
        ("ct_int8", "CompressedTensorsW8A8Int8MoEMethod"),
        ("ct_fp8", "CompressedTensorsW8A8Fp8MoEMethod"),
    ):
        result = getattr(state[provider], cls_name)().get_fused_moe_quant_config(layer)
        assert (result.gemm1_alpha, result.gemm1_beta) == (1.7, 0.3)
