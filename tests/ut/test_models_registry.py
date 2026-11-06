# SPDX-License-Identifier: Apache-2.0
"""Tests for PPU model registration into vLLM's ModelRegistry.

Bare-CPU tier: ``vllm_sail.models`` imports without vLLM (the hook must be
loadable in any process), and ``register_model`` is exercised against a
strict stub ModelRegistry following the ``tests/conftest.py`` no-blanket-mock
convention. With vLLM installed, registration is verified against the real
registry.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from types import SimpleNamespace
from unittest import mock

import pytest

import vllm_sail.models as models_pkg


class _RecordingRegistry:
    """Strict ModelRegistry stand-in mirroring upstream's contract."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.fail_archs: set[str] = set()
        self.models: dict[str, object] = {}

    def register_model(self, model_arch: str, model_cls) -> None:
        if not isinstance(model_arch, str):
            raise TypeError(f"`model_arch` should be a string, not {type(model_arch)}")
        if model_arch in self.fail_archs:
            raise ValueError(f"synthetic failure for {model_arch}")
        if not isinstance(model_cls, str) or model_cls.count(":") != 1:
            raise ValueError(f"Expected `<module>:<class>`, got {model_cls!r}")
        self.calls.append((model_arch, model_cls))
        self.models[model_arch] = model_cls


@pytest.fixture
def fresh_models_package(monkeypatch: pytest.MonkeyPatch):
    """Reload vllm_sail.models so the idempotency flag starts clear."""
    monkeypatch.setattr(models_pkg, "_registered", False, raising=False)
    return models_pkg


@pytest.fixture
def fake_vllm(monkeypatch: pytest.MonkeyPatch) -> _RecordingRegistry:
    registry = _RecordingRegistry()
    module = types.ModuleType("vllm")
    module.__spec__ = importlib.util.spec_from_loader("vllm", loader=None)
    module.ModelRegistry = registry
    monkeypatch.setitem(sys.modules, "vllm", module)
    return registry


EXPECTED_ARCHITECTURES = {
    "DeepseekV4ForCausalLM",
    "DeepSeekV4MTPModel",
    "DSparkDraftModel",
    "DSparkV41DraftModel",
    "GlmMoeDsaForCausalLM",
    "DeepseekV32MTPModel",
    "MiniMaxM3SparseForCausalLM",
    "MiniMaxM3SparseForConditionalGeneration",
}


def test_registered_architecture_inventory() -> None:
    """The inventory is the audited fork cross-check; change deliberately.

    * DeepseekV4ForCausalLM / DeepSeekV4MTPModel: the fork's A-status
      ppu/model.py + ppu/mtp.py (ported to vllm_sail/models/deepseek_v4/).
    * DSparkDraftModel: the PPU subclass injects quant mappings into the
      draft's independently-created quant config.
    * MiniMaxM3*: the fork adds SupportsQuant to the two upstream classes.
    """
    assert set(models_pkg._PPU_MODELS) == EXPECTED_ARCHITECTURES


def test_registration_targets_are_lazy_strings() -> None:
    for arch, target in models_pkg._PPU_MODELS.items():
        module_path, _, class_name = target.partition(":")
        if arch not in {"GlmMoeDsaForCausalLM", "DeepseekV32MTPModel"}:
            assert module_path.startswith("vllm_sail."), (arch, target)
        assert class_name, (arch, target)


@pytest.mark.parametrize(
    "arch,upstream_class,ppu_target",
    [
        (
            "GlmMoeDsaForCausalLM",
            "GlmMoeDsaForCausalLM",
            "vllm.model_executor.models.deepseek_v2:GlmMoeDsaForCausalLM",
        ),
        (
            "DeepseekV32MTPModel",
            "DeepSeekV32MTP",
            "vllm.model_executor.models.deepseek_mtp:DeepSeekMTP",
        ),
    ],
)
def test_glm_and_mtp_replace_fp8_fused_models(
    fresh_models_package, fake_vllm, arch, upstream_class, ppu_target
) -> None:
    fake_vllm.models[arch] = f"vllm.models.deepseek_v32:{upstream_class}"

    models_pkg.register_model()

    assert fake_vllm.models[arch] == ppu_target


def test_module_imports_without_vllm() -> None:
    """models/__init__.py must not import vllm at module level."""
    if "vllm" in sys.modules:
        pytest.skip("vLLM is installed here; the no-vLLM property is CI-mode A")
    assert models_pkg.register_model is not None


def test_register_model_registers_expected_architectures(
    fresh_models_package, fake_vllm: _RecordingRegistry
) -> None:
    models_pkg.register_model()
    registered = dict(fake_vllm.calls)
    assert set(registered) == EXPECTED_ARCHITECTURES
    assert (
        registered["DeepseekV4ForCausalLM"]
        == "vllm_sail.models.deepseek_v4.model:DeepseekV4ForCausalLM"
    )
    assert (
        registered["DeepSeekV4MTPModel"]
        == "vllm_sail.models.deepseek_v4.mtp:DeepSeekV4MTP"
    )
    assert (
        registered["DSparkDraftModel"]
        == "vllm_sail.models.deepseek_v4.dspark:DSparkDeepseekV4ForCausalLM"
    )


def test_register_model_is_idempotent(
    fresh_models_package, fake_vllm: _RecordingRegistry
) -> None:
    """vllm.general_plugins may call the hook more than once per process."""
    models_pkg.register_model()
    models_pkg.register_model()
    models_pkg.register_model()
    assert len(fake_vllm.calls) == len(EXPECTED_ARCHITECTURES)


def test_dspark_override_exposes_quant_mappings() -> None:
    pytest.importorskip("vllm", reason="requires vLLM model interfaces")
    from vllm.model_executor.models.interfaces import SupportsQuant

    from vllm_sail.models.deepseek_v4.dspark import DSparkDeepseekV4ForCausalLM

    assert issubclass(DSparkDeepseekV4ForCausalLM, SupportsQuant)
    assert set(DSparkDeepseekV4ForCausalLM.packed_modules_mapping) == {
        "gate_up_proj",
        "fused_wqa_wkv",
        "fused_wkv_wgate",
    }
    assert DSparkDeepseekV4ForCausalLM.hf_to_vllm_mapper is not None


def test_deepseek_mapper_selection_is_instance_local() -> None:
    """Building one quant variant must not mutate later model instances."""
    pytest.importorskip("vllm", reason="requires the vLLM DeepSeek V4 base")
    from vllm.models.deepseek_v4.nvidia.model import (
        DeepseekV4ForCausalLM as NvidiaDeepseekV4ForCausalLM,
    )

    from vllm_sail.models.deepseek_v4.model import DeepseekV4ForCausalLM

    class_mapper = DeepseekV4ForCausalLM.hf_to_vllm_mapper
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                expert_dtype="fp4",
                quantization_config={},
            )
        ),
        quant_config=SimpleNamespace(
            target_scheme_map={
                "model.layers": {"weights": SimpleNamespace(strategy="channel")}
            }
        ),
    )

    with mock.patch.object(NvidiaDeepseekV4ForCausalLM, "__init__", return_value=None):
        model = DeepseekV4ForCausalLM(vllm_config=vllm_config)

    assert DeepseekV4ForCausalLM.hf_to_vllm_mapper is class_mapper
    assert model.hf_to_vllm_mapper is not class_mapper
    assert (
        model.hf_to_vllm_mapper._map_name("layers.0.attn.wq_a.scale")
        == "model.layers.0.attn.wq_a.weight_scale"
    )


@pytest.mark.parametrize("tp_rank", [0, 3])
@pytest.mark.parametrize(
    "expert_dtype,channelwise_layers",
    [
        ("fp4", None),
        ("fp4", ["layers.0.attn.wq_a"]),
        ("fp8", None),
        ("int8", None),
        ("int4", None),
    ],
)
def test_deepseek_target_loader_filters_mtp_weights(
    monkeypatch, expert_dtype, channelwise_layers, tp_rank
) -> None:
    """The target loader must not receive parameters owned by the draft model."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from vllm.model_executor.model_loader import weight_utils
    from vllm.model_executor.models.utils import AutoWeightsLoader
    from vllm.models.deepseek_v4.nvidia import model as upstream

    from vllm_sail.models.deepseek_v4.model import (
        DeepseekV4ForCausalLM,
        _make_deepseek_v4_weights_mapper,
    )

    # Exercise real loader traversal with one tiny CPU parameter, without
    # constructing the full model or initializing a distributed device group.
    target = DeepseekV4ForCausalLM.__new__(DeepseekV4ForCausalLM)
    torch.nn.Module.__init__(target)
    body = upstream.DeepseekV4Model.__new__(upstream.DeepseekV4Model)
    torch.nn.Module.__init__(body)
    body.config = SimpleNamespace(num_attention_heads=4)
    body.quant_config = None
    body.use_sequence_parallel = False
    body.get_expert_mapping = lambda: []
    layer = torch.nn.Module()
    layer.attn = torch.nn.Module()
    monkeypatch.setattr(weight_utils, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(weight_utils, "get_tensor_model_parallel_rank", lambda: tp_rank)
    with torch.device("cpu"):
        layer.attn.attn_sink = weight_utils.make_attention_sink(3, torch.float32)
    registered_loader = layer.attn.attn_sink.weight_loader
    sink_loads = []

    def record_sink_load(param, loaded_weight):
        sink_loads.append(loaded_weight)
        registered_loader(param, loaded_weight)

    layer.attn.attn_sink.weight_loader = record_sink_load
    body.layers = torch.nn.ModuleList([layer])
    target.model = body
    monkeypatch.setattr(upstream, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(upstream, "is_pp_missing_parameter", lambda *args: False)

    mapper = _make_deepseek_v4_weights_mapper(expert_dtype, channelwise_layers)
    weights = [
        ("layers.0.attn.attn_sink", torch.arange(1, 5, dtype=torch.float32)),
        ("mtp.0.attn.attn_sink", torch.full((4,), 99.0)),
        ("mtp.0.attn.wq_a.scale", torch.ones(1)),
        ("mtp.0.ffn.experts.0.w1.scale", torch.ones(1)),
    ]
    loaded = AutoWeightsLoader(target).load_weights(weights, mapper=mapper)
    assert loaded == {"model.layers.0.attn.attn_sink"}
    assert len(sink_loads) == 1 and sink_loads[0] is weights[0][1]
    assert layer.attn.attn_sink[0].item() == tp_rank + 1
    assert torch.isneginf(layer.attn.attn_sink[1:]).all()


def test_dspark_draft_keeps_mtp_names_for_its_own_loader() -> None:
    pytest.importorskip("vllm")
    from vllm_sail.models.deepseek_v4.dspark import DSparkDeepseekV4ForCausalLM

    draft = DSparkDeepseekV4ForCausalLM.__new__(DSparkDeepseekV4ForCausalLM)
    key = "mtp.0.attn.attn_sink"
    assert draft._remap_dspark_name(key) == "model.layers.0.attn.attn_sink"
    assert draft._remap_dspark_name("layers.0.attn.attn_sink") is None
    # Quantization setup consumes renames, not target-only weight exclusions.
    rename_mapper = draft.hf_to_vllm_mapper.get_rename_mapper()
    assert rename_mapper._map_name(key) == "model.mtp.0.attn.attn_sink"


def test_one_broken_model_does_not_break_the_hook(
    fresh_models_package, fake_vllm: _RecordingRegistry, caplog
) -> None:
    """A missing optional model must be logged, not raised."""
    fake_vllm.fail_archs.add("MiniMaxM3SparseForCausalLM")
    models_pkg.register_model()
    registered = {arch for arch, _ in fake_vllm.calls}
    assert registered == EXPECTED_ARCHITECTURES - {"MiniMaxM3SparseForCausalLM"}


@pytest.mark.parametrize(
    "failed_arch",
    ["DeepSeekV4MTPModel", "GlmMoeDsaForCausalLM", "DeepseekV32MTPModel"],
)
def test_required_registration_failure_is_transactional(
    fresh_models_package, fake_vllm: _RecordingRegistry, failed_arch
) -> None:
    previous = {arch: object() for arch in models_pkg._REQUIRED_PPU_MODELS}
    fake_vllm.models.update(previous)
    fake_vllm.fail_archs.add(failed_arch)

    with pytest.raises(RuntimeError, match="restored the previous model registry"):
        models_pkg.register_model()

    assert fake_vllm.models == previous
    assert models_pkg._registered is False


def test_missing_vllm_fails_closed(
    fresh_models_package, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "vllm", None)  # forces ImportError
    with pytest.raises(RuntimeError, match="cannot register required PPU models"):
        models_pkg.register_model()


# ---------------------------------------------------------------------------
# With a real vLLM: the architectures must land in the actual registry.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_registry():
    pytest.importorskip("torch", reason="requires torch")
    vllm = pytest.importorskip("vllm", reason="requires vLLM")
    return vllm.ModelRegistry


def test_registration_lands_in_real_registry(real_registry) -> None:
    models_pkg._registered = False
    previous = {arch: real_registry.models.get(arch) for arch in EXPECTED_ARCHITECTURES}
    try:
        models_pkg.register_model()
        for arch, target in models_pkg._PPU_MODELS.items():
            module_name, class_name = target.split(":")
            registered = real_registry.models[arch]
            assert registered.module_name == module_name, arch
            assert registered.class_name == class_name, arch
    finally:
        for arch, old_model in previous.items():
            if old_model is None:
                real_registry.models.pop(arch, None)
            else:
                real_registry.models[arch] = old_model
        models_pkg._registered = True
