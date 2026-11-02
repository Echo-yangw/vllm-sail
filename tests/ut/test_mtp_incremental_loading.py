# SPDX-License-Identifier: Apache-2.0
"""MTP keeps unfinished FP8 pairs across public load_weights calls."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from uuid import uuid4

import pytest

from vllm_sail.patch.utils import PATCH_REGISTRY


@pytest.fixture
def install_loader(monkeypatch):
    before = len(PATCH_REGISTRY)

    def install(
        helper,
        *,
        torch=None,
        completeness=lambda: False,
        spec_layer=lambda config, name: 0,
        resolved_shared=False,
        global_shared=False,
        expert_mapping=lambda *a, **k: [],
        remap_scale=lambda name, parameters: name,
    ):
        modules = {}

        def register(name, **attributes):
            parts = name.split(".")
            for depth in range(1, len(parts) + 1):
                full = ".".join(parts[:depth])
                if full not in modules:
                    module = ModuleType(full)
                    module.__path__ = []
                    modules[full] = module
                    monkeypatch.setitem(sys.modules, full, module)
                    if depth > 1:
                        setattr(
                            modules[".".join(parts[: depth - 1])],
                            parts[depth - 1],
                            module,
                        )
            for key, value in attributes.items():
                setattr(modules[name], key, value)

        class Target:
            def __init__(self, *, vllm_config, prefix=""):
                self.quant_config = SimpleNamespace(
                    get_name=lambda: "modelopt_fp4" if vllm_config.fp4 else "fp8"
                )
                self.config = SimpleNamespace(n_routed_experts=0, n_shared_experts=0)
                self.model = SimpleNamespace(mtp_start_layer_idx=0, num_mtp_layers=1)
                self.parameters = vllm_config.parameters
                self.is_fused_shared_expert_enabled = resolved_shared

            def named_parameters(self):
                return self.parameters.items()

            def _rewrite_spec_layer_name(self, index, name):
                assert index == 0
                return name

            def load_weights(self, weights):
                raise AssertionError("the real plugin loader must replace this target")

        monkeypatch.setitem(sys.modules, "torch", torch or ModuleType("torch"))
        register(
            "vllm._aiter_ops",
            rocm_aiter_ops=SimpleNamespace(
                is_fusion_moe_shared_experts_enabled=lambda: global_shared
            ),
        )
        register(
            "vllm.model_executor.layers.fused_moe",
            fused_moe_make_expert_params_mapping=expert_mapping,
        )
        register(
            "vllm.model_executor.model_loader.mtp_validation",
            is_mtp_completeness_check_enabled=completeness,
        )
        register(
            "vllm.model_executor.model_loader.weight_utils",
            default_weight_loader=lambda *a: pytest.fail(
                "use the parameter's registered loader"
            ),
            maybe_remap_kv_scale_name=remap_scale,
        )
        register("vllm.model_executor.models.deepseek_mtp", DeepSeekMTP=Target)
        register(
            "vllm.model_executor.models.deepseek_v2", _try_load_fp8_indexer_wk=helper
        )
        register(
            "vllm.model_executor.models.utils",
            get_pp_missing_layer_names=lambda model: [],
            get_spec_layer_idx_from_weight_name=spec_layer,
        )
        name = "_sail_incremental_mtp_" + uuid4().hex
        source = (
            Path(__file__).parents[2]
            / "vllm_sail/patch/enhancement/models/deepseek_mtp.py"
        )
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return Target

    yield install
    del PATCH_REGISTRY[before:]


def test_loader_preserves_external_buffer_state_across_calls(install_loader):
    buffers = []

    def helper(name, tensor, pending, parameters, loaded, missing):
        buffers.append(pending)
        return True

    target = install_loader(helper)
    model = target(vllm_config=SimpleNamespace(fp4=True, parameters={}))
    model.load_weights([("model.layers.0.indexer.wk.weight", object())])
    marker = object()
    buffers[0]["unfinished"] = marker
    model.load_weights([("model.layers.0.indexer.wk.weight_scale_inv", object())])
    assert buffers[0] is buffers[1]
    assert buffers[1]["unfinished"] is marker


def test_loader_buffers_belong_to_each_model(install_loader):
    buffers = []

    def helper(name, tensor, pending, parameters, loaded, missing):
        buffers.append(pending)
        return True

    target = install_loader(helper)
    for _ in range(2):
        model = target(vllm_config=SimpleNamespace(fp4=True, parameters={}))
        model.load_weights([("model.layers.0.indexer.wk.weight", object())])
    assert buffers[0] is not buffers[1]


def test_non_fp4_loader_preserves_separate_weight_ownership(install_loader):
    calls = []

    def forbidden(*args):
        pytest.fail("non-FP4 weights must not enter the fused indexer helper")

    weight = object()
    parameter = SimpleNamespace(
        weight_loader=lambda param, value: calls.append((param, value))
    )
    name = "model.layers.0.indexer.wk.weight"
    target = install_loader(forbidden)
    model = target(vllm_config=SimpleNamespace(fp4=False, parameters={name: parameter}))
    assert model.load_weights([(name, weight)]) == {name}
    assert calls == [(parameter, weight)]


@pytest.mark.parametrize("fp4", [False, True])
def test_tied_top_level_embedding_loads_without_rewriting(install_loader, fp4):
    writes = []
    embedding_name = "model.embed_tokens.weight"
    ordinary_name = "model.layers.0.norm.weight"
    embedding = SimpleNamespace(
        weight_loader=lambda param, value: writes.append((param, value))
    )
    ordinary = SimpleNamespace(weight_loader=lambda param, value: None)
    value = object()
    target = install_loader(
        lambda *a: False,
        spec_layer=lambda config, name: 0 if ".layers.0." in name else None,
    )
    model = target(
        vllm_config=SimpleNamespace(
            fp4=fp4, parameters={embedding_name: embedding, ordinary_name: ordinary}
        )
    )
    loaded = model.load_weights(
        [
            (embedding_name, value),
            ("unowned.weight", object()),
            (ordinary_name, object()),
        ]
    )
    assert loaded == {embedding_name, ordinary_name}
    assert writes == [(embedding, value)]


@pytest.mark.parametrize("resolved_shared", [False, True])
@pytest.mark.parametrize("global_shared", [False, True])
def test_expert_mapping_uses_the_models_resolved_shared_flag(
    install_loader, resolved_shared, global_shared
):
    observed = []

    def mapping(model, **kwargs):
        observed.append(kwargs["num_experts"])
        return []

    name = "model.layers.0.norm.weight"
    target = install_loader(
        lambda *a: False,
        resolved_shared=resolved_shared,
        global_shared=global_shared,
        expert_mapping=mapping,
    )
    model = target(
        vllm_config=SimpleNamespace(
            fp4=False,
            parameters={name: SimpleNamespace(weight_loader=lambda *a: None)},
        )
    )
    model.config.n_routed_experts = 3
    model.config.n_shared_experts = 2
    assert model.load_weights([(name, object())]) == {name}
    assert observed == [5 if resolved_shared else 3]


def test_disabled_model_fusion_keeps_shared_projection_whole(install_loader):
    writes = []
    name = "model.layers.0.mlp.shared_experts.gate_up_proj.weight"
    parameter = SimpleNamespace(
        weight_loader=lambda param, value, shard: writes.append((param, value, shard))
    )
    target = install_loader(lambda *a: False, resolved_shared=False, global_shared=True)
    model = target(vllm_config=SimpleNamespace(fp4=False, parameters={name: parameter}))
    model.config.n_shared_experts = 2
    value = object()
    assert model.load_weights(
        [("model.layers.0.mlp.shared_experts.gate_proj.weight", value)]
    ) == {name}
    assert writes == [(parameter, value, 0)]


def test_ignored_kv_scale_preserves_string_name_for_layer_validation(install_loader):
    classified = []
    ordinary_name = "model.layers.0.norm.weight"
    ignored_name = "model.layers.0.self_attn.k_scale"

    def layer(config, name):
        assert isinstance(name, str)
        classified.append(name)
        return 0

    target = install_loader(
        lambda *a: False,
        spec_layer=layer,
        remap_scale=lambda name, parameters: None if name == ignored_name else name,
    )
    model = target(
        vllm_config=SimpleNamespace(
            fp4=False,
            parameters={ordinary_name: SimpleNamespace(weight_loader=lambda *a: None)},
        )
    )
    assert model.load_weights(
        [(ignored_name, object()), (ordinary_name, object())]
    ) == {
        ignored_name,
        ordinary_name,
    }
    assert classified.count(ignored_name) == 2


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "layout", ["tensor", "row-vector", "row-matrix", "block", "e8m0"]
)
def test_real_fp8_helper_pairs_split_calls_and_preserves_fused_padding(
    install_loader, reverse, layout
):
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from vllm.model_executor.model_loader.mtp_validation import (
        disable_mtp_completeness_check,
        is_mtp_completeness_check_enabled,
    )
    from vllm.model_executor.models.deepseek_v2 import _try_load_fp8_indexer_wk

    target = install_loader(
        _try_load_fp8_indexer_wk,
        torch=torch,
        completeness=is_mtp_completeness_check_enabled,
    )
    prefix = "model.layers.0.self_attn.indexer"
    fused_name = prefix + ".wk_weights_proj.weight"
    parameter = torch.nn.Parameter(
        torch.full((5, 4), -777.0, dtype=torch.bfloat16), requires_grad=False
    )
    writes = []

    def load_shard(param, value, shard):
        assert param is parameter and shard == 0
        writes.append(value.clone())
        param[:4].copy_(value)

    parameter.weight_loader = load_shard
    interleaved = []
    ordinary_name = "model.layers.0.norm.weight"
    ordinary = SimpleNamespace(
        weight_loader=lambda param, value: interleaved.append(value)
    )
    model = target(
        vllm_config=SimpleNamespace(
            fp4=True, parameters={fused_name: parameter, ordinary_name: ordinary}
        )
    )
    weight = torch.arange(1, 17, dtype=torch.float32).view(4, 4).to(torch.float8_e4m3fn)
    if layout == "tensor":
        scale, expanded = torch.tensor([[2.0]]), torch.full((4, 4), 2.0)
    elif layout in ("row-vector", "row-matrix", "e8m0"):
        scale = torch.tensor([1.0, 2.0, 4.0, 8.0])
        expanded = scale[:, None].expand(4, 4)
        if layout == "row-matrix":
            scale = scale[:, None]
        elif layout == "e8m0":
            scale = torch.tensor([[127], [128], [129], [130]], dtype=torch.uint8)
    else:
        scale = torch.tensor([[1.0, 2.0], [4.0, 8.0]])
        expanded = scale.repeat_interleave(2, 0).repeat_interleave(2, 1)
    pair = [(prefix + ".wk.weight", weight), (prefix + ".wk.weight_scale_inv", scale)]
    if reverse:
        pair.reverse()
    with torch.inference_mode(), disable_mtp_completeness_check():
        assert model.load_weights(pair[:1]) == set()
        assert not writes
        token = object()
        assert model.load_weights([(ordinary_name, token)]) == {ordinary_name}
        assert interleaved == [token]
    with torch.inference_mode():
        assert model.load_weights(pair[1:]) == {fused_name}
    expected = (weight.float() * expanded).to(torch.bfloat16)
    assert len(writes) == 1
    torch.testing.assert_close(parameter[:4], expected, rtol=0, atol=0)
    torch.testing.assert_close(
        parameter[4], torch.full((4,), -777.0, dtype=torch.bfloat16), rtol=0, atol=0
    )
    # A completed pair must not be reused by the next incremental load.
    with torch.inference_mode(), disable_mtp_completeness_check():
        assert model.load_weights(pair[:1]) == set()
        assert len(writes) == 1
    with torch.inference_mode():
        assert model.load_weights(pair[1:]) == {fused_name}
    assert len(writes) == 2
