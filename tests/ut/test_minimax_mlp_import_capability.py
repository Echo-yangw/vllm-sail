# SPDX-License-Identifier: Apache-2.0
"""Run the whole upstream model's MLP across the NVIDIA import boundary."""

import ast
import importlib.abc
import importlib.util
import sys
from types import SimpleNamespace as NS

import pytest

from tests.ut.test_ppu_kernel_capabilities import load_patch
from tests.ut.test_ppu_kernel_capabilities import modules as _modules

modules = _modules


def load_model(modules, monkeypatch, source, ppu):
    """Configure only external imports; execute the complete real model module."""
    path = source / "vllm/models/minimax_m3/nvidia/model.py"

    def unavailable(self, *args, **kwargs):
        raise AssertionError("Unconfigured external dependency was invoked")

    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.ImportFrom) and node.module.startswith(
            ("vllm", "transformers")
        ):
            attrs = {
                alias.name: type(alias.name, (), {"__init__": unavailable})
                for alias in node.names
            }
            modules(node.module, **attrs)
    modules("vllm", _custom_ops=modules("vllm._custom_ops"))
    modules("vllm.envs")

    class Module:
        def __init__(self):
            self.initialized = True

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    tensor = type("Tensor", (), {})
    nn = modules("torch.nn", Module=Module, Parameter=tensor, ModuleList=list)
    modules(
        "torch",
        nn=nn,
        Tensor=tensor,
        dtype=type("dtype", (), {}),
        device=type("device", (), {}),
    )
    modules("vllm.logger", init_logger=lambda name: NS())
    modules(
        "vllm.multimodal",
        MULTIMODAL_REGISTRY=NS(
            register_processor=lambda processor, *, info, dummy_inputs: lambda cls: cls
        ),
    )
    modules(
        "vllm.model_executor.models.utils",
        WeightsMapper=lambda *, orig_to_new_prefix, orig_to_new_substr: NS(),
    )
    modules(
        "vllm.compilation.breakable_cudagraph",
        eager_break_during_capture=lambda fn: fn,
    )
    modules(
        "vllm.platforms", current_platform=NS(is_ppu=lambda: ppu, is_cuda=lambda: True)
    )
    name = "vllm.models.minimax_m3.nvidia.model"
    modules(name)
    spec = importlib.util.spec_from_file_location(name, path)
    provider = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, provider)
    monkeypatch.setattr(sys.modules[name.rpartition(".")[0]], "model", provider)
    spec.loader.exec_module(provider)
    return provider


@pytest.mark.upstream_source
@pytest.mark.parametrize("ppu", [False, True])
@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("threshold", [16, 64])
@pytest.mark.parametrize("autotune", [False, True])
def test_registered_mlp_keeps_constructor_and_forward_contract(
    modules, monkeypatch, upstream_source_root, ppu, fuse, threshold, autotune
):
    provider = load_model(modules, monkeypatch, upstream_source_root, ppu)
    calls = []
    quant = object()
    gate_up, activated, output = object(), object(), object()

    class Linear:
        def __init__(self, *args, **kwargs):
            self.args, self.kwargs = args, kwargs
            calls.append(("linear", self))

        def __call__(self, x):
            calls.append(("linear-forward", self, x))
            return (
                gate_up
                if len(self.args) == 2 and isinstance(self.args[1], list)
                else output
            ), None

    class Activation:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __call__(self, x):
            assert x is gate_up
            calls.append(("act", x))
            return activated

    provider.MergedColumnParallelLinear = provider.RowParallelLinear = Linear
    provider.SiluAndMulWithClamp = Activation
    provider.envs = NS(VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=threshold)
    provider.is_flashinfer_autotuning = lambda: autotune
    config = NS(
        hidden_size=8,
        hidden_act="swigluoai",
        swiglu_limit=7,
        swiglu_alpha=1.702,
        swiglu_beta=1,
    )
    name = "vllm.models.minimax_m3.nvidia.ops"

    class Forbidden(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == name or fullname.startswith(name + "."):
                raise ModuleNotFoundError("NVIDIA-only dense import", name="quack")
            return None

    def capable(*args):
        calls.append(("capability", args))
        return fuse

    def fused(x, linear, act):
        calls.append(("fused-forward", x, linear, act))
        return activated

    if ppu:
        monkeypatch.setattr(sys, "meta_path", [Forbidden(), *sys.meta_path])
        with pytest.raises(ModuleNotFoundError, match="NVIDIA-only"):
            provider.MiniMaxM3MLP(config, 16)
        calls.clear()
    else:
        dense = modules(
            name + ".dense_gemm1_mxfp8",
            can_fuse_gemm1=capable,
            MIN_TOKENS=32,
            fused_gemm1=fused,
        )
        modules(name, dense_gemm1_mxfp8=dense)
    load_patch("vllm_sail/patch/enhancement/models/minimax_m3.py")
    disable = threshold == 64
    mlp = provider.MiniMaxM3MLP(config, 16, quant, not disable, disable, "layers.0.mlp")
    assert mlp.initialized
    assert mlp.gate_up_proj.args == (8, [16, 16])
    assert mlp.gate_up_proj.kwargs == dict(
        bias=False,
        quant_config=quant,
        disable_tp=disable,
        prefix="layers.0.mlp.gate_up_proj",
    )
    assert mlp.down_proj.args == (16, 8)
    assert mlp.down_proj.kwargs == dict(
        bias=False,
        quant_config=quant,
        reduce_results=not disable,
        disable_tp=disable,
        prefix="layers.0.mlp.down_proj",
    )
    assert mlp.act_fn.kwargs == dict(swiglu_limit=7, alpha=1.702, beta=1)
    assert mlp.fused_gemm1_min_tokens == (
        None if ppu or not fuse else max(32, threshold)
    )
    assert len([c for c in calls if c[0] == "capability"]) == (0 if ppu else 1)
    calls.clear()
    x = NS(shape=(64, 8))
    assert mlp(x) is output
    if not ppu and fuse and threshold < 64 and not autotune:
        assert calls == [
            ("fused-forward", x, mlp.gate_up_proj, mlp.act_fn),
            ("linear-forward", mlp.down_proj, activated),
        ]
    else:
        assert calls == [
            ("linear-forward", mlp.gate_up_proj, x),
            ("act", gate_up),
            ("linear-forward", mlp.down_proj, activated),
        ]


@pytest.mark.upstream_source
@pytest.mark.parametrize("ppu", [False, True])
def test_registered_mlp_retains_activation_validation(
    modules, monkeypatch, upstream_source_root, ppu
):
    provider = load_model(modules, monkeypatch, upstream_source_root, ppu)
    provider.MergedColumnParallelLinear = provider.RowParallelLinear = lambda *a, **k: (
        object()
    )
    load_patch("vllm_sail/patch/enhancement/models/minimax_m3.py")
    with pytest.raises(ValueError, match="Only swigluoai is supported"):
        provider.MiniMaxM3MLP(NS(hidden_size=8, hidden_act="unsupported"), 16)
