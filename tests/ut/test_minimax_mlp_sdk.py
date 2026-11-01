# SPDX-License-Identifier: Apache-2.0
"""Check the actual installed MiniMax constructor without device allocation."""

import importlib
import sys
from types import SimpleNamespace as NS

import pytest


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_installed_mlp_keeps_unfused_cpu_tensor_identity(monkeypatch, dtype_name):
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    import vllm_sail.patch

    vllm_sail.patch.install()
    provider = importlib.import_module("vllm.models.minimax_m3.nvidia.model")
    from vllm.platforms import current_platform

    if not current_platform.is_ppu():
        pytest.skip("requires the installed SAIL platform")
    calls = []

    class Linear(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.prefix = kwargs["prefix"]

        def forward(self, x):
            calls.append(self.prefix)
            return x, None

    monkeypatch.setattr(provider, "MergedColumnParallelLinear", Linear)
    monkeypatch.setattr(provider, "RowParallelLinear", Linear)
    monkeypatch.setattr(
        provider, "SiluAndMulWithClamp", lambda **kwargs: torch.nn.Identity()
    )
    config = NS(
        hidden_size=8,
        hidden_act="swigluoai",
        swiglu_limit=7,
        swiglu_alpha=1.702,
        swiglu_beta=1,
    )
    instance = provider.MiniMaxM3MLP(config, 16, prefix="model.layers.0.mlp")
    assert instance.fused_gemm1_min_tokens is None
    x = torch.ones(3, 8, dtype=getattr(torch, dtype_name), device="cpu")
    assert instance(x) is x
    assert calls == ["model.layers.0.mlp.gate_up_proj", "model.layers.0.mlp.down_proj"]
    assert "vllm.models.minimax_m3.nvidia.ops.dense_gemm1_mxfp8" not in sys.modules
