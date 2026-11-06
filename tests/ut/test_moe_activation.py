# SPDX-License-Identifier: Apache-2.0
"""MoE experts preserve resolved activation configuration and precedence."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from tests.support.source import ROOT, definition


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("mxfp4", [False, True])
def test_ppu_experts_reuse_base_activation_config(batched, mxfp4):
    """Custom quantizers must use the same resolved values as activation()."""
    basename = "batched_deep_gemm_moe" if batched else "deep_gemm_moe"
    name = "PPUBatchedDeepGemmExperts" if batched else "PPUDeepGemmExperts"
    if mxfp4:
        name += "MXFP4"
    path = ROOT / f"vllm_sail/model_executor/layers/fused_moe/experts/{basename}.py"
    init = definition(path, f"{name}.__init__")
    resolved = SimpleNamespace(clamp_limit=6.5, alpha=1.8, beta=0.25)

    class Base:
        def __init__(self, **kwargs):
            self.activation_config = resolved

    cls = ast.ClassDef(
        name=name,
        bases=[ast.Name(id="Base", ctx=ast.Load())],
        keywords=[],
        body=[init],
        decorator_list=[],
    )
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    namespace = {"Base": Base}
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    moe = SimpleNamespace(activation_situ_beta=None, activation_situ_linear_beta=None)
    quant = SimpleNamespace(
        block_shape=None, gemm1_clamp_limit=None, gemm1_alpha=None, gemm1_beta=None
    )
    kwargs = dict(moe_config=moe, quant_config=quant)
    if batched:
        kwargs.update(max_num_tokens=32, num_dispatchers=2)
    obj = namespace[name](**kwargs)
    assert (obj.gemm1_clamp_limit, obj.gemm1_alpha, obj.gemm1_beta) == (6.5, 1.8, 0.25)


@pytest.mark.upstream_source
def test_upstream_activation_config_precedence(upstream_source_root):
    """Execute upstream's resolver to pin quant > model > neutral precedence."""
    path = upstream_source_root / "vllm/model_executor/layers/fused_moe/activation.py"
    cls = definition(path, "ApplyMoEActivationConfig")
    module = ast.Module(body=[cls], type_ignores=[])
    namespace = {"dataclass": dataclass}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    config = namespace["ApplyMoEActivationConfig"]
    moe = SimpleNamespace(
        swiglu_limit=7.0,
        swiglu_alpha=1.7,
        swiglu_beta=0.3,
        activation_situ_beta=1.2,
        activation_situ_linear_beta=-0.8,
    )
    quant = SimpleNamespace(gemm1_clamp_limit=None, gemm1_alpha=None, gemm1_beta=None)
    model = config.from_configs(moe, quant)
    assert (model.clamp_limit, model.alpha, model.beta) == (7.0, 1.7, 0.3)
    quant.gemm1_clamp_limit, quant.gemm1_alpha, quant.gemm1_beta = 4.0, 0.0, 0.0
    override = config.from_configs(moe, quant)
    assert (override.clamp_limit, override.alpha, override.beta) == (4.0, 0.0, 0.0)
    assert (override.activation_situ_beta, override.activation_situ_linear_beta) == (
        1.2,
        -0.8,
    )
    moe.swiglu_limit = moe.swiglu_alpha = moe.swiglu_beta = None
    quant.gemm1_clamp_limit = quant.gemm1_alpha = quant.gemm1_beta = None
    neutral = config.from_configs(moe, quant)
    assert (neutral.clamp_limit, neutral.alpha, neutral.beta) == (None, 1.0, 0.0)
