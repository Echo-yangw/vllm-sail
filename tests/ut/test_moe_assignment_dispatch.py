# SPDX-License-Identifier: Apache-2.0
"""MoE host-offload assignment and expert override dispatch, without kernels."""

import functools
import importlib.util
import inspect
import math
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tests.support.source import function
from tests.ut.test_ppu_kernel_capabilities import modules as _modules

modules = _modules

PATCH = "vllm_sail/patch/enhancement/triton_moe.py"
FUSED = "vllm/model_executor/layers/fused_moe/fused_moe.py"
EXPERTS = "vllm/model_executor/layers/fused_moe/experts/triton_moe.py"


class Tensor:
    def __init__(self, shape, dtype="bf16", device="host-double"):
        self.shape = tuple(shape)
        self.dtype, self.device = dtype, device

    def size(self, axis=None):
        return self.shape if axis is None else self.shape[axis]

    def dim(self):
        return len(self.shape)

    def stride(self, axis):
        return math.prod(self.shape[axis % self.dim() + 1 :])

    def is_contiguous(self):
        return True

    def numel(self):
        return math.prod(self.shape)

    def view(self, *shape):
        if -1 in shape:
            size = self.numel() // -math.prod(shape)
            shape = tuple(size if n == -1 else n for n in shape)
        assert math.prod(shape) == self.numel()
        return Tensor(shape, self.dtype, self.device)

    def __getitem__(self, index):
        start, stop, step = index.indices(self.shape[0])
        assert step == 1
        return Tensor((stop - start, *self.shape[1:]), self.dtype, self.device)


def context(modules, resident, source_root=None, copied=True, functional=False):
    calls, assignments, alignments, events = [], [], [], []
    hidden, w1, w2 = Tensor((2, 4)), Tensor((32, 8, 4)), Tensor((32, 4, 4))
    ids, weights, output = Tensor((2, 2), "i32"), Tensor((2, 2)), Tensor((2, 4))
    if resident is not None:
        (w1, w2)[resident]._vllm_is_uva_offloaded = True
    config = {
        "BLOCK_SIZE_M": 16,
        "UP": {"BLOCK_SIZE_M": 32},
        "DOWN": {"BLOCK_SIZE_M": 64},
        "USE_VALU": True,
    }
    aligned = (Tensor((64,), "i32"), Tensor((4,), "i32"), Tensor((1,), "i32"))

    def empty(*shape, **kwargs):
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        return Tensor(shape, kwargs["dtype"], kwargs["device"])

    def full(shape, value, *, dtype, device):
        assert (shape, value, dtype, device) == ((1,), 64, "i32", ids.device)
        return Tensor(shape, dtype, device)

    def align(*args, **kwargs):
        alignments.append((args, kwargs))
        return aligned

    namespace = {
        "torch": NS(
            float32="f32",
            float16="f16",
            bfloat16="bf16",
            float8_e4m3fn="fp8",
            float8_e4m3fnuz="fp8uz",
            int8="i8",
            int32="i32",
            empty=empty,
            empty_like=lambda x: Tensor(x.shape, x.dtype, x.device),
            full=full,
        ),
        "tl": NS(float32="f32", float16="f16", bfloat16="bf16"),
        "functools": functools,
        "current_platform": NS(is_ppu=lambda: True),
        "try_get_optimal_moe_config": lambda *a, **kw: config,
        "_get_config_quant_dtype": lambda **kw: None,
        "_get_config_dtype_str": lambda **kw: None,
        "_resize_cache": lambda workspace, shape: Tensor(shape),
        "moe_kernel_quantize_input": lambda A, A_scale, *a, **kw: (A, A_scale),
        "MoEActivation": NS(SILU="silu", from_str=lambda x: x),
        "mk": NS(
            FusedMoEExpertsModular=NS(adjust_N_for_activation=lambda n, a: n // 2)
        ),
        "apply_moe_activation": lambda *a: events.append(("activation", a)),
        "ops": NS(moe_sum=lambda *a: events.append(("sum", a))),
        "moe_align_block_size": align,
    }
    if source_root is not None:
        owner = function(source_root / FUSED, "_weights_host_resident", namespace)
        prepare = function(source_root / FUSED, "_prepare_expert_assignment", namespace)
    else:

        def owner(*weights):
            return any(getattr(w, "_vllm_is_uva_offloaded", False) for w in weights)

        def prepare(*args, **kwargs):
            return aligned

    def assignment(*args, **kwargs):
        assignments.append((args, kwargs))
        return prepare(*args, **kwargs)

    namespace.update(
        _weights_host_resident=owner, _prepare_expert_assignment=assignment
    )
    kernel_name = (
        "dispatch_fused_moe_kernel" if functional else "invoke_fused_moe_triton_kernel"
    )
    kernel_path = PATCH if copied else source_root / FUSED
    signature = inspect.signature(function(kernel_path, kernel_name, {}))

    def kernel(*args, **kwargs):
        assert "weights_host_resident" not in kwargs
        signature.bind(*args, **kwargs)
        calls.append((args, kwargs))

    namespace[kernel_name] = kernel
    modules(
        "vllm_sail.model_executor.layers.fused_moe.triton_kernels",
        moe_sum_reduce_triton=lambda *a: pytest.fail("unexpected large-batch path"),
    )
    modules(
        "vllm.model_executor.layers.fused_moe.config",
        _get_config_dtype_str=lambda **kw: None,
    )
    return NS(
        namespace=namespace,
        hidden=hidden,
        w1=w1,
        w2=w2,
        ids=ids,
        weights=weights,
        output=output,
        config=config,
        aligned=aligned,
        calls=calls,
        assignments=assignments,
        alignments=alignments,
        events=events,
    )


def check_assignment(ctx, resident, actual, sorted_index):
    args, kwargs = ctx.assignments[0]
    assert len(ctx.assignments) == 1
    assert args == (ctx.ids, ctx.config, 2, 2, 32, None)
    assert kwargs["weights_host_resident"] is (resident is not None)
    assert kwargs["use_int8_w8a16"] is False and kwargs["use_int4_w4a16"] is False
    if actual:
        assert len(ctx.alignments) == int(resident is not None)
        if resident is not None:
            assert ctx.alignments[0][0] == (ctx.ids, 16, 32, None)
            assert ctx.alignments[0][1] == {
                "ignore_invalid_experts": kwargs["ignore_invalid_experts"]
            }
            assert all(call[0][sorted_index] is ctx.aligned[0] for call in ctx.calls)
        else:
            assert all(call[0][sorted_index] is None for call in ctx.calls)
    else:
        assert all(call[0][sorted_index] is ctx.aligned[0] for call in ctx.calls)


def check_apply(modules, resident, router, source_root=None, copied=True):
    ctx = context(modules, resident, source_root, copied)

    def route(value):
        ctx.events.append(("route", value))
        # Deliberately differs from the base's `not value` to detect bypass.
        return value

    def finalize(*args):
        ctx.events.append(("finalize", args))

    layer = NS(
        quant_config=NS(
            use_int4_w4a16=False,
            use_int8_w8a16=False,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            config_name=lambda dtype: None,
        ),
        expects_unquantized_inputs=False,
        quant_dtype=None,
        block_shape=None,
        per_act_token_quant=False,
        quantization_emulation=False,
        _lora_context=None,
        # This fixture exercises legacy host assignment, with no tail consumer.
        _uses_post_gemm2_hook=lambda tokens: False,
        moe_problem_size=lambda *a: (32, 2, 8, 4, 2),
        adjust_N_for_activation=lambda n, a: n // 2,
        activation_config=NS(clamp_limit=None),
        activation=lambda *a: ctx.events.append(("activation", a)),
        a1_scale=None,
        w1_scale=None,
        w2_scale=None,
        w1_bias=None,
        w2_bias=None,
        moe_sum=lambda *a: pytest.fail("finalize helper was bypassed"),
        _apply_router_weight_in_gemm2=route,
        _finalize_expert_output=finalize,
    )
    path, name = (
        (PATCH, "apply") if copied else (source_root / EXPERTS, "TritonExperts.apply")
    )
    function(path, name, ctx.namespace)(
        layer,
        ctx.output,
        ctx.hidden,
        ctx.w1,
        ctx.w2,
        ctx.weights,
        ctx.ids,
        "relu",
        -1,
        None,
        None,
        None,
        Tensor((32,)),
        Tensor((32,)),
        None,
        router,
    )
    check_assignment(ctx, resident, source_root is not None, 6)
    assert len(ctx.calls) == 2
    assert ctx.calls[0][0][9] is False and ctx.calls[1][0][9] is router
    assert [event[0] for event in ctx.events] == ["activation", "route", "finalize"]
    assert ctx.events[1] == ("route", router)
    cache, output, ids, expert_map = ctx.events[2][1]
    assert cache.shape == (2, 2, 4)
    assert output is ctx.output and ids is ctx.ids and expert_map is None
    if copied:
        assert ctx.calls[0][0][11] is ctx.config["UP"]
        assert ctx.calls[1][0][11] is ctx.config["DOWN"]
        assert all(call[1]["use_valu"] is True for call in ctx.calls)


@pytest.mark.parametrize("quantize", [False, True])
def test_empty_expert_batch_does_not_prepare_or_launch(
    modules, patch_utils_module, quantize
):
    ctx = context(modules, None)

    def unexpected(*args, **kwargs):
        pytest.fail("empty batches must not quantize, prepare assignments or launch")

    ctx.namespace["moe_kernel_quantize_input"] = unexpected
    ctx.namespace["try_get_optimal_moe_config"] = unexpected
    layer = NS(
        quant_config=NS(use_int4_w4a16=False),
        expects_unquantized_inputs=quantize,
        a1_scale=None,
        quant_dtype=None,
        per_act_token_quant=False,
        block_shape=None,
        quantization_emulation=False,
        moe_problem_size=unexpected,
    )
    provider = modules("vllm.model_executor.layers.fused_moe.fused_moe")
    for name in (
        "invoke_fused_moe_triton_kernel",
        "invoke_fused_moe_wna16_triton_kernel",
        "dispatch_fused_moe_kernel",
        "get_config_file_name",
        "get_moe_configs",
        "get_moe_wna16_block_config",
        "try_get_optimal_moe_config",
        "fused_experts_impl",
        "should_moe_wna16_use_cuda",
    ):
        setattr(provider, name, unexpected)
    experts = modules("vllm.model_executor.layers.fused_moe.experts.triton_moe")
    experts.__dict__.update(ctx.namespace)
    experts.TritonExperts = type("TritonExperts", (), {"apply": unexpected})
    modules("vllm.platforms", current_platform=NS(is_ppu=lambda: True))
    registry = patch_utils_module.PATCH_REGISTRY
    before = len(registry)
    spec = importlib.util.spec_from_file_location(
        "_empty_moe_patch", Path(__file__).parents[2] / PATCH
    )
    target = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(target)
        result = experts.TritonExperts.apply(
            layer,
            ctx.output[:0],
            ctx.hidden[:0],
            ctx.w1,
            ctx.w2,
            ctx.weights[:0],
            ctx.ids[:0],
            "relu",
            -1,
            None,
            None,
            None,
            Tensor((32,)),
            Tensor((32,)),
            None,
            False,
        )
    finally:
        del registry[before:]
    assert result is None
    assert not ctx.calls and not ctx.assignments and not ctx.events


@pytest.mark.parametrize("resident", [None, 0])
@pytest.mark.parametrize("router", [False, True])
def test_experts_use_host_assignment_and_override_helpers(modules, resident, router):
    check_apply(modules, resident, router)


@pytest.mark.upstream_source
@pytest.mark.parametrize("copied", [False, True])
@pytest.mark.parametrize("resident", [None, 0, 1])
@pytest.mark.parametrize("router", [False, True])
def test_actual_host_assignment_and_override_helpers(
    modules, upstream_source_root, copied, resident, router
):
    check_apply(modules, resident, router, upstream_source_root, copied)


def check_functional(modules, resident, source_root=None, copied=True):
    ctx = context(modules, resident, source_root, copied, functional=True)
    path = PATCH if copied else source_root / FUSED
    output = function(path, "fused_experts_impl", ctx.namespace)(
        ctx.hidden, ctx.w1, ctx.w2, ctx.weights, ctx.ids, activation="relu"
    )
    check_assignment(ctx, resident, source_root is not None, 7)
    assert output.shape == (2, 4)
    assert len(ctx.calls) == 2
    assert ctx.calls[0][0][10] is False and ctx.calls[1][0][10] is True
    assert [event[0] for event in ctx.events] == ["activation", "sum"]
    assert ctx.events[1][1][0].shape == (2, 2, 4)
    assert ctx.events[1][1][1] is output
    if copied:
        assert ctx.calls[0][0][12] is ctx.config["UP"]
        assert ctx.calls[1][0][12] is ctx.config["DOWN"]
        assert all(call[1]["use_valu"] is True for call in ctx.calls)


@pytest.mark.parametrize("resident", [None, 1])
def test_functional_experts_forward_host_residency(modules, resident):
    check_functional(modules, resident)


@pytest.mark.upstream_source
@pytest.mark.parametrize("copied", [False, True])
@pytest.mark.parametrize("resident", [None, 1])
def test_actual_functional_host_residency(
    modules, upstream_source_root, copied, resident
):
    check_functional(modules, resident, upstream_source_root, copied)
