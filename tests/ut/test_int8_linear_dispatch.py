# SPDX-License-Identifier: Apache-2.0
"""Import the production INT8 module with strict external dependency doubles."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def kernel(monkeypatch):
    def module(name, **attributes):
        value = ModuleType(name)
        value.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, value)
        return value

    base = type("ExternalLinearKernel", (), {})
    module("torch", Tensor=object, dtype=object, nn=SimpleNamespace(Module=object))
    module("vllm", _custom_ops=SimpleNamespace())
    dependencies = {
        "vllm.config": dict(get_current_vllm_config=Mock(spec=[])),
        "vllm.logger": dict(init_logger=lambda _: Mock(spec=["debug"])),
        "vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel": dict(
            Fp8BlockScaledMMLinearKernel=base,
        ),
        "vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel": dict(
            FP8ScaledMMLinearKernel=base,
            FP8ScaledMMLinearLayerConfig=object,
            Int8ScaledMMLinearKernel=base,
            Int8ScaledMMLinearLayerConfig=object,
        ),
        "vllm.model_executor.layers.quantization.input_quant_fp8": dict(
            QuantFP8=object
        ),
        "vllm.model_executor.layers.quantization.utils": dict(
            replace_parameter=Mock(spec=[])
        ),
        "vllm.model_executor.layers.quantization.utils.fp8_utils": dict(
            deepgemm_post_process_fp8_weight_block=Mock(spec=[])
        ),
        "vllm.model_executor.layers.quantization.utils.int8_utils": dict(
            per_token_group_quant_int8=Mock(spec=[])
        ),
        "vllm.model_executor.layers.quantization.utils.quant_utils": {
            name: object()
            for name in (
                "GroupShape",
                "kFp8DynamicTensorSym",
                "kFp8DynamicTokenSym",
                "kFp8StaticChannelSym",
                "kFp8StaticTensorSym",
            )
        },
        "vllm.model_executor.layers.quantization.utils.w8a8_utils": dict(
            convert_to_channelwise=Mock(spec=[])
        ),
        "vllm.platforms": dict(current_platform=SimpleNamespace(is_ppu=lambda: False)),
        "vllm.utils.torch_utils": dict(direct_register_custom_op=Mock(spec=[])),
        "vllm_sail.utils.deep_gemm": {
            name: Mock(spec=[])
            for name in (
                "fp8_gemm_nt",
                "get_deep_gemm_config",
                "int8_gemm_nt",
                "is_deep_gemm_supported",
                "should_use_deepgemm_for_fp8_linear",
            )
        },
    }
    for name, attributes in dependencies.items():
        module(name, **attributes)
    triton = module(
        "vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm",
        triton_scaled_mm=Mock(spec=[]),
    )
    path = (
        Path(__file__).parents[2]
        / "vllm_sail/model_executor/kernels/linear/scaled_mm/ppu.py"
    )
    spec = importlib.util.spec_from_file_location("_test_ppu_int8", path)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    return runtime.PPUInt8ScaledMMLinearKernel(), runtime.__dict__, triton


@pytest.mark.parametrize(
    ("asymmetric", "deepgemm", "acext", "bias", "row_major", "backend"),
    [
        (True, True, True, True, True, "azp"),
        (False, True, True, False, True, "deepgemm"),
        (False, True, True, True, True, "acext"),
        (False, True, False, True, True, "triton"),
        (False, False, True, True, True, "acext"),
        (False, False, False, False, False, "triton"),
        (False, False, False, True, True, "triton"),
        (False, False, False, True, False, "triton"),
    ],
)
@pytest.mark.parametrize("triton_quant", [False, True])
@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("static_input", [False, True])
def test_dispatch_preserves_bias_priority_layout_and_output_shape(
    kernel,
    asymmetric,
    deepgemm,
    acext,
    bias,
    row_major,
    backend,
    triton_quant,
    dtype,
    static_input,
):
    instance, environment, triton_module = kernel
    if backend == "deepgemm" and dtype == "fp16":
        backend = "acext" if acext else "triton"
    x = Mock(spec=["shape", "dtype", "reshape"], shape=(2, 3, 64), dtype=dtype)
    flat = Mock(spec=["shape", "dtype", "contiguous"], shape=(6, 64), dtype=dtype)
    x.reshape.return_value = flat
    flat.contiguous.return_value = flat
    output = Mock(spec=["shape", "view"], shape=(6, 128))
    calls = {
        name: Mock(spec=[], return_value=output)
        for name in ("azp", "deepgemm", "acext", "triton")
    }
    triton_module.triton_scaled_mm = calls["triton"]
    weight = Mock(spec=["t"])
    scale, quantized, input_scale = object(), object(), object()
    zero_point = object() if asymmetric else None
    adjustment = object() if asymmetric else None
    bias_tensor = object() if bias else None
    native_quant = Mock(spec=[], return_value=(quantized, input_scale, zero_point))
    triton = Mock(spec=[], return_value=(quantized, input_scale))
    environment.update(
        torch=SimpleNamespace(
            int8="int8",
            bfloat16="bf16",
            ops=SimpleNamespace(
                vllm=SimpleNamespace(
                    w8a8_int8_matmul_deepgemm=calls["deepgemm"],
                    w8a8_int8_matmul_acext=calls["acext"],
                )
            ),
        ),
        ops=SimpleNamespace(
            scaled_int8_quant=native_quant,
            cutlass_scaled_mm_azp=calls["azp"],
            cutlass_scaled_mm=Mock(
                spec=[], side_effect=AssertionError("unsupported CUTLASS")
            ),
        ),
        ppu_envs=SimpleNamespace(VLLM_SAIL_USE_TRITON_INT8_QUANT=triton_quant),
        per_token_group_quant_int8=triton,
    )
    stored_scale = object() if static_input else None
    stored_zp = zero_point if static_input else None
    instance._get_layer_params = lambda layer: (
        weight,
        scale,
        stored_scale,
        stored_zp,
        adjustment,
    )
    instance.use_deepgemm_int8_gemm = deepgemm
    instance.use_acext_int8_gemm = acext
    instance.weight_RowMajor = row_major

    assert instance.apply_weights(object(), x, bias_tensor) is output.view.return_value

    x.reshape.assert_called_once_with(-1, 64)
    output.view.assert_called_once_with(2, 3, 128)
    for name, call in calls.items():
        assert call.call_count == (1 if name == backend else 0)
    args, kwargs = calls[backend].call_args
    assert args[0] is quantized
    assert args[1] is (
        weight.t.return_value if backend == "triton" and row_major else weight
    )
    assert kwargs["out_dtype"] == dtype
    if backend == "deepgemm":
        assert "bias" not in kwargs
        assert kwargs["scale_x"] is input_scale
        assert kwargs["scale_w"] is scale
    else:
        assert kwargs["bias"] is bias_tensor
        assert kwargs["scale_a"] is input_scale
        assert kwargs["scale_b"] is scale
    if asymmetric:
        assert kwargs["azp_adj"] is adjustment
        assert kwargs["azp"] is (None if static_input else zero_point)
    if backend == "triton":
        assert kwargs["use_td"] is False
    uses_triton_quant = triton_quant and not asymmetric and not static_input
    assert triton.call_count == int(uses_triton_quant)
    assert native_quant.call_count == int(not uses_triton_quant)
    if not uses_triton_quant:
        native_quant.assert_called_once_with(
            flat,
            stored_scale,
            stored_zp,
            symmetric=not asymmetric,
        )


@pytest.mark.parametrize("weight_shape", [(128, 64), (1, 64)])
@pytest.mark.parametrize("backend", [None, "deepgemm", "acext", "triton"])
def test_weight_preparation_keeps_backend_layout(kernel, backend, weight_shape):
    instance, environment, _ = kernel
    weight = Mock(spec=["data", "t", "shape"], shape=weight_shape)
    scale = Mock(spec=["data"])
    layer = SimpleNamespace(weight=weight, scale=scale, logical_widths=[128])
    parameters = []

    def replace(layer, name, value):
        parameters.append((name, value))
        setattr(layer, name, value)

    environment.update(
        torch=SimpleNamespace(
            nn=SimpleNamespace(Parameter=lambda data, **kwargs: data)
        ),
        ppu_envs=SimpleNamespace(VLLM_SAIL_DENSE_BACKEND=backend),
        current_platform=SimpleNamespace(is_device_capability=lambda cap: True),
        _get_acext_int8_gemm=lambda: object(),
        is_deep_gemm_supported=lambda: True,
        replace_parameter=replace,
    )
    instance.config = SimpleNamespace(
        is_channelwise=True, is_static_input_scheme=False, input_symmetric=True
    )
    instance.layer_param_names = (
        "weight",
        "scale",
        "input_scale",
        "input_zp",
        "azp_adj",
    )

    instance.process_weights_after_loading(layer)

    # acext and DeepGEMM need both weight dims to be multiples of 16.
    aligned = weight_shape[0] % 16 == 0 and weight_shape[1] % 16 == 0
    row_major = aligned and backend != "triton"
    assert instance.weight_RowMajor is row_major
    assert parameters == [
        ("weight", weight.data if row_major else weight.t.return_value.data),
        ("scale", scale.data),
    ]
    assert weight.t.call_count == int(not row_major)
    assert instance.use_acext_int8_gemm is (aligned and backend in (None, "acext"))
    assert instance.use_deepgemm_int8_gemm is (
        aligned and backend in (None, "deepgemm")
    )
