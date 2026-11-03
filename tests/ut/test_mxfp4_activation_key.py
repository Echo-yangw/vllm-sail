# SPDX-License-Identifier: Apache-2.0
"""Carry checkpoint activation schemas through the MXFP4 converter seam."""

from types import SimpleNamespace as NS

import pytest

from tests.support.source import function

PATCH = "vllm_sail/patch/enhancement/mxfp4.py"


@pytest.mark.parametrize("separated", [False, True])
@pytest.mark.parametrize("with_key", [False, True])
def test_non_ppu_converter_preserves_checkpoint_activation_and_layout(
    separated, with_key
):
    calls = []
    expected, activation, key = object(), object(), object() if with_key else None
    inputs = tuple(object() for _ in range(8))

    def upstream(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    convert = function(
        PATCH,
        "install._convert_weight",
        {
            "_ppu_backends": lambda _: (),
            "_upstream_convert_weight": upstream,
        },
    )
    assert (
        convert(
            "HUMMING",
            *inputs,
            activation=activation,
            activation_key=key,
            use_separated_a4w4=separated,
        )
        is expected
    )
    assert calls == [
        (
            ("HUMMING", *inputs),
            dict(
                activation=activation, use_separated_a4w4=separated, activation_key=key
            ),
        )
    ]


def test_ppu_w4a4_converter_keeps_native_weight_conversion_with_activation_key():
    expected, key = object(), object()
    weights = tuple(object() for _ in range(6))
    calls = []

    def native(*args):
        calls.append(args)
        return expected

    convert = function(
        PATCH,
        "install._convert_weight",
        {
            "_ppu_backends": lambda kind: ("PPU",) if kind == "w4a4" else (),
            "_convert_ppu_weights": native,
        },
    )
    assert convert("PPU", object(), *weights, activation_key=key) is expected
    assert calls == [weights]


@pytest.mark.upstream_source
@pytest.mark.parametrize("override", ["", "fp8"])
def test_humming_converter_retains_checkpoint_key_unless_explicit_override(
    monkeypatch, upstream_source_root, override
):
    import sys
    from types import ModuleType

    path = upstream_source_root / "vllm/model_executor/layers/fused_moe/oracle/mxfp4.py"
    backend = NS(
        CUTLASS_MXFP4_MXFP4="cutlass", DEEPGEMM_MXFP4="deepgemm", HUMMING="humming"
    )
    key, schema = object(), object()
    key_calls, conversions = [], []
    module = ModuleType("vllm.model_executor.layers.quantization.utils.humming")

    def convert(layer, **kwargs):
        conversions.append((layer, kwargs))

    def to_schema(value):
        key_calls.append(value)
        return schema

    module.convert_to_humming_moe_kernel_format = convert
    module.quant_key_to_input_schema = to_schema
    monkeypatch.setitem(sys.modules, module.__name__, module)
    layer = NS(
        w13_weight=object(),
        w2_weight=object(),
        w13_weight_scale=object(),
        w2_weight_scale=object(),
    )
    namespace = {
        "Mxfp4MoeBackend": backend,
        "kMxfp8Dynamic": object(),
        "envs": NS(VLLM_HUMMING_INPUT_QUANT_CONFIG=override),
        "current_platform": NS(is_rocm=lambda: False),
        "B12X_BACKENDS": (),
        "QuantizationConfigArgs": type("QuantizationArgs", (), {}),
        "get_current_vllm_config": lambda: NS(
            model_config=NS(quantization_config=None)
        ),
    }
    for name in (
        "_user_moe_activation_override",
        "_resolve_activation_key",
        "_humming_input_schema",
    ):
        function(path, name, namespace)
    converter = function(path, "convert_weight_to_mxfp4_moe_kernel_format", namespace)
    converter("humming", layer, *[object() for _ in range(4)], activation_key=key)
    assert key_calls == ([] if override else [key])
    assert conversions == [
        (
            layer,
            dict(
                quant_config={"quant_method": "mxfp4"},
                input_schema=None if override else schema,
                allow_input_schema_fallback=False,
            ),
        )
    ]


@pytest.mark.upstream_source
@pytest.mark.parametrize("ppu", [False, True])
def test_w4a8_checkpoint_method_retains_inherited_activation_state(
    monkeypatch, upstream_source_root, ppu
):
    import ast
    import sys
    from types import ModuleType

    from tests.support.source import definition

    key = object()

    class Mxfp4MoEMethod:
        def __init__(self, activation_key):
            self.activation_key = activation_key

    path = (
        upstream_source_root
        / "vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a8_mxfp4.py"
    )
    node = definition(path, "CompressedTensorsW4A8Mxfp4MoEMethod")
    namespace = {"Mxfp4MoEMethod": Mxfp4MoEMethod}
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    cls = namespace["CompressedTensorsW4A8Mxfp4MoEMethod"]
    assert cls.__bases__ == (Mxfp4MoEMethod,)
    method = cls(key)
    platform = ModuleType("vllm.platforms")
    platform.current_platform = NS(is_ppu=lambda: ppu)
    monkeypatch.setitem(sys.modules, platform.__name__, platform)
    old_class_module = ModuleType(
        "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_mxfp4"
    )
    old_class_module.CompressedTensorsW4A4Mxfp4MoEMethod = type("W4A4", (), {})
    monkeypatch.setitem(sys.modules, old_class_module.__name__, old_class_module)
    calls = []

    def original(*args):
        calls.append(args)
        return method

    get_method = function(
        "vllm_sail/patch/enhancement/compressed_tensors_ppu.py",
        "install._get_moe_method",
        {"_upstream_get_moe_method": original},
    )
    quant, layer = object(), object()
    assert get_method(quant, layer, "model.experts") is method
    assert method.activation_key is key
    assert calls == [(quant, layer, "model.experts")]
