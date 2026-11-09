# SPDX-License-Identifier: Apache-2.0
"""V4.1 integration contracts; CPU tests never import torch or vLLM."""

from types import SimpleNamespace

import pytest

from tests.support.modules import load_patch


@pytest.fixture
def quant_module(modules):
    calls = []

    class Upstream:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        @classmethod
        def override_quantization_method(cls, *args):
            calls.append(("recognize", args))
            return "upstream-method"

        @classmethod
        def from_config(cls, config):
            calls.append(("config", config))
            return cls()

        def get_quant_method(self, layer, prefix):
            calls.append(("dispatch", layer, prefix))
            return "upstream-kernel"

    modules("vllm.models.deepseek_v41.quant_config", DeepseekV4FP8Config=Upstream)
    modules("vllm.model_executor.layers.fused_moe", RoutedExperts=type("MoE", (), {}))
    modules(
        "vllm.model_executor.layers.linear",
        LinearBase=type("Linear", (), {}),
        UnquantizedLinearMethod=object,
    )
    modules(
        "vllm.model_executor.layers.quantization.utils.quant_utils",
        is_layer_skipped=lambda **kwargs: False,
    )
    modules("vllm.platforms", current_platform=SimpleNamespace(is_ppu=lambda: True))
    return load_patch("vllm_sail/models/deepseek_v41/quant_config.py"), calls


def test_standard_quantization_delegates_without_rewriting_config(quant_module):
    module, calls = quant_module
    cls = module.DeepseekV41FP8Config
    checkpoint = {"quant_method": "fp8", "weight_block_size": [32, 32]}
    hf = SimpleNamespace(model_type="deepseek_v41")
    assert cls.override_quantization_method(checkpoint, None, hf) == "deepseek_v41_fp8"
    config = cls.from_config(checkpoint)
    layer = object()
    assert config.get_quant_method(layer, "model.ffn") == "upstream-kernel"
    assert calls == [
        ("recognize", (checkpoint, None, hf)),
        ("config", checkpoint),
        ("dispatch", layer, "model.ffn"),
    ]
    assert calls[1][1] is checkpoint


@pytest.mark.parametrize("quantized_drafts", [(), (0, 1), (0,)])
def test_dspark_context_projection_inherits_source_wkv_precision(
    quant_module, modules, quantized_drafts
):
    import sys

    module, _ = quant_module
    helpers = load_patch("vllm_sail/models/deepseek_v41/config.py")
    modules(
        "vllm_sail.models.deepseek_v41.config",
        checkpoint_layer_name=helpers.checkpoint_layer_name,
    )
    hf = SimpleNamespace(num_hidden_layers=40, n_mtp_layers=2)
    modules(
        "vllm.config",
        get_current_vllm_config=lambda: SimpleNamespace(
            model_config=SimpleNamespace(hf_config=hf)
        ),
    )
    module.is_layer_skipped = (
        lambda *, prefix, ignored_layers, **kwargs: prefix in ignored_layers
    )
    modules(
        "compressed_tensors.quantization",
        QuantizationArgs=lambda **kwargs: kwargs,
        QuantizationStrategy=SimpleNamespace(CHANNEL="channel"),
        QuantizationType=SimpleNamespace(FLOAT="float"),
    )
    modules(
        "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors",
        CompressedTensorsLinearMethod=lambda config: ("channelwise", config),
    )
    modules(
        "vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8",
        CompressedTensorsW8A8Fp8=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    config = module.DeepseekV41FP8Config()
    config.fp8_channelwise_layers = ["layers.0.attn.wq_a"] + [
        f"mtp.{index}.attn.wkv" for index in quantized_drafts
    ]
    layer = sys.modules["vllm.model_executor.layers.linear"].LinearBase()
    if len(quantized_drafts) == 1:
        with pytest.raises(ValueError, match="matching WKV precision"):
            config.get_quant_method(layer, "model.context_wkv_proj")
    elif quantized_drafts:
        assert config.get_quant_method(layer, "model.context_wkv_proj") == (
            "channelwise",
            config,
        )
        assert layer.scheme.weight_quant["strategy"] == "channel"
    else:
        assert type(config.get_quant_method(layer, "model.context_wkv_proj")) is object


@pytest.mark.parametrize(
    "model_type,architectures",
    [
        ("deepseek_v41", ["DeepseekV41ForCausalLM"]),
        ("deepseek_v41_text", []),
        # SpeculativeConfig verifies this temporary config before rewriting
        # it to deepseek_v41 / DSparkV41DraftModel.
        ("deepseek_mtp", ["DeepseekV41ForCausalLM"]),
    ],
)
def test_qlean_metadata_is_translated_without_editing_checkpoint(
    quant_module, model_type, architectures
):
    module, calls = quant_module
    cls = module.DeepseekV41FP8Config
    cls.__bases__[0].override_quantization_method = classmethod(lambda *args: None)
    checkpoint = {
        "quant_method": "mxfp4",
        "expert_dtype": "mxfp4",
        "fp8_channelwise_layers": ["layers.0.attn.wo_a"],
        "ignore": ["layers.0.attn.compressor.wkv"],
    }
    assert (
        cls.override_quantization_method(
            checkpoint,
            None,
            SimpleNamespace(model_type=model_type, architectures=architectures),
        )
        == "deepseek_v41_fp8"
    )
    result = cls.from_config(checkpoint)
    translated = calls[-1][1]
    assert translated["quant_method"] == "fp8"
    assert translated["activation_scheme"] == "dynamic"
    assert translated["ignored_layers"] == checkpoint["ignore"]
    assert result.fp8_channelwise_layers == checkpoint["fp8_channelwise_layers"]
    assert (
        checkpoint["quant_method"] == "mxfp4" and "activation_scheme" not in checkpoint
    )


@pytest.mark.parametrize("quant_method", ["fp8", "mxfp4"])
@pytest.mark.parametrize(
    "ppu,model_type,architectures",
    [
        (False, "deepseek_v41", ["DeepseekV41ForCausalLM"]),
        (True, "deepseek_v4", ["DeepseekV4ForCausalLM"]),
        (True, "deepseek_v4_text", []),
        (True, "deepseek_mtp", ["DeepSeekV4MTPModel"]),
        (True, "deepseek_mtp", ["DeepSeekMTPModel"]),
        (True, "other", []),
    ],
)
def test_v41_quant_override_leaves_other_models_and_platforms_untouched(
    quant_module, quant_method, ppu, model_type, architectures
):
    module, calls = quant_module
    module.current_platform.is_ppu = lambda: ppu
    assert (
        module.DeepseekV41FP8Config.override_quantization_method(
            {
                "quant_method": quant_method,
                "fp8_channelwise_layers": ["layers.0.attn.wq_a"],
            },
            None,
            SimpleNamespace(model_type=model_type, architectures=architectures),
        )
        is None
    )
    assert calls == []


@pytest.mark.parametrize("ppu", [False, True])
def test_quant_registration_is_lazy_idempotent_and_preserves_v4(modules, ppu):
    calls = []
    modules("vllm.logger", init_logger=lambda _: SimpleNamespace(debug=lambda *a: None))
    modules("vllm.platforms", current_platform=SimpleNamespace(is_ppu=lambda: ppu))
    modules(
        "vllm.model_executor.layers.quantization",
        QUANTIZATION_METHODS=[],
        register_quantization_config=lambda name: lambda cls: calls.append((name, cls)),
    )
    mixed = type("Mixed", (), {"get_name": staticmethod(lambda: "mixed_precision_w4")})
    v4 = type("V4", (), {})
    v41 = type("V41", (), {"get_name": staticmethod(lambda: "deepseek_v41_fp8")})
    modules(
        "vllm_sail.model_executor.layers.quantization.mixed_precision_w4",
        MixedPrecisionW4Config=mixed,
    )
    modules("vllm_sail.models.deepseek_v4.quant_config", DeepseekV4FP8Config=v4)
    modules("vllm_sail.models.deepseek_v41.quant_config", DeepseekV41FP8Config=v41)
    conversions = []
    for module_name, class_name, name in (
        ("fp8_to_int8", "Fp8ToInt8Config", "fp8-to-int8"),
        ("fp8_to_bf16", "Fp8ToBf16Config", "fp8-to-bf16"),
        ("bf16_to_int8", "Bf16ToInt8Config", "bf16-to-int8"),
    ):
        cls = type(class_name, (), {"get_name": staticmethod(lambda name=name: name)})
        modules(
            f"vllm_sail.model_executor.layers.quantization.{module_name}",
            **{class_name: cls},
        )
        conversions.append((name, cls))
    registry = load_patch("vllm_sail/registry/quant_config/__init__.py")
    assert calls == []
    registry.register()
    registry.register()
    assert calls == [("mixed_precision_w4", mixed)] + (
        conversions + [("deepseek_v4_fp8", v4), ("deepseek_v41_fp8", v41)]
        if ppu
        else []
    )


def test_v41_selector_preserves_cuda_and_other_backends(modules):
    upstream, ppu, other = (type(name, (), {}) for name in ("Upstream", "PPU", "Other"))
    platform = SimpleNamespace(ppu=True)
    modules(
        "vllm.platforms", current_platform=SimpleNamespace(is_ppu=lambda: platform.ppu)
    )
    modules(
        "vllm.models.deepseek_v41.nvidia.flashmla", DeepseekV4FlashMLAAttention=upstream
    )
    modules("vllm_sail.models.deepseek_v41.flashmla", DeepseekV41FlashMLAAttention=ppu)

    def original(config):
        return config

    provider = modules(
        "vllm.models.deepseek_v41.nvidia.model",
        _select_dsv4_attn_cls=original,
        _linear_scale_param_name=lambda *args: "weight_scale_inv",
    )
    original_scale = provider._linear_scale_param_name
    draft = modules(
        "vllm.models.deepseek_v41.nvidia.dspark",
        _linear_scale_param_name=original_scale,
    )
    vl = modules(
        "vllm.models.deepseek_v41.nvidia.vl_model",
        _linear_scale_param_name=original_scale,
    )
    replacement = load_patch("vllm_sail/patch/enhancement/models/deepseek_v41.py")
    assert provider._select_dsv4_attn_cls is replacement._select_dsv4_attn_cls
    marker = getattr(provider._select_dsv4_attn_cls, replacement.PATCH_MARKER)
    assert marker[replacement._TARGET] is original
    assert provider._select_dsv4_attn_cls(upstream) is ppu
    assert provider._select_dsv4_attn_cls(other) is other
    config = SimpleNamespace(
        quant_config=SimpleNamespace(fp8_channelwise_layers=["attn.wo_a"])
    )
    assert (
        draft._linear_scale_param_name
        is vl._linear_scale_param_name
        is provider._linear_scale_param_name
    )
    assert draft._linear_scale_param_name(config, "fp4") == "weight_scale"
    platform.ppu = False
    assert provider._select_dsv4_attn_cls(upstream) is upstream
    assert draft._linear_scale_param_name(config, "fp4") == "weight_scale_inv"
    with pytest.raises(RuntimeError, match="already patched"):
        load_patch("vllm_sail/patch/enhancement/models/deepseek_v41.py")


@pytest.mark.parametrize(
    "dequant_at_load,block_size", [(True, 32), (False, 32), (True, 128)]
)
def test_v41_projection_preserves_weights_and_uses_bf16_helper(
    modules, dequant_at_load, block_size
):
    calls = []
    loaded = SimpleNamespace(element_size=lambda: 2 if dequant_at_load else 1)
    bf16 = loaded if dequant_at_load else object()
    modules(
        "vllm.model_executor.layers.quantization.utils.mxfp8_utils",
        dequant_mxfp8_to_bf16=lambda *args: bf16,
    )
    modules(
        "vllm.models.deepseek_v4.nvidia.ops.o_proj",
        deep_gemm_fp8_o_proj=lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    class Upstream:
        forward_mqa = object()
        _forward_prefill = object()

    modules(
        "vllm.models.deepseek_v41.nvidia.flashmla", DeepseekV4FlashMLAAttention=Upstream
    )
    modules("vllm_sail.models.deepseek_v4.flashmla", DeepseekV4FlashMLAAttention=object)
    modules("vllm.platforms", current_platform=SimpleNamespace(is_ppu=lambda: True))
    module = load_patch("vllm_sail/models/deepseek_v41/flashmla.py")
    attn = module.DeepseekV41FlashMLAAttention()
    attn.__dict__.update(
        _o_proj_block_size=block_size,
        wo_a=SimpleNamespace(weight=loaded, weight_scale=object()),
        wo_b=object(),
        rotary_emb=SimpleNamespace(cos_sin_cache=object()),
        n_local_groups=2,
        n_local_heads=16,
        nope_head_dim=448,
        rope_head_dim=64,
        o_lora_rank=1024,
    )
    o, pos = object(), object()
    attn._o_proj(o, pos)
    args, kwargs = calls.pop()
    assert args[:2] == (o, pos) and args[3].weight is bf16 and args[4] is attn.wo_b
    assert kwargs["heads_per_group"] == 8 and kwargs["einsum_recipe"] == (
        1,
        1,
        block_size,
    )
    assert attn.wo_a.weight is loaded
    assert attn.forward_mqa is Upstream.forward_mqa
    assert attn._forward_prefill is Upstream._forward_prefill


def test_v41_draft_inherits_its_own_loader_and_execution(modules):
    class Upstream:
        load_weights = object()
        forward = object()

    class SupportsQuant:
        pass

    mapper_args = []
    mapper = object()

    def make_mapper(*args):
        mapper_args.append(args)
        return mapper

    modules("vllm.model_executor.models.interfaces", SupportsQuant=SupportsQuant)
    modules(
        "vllm.models.deepseek_v41.nvidia.dspark", DSparkDeepseekV4ForCausalLM=Upstream
    )
    modules(
        "vllm.models.deepseek_v41.nvidia.model",
        _make_deepseek_v4_weights_mapper=make_mapper,
    )
    draft = load_patch(
        "vllm_sail/models/deepseek_v41/dspark.py"
    ).DSparkDeepseekV41ForCausalLM
    assert issubclass(draft, SupportsQuant)
    assert (
        draft.load_weights is Upstream.load_weights
        and draft.forward is Upstream.forward
    )
    assert draft.hf_to_vllm_mapper is mapper
    assert mapper_args == [("fp4", "weight_scale")]
    assert draft.packed_modules_mapping["fused_wqa_wkv"] == ["wq_a", "wkv"]
