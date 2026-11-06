# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Upstream DeepSeek V4.1 quantization with PPU channelwise dense weights."""

from __future__ import annotations

from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.models.deepseek_v41.quant_config import (
    DeepseekV4FP8Config as UpstreamDeepseekV41FP8Config,
)
from vllm.platforms import current_platform


class DeepseekV41FP8Config(UpstreamDeepseekV41FP8Config):
    """Reuse upstream MXFP8 and MoE dispatch; specialize PPU mixed exports."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Keep these names in checkpoint space. Matching after reversing the
        # runtime prefix preserves the exact target/MTP layer precision.
        self.fp8_channelwise_layers: list[str] = []
        self._checkpoint_ignored_layers: list[str] = []
        # The multimodal wrapper has no class-level packed mapping.
        self.packed_modules_mapping = {
            **getattr(self, "packed_modules_mapping", {}),
            "gate_up_proj": ["w1", "w3"],
            "fused_wqa_wkv": ["wq_a", "wkv"],
            "fused_wkv_wgate": ["wkv", "wgate"],
        }

    @classmethod
    def get_name(cls) -> str:
        return "deepseek_v41_fp8"

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        if not current_platform.is_ppu() or not isinstance(hf_quant_cfg, dict):
            return None
        model_type = getattr(hf_config, "model_type", None)
        # SpeculativeConfig validates the draft before DSpark restores its
        # model_type to deepseek_v41. The target architecture is still present.
        is_dspark_draft = model_type == "deepseek_mtp" and (
            "DeepseekV41ForCausalLM"
            in (getattr(hf_config, "architectures", None) or [])
        )
        if (
            model_type not in ("deepseek_v41", "deepseek_v41_text")
            and not is_dspark_draft
        ):
            return None
        if hf_quant_cfg.get("quant_method") == "mxfp4" and hf_quant_cfg.get(
            "fp8_channelwise_layers"
        ):
            return cls.get_name()
        # Use upstream checkpoint recognition, including MXFP8 and Quark.
        # Its public name is still deepseek_v4_fp8 in the V4.1 module.
        upstream_quant = (
            "deepseek_v4_fp8"
            if is_dspark_draft or user_quant == cls.get_name()
            else user_quant
        )
        if super().override_quantization_method(
            hf_quant_cfg, upstream_quant, hf_config
        ):
            return cls.get_name()
        return None

    @classmethod
    def from_config(cls, config: dict) -> DeepseekV41FP8Config:
        if not config.get("fp8_channelwise_layers"):
            return super().from_config(config)
        # Mixed exports name the MoE dtype; the upstream parser needs the dense
        # FP8 format. Do not rewrite the checkpoint's quantization metadata.
        dense_config = {
            **config,
            "quant_method": "fp8",
            "activation_scheme": config.get("activation_scheme", "dynamic"),
            "ignored_layers": config.get("ignored_layers")
            or config.get("ignore")
            or [],
        }
        result = super().from_config(dense_config)
        result.fp8_channelwise_layers = list(config["fp8_channelwise_layers"])
        result._checkpoint_ignored_layers = list(dense_config["ignored_layers"])
        return result

    def get_quant_method(self, layer, prefix):
        if (
            current_platform.is_ppu()
            and isinstance(layer, LinearBase)
            and self.fp8_channelwise_layers
        ):
            from vllm.config import get_current_vllm_config

            from vllm_sail.models.deepseek_v41.config import checkpoint_layer_name

            hf = get_current_vllm_config().model_config.hf_config
            prefix = checkpoint_layer_name(prefix, hf.num_hidden_layers)
            if prefix == "context_wkv_proj":
                # DSpark copies each draft layer's WKV weights and scales into
                # one merged projection. Its shards must share one scheme.
                count = (
                    getattr(hf, "n_mtp_layers", None)
                    or getattr(hf, "num_nextn_predict_layers", None)
                    or 3
                )
                names = [f"mtp.{index}.attn.wkv" for index in range(count)]

                def channelwise(name):
                    return not is_layer_skipped(
                        prefix=name,
                        ignored_layers=self._checkpoint_ignored_layers,
                        fused_mapping=self.packed_modules_mapping,
                        match_mode="suffix",
                    ) and is_layer_skipped(
                        prefix=name,
                        ignored_layers=self.fp8_channelwise_layers,
                        fused_mapping=self.packed_modules_mapping,
                        match_mode="suffix",
                    )

                schemes = [channelwise(name) for name in names]
                if any(scheme != schemes[0] for scheme in schemes):
                    raise ValueError(
                        "DSpark context_wkv_proj requires matching WKV precision across draft layers"
                    )
                prefix = names[0]
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self._checkpoint_ignored_layers,
                fused_mapping=self.packed_modules_mapping,
                match_mode="suffix",
            ):
                return UnquantizedLinearMethod()
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.fp8_channelwise_layers,
                fused_mapping=self.packed_modules_mapping,
                match_mode="suffix",
            ):
                from compressed_tensors.quantization import (
                    QuantizationArgs,
                    QuantizationStrategy,
                    QuantizationType,
                )
                from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
                    CompressedTensorsLinearMethod,
                )
                from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (
                    CompressedTensorsW8A8Fp8,
                )

                layer.scheme = CompressedTensorsW8A8Fp8(
                    weight_quant=QuantizationArgs(
                        strategy=QuantizationStrategy.CHANNEL,
                        type=QuantizationType.FLOAT,
                        num_bits=8,
                        symmetric=True,
                    ),
                    is_static_input_scheme=False,
                )
                return CompressedTensorsLinearMethod(self)
            # Unlisted dense layers have BF16 weights and no FP8 scale to load.
            return UnquantizedLinearMethod()
        return super().get_quant_method(layer, prefix)
