# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PPU-specific DeepSeek V4 model.

Inherits from the NVIDIA implementation and configures weight mapping
to support PPU-specific quantization schemes (INT8/INT4 channelwise,
FP8 channelwise dense + FP4 MoE mixed-precision).
"""

import re

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.config.kernel import MEGA_MOE_BACKENDS
from vllm.model_executor.models.utils import WeightsMapper
from vllm.models.deepseek_v4.nvidia.model import (
    DeepseekV4ForCausalLM as NvidiaDeepseekV4ForCausalLM,
)


def _make_deepseek_v4_weights_mapper(
    expert_dtype: str,
    fp8_channelwise_layers: list[str] | None = None,
) -> WeightsMapper:
    """Fork's extended mapper (upstream lacks the int8/int4 branches).

    INT8/INT4 checkpoints store scales that must land on ``weight_scale``
    (no ``_inv`` suffix); the upstream mapper would send them to
    ``weight_scale_inv`` and weight loading dies with a KeyError.
    """
    if expert_dtype == "fp4":
        if fp8_channelwise_layers:
            # Mixed-precision checkpoint (mxfp4 MoE + fp8 channelwise dense):
            # both register ``weight_scale`` (no ``_inv`` suffix).
            scale_regex: dict[re.Pattern, str] = {
                re.compile(r"\.scale$"): ".weight_scale",
            }
        else:
            # MXFP4 experts register ``w{1,2,3}_weight_scale``; FP8 linear and
            # shared experts register ``weight_scale_inv``.
            scale_regex = {
                re.compile(r"(\.experts\.\d+\.w[123])\.scale$"): (r"\1.weight_scale"),
                re.compile(r"\.scale$"): ".weight_scale_inv",
            }
    elif expert_dtype == "fp8":
        scale_regex = {
            re.compile(r"\.scale$"): ".weight_scale_inv",
        }
    elif expert_dtype in ("int8", "int4"):
        scale_regex = {
            re.compile(r"\.scale$"): ".weight_scale",
        }
    else:
        raise ValueError(f"Invalid expert_dtype: {expert_dtype}")
    return WeightsMapper(
        orig_to_new_prefix={
            "layers.": "model.layers.",
            "embed.": "model.embed.",
            "norm.": "model.norm.",
            "hc_head": "model.hc_head",
            "mtp.": "model.mtp.",
        },
        orig_to_new_regex=scale_regex,
        orig_to_new_suffix={
            "head.weight": "lm_head.weight",
            "embed.weight": "embed_tokens.weight",
            ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
        },
        orig_to_new_substr={
            ".shared_experts.w2": ".shared_experts.down_proj",
            # Draft weights are loaded separately by the MTP/DSpark model.
            "mtp.": None,
        },
    )


def _validate_pcp_config(vllm_config: VllmConfig) -> None:
    parallel_config = vllm_config.parallel_config
    if parallel_config.prefill_context_parallel_size <= 1:
        return
    unsupported = (
        parallel_config.decode_context_parallel_size != 1
        or parallel_config.pipeline_parallel_size != 1
        or parallel_config.data_parallel_size != 1
        or parallel_config.nnodes != 1
        or parallel_config.use_ubatching
        or vllm_config.speculative_config is not None
        or vllm_config.kv_transfer_config is not None
        or vllm_config.kernel_config.moe_backend in MEGA_MOE_BACKENDS
    )
    if unsupported:
        raise NotImplementedError(
            "DeepSeek V4 PCP on PPU requires one node, DCP=PP=DP=1, no "
            "speculation, KV transfer, microbatching, or MegaMoE."
        )
    if vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
        raise NotImplementedError(
            "DeepSeek V4 PCP on PPU requires -cc.cudagraph_mode=NONE."
        )
    cache_dtype = str(vllm_config.cache_config.cache_dtype)
    if cache_dtype != "auto" and not cache_dtype.startswith("fp8"):
        raise NotImplementedError(
            "DeepSeek V4 PCP on PPU requires an FP8 DeepSeek MLA KV cache."
        )


class DeepseekV4ForCausalLM(NvidiaDeepseekV4ForCausalLM):
    """PPU-specific DeepSeek V4 model.

    Differences from NVIDIA version:
    1. Adds packed_modules_mapping for fused weight loading
    2. Detects fp8_channelwise_layers and int8/int4 expert_dtype for mapper
    """

    packed_modules_mapping = {
        "gate_up_proj": ["w1", "w3"],
        "fused_wqa_wkv": ["wq_a", "wkv"],
        "fused_wkv_wgate": ["wkv", "wgate"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        _validate_pcp_config(vllm_config)
        config = vllm_config.model_config.hf_config
        expert_dtype = getattr(config, "expert_dtype", "fp4")

        # Detect fp8 channelwise layers from quantization config
        fp8_channelwise_layers: list[str] = []
        quant_cfg = getattr(config, "quantization_config", {}) or {}
        fp8_channelwise_layers = quant_cfg.get("fp8_channelwise_layers", []) or []

        # Detect full channelwise model → use int8 mapper
        is_full_channelwise = False
        quant_config = getattr(vllm_config, "quant_config", None)
        target_scheme_map = getattr(quant_config, "target_scheme_map", None)
        if target_scheme_map:
            for scheme_dict in target_scheme_map.values():
                weights_args = scheme_dict.get("weights")
                strategy = getattr(weights_args, "strategy", None)
                if strategy is not None and "channel" in str(strategy).lower():
                    is_full_channelwise = True
                    break

        # Select the mapper now, but install it on this instance only after the
        # NVIDIA initializer. Mutating ``self.__class__`` leaks one model's
        # quantization scheme into later instances in the same process.
        selected_mapper = None
        if is_full_channelwise:
            selected_mapper = _make_deepseek_v4_weights_mapper("int8")
        elif expert_dtype != "fp4" or fp8_channelwise_layers:
            selected_mapper = _make_deepseek_v4_weights_mapper(
                expert_dtype,
                fp8_channelwise_layers=(
                    fp8_channelwise_layers if expert_dtype == "fp4" else None
                ),
            )

        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if selected_mapper is not None:
            self.hf_to_vllm_mapper = selected_mapper

    def get_model_state_cls(self):
        from vllm.v1.worker.gpu.model_states.default import DefaultModelState

        if self.model.parallel_config.prefill_context_parallel_size <= 1:
            return DefaultModelState
        from vllm_sail.models.deepseek_v4.pcp import DeepseekV4PCPModelState

        return DeepseekV4PCPModelState
