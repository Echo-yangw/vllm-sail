# SPDX-License-Identifier: Apache-2.0
"""Dependency-free normalization for PPU V4.1 mixed-precision exports."""

import re


def normalize_expert_dtype(vllm_config):
    """Canonicalize the exporter alias in resolved configs, never checkpoint files."""
    configs = [getattr(vllm_config, "model_config", None)]
    speculative = getattr(vllm_config, "speculative_config", None)
    configs.append(getattr(speculative, "draft_model_config", None))
    for model in configs:
        for attr in ("hf_config", "hf_text_config"):
            hf = getattr(model, attr, None)
            if getattr(hf, "model_type", None) not in (
                "deepseek_v41",
                "deepseek_v41_text",
            ):
                continue
            quant = getattr(hf, "quantization_config", None) or {}
            if quant.get("quant_method") == "mxfp4" and quant.get(
                "fp8_channelwise_layers"
            ):
                if getattr(hf, "expert_dtype", None) == "mxfp4":
                    hf.expert_dtype = "fp4"


def checkpoint_layer_name(prefix: str, num_hidden_layers: int) -> str:
    """Reverse V4.1 runtime prefixes, including DSpark's absolute layer indices."""
    prefix = prefix.removeprefix("language_model.").removeprefix("model.")
    match = re.match(r"layers\.(\d+)\.(.*)", prefix)
    if match and int(match[1]) >= num_hidden_layers:
        prefix = f"mtp.{int(match[1]) - num_hidden_layers}.{match[2]}"
    if prefix == "main_proj":
        prefix = "mtp.0.main_proj"
    return prefix.replace(".shared_experts.down_proj", ".shared_experts.w2")
