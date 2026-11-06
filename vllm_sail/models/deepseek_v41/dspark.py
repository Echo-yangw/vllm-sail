# SPDX-License-Identifier: Apache-2.0
"""Attach quant mappings to the upstream V4.1 DSpark model before construction."""

from vllm.model_executor.models.interfaces import SupportsQuant
from vllm.models.deepseek_v41.nvidia.dspark import (
    DSparkDeepseekV4ForCausalLM as UpstreamDSparkV41,
)
from vllm.models.deepseek_v41.nvidia.model import _make_deepseek_v4_weights_mapper


class DSparkDeepseekV41ForCausalLM(UpstreamDSparkV41, SupportsQuant):
    """Preserve V4.1 draft layers, weight loader and target-state sharing."""

    packed_modules_mapping = {
        "gate_up_proj": ["w1", "w3"],
        "fused_wqa_wkv": ["wq_a", "wkv"],
        "fused_wkv_wgate": ["wkv", "wgate"],
    }

    # SupportsQuant uses only the rename mapper for quant-config patterns.
    # The upstream V4.1 loader independently resolves the runtime scale suffix,
    # expert count and TP slicing from its draft config.
    hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper("fp4", "weight_scale")
