# SPDX-License-Identifier: Apache-2.0
"""Retain ordinary MiniMax MoE finalize on PPU."""

from vllm.models.minimax_m3.nvidia.model import MiniMaxM3Model
from vllm.platforms import current_platform

from vllm_sail.patch.utils import patch

_original_init = MiniMaxM3Model._init_fused_moe_finalize


@patch(
    "vllm.models.minimax_m3.nvidia.model",
    "MiniMaxM3Model._init_fused_moe_finalize",
    reason="PPU inherits CUDA identity but does not implement NVIDIA's collective MoE finalize kernel or its dependencies.",
    affected_versions=">=0.32.0,<0.33.0",
    remove_when="MiniMax gates fused finalize through a platform implementation capability or PPU implements and verifies that collective kernel.",
)
def init_fused_moe_finalize(self, vllm_config):
    if current_platform.is_ppu():
        return None
    return _original_init(self, vllm_config)
