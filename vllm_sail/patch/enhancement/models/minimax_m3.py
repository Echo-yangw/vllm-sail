# SPDX-License-Identifier: Apache-2.0
"""Retain ordinary MiniMax MoE finalize on PPU."""

from vllm.models.minimax_m3.nvidia import model as _model
from vllm.models.minimax_m3.nvidia.model import MiniMaxM3Model
from vllm.platforms import current_platform

from vllm_sail.models.minimax_m3_mlp import init_mlp as _mlp_body
from vllm_sail.patch.bodies import bind_body
from vllm_sail.patch.utils import patch

_original_init = MiniMaxM3Model._init_fused_moe_finalize
_original_mlp_init = _model.MiniMaxM3MLP.__init__
_ppu_mlp_init = bind_body(_mlp_body, _model)


@patch(
    "vllm.models.minimax_m3.nvidia.model",
    "MiniMaxM3MLP.__init__",
    reason="The dense MLP eagerly imports an SM100-only MXFP8 optimization before checking its capability. PPU keeps the original linear and activation path without importing NVIDIA kernels.",
    affected_versions=">=0.32.0,<0.33.0",
    remove_when="MiniMax dense MLP gates the NVIDIA-only optimization import through a platform capability, or PPU implements that optimization.",
)
def init_mlp(
    self,
    config,
    intermediate_size,
    quant_config=None,
    reduce_results=True,
    disable_tp=False,
    prefix="",
):
    function = _ppu_mlp_init if current_platform.is_ppu() else _original_mlp_init
    return function(
        self,
        config,
        intermediate_size,
        quant_config,
        reduce_results,
        disable_tp,
        prefix,
    )


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
