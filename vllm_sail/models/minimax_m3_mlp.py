# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bind the upstream MLP constructor with a PPU import capability boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.models.minimax_m3.nvidia.model import (
        MergedColumnParallelLinear,
        MiniMaxM3MLP,
        PreTrainedConfig,
        QuantizationConfig,
        RowParallelLinear,
        SiluAndMulWithClamp,
        current_platform,
        envs,
    )


def init_mlp(
    self,
    config: PreTrainedConfig,
    intermediate_size: int,
    quant_config: QuantizationConfig | None = None,
    reduce_results: bool = True,
    disable_tp: bool = False,
    prefix: str = "",
) -> None:
    super(MiniMaxM3MLP, self).__init__()
    self.gate_up_proj = MergedColumnParallelLinear(
        config.hidden_size,
        [intermediate_size] * 2,
        bias=False,
        quant_config=quant_config,
        disable_tp=disable_tp,
        prefix=f"{prefix}.gate_up_proj",
    )
    self.down_proj = RowParallelLinear(
        intermediate_size,
        config.hidden_size,
        bias=False,
        quant_config=quant_config,
        reduce_results=reduce_results,
        disable_tp=disable_tp,
        prefix=f"{prefix}.down_proj",
    )
    if config.hidden_act != "swigluoai":
        raise ValueError(
            f"Unsupported activation: {config.hidden_act}. Only swigluoai is supported."
        )
    # gate * sigmoid(alpha * gate) * (up + beta), with both halves clamped.
    self.act_fn = SiluAndMulWithClamp(
        swiglu_limit=config.swiglu_limit,
        alpha=config.swiglu_alpha,
        beta=config.swiglu_beta,
    )
    # At or below the shared-experts stream threshold the persistent kernel
    # would contend with the routed experts on the aux stream.
    self.fused_gemm1_min_tokens: int | None = None
    if not current_platform.is_ppu():
        from vllm.models.minimax_m3.nvidia.ops import dense_gemm1_mxfp8

        if dense_gemm1_mxfp8.can_fuse_gemm1(
            self.gate_up_proj, self.down_proj, self.act_fn
        ):
            self.fused_gemm1_min_tokens = max(
                dense_gemm1_mxfp8.MIN_TOKENS,
                envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD,
            )
