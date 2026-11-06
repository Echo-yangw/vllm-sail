# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PPU-specific deltas for vLLM's DeepSeek V4 FlashMLA implementation."""

from __future__ import annotations

import torch
from vllm.models.deepseek_v4.nvidia.flashmla import (
    DeepseekV4FlashMLAAttention as NvidiaDeepseekV4FlashMLAAttention,
)
from vllm.platforms import current_platform

from vllm_sail.models.deepseek_v4.ops.o_proj import (
    compute_fp8_einsum_recipe,
    deep_gemm_fp8_channel_o_proj,
    deep_gemm_fp8_o_proj,
    deep_gemm_int8_o_proj,
)


class DeepseekV4FlashMLAAttention(NvidiaDeepseekV4FlashMLAAttention):
    """Reuse vLLM's FlashMLA path and override only PPU-specific behavior."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._einsum_recipe, self._tma_aligned_scales = compute_fp8_einsum_recipe()

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self.wo_a.weight.dtype == torch.int8:
            return deep_gemm_int8_o_proj(
                o,
                positions,
                self.rotary_emb.cos_sin_cache,
                self.wo_a,
                getattr(self, "_wo_b_proj", self.wo_b),
                n_groups=self.n_local_groups,
                heads_per_group=self.n_local_heads // self.n_local_groups,
                nope_dim=self.nope_head_dim,
                rope_dim=self.rope_head_dim,
                o_lora_rank=self.o_lora_rank,
                einsum_recipe=self._einsum_recipe,
            )
        if self.wo_a.weight.dtype == torch.float8_e4m3fn and hasattr(
            self.wo_a, "weight_scale"
        ):
            return deep_gemm_fp8_channel_o_proj(
                o,
                positions,
                self.rotary_emb.cos_sin_cache,
                self.wo_a,
                getattr(self, "_wo_b_proj", self.wo_b),
                n_groups=self.n_local_groups,
                heads_per_group=self.n_local_heads // self.n_local_groups,
                nope_dim=self.nope_head_dim,
                rope_dim=self.rope_head_dim,
                o_lora_rank=self.o_lora_rank,
                einsum_recipe=self._einsum_recipe,
            )
        return deep_gemm_fp8_o_proj(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.wo_a,
            getattr(self, "_wo_b_proj", self.wo_b),
            n_groups=self.n_local_groups,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
            o_lora_rank=self.o_lora_rank,
            einsum_recipe=self._einsum_recipe,
            tma_aligned_scales=self._tma_aligned_scales,
        )

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        if num_heads > 128:
            raise ValueError(
                f"DeepseekV4 FlashMLA does not support {num_heads} heads "
                "(FP8 decode kernel requires h_q in {64, 128})."
            )
        if current_platform.is_ppu():
            return num_heads
        return super().get_padded_num_q_heads(num_heads)
