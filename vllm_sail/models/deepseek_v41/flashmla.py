# SPDX-License-Identifier: Apache-2.0
"""Keep V4.1 cache sharing and override only the PPU projection contract."""

from types import SimpleNamespace

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    dequant_mxfp8_to_bf16,
)
from vllm.models.deepseek_v4.nvidia.ops.o_proj import deep_gemm_fp8_o_proj
from vllm.models.deepseek_v41.nvidia.flashmla import (
    DeepseekV4FlashMLAAttention as UpstreamFlashMLAAttention,
)
from vllm.platforms import current_platform

from vllm_sail.models.deepseek_v4.flashmla import (
    DeepseekV4FlashMLAAttention as PPUV4FlashMLAAttention,
)


class DeepseekV41FlashMLAAttention(UpstreamFlashMLAAttention):
    """Upstream V4.1 metadata/forward, PPU head count and dense fallback."""

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        if current_platform.is_ppu():
            return PPUV4FlashMLAAttention.get_padded_num_q_heads(num_heads)
        return super().get_padded_num_q_heads(num_heads)

    def _o_proj(self, o, positions):
        if not current_platform.is_ppu():
            return super()._o_proj(o, positions)
        if self._o_proj_block_size != 32 and self.wo_a.weight.element_size() != 2:
            return PPUV4FlashMLAAttention._o_proj(self, o, positions)

        # Upstream ModelOpt expands the checkpoint's 32x32 scales to 1x32.
        # Its BMM selector falls back to EmulationMxfp8LinearKernel on PPU.
        # Usually that already dequantized at load time. Also honor upstream's
        # opt-out by dequantizing here, without mutating the module in forward.
        weight = self.wo_a.weight
        if weight.element_size() == 1:
            weight = dequant_mxfp8_to_bf16(weight, self.wo_a.weight_scale)
        # The upstream helper uses BF16 inverse RoPE + grouped BMM when the
        # weight is BF16, preserving V4.1's output layout without CUDA DeepGEMM.
        return deep_gemm_fp8_o_proj(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            SimpleNamespace(weight=weight),
            getattr(self, "_wo_b_proj", self.wo_b),
            n_groups=self.n_local_groups,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
            o_lora_rank=self.o_lora_rank,
            einsum_recipe=(1, 1, self._o_proj_block_size or 128),
            tma_aligned_scales=False,
        )
