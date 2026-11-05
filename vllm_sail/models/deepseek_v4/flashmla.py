# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PPU-specific deltas for vLLM's DeepSeek V4 FlashMLA implementation."""

from __future__ import annotations

import torch
from vllm.distributed import get_pcp_group
from vllm.models.deepseek_v4.nvidia.flashmla import (
    DeepseekSparseSWAFlashMLABackend,
    DeepseekSparseSWAFlashMLAMetadataBuilder,
)
from vllm.models.deepseek_v4.nvidia.flashmla import (
    DeepseekV4FlashMLAAttention as NvidiaDeepseekV4FlashMLAAttention,
)
from vllm.models.deepseek_v4.sparse_mla import (
    DeepseekV4FlashMLABackend,
    DeepseekV4FlashMLAMetadataBuilder,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backend import CommonAttentionMetadata

from vllm_sail.models.deepseek_v4.ops.o_proj import (
    compute_fp8_einsum_recipe,
    deep_gemm_fp8_channel_o_proj,
    deep_gemm_fp8_o_proj,
    deep_gemm_int8_o_proj,
)


def _gather_padded_compressed_slots(
    compressed_slot_mapping_buffer: torch.Tensor,
    local_slot_mapping: torch.Tensor,
) -> torch.Tensor:
    local_slots = compressed_slot_mapping_buffer[: local_slot_mapping.numel()]
    return get_pcp_group().all_gather(local_slots.contiguous(), dim=0)


class PPUDeepseekSparseSWAMetadataBuilder(DeepseekSparseSWAFlashMLAMetadataBuilder):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pcp_size = self.vllm_config.parallel_config.prefill_context_parallel_size
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        if self.pcp_size > 1:
            self.no_replay_start = torch.zeros(
                self.vllm_config.scheduler_config.max_num_seqs * 2 + 1,
                dtype=torch.int32,
                device=self.device,
            )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        replay_start: torch.Tensor | None = None,
        pcp_shared_metadata=None,
    ):
        if self.pcp_size <= 1:
            return super().build(
                common_prefix_len,
                common_attn_metadata,
                fast_build,
                replay_start,
            )
        global_slot_mapping = common_attn_metadata.slot_mapping
        local_slot_mapping = global_slot_mapping.chunk(self.pcp_size)[self.pcp_rank]
        local_metadata = common_attn_metadata.replace(slot_mapping=local_slot_mapping)
        metadata = super().build(
            common_prefix_len,
            local_metadata,
            fast_build,
            replay_start,
        )
        assert local_metadata.positions is not None
        metadata.cache_slot_mapping = global_slot_mapping
        metadata.cache_positions = (
            pcp_shared_metadata.get_cache_positions(local_metadata)
            if pcp_shared_metadata is not None
            else get_pcp_group().all_gather(
                local_metadata.positions.contiguous(), dim=0
            )
        )
        return metadata


class PPUDeepseekSparseSWAFlashMLABackend(DeepseekSparseSWAFlashMLABackend):
    @classmethod
    def supports_pcp(cls) -> bool:
        return True

    @staticmethod
    def get_builder_cls() -> type[PPUDeepseekSparseSWAMetadataBuilder]:
        return PPUDeepseekSparseSWAMetadataBuilder


class PPUDeepseekV4FlashMLAMetadataBuilder(DeepseekV4FlashMLAMetadataBuilder):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pcp_size = self.vllm_config.parallel_config.prefill_context_parallel_size
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ):
        if self.pcp_size <= 1:
            return super().build(
                common_prefix_len,
                common_attn_metadata,
                fast_build,
            )
        global_slot_mapping = common_attn_metadata.slot_mapping
        local_slot_mapping = global_slot_mapping.chunk(self.pcp_size)[self.pcp_rank]
        metadata = super().build(
            common_prefix_len,
            common_attn_metadata.replace(slot_mapping=local_slot_mapping),
            fast_build,
        )
        metadata.slot_mapping = _gather_padded_compressed_slots(
            self.compressed_slot_mapping_buffer,
            local_slot_mapping,
        )
        return metadata


class PPUDeepseekV4FlashMLABackend(DeepseekV4FlashMLABackend):
    @classmethod
    def supports_pcp(cls) -> bool:
        return True

    @staticmethod
    def get_builder_cls() -> type[PPUDeepseekV4FlashMLAMetadataBuilder]:
        return PPUDeepseekV4FlashMLAMetadataBuilder


class DeepseekV4FlashMLAAttention(NvidiaDeepseekV4FlashMLAAttention):
    """Reuse vLLM's FlashMLA path and override only PPU-specific behavior."""

    backend_cls = PPUDeepseekV4FlashMLABackend
    swa_backend_cls = PPUDeepseekSparseSWAFlashMLABackend

    def __init__(self, *args, **kwargs) -> None:
        vllm_config = kwargs.get("vllm_config", args[0] if args else None)
        use_pcp = (
            vllm_config is not None
            and vllm_config.parallel_config.prefill_context_parallel_size > 1
        )
        if use_pcp:
            if len(args) > 3:
                args = (*args[:3], None, *args[4:])
            else:
                kwargs["aux_stream_list"] = None
        super().__init__(*args, **kwargs)
        self.use_pcp = use_pcp
        self._einsum_recipe, self._tma_aligned_scales = compute_fp8_einsum_recipe()

    def _fused_qnorm_rope_kv_insert(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata,
    ) -> torch.Tensor:
        if not self.use_pcp or not isinstance(attn_metadata, dict):
            return super()._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)
        swa_metadata = attn_metadata[self.swa_cache_layer.prefix]
        cache_slot_mapping = getattr(swa_metadata, "cache_slot_mapping", None)
        cache_positions = getattr(swa_metadata, "cache_positions", None)
        assert cache_slot_mapping is not None
        assert cache_positions is not None
        swa_kv_cache = self.swa_cache_layer.kv_cache
        assert swa_kv_cache.dtype == torch.uint8
        gathered_kv = get_pcp_group().all_gather(kv.contiguous(), dim=0)
        swa_kv_cache_2d = swa_kv_cache.view(swa_kv_cache.shape[0], -1)
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            gathered_kv.unsqueeze(1).contiguous(),
            gathered_kv,
            swa_kv_cache_2d,
            cache_slot_mapping,
            cache_positions,
            self.rotary_emb.cos_sin_cache,
            0,
            self.eps,
            swa_metadata.block_size,
            False,
            False,
            False,
        )
        return torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            q,
            kv,
            swa_kv_cache_2d,
            swa_metadata.slot_mapping[:0],
            positions,
            self.rotary_emb.cos_sin_cache,
            self.padded_heads,
            self.eps,
            swa_metadata.block_size,
        )

    def _forward_decode(self, q: torch.Tensor, *args, **kwargs) -> None:
        if q.shape[0] == 0:
            return
        super()._forward_decode(q, *args, **kwargs)

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self.wo_a.weight.dtype == torch.int8:
            return deep_gemm_int8_o_proj(
                o,
                positions,
                self.rotary_emb.cos_sin_cache,
                self.wo_a,
                self.wo_b,
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
                self.wo_b,
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
            self.wo_b,
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
