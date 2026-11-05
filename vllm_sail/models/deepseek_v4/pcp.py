# SPDX-License-Identifier: Apache-2.0
"""Share rank-local PCP layout metadata within one attention build."""

from __future__ import annotations

import torch
from vllm.config import CUDAGraphMode
from vllm.distributed import get_pcp_group
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata


class PCPSharedMetadata(ModelSpecificAttnMetadata):
    def __init__(self) -> None:
        self.cache_positions: torch.Tensor | None = None
        self.gather_num_reqs: int | None = None
        self.token_to_req_indices: torch.Tensor | None = None

    def get_extra_attn_kwargs(self, attn_metadata_builder, num_reqs) -> dict:
        from vllm.models.deepseek_v4.compressor import CompressorMetadataBuilder

        from vllm_sail.models.deepseek_v4.flashmla import (
            PPUDeepseekSparseSWAMetadataBuilder,
        )

        if isinstance(
            attn_metadata_builder,
            CompressorMetadataBuilder | PPUDeepseekSparseSWAMetadataBuilder,
        ):
            return {"pcp_shared_metadata": self}
        return {}

    def get_cache_positions(self, cm: CommonAttentionMetadata) -> torch.Tensor:
        if self.cache_positions is None:
            assert cm.positions is not None
            self.cache_positions = get_pcp_group().all_gather(
                cm.positions.contiguous(), dim=0
            )
        return self.cache_positions

    def get_gather_num_reqs(self, cm: CommonAttentionMetadata, device) -> int:
        if self.gather_num_reqs is None:
            local_num_reqs = torch.tensor(
                [cm.num_reqs], dtype=torch.int32, device=device
            )
            self.gather_num_reqs = max(
                1,
                int(get_pcp_group().all_gather(local_num_reqs, dim=0).max().item()),
            )
        return self.gather_num_reqs

    def get_token_to_req_indices(
        self,
        cm: CommonAttentionMetadata,
        local_tokens: int,
        token_to_req_indices: torch.Tensor,
        local_buffer: torch.Tensor,
        gathered_buffer: torch.Tensor,
    ) -> torch.Tensor:
        if self.token_to_req_indices is None:
            group = get_pcp_group()
            gather_num_reqs = self.get_gather_num_reqs(cm, local_buffer.device)
            local_req_indices = local_buffer[:local_tokens]
            local_req_indices.zero_()
            local_req_indices[: cm.num_actual_tokens].copy_(token_to_req_indices)
            local_req_indices += group.rank_in_group * gather_num_reqs
            gathered = gathered_buffer[: local_tokens * group.world_size]
            gathered.copy_(group.all_gather(local_req_indices, dim=0))
            self.token_to_req_indices = gathered
        return self.token_to_req_indices


class DeepseekV4PCPModelState(DefaultModelState):
    def prepare_attn(
        self,
        input_batch,
        cudagraph_mode,
        block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture=False,
        ubatch_idx=0,
        model_specific_attn_metadata=None,
    ):
        shared_metadata = model_specific_attn_metadata
        if (
            input_batch.pcp is not None
            and cudagraph_mode == CUDAGraphMode.NONE
            and not for_capture
        ):
            assert model_specific_attn_metadata is None
            # A fresh object prevents reuse after mutable input buffers advance.
            shared_metadata = PCPSharedMetadata()
        return super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture=for_capture,
            ubatch_idx=ubatch_idx,
            model_specific_attn_metadata=shared_metadata,
        )
