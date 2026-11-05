# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_sail.attention.ops.mla_sparse import (
    ppu_sparse_attn_indexer,
    ppu_sparse_attn_indexer_fake,
)
from vllm_sail.attention.ops.mla_sparse_kpool import (
    ppu_sparse_attn_indexer_kpool,
    ppu_sparse_attn_indexer_kpool_fake,
)

FP8_DTYPE = current_platform.fp8_dtype()

# Global flag to ensure ops are registered only once
_OPS_REGISTERED = False


class ppu_ops:
    @staticmethod
    def register_ops_once() -> None:
        global _OPS_REGISTERED
        if not _OPS_REGISTERED:
            # register all the custom ops here
            direct_register_custom_op(
                op_name="ppu_sparse_attn_indexer",
                op_func=ppu_sparse_attn_indexer,
                mutates_args=["topk_indices_buffer", "candidate_blocks"],
                fake_impl=ppu_sparse_attn_indexer_fake,
                dispatch_key=current_platform.dispatch_key,
            )
            direct_register_custom_op(
                op_name="ppu_sparse_attn_indexer_kpool",
                op_func=ppu_sparse_attn_indexer_kpool,
                mutates_args=[
                    "kv_cache",
                    "tail_kv_cache",
                    "topk_indices_buffer",
                ],
                fake_impl=ppu_sparse_attn_indexer_kpool_fake,
                dispatch_key=current_platform.dispatch_key,
            )

            _OPS_REGISTERED = True


ppu_ops.register_ops_once()
