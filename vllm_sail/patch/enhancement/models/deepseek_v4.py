# SPDX-License-Identifier: Apache-2.0
"""Select PPU FlashMLA and the PPU 1.0 INT8 sparse-indexer kernels."""

from __future__ import annotations

import sys

import torch
from vllm.distributed import get_pcp_group
from vllm.model_executor.layers.fused_moe.runner import moe_runner as _moe_runner
from vllm.models.deepseek_v4.nvidia.flashmla import (
    DeepseekV4FlashMLAAttention as NvidiaDeepseekV4FlashMLAAttention,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from vllm_sail.models.deepseek_v4.flashmla import (
    DeepseekV4FlashMLAAttention as PPUDeepseekV4FlashMLAAttention,
)
from vllm_sail.patch.utils import PATCH_MARKER, patch

_AFFECTED = ">=0.32.0,<0.33.0"

_SELECTOR_TARGET = "vllm.models.deepseek_v4.nvidia.model._select_dsv4_attn_cls"


@patch(
    "vllm.models.deepseek_v4.nvidia.model",
    "_select_dsv4_attn_cls",
    reason=(
        "The upstream DeepSeek V4 selector returns the NVIDIA FlashMLA class. "
        "On PPU, substitute the plugin implementation for its DeepGEMM output "
        "projection and unpadded head-count contract while preserving every "
        "non-PPU and non-FlashMLA selection."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "vLLM exposes a DeepSeek V4 attention-class registration hook, or its "
        "selector natively resolves the PPU plugin FlashMLA implementation."
    ),
)
def _select_dsv4_attn_cls(vllm_config):
    original = getattr(_select_dsv4_attn_cls, PATCH_MARKER)[_SELECTOR_TARGET]
    parallel_config = getattr(vllm_config, "parallel_config", None)
    use_pcp = (
        parallel_config is not None
        and parallel_config.prefill_context_parallel_size > 1
    )
    if current_platform.is_ppu() and use_pcp:
        backend = vllm_config.attention_config.backend
        if backend not in (
            None,
            AttentionBackendEnum.FLASHMLA_SPARSE,
            AttentionBackendEnum.FLASHMLA_SPARSE_DSV4,
        ):
            raise NotImplementedError(
                "DeepSeek V4 PCP on PPU requires FlashMLA sparse attention."
            )
        return PPUDeepseekV4FlashMLAAttention
    selected = original(vllm_config)
    if current_platform.is_ppu() and selected is NvidiaDeepseekV4FlashMLAAttention:
        return PPUDeepseekV4FlashMLAAttention
    return selected


_upstream_apply_quant_method = _moe_runner.MoERunner._apply_quant_method


@patch(
    "vllm.model_executor.layers.fused_moe.runner.moe_runner",
    "MoERunner._apply_quant_method",
    reason=(
        "PCP gathers routed activations and logits before expert selection; "
        "DeepSeek V4 hash routing needs input_ids in the same rank-major order."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream PR #65305 lands with PCP input_ids gathering in MoERunner.",
)
def _apply_quant_method(
    self,
    hidden_states,
    router_logits,
    shared_experts_input,
    input_ids=None,
    shared_experts_overlapping=False,
):
    if (
        current_platform.is_ppu()
        and self.moe_config.pcp_size > 1
        and not self.moe_config.moe_parallel_config.use_all2all_kernels
        and input_ids is not None
    ):
        input_ids = get_pcp_group().all_gather(input_ids.contiguous(), dim=0)
    return _upstream_apply_quant_method(
        self,
        hidden_states,
        router_logits,
        shared_experts_input,
        input_ids,
        shared_experts_overlapping,
    )


_Q_MODULE = "vllm.models.deepseek_v4.common.ops.fused_indexer_q"
_K_MODULE = "vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache"
_INDEXER_ALIAS_CONSUMERS = {
    "fused_indexer_q_rope_quant": (
        "vllm.models.deepseek_v4.common.ops",
        "vllm.models.deepseek_v4.attention",
    ),
    "compress_norm_rope_store_triton": ("vllm.models.deepseek_v4.compressor",),
}
_INDEXER_REMOVE_WHEN = (
    "vLLM exposes a DeepSeek V4 indexer quantization registration hook or "
    "natively supports paired INT8 query and compressed-key quantization on PPU 1.0."
)


@patch(
    _Q_MODULE,
    "fused_indexer_q_rope_quant",
    reason=(
        "PPU 1.0 Triton cannot compile upstream's fp8e4nv indexer query cast. "
        "Its DeepGEMM indexer consumes INT8 Q and K; route Q through the PPU "
        "INT8 quantizer with its scale folded into the logits weights. "
        "The directly called upstream function has no registration hook."
    ),
    affected_versions=_AFFECTED,
    remove_when=_INDEXER_REMOVE_WHEN,
)
def fused_indexer_q_rope_quant(
    positions,
    index_q,
    index_q_cos_sin_cache,
    index_weights,
    index_weights_softmax_scale,
    index_weights_head_scale,
    use_fp4=False,
    weights_out_dtype=torch.float32,
    *,
    output_buffers=None,
):
    original = getattr(fused_indexer_q_rope_quant, PATCH_MARKER)[
        f"{_Q_MODULE}.fused_indexer_q_rope_quant"
    ]
    if (
        current_platform.is_ppu()
        and current_platform.is_device_capability((8, 0))
        and not use_fp4
    ):
        assert weights_out_dtype == torch.float32, weights_out_dtype
        from vllm_sail.models.deepseek_v4.ops.indexer import (
            fused_indexer_q_rope_quant_int8,
        )

        return fused_indexer_q_rope_quant_int8(
            positions,
            index_q,
            index_q_cos_sin_cache,
            index_weights,
            index_weights_softmax_scale,
            index_weights_head_scale,
            output_buffers=output_buffers,
        )
    if output_buffers is not None:
        raise ValueError("output_buffers is supported only by the PPU 1.0 INT8 path")
    return original(
        positions,
        index_q,
        index_q_cos_sin_cache,
        index_weights,
        index_weights_softmax_scale,
        index_weights_head_scale,
        use_fp4=use_fp4,
        weights_out_dtype=weights_out_dtype,
    )


@patch(
    _K_MODULE,
    "compress_norm_rope_store_triton",
    reason=(
        "PPU 1.0 INT8 indexer queries must read INT8 compressed keys. "
        "Upstream writes FP8 bytes using an unsupported fp8e4nv cast; "
        "emit signed INT8 bytes and a float32 scale in the same paged layout. "
        "The compressor launcher has no backend registration hook."
    ),
    affected_versions=_AFFECTED,
    remove_when=_INDEXER_REMOVE_WHEN,
)
def compress_norm_rope_store_triton(
    state_cache,
    num_actual,
    token_to_req_indices,
    positions,
    slot_mapping,
    block_table,
    block_size,
    state_width,
    cos_sin_cache,
    kv_cache,
    k_cache_metadata,
    pdl_kwargs,
    head_dim,
    rope_head_dim,
    compress_ratio,
    overlap,
    use_fp4_cache,
    rms_norm_weight,
    rms_norm_eps,
    quant_block,
    token_stride,
    scale_dim,
):
    implementation = getattr(compress_norm_rope_store_triton, PATCH_MARKER)[
        f"{_K_MODULE}.compress_norm_rope_store_triton"
    ]
    if (
        current_platform.is_ppu()
        and current_platform.is_device_capability((8, 0))
        and head_dim == 128
        and not use_fp4_cache
    ):
        from vllm_sail.models.deepseek_v4.ops.indexer import (
            compress_indexer_rope_store_int8,
        )

        implementation = compress_indexer_rope_store_int8
    elif (
        current_platform.is_ppu()
        and current_platform.is_device_capability((8, 0))
        and head_dim == 512
    ):
        from vllm_sail.models.deepseek_v4.ops.cache import compress_mla_rope_store_fp8

        implementation = compress_mla_rope_store_fp8
    return implementation(
        state_cache,
        num_actual,
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        block_size,
        state_width,
        cos_sin_cache,
        kv_cache,
        k_cache_metadata,
        pdl_kwargs,
        head_dim,
        rope_head_dim,
        compress_ratio,
        overlap,
        use_fp4_cache,
        rms_norm_weight,
        rms_norm_eps,
        quant_block,
        token_stride,
        scale_dim,
    )


def _rebind_indexer_aliases():
    # Attention (also used by MTP) and compressor may load before this patch.
    # Later imports naturally receive the patched provider/re-export.
    for provider, replacement in (
        (_Q_MODULE, fused_indexer_q_rope_quant),
        (_K_MODULE, compress_norm_rope_store_triton),
    ):
        name = replacement.__name__
        original = getattr(replacement, PATCH_MARKER)[f"{provider}.{name}"]
        for consumer in _INDEXER_ALIAS_CONSUMERS[name]:
            module = sys.modules.get(consumer)
            if module is None or getattr(module, name, None) is not original:
                continue
            patch(
                consumer,
                name,
                reason=(
                    "This DeepSeek V4 consumer captured its indexer operation "
                    "before PPU installation and would bypass PPU 1.0 INT8 dispatch."
                ),
                affected_versions=_AFFECTED,
                remove_when=(
                    "vLLM resolves DeepSeek V4 indexer operations through module "
                    "lookups or a registry instead of importing them by value."
                ),
            )(replacement)


_rebind_indexer_aliases()
