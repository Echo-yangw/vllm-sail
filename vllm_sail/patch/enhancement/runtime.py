# ruff: noqa: E731, W291, W293, UP037
# ruff: noqa: F821
# Copied bodies resolve globals in their upstream module through bind_body.
# SPDX-License-Identifier: Apache-2.0
"""PPU input-batch invalidation and race-free speculative sample insertion."""

from __future__ import annotations

import functools
import sys

from vllm.platforms import current_platform
from vllm.v1.worker import gpu_model_runner as _runner
from vllm.v1.worker.gpu.spec_decode import rejection_sampler_utils as _sampler

from vllm_sail.patch.bodies import bind_body
from vllm_sail.patch.utils import patch


def may_reinitialize_input_batch(self, kv_cache_config: KVCacheConfig) -> None:
    """Re-initialize the input batch if the block sizes are different from
    what it was originally created with. This happens when the final
    block size (determined after model loading) differs from the
    placeholder used during __init__, or when there are multiple
    KV cache groups.

    Args:
        kv_cache_config: The KV cache configuration.

    """
    block_sizes = []
    max_num_blocks = []
    slot_mapping_modes = []
    max_model_len = max(self.max_model_len, self.max_encoder_len)
    for kv_cache_group in kv_cache_config.kv_cache_groups:
        kv_cache_spec = kv_cache_group.kv_cache_spec
        kv_cache_spec_kind = get_kv_cache_spec_kind(kv_cache_spec)
        if kv_cache_spec_kind == KVCacheSpecKind.ENCODER_ONLY_ATTENTION:
            continue
        block_size = kv_cache_spec.block_size
        block_sizes.append(block_size)
        if kv_cache_spec_kind == KVCacheSpecKind.MAMBA:
            slot_mapping_modes.append(SlotMappingMode.NONE)
        else:
            slot_mapping_modes.append(SlotMappingMode.TOKEN_TO_KV_SLOT)
        max_num_blocks_per_req = kv_cache_spec.max_num_blocks_per_req(
            self.vllm_config, max_model_len
        )
        max_num_blocks.append(max_num_blocks_per_req)

    if (
        block_sizes != self._init_block_sizes
        or max_num_blocks != self._init_max_num_blocks
        or slot_mapping_modes != self._init_slot_mapping_modes
        or self.cp_kv_cache_interleave_size
        != self.parallel_config.cp_kv_cache_interleave_size
        # PPU MODIFICATION: begin
        or max_model_len != self.input_batch.max_model_len
        # PPU MODIFICATION: end
    ):
        self._init_block_sizes = block_sizes
        self._init_max_num_blocks = max_num_blocks
        self._init_slot_mapping_modes = slot_mapping_modes
        self.cp_kv_cache_interleave_size = (
            self.parallel_config.cp_kv_cache_interleave_size
        )
        # Capture warmup providers registered after final KV-cache geometry is known
        with self.jit_warmup_registry.activate():  # type: ignore[attr-defined]
            self.input_batch = InputBatch(
                max_num_reqs=self.max_num_reqs,
                max_model_len=max_model_len,
                max_num_batched_tokens=self.max_num_tokens,
                device=self.device,
                vocab_size=self.model_config.get_vocab_size(),
                block_sizes=block_sizes,
                max_num_blocks_per_req=max_num_blocks,
                num_spec_tokens=self.num_spec_tokens,
                logitsprocs=self.input_batch.logitsprocs,
                logitsprocs_need_output_token_ids=self.input_batch.logitsprocs_need_output_token_ids,
                is_pooling_model=self.is_pooling_model,
                cp_kv_cache_interleave_size=self.parallel_config.cp_kv_cache_interleave_size,
                reasoning_config=self.vllm_config.reasoning_config,
                use_replayssm=self.cache_config.use_replayssm,
                slot_mapping_modes=slot_mapping_modes,
            )

    assert self._init_block_sizes == block_sizes, (
        f"InputBatch block_sizes {self._init_block_sizes} != "
        f"kv_cache block_sizes {block_sizes}"
    )


def rejection_sample(
    # [num_logits, V]
    target_logits: torch.Tensor,
    # [max_num_reqs, num_speculative_steps, V], or this rank's rows under
    # batch-sharded sampling, indexed by req_state_idx // draft_logits_num_shards.
    draft_logits: torch.Tensor | None,
    # [num_logits]
    draft_sampled: torch.Tensor,
    # [num_reqs + 1]
    cu_num_logits: torch.Tensor,
    # [num_logits]
    pos: torch.Tensor,
    # [num_reqs]
    idx_mapping: torch.Tensor,
    # [num_logits]
    expanded_idx_mapping: torch.Tensor,
    # [num_logits]
    expanded_local_pos: torch.Tensor,
    # [max_num_reqs]
    temperature: torch.Tensor,
    # [max_num_reqs]
    seed: torch.Tensor,
    num_speculative_steps: int,
    # [num_speculative_steps]
    synthetic_conditional_rates: torch.Tensor | None = None,
    use_fp64: bool = False,
    use_block_verification: bool = False,
    contexts: torch.Tensor | None = None,
    watermarking: torch.Tensor | None = None,
    watermarking_skip_mask: torch.Tensor | None = None,
    watermark_key: int | None = None,
    fly_window_size: int = 0,
    fly_entropy_threshold: float = 0.3,
    fly_entropy_top_k: int = 3,
    draft_logits_num_shards: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert target_logits.ndim == 2 and target_logits.stride(-1) == 1
    assert draft_logits is None or (
        draft_logits.ndim == 3 and draft_logits.stride(-1) == 1
    )
    num_reqs = cu_num_logits.shape[0] - 1
    num_logits, vocab_size = target_logits.shape

    watermark = contexts is not None
    assert watermark == (watermarking is not None) == (watermark_key is not None), (
        "contexts, watermarking and watermark_key must be set together."
    )
    assert watermarking_skip_mask is None or watermark, (
        "watermarking_skip_mask requires watermarking."
    )
    contexts_stride = 0
    context_width = 1
    watermarking_bytes: torch.Tensor | None = None
    watermarking_skip_mask_bytes: torch.Tensor | None = None
    watermark_key_0 = 0
    watermark_key_1 = 0
    if contexts is not None:
        assert watermarking is not None and watermark_key is not None
        assert contexts.ndim == 2 and contexts.shape[0] == num_logits
        # Context words are hashed as uint32, so int32 (the request-state token
        # dtype) and int64 both land on the same PRF stream, -1 included.
        assert contexts.dtype in (torch.int32, torch.int64)
        if contexts.stride(-1) != 1:
            contexts = contexts.contiguous()
        assert watermarking.ndim == 1 and watermarking.dtype == torch.bool
        contexts_stride = contexts.stride(0)
        context_width = contexts.shape[-1]
        watermarking_bytes = watermarking.view(torch.uint8)
        if watermarking_skip_mask is not None:
            assert watermarking_skip_mask.shape == (num_logits,)
            assert watermarking_skip_mask.dtype == torch.bool
            if not watermarking_skip_mask.is_contiguous():
                watermarking_skip_mask = watermarking_skip_mask.contiguous()
            watermarking_skip_mask_bytes = watermarking_skip_mask.view(torch.uint8)
        watermark_key_0 = watermark_key & 0xFFFFFFFF
        watermark_key_1 = watermark_key >> 32
    draft_logits_stride_0 = 0
    draft_logits_stride_1 = 0
    if has_draft_logits := draft_logits is not None:
        draft_logits_stride_0 = draft_logits.stride(0)
        draft_logits_stride_1 = draft_logits.stride(1)
        # In some cases (e.g. MiMo v2.5 Pro + DFlash) the target model's
        # vocab size is larger than the draft's due to padding.
        vocab_size = min(vocab_size, draft_logits.size(-1))

    fly_entropy = None
    if fly_window_size > 0:
        assert synthetic_conditional_rates is None and not use_block_verification
        fly_entropy = compute_fly_entropy(
            target_logits[:, :vocab_size], fly_entropy_top_k, from_logits=True
        )

    # Compute the per-vocab-block logits stats, such as target argmax
    # (for greedy requests), and target max + softmax exponential
    # (for non-greedy requests).
    VOCAB_BLOCK_SIZE = 8192
    vocab_num_blocks = triton.cdiv(vocab_size, VOCAB_BLOCK_SIZE)
    padded_vocab_num_blocks = triton.next_power_of_2(vocab_num_blocks)
    target_local_argmax = target_logits.new_empty(
        num_logits, vocab_num_blocks, dtype=torch.int64
    )
    target_local_max = target_logits.new_empty(
        num_logits, vocab_num_blocks, dtype=torch.float32
    )
    target_local_sumexp = target_logits.new_empty(
        num_logits, vocab_num_blocks, dtype=torch.float32
    )
    draft_local_max = target_logits.new_empty(
        num_logits, vocab_num_blocks, dtype=torch.float32
    )
    draft_local_sumexp = target_logits.new_empty(
        num_logits, vocab_num_blocks, dtype=torch.float32
    )
    _compute_local_logits_stats_kernel[(num_logits, vocab_num_blocks)](
        target_local_argmax,
        target_local_argmax.stride(0),
        target_local_max,
        target_local_max.stride(0),
        target_local_sumexp,
        target_local_sumexp.stride(0),
        draft_local_max,
        draft_local_max.stride(0),
        draft_local_sumexp,
        draft_local_sumexp.stride(0),
        target_logits,
        target_logits.stride(0),
        draft_logits,
        draft_logits_stride_0,
        draft_logits_stride_1,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        vocab_size,
        num_logits,
        BLOCK_SIZE=VOCAB_BLOCK_SIZE,
        HAS_DRAFT_LOGITS=has_draft_logits,
        DRAFT_LOGITS_NUM_SHARDS=draft_logits_num_shards,
    )

    # Precompute the running joint ratio and residual mass for block
    # verification.
    if use_block_verification:
        assert synthetic_conditional_rates is None, (
            "Block verification is incompatible with synthetic acceptance rates."
        )

        # Compute the log of the running joint ratio, p_i.
        # cumulative_log_p[start + i] = log(p_{i+1}), the cumulative ratio after
        # the (i+1)-th draft token.
        cumulative_log_p = target_logits.new_empty(num_logits, dtype=torch.float32)
        _compute_cumulative_log_p_kernel[(num_reqs,)](
            cumulative_log_p,
            target_logits,
            target_logits.stride(0),
            target_local_max,
            target_local_max.stride(0),
            target_local_sumexp,
            target_local_sumexp.stride(0),
            draft_sampled,
            draft_logits,
            draft_logits_stride_0,
            draft_logits_stride_1,
            draft_local_max,
            draft_local_max.stride(0),
            draft_local_sumexp,
            draft_local_sumexp.stride(0),
            cu_num_logits,
            idx_mapping,
            temperature,
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
            HAS_DRAFT_LOGITS=has_draft_logits,
            DRAFT_LOGITS_NUM_SHARDS=draft_logits_num_shards,
            num_warps=1,
        )

        # Compute the per-vocab-block partials of the residual mass, later reduced
        # to the total by _compute_global_residual_mass. Only launched for full
        # draft logits distributions. One-hot drafts used a closed-form residual
        # mass instead.
        if has_draft_logits:
            local_residual_mass = target_logits.new_empty(
                num_logits, vocab_num_blocks, dtype=torch.float32
            )
            _compute_local_residual_mass_kernel[(num_logits, vocab_num_blocks)](
                local_residual_mass,
                local_residual_mass.stride(0),
                cumulative_log_p,
                target_logits,
                target_logits.stride(0),
                target_local_max,
                target_local_max.stride(0),
                target_local_sumexp,
                target_local_sumexp.stride(0),
                draft_logits,
                draft_logits_stride_0,
                draft_logits_stride_1,
                draft_local_max,
                draft_local_max.stride(0),
                draft_local_sumexp,
                draft_local_sumexp.stride(0),
                draft_sampled,
                expanded_idx_mapping,
                expanded_local_pos,
                temperature,
                vocab_size,
                num_logits,
                vocab_num_blocks,
                BLOCK_SIZE=VOCAB_BLOCK_SIZE,
                PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
                DRAFT_LOGITS_NUM_SHARDS=draft_logits_num_shards,
            )
        else:
            local_residual_mass = None
    else:
        cumulative_log_p = None
        local_residual_mass = None

    # Sample up until the first rejected/bonus token, and store
    # the step.
    sampled = draft_sampled.new_empty(
        num_reqs, num_speculative_steps + 1, dtype=torch.int64
    )
    num_sampled = sampled.new_empty(num_reqs, dtype=torch.int32)
    target_rejected_logsumexp = target_logits.new_empty(num_reqs, dtype=torch.float32)
    draft_rejected_logsumexp = target_logits.new_empty(num_reqs, dtype=torch.float32)
    _rejection_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        target_rejected_logsumexp,
        draft_rejected_logsumexp,
        target_logits,
        target_logits.stride(0),
        target_local_argmax,
        target_local_argmax.stride(0),
        target_local_max,
        target_local_max.stride(0),
        target_local_sumexp,
        target_local_sumexp.stride(0),
        draft_sampled,
        draft_logits,
        draft_logits_stride_0,
        draft_logits_stride_1,
        draft_local_max,
        draft_local_max.stride(0),
        draft_local_sumexp,
        draft_local_sumexp.stride(0),
        cu_num_logits,
        idx_mapping,
        temperature,
        seed,
        pos,
        synthetic_conditional_rates,
        cumulative_log_p,
        local_residual_mass,
        local_residual_mass.stride(0) if local_residual_mass is not None else 0,
        fly_entropy,
        fly_entropy_threshold,
        vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
        HAS_DRAFT_LOGITS=has_draft_logits,
        DRAFT_LOGITS_NUM_SHARDS=draft_logits_num_shards,
        SYNTHETIC_MODE=synthetic_conditional_rates is not None,
        USE_BLOCK_VERIFICATION=use_block_verification,
        FLY_WINDOW_SIZE=fly_window_size,
        num_warps=1,
    )

    # Resample the rejected/bonus tokens.
    RESAMPLE_BLOCK_SIZE = 1024
    resample_num_blocks = triton.cdiv(vocab_size, RESAMPLE_BLOCK_SIZE)
    padded_resample_num_blocks = triton.next_power_of_2(resample_num_blocks)
    resampled_local_argmax = target_logits.new_empty(
        num_reqs, resample_num_blocks, dtype=torch.int64
    )
    resampled_local_max = target_logits.new_empty(
        num_reqs,
        resample_num_blocks,
        dtype=torch.float64 if use_fp64 else torch.float32,
    )
    _resample_kernel[(num_reqs, resample_num_blocks)](
        resampled_local_argmax,
        resampled_local_argmax.stride(0),
        resampled_local_max,
        resampled_local_max.stride(0),
        target_logits,
        target_logits.stride(0),
        target_rejected_logsumexp,
        draft_logits,
        draft_logits_stride_0,
        draft_logits_stride_1,
        draft_rejected_logsumexp,
        num_sampled,
        cu_num_logits,
        expanded_idx_mapping,
        draft_sampled,
        temperature,
        seed,
        pos,
        cumulative_log_p,
        contexts,
        contexts_stride,
        watermarking_bytes,
        watermarking_skip_mask_bytes,
        watermark_key_0,
        watermark_key_1,
        vocab_size,
        BLOCK_SIZE=RESAMPLE_BLOCK_SIZE,
        HAS_DRAFT_LOGITS=has_draft_logits,
        DRAFT_LOGITS_NUM_SHARDS=draft_logits_num_shards,
        USE_FP64=use_fp64,
        USE_BLOCK_VERIFICATION=use_block_verification,
        CONTEXT_WIDTH=context_width,
        WATERMARK=watermark,
        DEDUPLICATE_CONTEXTS=watermarking_skip_mask is not None,
    )

    # Insert the resampled tokens into the output sampled.
    # PPU MODIFICATION: begin
    # NOTE: num_warps=1 keeps the read-modify-write of num_sampled within a
    # single warp, so no thread can observe another warp's increment.
    # PPU MODIFICATION: end
    _insert_resampled_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        resampled_local_argmax,
        resampled_local_argmax.stride(0),
        resampled_local_max,
        resampled_local_max.stride(0),
        resample_num_blocks,
        cu_num_logits,
        expanded_idx_mapping,
        temperature,
        PADDED_RESAMPLE_NUM_BLOCKS=padded_resample_num_blocks,
        # PPU MODIFICATION: begin
        num_warps=1,
        # PPU MODIFICATION: end
    )
    return sampled, num_sampled


def _install(module, owner, attribute, body):
    original = getattr(owner, attribute)
    replacement = bind_body(body, module)

    @functools.wraps(original)
    def dispatch(*args, **kwargs):
        if current_platform.is_ppu():
            return replacement(*args, **kwargs)
        return original(*args, **kwargs)

    target = attribute if owner is module else owner.__name__ + "." + attribute
    metadata = dict(
        reason="PPU must reinitialize batches on context-length changes and insert speculative tokens within one warp.",
        affected_versions=">=0.32.0,<0.33.0",
        remove_when="Upstream includes the context-length invalidation and one-warp insertion fixes.",
    )
    patch(module.__name__, target, **metadata)(dispatch)
    if owner is module:
        for name in _CONSUMERS:
            consumer = sys.modules.get(name)
            if consumer is not None and getattr(consumer, attribute, None) is original:
                patch(name, attribute, **metadata)(dispatch)


_CONSUMERS = [
    "vllm.v1.worker.gpu.spec_decode.rejection_sampler",
    "vllm.v1.watermarking.spec_decode",
]
_install(
    _runner,
    _runner.GPUModelRunner,
    "may_reinitialize_input_batch",
    may_reinitialize_input_batch,
)
_install(_sampler, _sampler, "rejection_sample", rejection_sample)
