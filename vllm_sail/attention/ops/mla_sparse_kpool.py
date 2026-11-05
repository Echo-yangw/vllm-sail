# SPDX-License-Identifier: Apache-2.0
"""PPU INT8 implementation of the glm5_next kpool sparse-attention indexer.

The live model is ``vllm.models.glm5next`` (registry entry
``Glm5NextForConditionalGeneration``): ``common.model`` -> ``common.attention``
-> ``glm5next/sparse_indexer.py`` selects ``nvidia/sparse_indexer.py`` on every
non-ROCm platform, including PPU. That module function writes fp8 pools with
CUDA Triton kernels and reads them back through CUDA DeepGEMM's fp8/fp4
MQA-logits kernels. PPU 1.0 cannot compile the fp8 Triton stores, and PPU
DeepGEMM's indexer consumes INT8 Q/K, so this module mirrors the same data flow
on the INT8 stack already used by the DSA indexer
(``vllm_sail.attention.ops.mla_sparse``):

* a pool cache entry keeps the exact CUDA layout -- 128 values followed by a
  4-byte fp32 scale per entry, value plane then scale plane per page -- so the
  existing INT8 scatter, ``cp_gather_indexer_k_quant_cache`` and paged MQA
  logits kernels are reused unchanged;
* prefill pool compression and the paged tail update are plain PyTorch
  replicas of the CUDA Triton kernels' semantics (correctness first; see the
  ``TRITON-FAST-PATH`` markers for the later optimisation);
* decode logits go straight through the paged ``int8_paged_mqa_logits`` kernel
  over the pool cache; there is no gathered-K fallback. An earlier report that
  the kernel "writes only 2 of every 64 logits" was a probe bug: the probe
  built each page as per-token ``[128B K | 4B scale]`` interleaved records,
  while the kernel contract is page-planar (a ``64 x 128B`` K plane followed by
  64 ``fp32`` scales). With the planar layout every column is written, and the
  kernel is bit-identical to the non-paged INT8 kernel used by prefill;
* Q is rotated with the same Hadamard-128 matrix as the pooled K and
  quantized with ``per_token_group_quant_int8``; the model folds its scale into
  ``weights`` before calling the op;
* the variant's ``get_indexer_topk`` dispatcher is bypassed in favour of the
  ``torch.ops._C`` top-k kernels that the DSA INT8 path already exercises on
  PPU.

Only the shared helpers from ``vllm.models.glm5next.common.sparse_indexer`` are
imported, never the same-named CustomOp in
``vllm.model_executor.layers.sparse_attn_indexer_kpool``: that module registers
the identical name and is not on the live path.
"""

# No ``from __future__ import annotations``: the registered op is wrapped by
# ``@eager_break_during_capture`` and torch's schema inference evaluates
# string annotations in the *wrapper's* module globals, where ``LayerNameType``
# does not exist (EngineCore fails with NameError at registration).
import torch
import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import get_current_vllm_config_or_none
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.int8_utils import (
    per_token_group_quant_int8,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import (
    LayerNameType,
    _resolve_layer_name,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.worker.workspace import current_workspace_manager

from vllm_sail.attention.ops.mla_sparse import (
    _scatter_indexer_int8_cache,
    indexer_k_quant_and_cache_int8,
)
from vllm_sail.models.glm5_next.ops.kpool_quant import rotate128

if current_platform.is_cuda_alike():
    from vllm import _custom_ops as ops

if current_platform.is_ppu():
    from vllm_sail.utils.deep_gemm import (
        int8_mqa_logits,
        int8_paged_mqa_logits,
    )

logger = init_logger(__name__)

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024


def _shared():
    """Helpers shared with the live glm5next indexer (no CustomOp side effects)."""
    from vllm.models.glm5next.common import sparse_indexer as shared

    return shared


def _kpool_ops():
    from vllm.models.glm5next.nvidia.ops import kpool_compress

    return kpool_compress


def _gather_workspace_shapes(
    total_seq_lens: int, head_dim: int
) -> tuple[tuple[tuple[int, int], torch.dtype], tuple[tuple[int, int], torch.dtype]]:
    """INT8 K-gather workspace: ``(T, head_dim)`` values + ``(T, 4)`` scales.

    Pool entries are stored as int8 values with one fp32 scale per head
    vector, matching the CUDA kpool cache layout.
    """
    return (
        ((total_seq_lens, head_dim), torch.int8),
        ((total_seq_lens, 4), torch.uint8),
    )


def _scatter_int8_cache(
    kv_cache: torch.Tensor,
    values: torch.Tensor,
    scales: torch.Tensor,
    slots: torch.Tensor,
) -> None:
    """Write ``(values, scales)`` at pool-granular ``slots``.

    Reuses the DSA INT8 scatter so the on-device layout matches
    ``indexer_k_quant_and_cache_int8`` byte for byte.
    """
    if values.shape[0] == 0:
        return
    assert values.shape[1] == 128
    _scatter_indexer_int8_cache[(values.shape[0],)](
        values.view(torch.uint8),
        scales.view(torch.uint8),
        kv_cache,
        slots,
        kv_cache.shape[1],
        kv_cache.stride(0),
        kv_cache.shape[0],
        num_warps=4,
    )


@triton.jit
def _stash_tail_kernel(
    tail_ptr,
    blk_ptr,
    phys_ptr,
    key_ptr,
    gate_ptr,
    valid_ptr,
    key_stride_b,
    key_off,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    RING: tl.constexpr,
    TAIL_STRIDE_B: tl.constexpr,
):
    """Masked tail-ring stash: one program per token, no host sync.

    ``valid == 0`` rows (padding / no tail slot) are skipped entirely, so this
    is capture-safe and can be used inside CUDA graph capture. ``key_ptr`` may
    be a ``[B, N, HEAD_DIM]`` tensor: ``key_stride_b``/``key_off`` select the
    token row so no per-token copy is needed.
    """
    row = tl.program_id(0)
    if tl.load(valid_ptr + row) != 0:
        offs = tl.arange(0, BLOCK_D)
        mask = offs < HEAD_DIM
        blk = tl.load(blk_ptr + row).to(tl.int64)
        phys = tl.load(phys_ptr + row).to(tl.int64)
        key = tl.load(
            key_ptr + row * key_stride_b + key_off + offs, mask=mask, other=0.0
        )
        gate = tl.load(
            gate_ptr + row * key_stride_b + key_off + offs, mask=mask, other=0.0
        )
        base = blk * TAIL_STRIDE_B + phys * HEAD_DIM
        tl.store(tail_ptr + base + offs, key, mask=mask)
        tl.store(tail_ptr + base + RING * HEAD_DIM + offs, gate, mask=mask)


def _stash_tail(
    tail_kv_cache: torch.Tensor,
    blk: torch.Tensor,
    phys: torch.Tensor,
    key: torch.Tensor,
    gate: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    """Write ``key``/``gate`` into each row's tail ring slot if ``valid``.

    ``key``/``gate`` may be ``[rows, head_dim]`` (prefill seed) or a token
    slice of ``[rows, next_n, head_dim]`` (decode); the caller passes the
    matching ``view(-1)`` via :func:`_stash_tail_t`.
    """
    rows = key.shape[0]
    if rows == 0:
        return
    _stash_tail_kernel[(rows,)](
        tail_kv_cache,
        blk,
        phys,
        key,
        gate,
        valid.to(torch.int8),
        key.stride(0),
        0,
        HEAD_DIM=key.shape[-1],
        BLOCK_D=triton.next_power_of_2(key.shape[-1]),
        RING=tail_kv_cache.shape[2],
        TAIL_STRIDE_B=2 * tail_kv_cache.shape[2] * key.shape[-1],
        num_warps=4,
    )


def _stash_tail_t(
    tail_kv_cache: torch.Tensor,
    blk: torch.Tensor,
    phys: torch.Tensor,
    key: torch.Tensor,
    gate: torch.Tensor,
    valid: torch.Tensor,
    step: int,
) -> None:
    """Decode variant: stash token ``step`` of ``[B, next_n, head_dim]``."""
    rows, next_n, head_dim = key.shape
    if rows == 0:
        return
    _stash_tail_kernel[(rows,)](
        tail_kv_cache,
        blk,
        phys,
        key,
        gate,
        valid.to(torch.int8),
        key.stride(0),
        step * key.stride(1),
        HEAD_DIM=head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        RING=tail_kv_cache.shape[2],
        TAIL_STRIDE_B=2 * tail_kv_cache.shape[2] * head_dim,
        num_warps=4,
    )


def _pool_weighted(
    k_win: torch.Tensor,
    gate_win: torch.Tensor,
    ape: torch.Tensor,
    kpool: int,
) -> torch.Tensor:
    """softmax(gate + ape)-weighted pool sum, mirroring the CUDA kernel.

    ``k_win``/``gate_win`` are ``[rows, kpool, head_dim]`` bf16, ``ape`` is
    ``[kpool, head_dim]`` fp32. The reduction runs in fp32 and the result is
    rounded to bf16 before the Hadamard/quantize stage, exactly like
    ``_kpool_softmax_rotate_write_cache_kernel``.
    """
    assert k_win.shape == gate_win.shape and k_win.shape[1] == kpool
    score = gate_win.to(torch.float32) + ape.to(torch.float32).unsqueeze(0)
    prob = torch.exp(score - score.max(dim=1, keepdim=True).values)
    denom = prob.sum(dim=1)
    acc = (k_win.to(torch.float32) * prob).sum(dim=1)
    return (acc / denom).to(torch.bfloat16)


def _kpool_compress_insert_int8(
    k: torch.Tensor,
    gate_score: torch.Tensor,
    ape: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kpool: int,
    head_dim: int,
) -> None:
    """Pool ``kpool`` consecutive prefill tokens into one INT8 cache entry.

    ``slot_mapping`` is pool-granular: only the last token of a complete pool
    carries a valid slot. Mirrors ``_kpool_compress_insert`` +
    ``kpool_compress_and_write_cache`` from the CUDA path.
    """
    n = slot_mapping.shape[0]
    if n < kpool:
        return
    pos = torch.arange(n, device=k.device)
    write_mask = (slot_mapping >= 0) & (pos >= kpool - 1)
    offs = torch.arange(kpool, device=k.device)
    window = (pos - (kpool - 1)).clamp_min(0)[:, None] + offs[None, :]
    # Pool every row (static shapes, capture-safe) and let the INT8 scatter
    # drop rows whose pool-granular slot is invalid.
    pooled = _pool_weighted(k[window], gate_score[window], ape, kpool)
    values, scales = per_token_group_quant_int8(rotate128(pooled), head_dim)
    slots = torch.where(
        write_mask,
        slot_mapping.to(torch.int64),
        torch.full_like(slot_mapping, -1, dtype=torch.int64),
    )
    _scatter_int8_cache(kv_cache, values, scales, slots)


def _kpool_seed_tail(
    tail_kv_cache: torch.Tensor,
    key: torch.Tensor,
    gate_score: torch.Tensor,
    tslot: torch.Tensor,
    kpool: int,
    head_dim: int,
) -> None:
    """Seed each request's trailing incomplete pool into its tail block.

    Mirrors ``_kpool_tail_seed_kernel``: token ``i`` is among its request's
    last ``kpool`` tokens iff the token ``kpool`` ahead belongs to a different
    tail block (or is past the batch / padded).
    """
    n = tslot.shape[0]
    if n == 0:
        return
    ring = tail_kv_cache.shape[2]
    blk = torch.div(tslot.clamp_min(0), ring, rounding_mode="floor")
    ahead = torch.full_like(tslot, -1)
    if n > kpool:
        ahead[: n - kpool] = tslot[kpool:]
    same_block = (ahead >= 0) & (
        torch.div(ahead.clamp_min(0), ring, rounding_mode="floor") == blk
    )
    valid = (tslot >= 0) & ~same_block
    _stash_tail(
        tail_kv_cache,
        blk,
        tslot.clamp_min(0) % ring,
        key,
        gate_score,
        valid,
    )


def _decode_update_torch(
    kv_cache: torch.Tensor,
    tail_kv_cache: torch.Tensor,
    tail_slot: torch.Tensor,
    key: torch.Tensor,
    gate_score: torch.Tensor,
    ape: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    kpool: int,
    head_dim: int,
) -> None:
    """Per-request tail stash + completed-pool compression for one decode step.

    Inputs are grouped per request (``[B, next_n, ...]``). Tokens are visited
    in position order and each token's completion read precedes its stash,
    which is what lets spec-decode tokens build a pool from earlier tokens of
    the same step. Mirrors ``_kpool_decode_update_batched_kernel``.
    """
    ring = tail_kv_cache.shape[2]
    num_requests, next_n = key.shape[0], key.shape[1]
    if num_requests == 0 or next_n == 0:
        return
    offs = torch.arange(kpool, device=key.device)
    neg_slot = torch.full_like(slot_mapping, -1, dtype=torch.int64)
    values: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []
    slots: list[torch.Tensor] = []

    # TRITON-FAST-PATH: replace with the CUDA batched kernel once PPU Triton
    # supports it; this loop is the semantic reference. Every step is fully
    # tensorised (no .tolist()/.item()), so it is valid inside CUDA graph
    # capture; ``next_n`` is a Python int (number of spec tokens).
    for t in range(next_n):
        pos = positions[:, t].to(torch.int64)
        tslot = tail_slot[:, t].to(torch.int64)
        valid = (pos >= 0) & (tslot >= 0)
        safe_pos = pos.clamp_min(0)
        blk = tslot.clamp_min(0) // ring
        phys_slot = safe_pos % ring
        slot_in_pool = safe_pos % kpool
        pool_start = safe_pos - slot_in_pool
        complete = valid & (slot_in_pool == kpool - 1)
        cache_loc = slot_mapping[:, t].to(torch.int64)
        write = complete & (cache_loc >= 0)

        # Ring gather for the whole batch; invalid rows read clamped indices
        # and are dropped at the scatter below.
        phys = (pool_start[:, None] + offs[None, :]) % ring
        k_buf = tail_kv_cache[blk[:, None], 0, phys]  # [B, kpool, D]
        g_buf = tail_kv_cache[blk[:, None], 1, phys]
        current = slot_in_pool[:, None] == offs[None, :]
        k_win = torch.where(current[:, :, None], key[:, t, :].unsqueeze(1), k_buf)
        g_win = torch.where(
            current[:, :, None], gate_score[:, t, :].unsqueeze(1), g_buf
        )
        pooled = _pool_weighted(k_win, g_win, ape, kpool)
        pool_values, pool_scales = per_token_group_quant_int8(
            rotate128(pooled), head_dim
        )
        values.append(pool_values)
        scales.append(pool_scales)
        slots.append(torch.where(write, cache_loc, neg_slot[:, t]))

        # Completion read first, stash second (CUDA kernel order): later
        # tokens of this step must see earlier stashes through the ring.
        _stash_tail_t(
            tail_kv_cache,
            blk,
            phys_slot,
            key,
            gate_score,
            valid,
            t,
        )

    _scatter_int8_cache(
        kv_cache,
        torch.cat(values, dim=0),
        torch.cat(scales, dim=0),
        torch.cat(slots, dim=0),
    )


def _kpool_decode_update(
    kv_cache: torch.Tensor,
    tail_kv_cache: torch.Tensor,
    attn_metadata: dict,
    decode_metadata,
    k: torch.Tensor,
    gate_score: torch.Tensor,
    compress_ape: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    num_decode_tokens: int,
    num_requests: int,
    kpool: int,
    head_dim: int,
    tail_prefix: str,
    shared,
) -> None:
    """Group the flattened decode batch per request and run the tail update."""
    tail_meta = attn_metadata.get(_resolve_layer_name(tail_prefix))
    if tail_meta is None or tail_kv_cache is None:
        return
    assert isinstance(tail_meta, DeepseekV32IndexerMetadata)
    per_req_lens = decode_metadata.per_req_decode_lens
    if per_req_lens is not None:
        use_uniform = (
            decode_metadata.decode_is_uniform
            and num_decode_tokens == num_requests * decode_metadata.write_max_decode_len
        )
        group_lens = per_req_lens
        lmax = decode_metadata.write_max_decode_len
    else:
        # Legacy metadata without per-request lens. Unreached with the current
        # builder, kept defensive like the CUDA path.
        use_uniform = not decode_metadata.requires_padding
        group_lens = decode_metadata.decode_lens
        lmax = int(decode_metadata.decode_lens.max().item())

    if not use_uniform:
        scatter_idx = shared._build_decode_scatter_indices(
            group_lens, num_requests, num_decode_tokens
        )
        dec_k = shared._scatter_decode_tokens_by_request(
            k[:num_decode_tokens], 0, num_requests, lmax, scatter_idx
        )
        dec_gate = shared._scatter_decode_tokens_by_request(
            gate_score[:num_decode_tokens], 0, num_requests, lmax, scatter_idx
        )
        dec_slot = shared._scatter_decode_tokens_by_request(
            slot_mapping[:num_decode_tokens], -1, num_requests, lmax, scatter_idx
        )
        dec_pos = shared._scatter_decode_tokens_by_request(
            positions[:num_decode_tokens].to(torch.int32),
            -1,
            num_requests,
            lmax,
            scatter_idx,
        )
        dec_tail = shared._scatter_decode_tokens_by_request(
            tail_meta.slot_mapping[:num_decode_tokens],
            -1,
            num_requests,
            lmax,
            scatter_idx,
        )
    else:
        next_n = num_decode_tokens // num_requests
        shape = (num_requests, next_n)
        dec_k = k[:num_decode_tokens].view(*shape, head_dim)
        dec_gate = gate_score[:num_decode_tokens].view(*shape, head_dim)
        dec_slot = slot_mapping[:num_decode_tokens].view(shape)
        dec_pos = positions[:num_decode_tokens].to(torch.int32).view(shape)
        dec_tail = tail_meta.slot_mapping[:num_decode_tokens].view(shape)

    _decode_update_torch(
        kv_cache,
        tail_kv_cache,
        dec_tail,
        dec_k,
        dec_gate,
        compress_ape,
        dec_slot,
        dec_pos,
        kpool,
        head_dim,
    )


def _expand_pools(
    pool_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    kpool: int,
    topk_tokens: int,
) -> torch.Tensor:
    """Expand selected pool ids to token ids and append the incomplete tail.

    Two-step torch equivalent of ``expand_pools_and_append_tail``'s identity
    path, both already provided by the model's kpool module.
    """
    kpool_ops = _kpool_ops()
    expanded = kpool_ops.expand_pools_to_tokens(
        pool_ids, pool_ids >= 0, topk_tokens, kpool
    )
    return kpool_ops.append_tail_to_topk(expanded, seq_lens, seq_lens // kpool, kpool)


def _decode_logits_paged(
    kv_cache_view: torch.Tensor,
    padded_q: torch.Tensor,
    padded_weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_pool_len: int,
) -> torch.Tensor:
    """Decode logits straight from the paged pool cache.

    This is the only decode-logits path: no K gather, no dense workspace.
    """
    return int8_paged_mqa_logits(
        padded_q,
        kv_cache_view,
        padded_weights,
        seq_lens,
        block_table,
        schedule_metadata,
        max_model_len=max_pool_len,
        clean_logits=False,
    )


@eager_break_during_capture
def ppu_sparse_attn_indexer_kpool(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_pool_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
    gate_score: torch.Tensor | None = None,
    compress_ape: torch.Tensor | None = None,
    index_kpool: int = 1,
    positions: torch.Tensor | None = None,
    tail_kv_cache: torch.Tensor | None = None,
    tail_prefix: str | None = None,
) -> torch.Tensor:
    """PPU INT8 port of ``sparse_attn_indexer_kpool``.

    ``max_pool_len`` is the pool-granular logits width the live model already
    computes (``max_model_len // index_kpool``); pool sequence lengths never
    exceed it, including the floor-divided tail pool.
    """
    shared = _shared()
    attn_metadata = get_forward_context().attn_metadata
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    if use_fp4_cache:
        raise NotImplementedError(
            "PPU kpool indexer does not support the MXFP4 cache path"
        )
    if q_quant.dtype != torch.int8:
        raise RuntimeError(
            "PPU kpool indexer expects INT8 q_quant (the PPU q-quant redirect "
            f"produces it); got {q_quant.dtype}"
        )
    if q_scale is not None:
        raise RuntimeError(
            "PPU kpool indexer folds the query scale into weights; q_scale must be None"
        )

    if not isinstance(attn_metadata, dict):
        # Dummy/profiling run: reserve the same workspace shapes the real path
        # uses, then return without touching any cache.
        values_spec, scales_spec = _gather_workspace_shapes(total_seq_lens, head_dim)
        current_workspace_manager().get_simultaneous(
            values_spec,
            scales_spec,
            ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
        )
        cfg = get_current_vllm_config_or_none()
        worst_decode_tokens = 0
        if cfg is not None:
            sched = cfg.scheduler_config
            num_spec = (
                cfg.speculative_config.num_speculative_tokens
                if cfg.speculative_config is not None
                else 0
            )
            worst_decode_tokens = min(
                sched.max_num_seqs * (num_spec + 1),
                sched.max_num_batched_tokens,
            )
        # Pool-granular width: 4B per logit.
        decode_logits_elems = worst_decode_tokens * max_pool_len * 4
        prefill_cap_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        _ = torch.empty(
            max(decode_logits_elems, prefill_cap_elems),
            dtype=torch.uint8,
            device=hidden_states.device,
        )
        return ppu_sparse_attn_indexer_kpool_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_quant,
            q_scale,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_pool_len,
            total_seq_lens,
            topk_indices_buffer,
            skip_k_cache_insert,
            use_fp4_cache,
            gate_score,
            compress_ape,
            index_kpool,
            positions,
            tail_kv_cache,
            tail_prefix,
        )

    meta = attn_metadata[k_cache_prefix]
    assert isinstance(meta, DeepseekV32IndexerMetadata)
    slot_mapping = meta.slot_mapping
    num_decode_tokens = meta.num_decode_tokens
    num_tokens = slot_mapping.shape[0]
    if k is not None:
        k = k[:num_tokens]

    if not skip_k_cache_insert:
        if index_kpool > 1 and gate_score is not None and compress_ape is not None:
            # Decode tokens (batched first) cannot be pooled in this batch;
            # they are handled by the tail update in the decode branch.
            n_prefill = num_tokens - num_decode_tokens
            if n_prefill > 0:
                prefill_slice = slice(num_decode_tokens, num_tokens)
                _kpool_compress_insert_int8(
                    k[prefill_slice],
                    gate_score[prefill_slice],
                    compress_ape,
                    kv_cache,
                    slot_mapping[prefill_slice],
                    index_kpool,
                    head_dim,
                )
                if tail_kv_cache is not None and tail_prefix is not None:
                    tail_meta = attn_metadata.get(_resolve_layer_name(tail_prefix))
                    if tail_meta is not None:
                        assert isinstance(tail_meta, DeepseekV32IndexerMetadata)
                        _kpool_seed_tail(
                            tail_kv_cache,
                            k[prefill_slice],
                            gate_score[prefill_slice],
                            tail_meta.slot_mapping[prefill_slice],
                            index_kpool,
                            head_dim,
                        )
        else:
            # Standard per-token insert (no pooling).
            assert scale_fmt is not None
            indexer_k_quant_and_cache_int8(k, kv_cache, slot_mapping, quant_block_size)

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if meta.num_prefills > 0:
        prefill_metadata = meta.prefill
        assert prefill_metadata is not None
        n_prefill_sf = num_tokens - num_decode_tokens
        if prefill_metadata.max_prefill_seq_len >= 0:
            short_prefill = (
                n_prefill_sf > 0
                and positions is not None
                and prefill_metadata.max_prefill_seq_len <= topk_tokens
            )
        else:
            short_prefill = (
                n_prefill_sf > 0
                and positions is not None
                and int(positions[num_decode_tokens:num_tokens].max().item()) + 1
                <= topk_tokens
            )
        if short_prefill:
            assert positions is not None
            shared._fill_causal_indices(
                topk_indices_buffer[num_decode_tokens:num_tokens],
                positions[num_decode_tokens:num_tokens].to(torch.int32),
            )

        workspace_manager = current_workspace_manager()
        values_spec, scales_spec = _gather_workspace_shapes(total_seq_lens, head_dim)
        k_quant_full, k_scale_full = workspace_manager.get_simultaneous(
            values_spec,
            scales_spec,
        )
        for chunk in prefill_metadata.chunks if not short_prefill else ():
            k_quant = k_quant_full[: chunk.total_seq_lens]
            k_scale = k_scale_full[: chunk.total_seq_lens]
            if not chunk.skip_kv_gather:
                ops.cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_quant,
                    k_scale,
                    chunk.block_table,
                    chunk.cu_seq_lens,
                )
            q_slice = q_quant[chunk.token_start : chunk.token_end]
            k_scale_cast = k_scale.view(torch.float32).squeeze(-1)
            logits = int8_mqa_logits(
                q_slice,
                (k_quant, k_scale_cast),
                weights[chunk.token_start : chunk.token_end],
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                clean_logits=False,
            )
            num_rows = logits.shape[0]
            # kpool: logits are pool-granular, so topk selects pools.
            select_k = topk_tokens // index_kpool if index_kpool > 1 else topk_tokens
            if index_kpool > 1:
                pool_topk = torch.empty(
                    (num_rows, select_k),
                    dtype=torch.int32,
                    device=logits.device,
                )
                topk_dst = pool_topk
            else:
                topk_dst = topk_indices_buffer[
                    chunk.token_start : chunk.token_end, :topk_tokens
                ]
            torch.ops._C.top_k_per_row_prefill(
                logits,
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                topk_dst,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                select_k,
            )
            if index_kpool > 1:
                pool_ids = pool_topk.to(torch.int64)
                if positions is not None:
                    q_seq = (
                        positions[chunk.token_start : chunk.token_end].to(torch.int32)
                        + 1
                    )
                    expanded = _expand_pools(pool_ids, q_seq, index_kpool, topk_tokens)
                else:
                    expanded = _kpool_ops().expand_pools_to_tokens(
                        pool_ids,
                        pool_ids >= 0,
                        topk_tokens,
                        index_kpool,
                    )
                topk_indices_buffer[
                    chunk.token_start : chunk.token_end, : expanded.shape[-1]
                ] = expanded

    if meta.num_decodes > 0:
        decode_metadata = meta.decode
        assert decode_metadata is not None
        kv_cache_view = shared.kv_cache_as_quant_view(kv_cache, head_dim, False)

        if (
            index_kpool > 1
            and gate_score is not None
            and compress_ape is not None
            and positions is not None
            and not skip_k_cache_insert
            and tail_kv_cache is not None
            and tail_prefix is not None
        ):
            _kpool_decode_update(
                kv_cache,
                tail_kv_cache,
                attn_metadata,
                decode_metadata,
                k,
                gate_score,
                compress_ape,
                slot_mapping,
                positions,
                num_decode_tokens,
                meta.num_decodes,
                index_kpool,
                head_dim,
                tail_prefix,
                shared,
            )

        if shared._fill_short_decode_causal_indices(
            topk_indices_buffer,
            positions,
            num_decode_tokens,
            meta.max_seq_len,
            topk_tokens,
        ):
            return topk_indices_buffer

        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            padded_q_quant_decode_tokens = pack_seq_triton(
                q_quant[:num_decode_tokens], decode_lens
            )
            padded_weights = pack_seq_triton(
                weights[:num_decode_tokens], decode_lens, pad_value=0
            ).reshape(-1, *weights.shape[1:])
        else:
            padded_q_quant_decode_tokens = q_quant[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_quant.shape[1:]
            )
            padded_weights = weights[:num_decode_tokens]
        batch_size = padded_q_quant_decode_tokens.shape[0]
        next_n = padded_q_quant_decode_tokens.shape[1]
        num_padded_tokens = batch_size * next_n
        seq_lens = decode_metadata.seq_lens[:batch_size]
        select_k = topk_tokens // index_kpool if index_kpool > 1 else topk_tokens

        # Fixed-length PPU DeepGEMM metadata: decode indices stay None.
        assert getattr(decode_metadata, "indices", None) is None
        logits = _decode_logits_paged(
            kv_cache_view,
            padded_q_quant_decode_tokens,
            padded_weights[:num_padded_tokens],
            seq_lens,
            decode_metadata.block_table,
            decode_metadata.schedule_metadata,
            max_pool_len,
        )
        num_rows = logits.shape[0]

        if index_kpool > 1:
            pool_topk = torch.empty(
                (num_rows, select_k), dtype=torch.int32, device=logits.device
            )
            topk_dst = pool_topk
        else:
            topk_dst = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        if select_k in (512, 1024, 2048):
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.persistent_topk(
                logits,
                seq_lens,
                topk_dst,
                topk_workspace,
                select_k,
                max_pool_len,
            )
        else:
            torch.ops._C.top_k_per_row_decode(
                logits,
                next_n,
                seq_lens,
                topk_dst,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                select_k,
            )

        if index_kpool > 1:
            pool_ids = pool_topk.to(torch.int64)
            if positions is not None:
                dec_seq = shared._decode_topk_seq_lens(
                    positions,
                    decode_lens,
                    num_decode_tokens,
                    batch_size,
                    next_n,
                    decode_metadata.requires_padding,
                )
            else:
                dec_seq = decode_metadata.seq_lens[:num_rows]
                if dec_seq.ndim == 2:
                    dec_seq = dec_seq[:, -1]
                dec_seq = dec_seq.to(torch.int32)
            out = _expand_pools(pool_ids, dec_seq, index_kpool, topk_tokens)
        else:
            out = topk_dst

        if decode_metadata.requires_padding:
            out = unpack_seq_triton(
                out.reshape(batch_size, -1, out.shape[-1]), decode_lens
            )
        topk_indices_buffer[: out.shape[0], : out.shape[-1]] = out

    return topk_indices_buffer


def ppu_sparse_attn_indexer_kpool_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_pool_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
    gate_score: torch.Tensor | None = None,
    compress_ape: torch.Tensor | None = None,
    index_kpool: int = 1,
    positions: torch.Tensor | None = None,
    tail_kv_cache: torch.Tensor | None = None,
    tail_prefix: str | None = None,
) -> torch.Tensor:
    return topk_indices_buffer
