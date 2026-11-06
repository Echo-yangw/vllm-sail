# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepGEMM MXFP4 scoring for the Qwen4Exp QSA indexer.

Replaces upstream's Triton MQA kernels with DeepGEMM's FP4 average-MQA logits:
the paged kernel for uniform decode and the non-paged kernel for prefill, which
is the only route to a compressed-key scoring pass because the paged entry caps
``next_n`` at six query rows per request.

Both paths hand ``persistent_topk`` the same request-relative compressed column
indices upstream produces, so selection and index expansion are unchanged. The
public signatures mirror ``vllm.models.qwen4_exp.nvidia.ops.qsa_indexer`` so
the indexer's dispatch needs no PPU-specific arguments.
"""

from __future__ import annotations

import torch

from vllm_sail.models.qwen4_exp.ops.mxfp4 import (
    MXFP4_LOGICAL_HEAD_DIM,
    MXFP4_PACKED_HEAD_DIM,
    MXFP4_PAGE_ROWS,
    MXFP4_ROW_BYTES,
    as_qsa_fp4_pages,
    expand_qsa_fp4_block_table,
    gather_qsa_fp4_keys,
    qsa_fp4_pages_per_block,
    qsa_quant_q_mxfp4,
)
from vllm_sail.utils.deep_gemm import (
    fp4_avg_metadata_extra,
    fp4_mqa_avg_logits,
    fp4_paged_mqa_avg_logits,
    get_num_sms,
    get_paged_mqa_logits_metadata,
)

_TOPK_WORKSPACE_BYTES = 1024 * 1024
# The logits row stride stays cooperative-topk friendly at this granularity,
# which is also the width upstream rounds its prefill logits to.
_LOGITS_WIDTH_GRANULARITY = 64
_FP4_AVG_NUM_HEADS = 4
# The paged kernel hard-caps the query rows it packs into one request tile.
_MAX_PAGED_NEXT_N = 6


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def _topk(
    logits: torch.Tensor,
    visible_blocks: torch.Tensor,
    block_indices: torch.Tensor,
    block_topk: int,
) -> None:
    # Zhenwu reports capability (8, 9), so upstream's cooperative_topk branch —
    # gated on has_device_capability(90) — would not select here either.
    workspace = torch.empty(
        (_TOPK_WORKSPACE_BYTES,), dtype=torch.uint8, device=logits.device
    )
    torch.ops._C.persistent_topk(
        logits,
        visible_blocks,
        block_indices,
        workspace,
        block_topk,
        logits.shape[1],
    )


def _fp4_pages_and_table(
    k_cache: torch.Tensor, page_table: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    fp4_cache = as_qsa_fp4_pages(k_cache)
    pages_per_block = qsa_fp4_pages_per_block(k_cache.shape[1])
    return fp4_cache, expand_qsa_fp4_block_table(page_table, pages_per_block)


def qsa_select_paged_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    visible_blocks: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    decode_query_len: int,
    block_indices: torch.Tensor,
) -> None:
    """Score compressed decode blocks with the paged FP4 average kernel.

    ``visible_blocks`` is each row's committed-group count, which is exactly the
    paged kernel's per-row context length, so decode needs no gather: the kernel
    walks the page table itself and writes request-relative columns.
    """
    assert token_topk % compress_ratio == 0
    assert block_indices.shape == (q.shape[0], token_topk // compress_ratio)
    assert decode_query_len > 0 and q.shape[0] % decode_query_len == 0
    if decode_query_len > _MAX_PAGED_NEXT_N:
        raise NotImplementedError(
            "DeepGEMM's paged FP4 average kernel packs at most "
            f"{_MAX_PAGED_NEXT_N} decode query rows per request, got "
            f"{decode_query_len}; lower --num-speculative-tokens"
        )
    num_requests = q.shape[0] // decode_query_len
    num_heads = q.shape[1]

    fp4_cache, fp4_block_table = _fp4_pages_and_table(k_cache, page_table)
    columns = fp4_block_table.shape[1] * MXFP4_PAGE_ROWS
    # The kernel and its schedule never see a zero-length row; top-k still uses
    # the raw counts, so an empty row selects nothing.
    context_lens = visible_blocks.view(num_requests, decode_query_len).clamp(min=1)

    q_values, q_sf = qsa_quant_q_mxfp4(q)
    logits = fp4_paged_mqa_avg_logits(
        (
            q_values.view(
                num_requests, decode_query_len, num_heads, MXFP4_PACKED_HEAD_DIM
            ),
            q_sf.view(num_requests, decode_query_len, num_heads),
        ),
        fp4_cache,
        context_lens,
        fp4_block_table,
        get_paged_mqa_logits_metadata(
            context_lens,
            MXFP4_PAGE_ROWS,
            get_num_sms(),
            fp4_avg_metadata_extra(decode_query_len),
        ),
        columns,
    )
    _topk(logits, visible_blocks, block_indices, token_topk // compress_ratio)


def qsa_select_paged_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    visible_blocks: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    max_query_len: int,
    block_indices: torch.Tensor,
    max_seq_len: int,
) -> None:
    """Score compressed prefill blocks with the non-paged FP4 average kernel.

    Each request's visible compressed rows are gathered into one packed key
    space and ``cu_seqlen_ks`` rebases every row onto its own request, so a
    single launch covers the whole batch. ``max_seqlen_k`` is upstream's
    ``logits_width``: a host-side bound on any row's span, which keeps the
    transient logits proportional to the longest context rather than to the
    packed space.

    Every derived size comes from ``max_seq_len`` and tensor shapes, so the
    per-request offsets are built on device and nothing here synchronizes.
    ``max_query_len`` is unused — this launch is not chunked over query rows.
    """
    del max_query_len
    assert token_topk % compress_ratio == 0
    assert block_indices.shape == (q.shape[0], token_topk // compress_ratio)
    assert query_start_loc.shape == (page_table.shape[0] + 1,)
    assert visible_blocks.shape == (q.shape[0],)

    rows = q.shape[0]
    num_requests = page_table.shape[0]
    device = q.device

    logits_width = _round_up(
        -(-max_seq_len // compress_ratio), _LOGITS_WIDTH_GRANULARITY
    )
    logits_width = min(
        max(_LOGITS_WIDTH_GRANULARITY, logits_width),
        page_table.shape[1] * k_cache.shape[1],
    )
    packed_key_capacity = num_requests * logits_width

    # ``query_start_loc`` is a rebased slice and may share a larger backing
    # tensor, so re-zero it before searching.
    starts = query_start_loc - query_start_loc[0]
    row_positions = torch.arange(rows, dtype=starts.dtype, device=device)
    req_per_row = torch.searchsorted(starts[1:].contiguous(), row_positions, right=True)
    # A request only needs as many gathered rows as its widest query sees, which
    # is the segmented max of ``visible_blocks`` rather than its full context.
    keys_per_req = torch.zeros(num_requests, dtype=torch.int64, device=device)
    keys_per_req.scatter_reduce_(
        0, req_per_row, visible_blocks.to(torch.int64), reduce="amax"
    )
    keys_per_req.clamp_(max=logits_width)
    cu_seqlens = torch.zeros(num_requests + 1, dtype=torch.int64, device=device)
    torch.cumsum(keys_per_req, 0, dtype=torch.int64, out=cu_seqlens[1:])

    # The gather reads the framework-block-planar layout the prepare kernel
    # writes, so prefill takes the bound cache and block table as they are;
    # re-paging onto 64-row FP4 pages is only what the paged kernel needs.
    k_values, k_sf = gather_qsa_fp4_keys(
        k_cache, page_table, cu_seqlens, packed_key_capacity
    )

    row_starts = cu_seqlens.index_select(0, req_per_row).to(torch.int32)
    row_ends = row_starts + visible_blocks.to(torch.int32)

    q_values, q_sf = qsa_quant_q_mxfp4(q)
    logits = fp4_mqa_avg_logits(
        (q_values, q_sf), (k_values, k_sf), row_starts, row_ends, logits_width
    )
    _topk(logits, visible_blocks, block_indices, token_topk // compress_ratio)


def warmup_qsa_mqa_paged_decode(
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    max_decode_query_len: int,
    max_num_reqs: int,
    max_num_batched_tokens: int,
) -> tuple[tuple[int, int], ...]:
    """JIT-compile every FP4 kernel reachable from the QSA indexer.

    DeepGEMM's schedule and logits kernels specialize on ``next_n`` alone, and
    neither may JIT during CUDA-graph capture, so every capturable decode
    profile launches once against a one-page dummy request. The prefill and
    quantization kernels are not captured but compile here too, since a
    first-call JIT inside a serving step is a visible stall.
    """
    del page_table, max_num_reqs, max_num_batched_tokens
    require_fp4_mqa_avg_logits()
    if num_heads != _FP4_AVG_NUM_HEADS or head_dim != MXFP4_LOGICAL_HEAD_DIM:
        raise NotImplementedError(
            "DeepGEMM's FP4 average MQA logits kernels score "
            f"{_FP4_AVG_NUM_HEADS} heads of {MXFP4_LOGICAL_HEAD_DIM} logical "
            f"dims only, got {num_heads} heads of {head_dim}"
        )

    device = k_cache.device
    num_sms = get_num_sms()
    page = as_qsa_fp4_pages(k_cache)[:1].contiguous()
    block_table = torch.zeros((1, 1), dtype=torch.int32, device=device)
    q_values = torch.zeros(
        (_MAX_PAGED_NEXT_N, _FP4_AVG_NUM_HEADS, MXFP4_PACKED_HEAD_DIM),
        dtype=torch.int8,
        device=device,
    )
    q_sf = torch.zeros(
        (_MAX_PAGED_NEXT_N, _FP4_AVG_NUM_HEADS), dtype=torch.int32, device=device
    )

    profiles: list[tuple[int, int]] = []
    for next_n in range(1, min(max_decode_query_len, _MAX_PAGED_NEXT_N) + 1):
        context_lens = torch.ones((1, next_n), dtype=torch.int32, device=device)
        fp4_paged_mqa_avg_logits(
            (
                q_values[:next_n].view(
                    1, next_n, _FP4_AVG_NUM_HEADS, MXFP4_PACKED_HEAD_DIM
                ),
                q_sf[:next_n].view(1, next_n, _FP4_AVG_NUM_HEADS),
            ),
            page,
            context_lens,
            block_table,
            get_paged_mqa_logits_metadata(
                context_lens, MXFP4_PAGE_ROWS, num_sms, fp4_avg_metadata_extra(next_n)
            ),
            MXFP4_PAGE_ROWS,
        )
        profiles.append((next_n, 1))

    bf16_q = torch.zeros(
        (1, _FP4_AVG_NUM_HEADS, MXFP4_LOGICAL_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    packed_q, packed_q_sf = qsa_quant_q_mxfp4(bf16_q)
    k_values = torch.zeros(
        (MXFP4_PAGE_ROWS, MXFP4_PACKED_HEAD_DIM), dtype=torch.int8, device=device
    )
    k_sf = torch.zeros(MXFP4_PAGE_ROWS, dtype=torch.int32, device=device)
    row_starts = torch.zeros(1, dtype=torch.int32, device=device)
    fp4_mqa_avg_logits(
        (packed_q, packed_q_sf),
        (k_values, k_sf),
        row_starts,
        row_starts + 1,
        MXFP4_PAGE_ROWS,
    )
    return tuple(profiles)


_FP4_UNAVAILABLE_MSG = (
    "The Qwen4Exp QSA indexer stores its compressed key cache as MXFP4, which "
    "only DeepGEMM's FP4 average MQA logits kernels can score. This build of "
    "deep_gemm does not export fp8_fp4_mqa_avg_logits / "
    "fp8_fp4_paged_mqa_avg_logits; install a DeepGEMM with the FP4 paged "
    "average entry point."
)


def require_fp4_mqa_avg_logits() -> None:
    """Fail at model load when DeepGEMM cannot score the MXFP4 cache."""
    from vllm_sail.utils.deep_gemm import (
        has_fp4_mqa_avg_logits,
        has_fp4_paged_mqa_avg_logits,
    )

    if not (has_fp4_mqa_avg_logits() and has_fp4_paged_mqa_avg_logits()):
        raise RuntimeError(_FP4_UNAVAILABLE_MSG)


def qsa_fp4_row_bytes() -> int:
    """Allocation width of one MXFP4 compressed cache row."""
    return MXFP4_ROW_BYTES


__all__ = [
    "qsa_fp4_row_bytes",
    "qsa_select_paged_decode",
    "qsa_select_paged_prefill",
    "require_fp4_mqa_avg_logits",
    "warmup_qsa_mqa_paged_decode",
]
