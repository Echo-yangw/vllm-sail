# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP4 packing for the Qwen4Exp QSA indexer's compressed key cache.

DeepGEMM's FP4 average MQA logits kernels read E2M1 values two per byte plus
one UE8M0 scale per 32 logical values. A QSA indexer head is 128 logical dims,
so a packed row is 64 value bytes and its four block scales fit in one int32 —
which is why the kernels take ``q_sf``/``k_sf`` as int32 rather than as bytes.

The paged kernel reads each cache page planar (all value rows, then all scale
rows) rather than per-row interleaved, so ``MXFP4_ROW_BYTES`` describes the
allocation width and not the addressing stride; see ``vllm_sail.utils.deep_gemm``
for the strides the kernel actually applies.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

MXFP4_BLOCK_SIZE = 32
MXFP4_E2M1_MAX = 6.0
MXFP4_LOGICAL_HEAD_DIM = 128
MXFP4_NUM_SCALE_BLOCKS = MXFP4_LOGICAL_HEAD_DIM // MXFP4_BLOCK_SIZE
MXFP4_PACKED_HEAD_DIM = MXFP4_LOGICAL_HEAD_DIM // 2
MXFP4_SCALE_BYTES = MXFP4_NUM_SCALE_BLOCKS
MXFP4_ROW_BYTES = MXFP4_PACKED_HEAD_DIM + MXFP4_SCALE_BYTES
MXFP4_PAGE_ROWS = 64

# Triton resolves a @jit body's globals against the defining module and rejects
# any that are not constexpr *instances*, so the kernels read this alias; the
# plain float above stays the host-visible value.
_E2M1_MAX = tl.constexpr(MXFP4_E2M1_MAX)


@triton.jit
def _fp32x2_to_fp4x2(x_lo, x_hi):
    """Pack two fp32 values into one byte of E2M1.

    Mirrors ``vllm.models.deepseek_v4.common.ops.fused_indexer_q`` so the QSA
    indexer rounds identically to the DSA indexer on the same build.
    """
    # NOTE: $1 is high nibble, $2 is low nibble
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b8 tmp;
            cvt.rn.satfinite.e2m1x2.f32 tmp, $1, $2;
            cvt.u32.u8 $0, tmp;
        }
        """,
        constraints="=r,f,f",
        args=[x_hi, x_lo],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    ).to(tl.uint8)


@triton.jit
def _quantize_mxfp4_pair(x_lo, x_hi):
    """Quantize one MXFP4 block given as its even and odd interleaved halves.

    Returns ``(packed, ue8m0)`` where ``packed`` is uint8[BLOCK/2] with the low
    nibble holding ``x_lo`` and ``ue8m0`` is the scalar block scale byte, so the
    block scale is ``2^(ue8m0 - 127)``.
    """
    amax = tl.maximum(tl.max(tl.abs(x_lo)), tl.max(tl.abs(x_hi)))
    # 6 * 2^-126 keeps a all-zero block from producing a -inf exponent.
    amax = tl.maximum(amax, _E2M1_MAX * (2**-126))
    log2_ratio = tl.ceil(tl.log2(amax * (1.0 / _E2M1_MAX)))
    log2_ratio = tl.minimum(tl.maximum(log2_ratio, -127.0), 127.0)
    inv_scale = tl.exp2(-log2_ratio)
    ue8m0 = (log2_ratio + 127.0).to(tl.uint8)
    return _fp32x2_to_fp4x2(x_lo * inv_scale, x_hi * inv_scale), ue8m0


@triton.jit
def _qsa_quant_q_mxfp4_kernel(
    q_ptr,
    values_ptr,
    scales_ptr,
    stride_q_row,
    stride_q_head,
    stride_val_row,
    stride_val_head,
    stride_sf_row,
    stride_sf_head,
    NUM_SCALE_BLOCKS: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    HALF: tl.constexpr = QUANT_BLOCK // 2
    src = q_ptr + row * stride_q_row + head * stride_q_head
    dst = values_ptr + row * stride_val_row + head * stride_val_head
    pair = tl.arange(0, HALF)

    # Little-endian byte j holds block j's scale, matching the int32 the kernel
    # reads back out of the cache's planar scale plane.
    packed_scale = 0
    for block in tl.static_range(NUM_SCALE_BLOCKS):
        base = block * QUANT_BLOCK
        even = tl.load(src + base + 2 * pair).to(tl.float32)
        odd = tl.load(src + base + 2 * pair + 1).to(tl.float32)
        packed, ue8m0 = _quantize_mxfp4_pair(even, odd)
        tl.store(dst + base // 2 + pair, packed)
        packed_scale = packed_scale | (ue8m0.to(tl.uint32) << (8 * block))

    tl.store(
        scales_ptr + row * stride_sf_row + head * stride_sf_head,
        packed_scale.to(tl.int32, bitcast=True),
    )


def qsa_quant_q_mxfp4(
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize indexer queries to the packed MXFP4 the avg kernels consume.

    Args:
        q: ``[num_rows, num_heads, 128]`` normalized and RoPE'd query, bf16 or
            fp32. ``num_heads`` must be 4, which is all the average-mode kernel
            supports.

    Returns:
        ``(values, scales)`` with ``values`` ``[num_rows, num_heads, 64]`` int8
        and ``scales`` ``[num_rows, num_heads]`` int32.
    """
    num_rows, num_heads, head_dim = q.shape
    if head_dim != MXFP4_LOGICAL_HEAD_DIM:
        raise ValueError(
            f"QSA MXFP4 packing requires a {MXFP4_LOGICAL_HEAD_DIM}-dim indexer "
            f"head, got {head_dim}"
        )
    device = q.device
    values = torch.empty(
        (num_rows, num_heads, MXFP4_PACKED_HEAD_DIM),
        dtype=torch.uint8,
        device=device,
    )
    scales = torch.empty((num_rows, num_heads), dtype=torch.int32, device=device)
    if num_rows:
        _qsa_quant_q_mxfp4_kernel[(num_rows, num_heads)](
            q,
            values,
            scales,
            q.stride(0),
            q.stride(1),
            values.stride(0),
            values.stride(1),
            scales.stride(0),
            scales.stride(1),
            NUM_SCALE_BLOCKS=MXFP4_NUM_SCALE_BLOCKS,
            QUANT_BLOCK=MXFP4_BLOCK_SIZE,
        )
    return values.view(torch.int8), scales


def expand_qsa_fp4_block_table(
    block_table: torch.Tensor, pages_per_block: int
) -> torch.Tensor:
    """Map a framework-geometry block table onto 64-row FP4 pages.

    FP4 page ``b * pages_per_block + j`` mirrors framework block ``b`` rows
    ``[j * 64, (j + 1) * 64)``; one page per block makes this the identity.
    """
    if pages_per_block == 1:
        return block_table
    cols = block_table.shape[1]
    expanded = block_table.repeat_interleave(pages_per_block, dim=1)
    pattern = (
        torch.arange(pages_per_block, device=block_table.device)
        .repeat(cols)
        .to(expanded.dtype)
    )
    return expanded * pages_per_block + pattern


def qsa_fp4_pages_per_block(block_rows: int) -> int:
    """FP4 pages covering one framework block of ``block_rows`` cache rows."""
    return block_rows // MXFP4_PAGE_ROWS


def as_qsa_fp4_pages(kv_cache: torch.Tensor) -> torch.Tensor:
    """Reshape a bound compressed cache onto 64-row FP4 pages.

    ``_QSAIndexerCache.bind_kv_cache`` hands the cache over as
    ``[num_blocks, block_rows, 1, row_bytes]``; a framework block of several FP4
    pages becomes that many pages here, and the framework block table is
    expanded to match by ``expand_qsa_fp4_block_table``.
    """
    num_blocks, block_rows, num_kv_heads, row_bytes = kv_cache.shape
    if num_kv_heads != 1:
        raise ValueError(f"QSA MXFP4 pages hold one kv head, got {num_kv_heads}")
    if row_bytes != MXFP4_ROW_BYTES:
        raise ValueError(
            f"QSA MXFP4 rows are {MXFP4_ROW_BYTES} bytes wide, got {row_bytes}"
        )
    if block_rows % MXFP4_PAGE_ROWS:
        raise ValueError(
            f"QSA MXFP4 pages hold {MXFP4_PAGE_ROWS} rows, which must divide the "
            f"{block_rows} compressed rows per framework block; the QSA block "
            f"size must be a multiple of {MXFP4_PAGE_ROWS} * compress_ratio"
        )
    return kv_cache.reshape(
        num_blocks * qsa_fp4_pages_per_block(block_rows),
        MXFP4_PAGE_ROWS,
        num_kv_heads,
        row_bytes,
    )


def gather_qsa_fp4_keys(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens: torch.Tensor,
    packed_key_capacity: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather compressed rows into one packed key space for the non-paged kernel.

    ``cu_seqlens`` is the exclusive scan of each request's packed length, so
    request ``i`` occupies ``[cu_seqlens[i], cu_seqlens[i + 1])``. That is what
    lets a single non-paged launch score the whole batch: the kernel receives
    per-row ``cu_seqlen_ks`` rebased onto the request each row belongs to, and
    the columns it writes are already request-relative.

    ``kv_cache`` is the bound compressed cache and ``block_table`` the framework
    block table, both in framework-block geometry rather than re-paged onto
    64-row FP4 pages. ``_qsa_prepare_kernel`` takes its page size from
    ``compressed_cache.shape[1]``, so it stores a framework block planar: every
    row's 64 value bytes, then every row's 4 UE8M0 scale bytes. Reading that
    back as 68 contiguous bytes per row would take the scales out of the value
    plane and shift every row past the first, so the two planes are gathered at
    their own block-relative offsets.

    The cache is one slice of the combined hybrid KV block, so its block pitch
    ``stride(0)`` also spans the other caches and is much larger than the
    ``block_rows * 68`` bytes this cache occupies. The writer steps blocks by the
    same ``stride(0)`` and never uses ``stride(1)``, so that padding is not
    something to reject -- only a pitch too small to hold a block is.

    Returns int8 ``[packed_key_capacity, 64]`` and int32
    ``[packed_key_capacity]``. Slots past ``cu_seqlens[-1]`` alias the last
    request; their bytes are never referenced because every row end stays inside
    its own request, and the clamps keep the gather indices in bounds. All sizes
    come from the host-side capacity bound, so nothing here synchronizes.
    """
    num_blocks, block_rows, num_kv_heads, row_bytes = kv_cache.shape
    if num_kv_heads != 1 or row_bytes != MXFP4_ROW_BYTES:
        raise ValueError(
            "QSA MXFP4 gather needs the bound compressed cache as "
            f"[blocks, rows, 1, {MXFP4_ROW_BYTES}], got {tuple(kv_cache.shape)}"
        )
    stride_block = kv_cache.stride(0)
    block_bytes = block_rows * row_bytes
    # ``stride(1) == row_bytes`` is not a tight-packing demand: the block pitch
    # may exceed ``block_bytes`` arbitrarily. It only pins the rows to the
    # allocation's own pitch, which is what makes the byte view below provably
    # inside the tensor's footprint rather than merely inside the storage.
    if (
        kv_cache.stride(3) != 1
        or kv_cache.stride(1) != row_bytes
        or stride_block < block_bytes
    ):
        raise ValueError(
            "QSA MXFP4 gather needs each framework block's "
            f"{block_bytes} bytes at a {row_bytes}-byte row pitch inside its own "
            f"block pitch, got strides {tuple(kv_cache.stride())}"
        )
    # ``bind_kv_cache`` hands the cache over transposed, so dim 2 is a size-1
    # artifact whose stride is the block pitch rather than 0. Span exactly the
    # tensor's own footprint -- the last block's last byte -- so the view stops
    # before the padding after it; the gather never reads that gap.
    flat = kv_cache.as_strided(
        ((num_blocks - 1) * stride_block + block_bytes,),
        (1,),
        kv_cache.storage_offset(),
    )
    num_reqs = cu_seqlens.shape[0] - 1
    device = kv_cache.device
    packed_positions = torch.arange(
        packed_key_capacity, dtype=torch.int64, device=device
    )
    packed_req = torch.searchsorted(
        cu_seqlens[1:].contiguous(), packed_positions, right=True
    )
    packed_req.clamp_(max=num_reqs - 1)
    packed_row = packed_positions - cu_seqlens.index_select(0, packed_req)
    packed_row.clamp_(0, block_table.shape[1] * block_rows - 1)
    block_ids = block_table[packed_req, packed_row // block_rows].long()
    slot = packed_row % block_rows
    value_base = block_ids * stride_block + slot * MXFP4_PACKED_HEAD_DIM
    scale_base = (
        block_ids * stride_block
        + block_rows * MXFP4_PACKED_HEAD_DIM
        + slot * MXFP4_SCALE_BYTES
    )
    values = flat[
        value_base[:, None]
        + torch.arange(MXFP4_PACKED_HEAD_DIM, dtype=torch.int64, device=device)
    ].view(torch.int8)
    scales = (
        flat[
            scale_base[:, None]
            + torch.arange(MXFP4_SCALE_BYTES, dtype=torch.int64, device=device)
        ]
        .view(torch.int32)
        .squeeze(1)
    )
    return values, scales


__all__ = [
    "MXFP4_BLOCK_SIZE",
    "MXFP4_E2M1_MAX",
    "MXFP4_LOGICAL_HEAD_DIM",
    "MXFP4_NUM_SCALE_BLOCKS",
    "MXFP4_PACKED_HEAD_DIM",
    "MXFP4_PAGE_ROWS",
    "MXFP4_ROW_BYTES",
    "MXFP4_SCALE_BYTES",
    "as_qsa_fp4_pages",
    "expand_qsa_fp4_block_table",
    "gather_qsa_fp4_keys",
    "qsa_fp4_pages_per_block",
    "qsa_quant_q_mxfp4",
]
