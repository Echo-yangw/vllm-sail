# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preserve V4.1 Engram lookup semantics for PPU FP32 per-row scales."""

from vllm.config import get_current_vllm_config
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from vllm_sail.patch.utils import PATCH_MARKER, patch

_AFFECTED = ">=0.32.0,<0.33.0"
_FACTORY = "vllm.models.deepseek_v41.common.engram"


@patch(
    "vllm.models.deepseek_v41.common.engram",
    reason="PPU channelwise Engram exports carry FP32 row scales rather than E8M0 block scales.",
    affected_versions=_AFFECTED,
    remove_when="Upstream Engram lookup supports FP32 scales alongside E8M0 scales.",
)
@triton.jit(
    do_not_specialize=[
        "vocab_start",
        "vocab_end",
        "num_rows",
        "ids_stride_t",
        "ids_stride_h",
        "GRID",
    ]
)
def _engram_lookup_kernel(
    weight,
    scales,
    ids,
    sorted_dst,
    out,
    vocab_start,
    vocab_end,
    num_rows,
    ids_stride_t,
    ids_stride_h,
    HEAD_START: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    TOTAL_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    GRID,
    SORTED: tl.constexpr,
    PACKED: tl.constexpr,
    OUT_STRIDE: tl.constexpr,
    PEER_ROWS: tl.constexpr = 0,
):
    """Gather FP8 rows using the current upstream sorted/packed contract.

    Only this rank's heads are read; padded heads write zeros for all-gather.
    `weight`/`scales` may address pinned host memory through UVA.
    """
    cols = tl.arange(0, DIM)
    scale_cols = cols // QUANT_BLOCK
    for base in tl.range(tl.program_id(0) * BLOCK_R, num_rows, GRID * BLOCK_R):
        rows = base + tl.arange(0, BLOCK_R)
        valid = rows < num_rows
        if SORTED:
            local = tl.load(ids + rows, mask=valid, other=-1).to(tl.int64)
            dst = tl.load(sorted_dst + rows, mask=valid, other=0).to(tl.int64)
            owned = (local >= 0) & (local < vocab_end - vocab_start)
        else:
            head = HEAD_START + rows % LOCAL_HEADS
            token = (rows // LOCAL_HEADS).to(tl.int64)
            index = tl.load(
                ids + token * ids_stride_t + head * ids_stride_h,
                mask=valid & (head < TOTAL_HEADS),
                other=-1,
            ).to(tl.int64)
            dst = rows.to(tl.int64)
            owned = valid & (head < TOTAL_HEADS)
            owned &= (index >= vocab_start) & (index < vocab_end)
            local = index - vocab_start
        local = tl.where(owned, local, 0)
        if PEER_ROWS:
            owner = local // PEER_ROWS
            peer = tl.load(weight + owner, mask=owned, other=0)
            local -= owner * PEER_ROWS
            row_weight = peer.to(tl.pointer_type(tl.float8e4nv))
            row_scales = (peer + PEER_ROWS * DIM).to(tl.pointer_type(tl.uint8))
        else:
            row_weight, row_scales = weight, scales
        row_weight += local * DIM
        row_scales += local * (DIM // QUANT_BLOCK)
        values = tl.load(
            row_weight[:, None] + cols[None, :],
            mask=owned[:, None],
            other=0.0,
        )
        scale = tl.load(
            row_scales[:, None] + scale_cols[None, :],
            mask=owned[:, None],
            other=0,
        )
        if PACKED:
            tl.static_assert(
                row_scales.dtype.element_ty == tl.uint8,
                "Packed Engram output requires E8M0 byte scales",
            )
            width: tl.constexpr = DIM + DIM // QUANT_BLOCK
            dst = dst // LOCAL_HEADS * OUT_STRIDE + dst % LOCAL_HEADS * width
            tl.store(
                out + dst[:, None] + cols[None, :],
                values.to(tl.uint8, bitcast=True),
                mask=valid[:, None],
            )
            tl.store(
                out + dst[:, None] + DIM + scale_cols[None, :],
                scale,
                mask=valid[:, None] & (cols[None, :] % QUANT_BLOCK == 0),
            )
        else:
            if row_scales.dtype.element_ty != tl.float32:
                scale = (scale.to(tl.int32) << 23).to(tl.float32, bitcast=True)
            tl.store(
                out + dst[:, None] * DIM + cols[None, :],
                (values.to(tl.float32) * scale).to(tl.bfloat16),
                mask=valid[:, None],
            )


@patch(
    _FACTORY,
    "Engram._create_embedding",
    reason="Select FP32 row-scale storage for channelwise Engram while reusing CUDA offload and prefetch.",
    affected_versions=_AFFECTED,
    remove_when="The Engram embedding factory exposes quantization-aware backend registration.",
)
def _create_embedding(self, layout, layer_hash_index):
    vllm_config = get_current_vllm_config()
    quant = (
        getattr(vllm_config.model_config.hf_config, "quantization_config", None) or {}
    )
    name = f"layers.{layout.layer_ids[layer_hash_index]}.engram.embed"
    if current_platform.is_ppu() and name in (
        quant.get("fp8_channelwise_layers") or []
    ):
        from vllm_sail.models.deepseek_v41.engram import ChannelwiseEngramEmbedding

        config = vllm_config.engram_config
        return ChannelwiseEngramEmbedding(
            layout.num_embeddings[layer_hash_index],
            layout.head_dim,
            tuple(size for order in layout.primes[layer_hash_index] for size in order),
            cpu_offload=config.cpu_offload,
            dp_shared_memory=config.dp_shared_memory,
        )
    original = getattr(_create_embedding, PATCH_MARKER)[
        f"{_FACTORY}.Engram._create_embedding"
    ]
    return original(self, layout, layer_hash_index)
