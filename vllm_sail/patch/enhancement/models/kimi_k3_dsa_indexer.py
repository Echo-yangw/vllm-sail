# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F821
# Copied Triton bodies resolve ``tl``/``triton`` in *this* module: a
# ``triton.jit`` function captures its defining module's globals, so
# ``bind_body`` cannot re-point them at the upstream module.
"""810E-class target support for the Kimi K3.1 DSA indexer.

Two hardware gaps on capability ``(8, 0)`` PPU targets (e.g. 810E) break the
stock DSA indexer, which otherwise only needs SM89+:

* Triton's CUDA backend rejects ``tl.float8e4nv`` below capability 8.9 and the
  target has no f32->e4m3 cvt instruction, so the fused prologue's FP8 stores
  fail to compile. The replacement wrappers quantize through a software RNE
  e4m3fn encoder instead (bit-identical to the hardware cast).
* Targets without FP8 MMA also have no FP8 DeepGEMM ``*_mqa_logits`` kernel.
  The wrappers then quantize the indexer Q/K to int8 (ue8m0 scales folded into
  the folded weights, mirroring the DeepSeek V3 PPU int8-indexer gate) and
  route prefill scoring through ``deep_gemm.int8_mqa_logits`` and decode/paged
  scoring through ``int8_paged_mqa_logits``.

Only the three entry points are replaced, with minimal diffs: the kernel
signatures and inner call sites stay byte-identical to upstream (the fork's
SW_FP8/USE_INT8/INT8_MAX knobs ride along as defaulted ``tl.constexpr``
parameters), so every launch reuses the upstream text except the quant
branch and the raw-byte cache views. The stock ``@triton.jit`` kernel objects
stay in place because ``dsa_indexer_mxfp4`` calls
``_dsa_fused_prologue_kernel`` with the old signature; targets with FP8
support delegate to upstream.

The same commit also branches ``vllm.model_executor.layers.sparse_attn_indexer``
on the int8 Q dtype; rather than copying that ~650-line function, its three
module-level scoring helpers are rebound here (see ``_install_int8_scoring``).
"""

from __future__ import annotations

import functools

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

import vllm.models.kimi_k3.nvidia.ops.dsa_indexer as _dsa_indexer
import vllm.model_executor.layers.sparse_attn_indexer as _sparse
from vllm.v1.attention.ops.common import pack_ue8m0_x4
from vllm_sail.patch.utils import patch

_AFFECTED = ">=0.32.0,<0.33.0"
_DSA_MODULE = "vllm.models.kimi_k3.nvidia.ops.dsa_indexer"
_SPARSE_MODULE = "vllm.model_executor.layers.sparse_attn_indexer"

logger = init_logger(__name__)

# Same gates as the fork: sm80-class targets need the software encoder, and
# PPU targets without FP8 MMA need the int8 indexer path.
_SW_FP8_CAST = not current_platform.has_device_capability((8, 9))
_USE_INT8_INDEXER = current_platform.is_ppu() and not current_platform.supports_fp8()
_FP8_DTYPE = current_platform.fp8_dtype()
_USE_FNUZ = torch.float8_e4m3fnuz == _FP8_DTYPE
_FP8_MAX = 224.0 if _USE_FNUZ else 448.0
_INT8_MAX = 127.0

_int8_mqa_logits = None
_int8_paged_mqa_logits = None
if _USE_INT8_INDEXER:
    try:
        from deep_gemm import int8_mqa_logits as _int8_mqa_logits
        from deep_gemm import int8_paged_mqa_logits as _int8_paged_mqa_logits
    except ImportError:  # pragma: no cover - depends on the SAIL image
        _USE_INT8_INDEXER = False
        logger.warning(
            "Skipping the Kimi K3.1 int8 DSA indexer path: deep_gemm does not "
            "export int8_mqa_logits/int8_paged_mqa_logits on this image."
        )

# Captured before patching so non-affected targets delegate to upstream.
_upstream_k_insert = _dsa_indexer.indexer_k_norm_quant_insert
_upstream_q_quant = _dsa_indexer.indexer_q_quant_fold_weights
_upstream_prologue = _dsa_indexer.dsa_fused_prologue

# Unchanged upstream jit helpers, reused in place: a jit call resolves through
# the callee object itself, so cross-module references are safe.
_ue8m0_scale = _dsa_indexer._ue8m0_scale
_fill_topk = _dsa_indexer._fill_topk
_rms_norm_row = _dsa_indexer._rms_norm_row


# ---------------------------------------------------------------------------
# Fork-only quant helpers (the whole point of the patch)
# ---------------------------------------------------------------------------


@triton.jit
def _encode_e4m3fn(x):
    """Encode pre-clamped fp32 to FP8 E4M3FN as uint8 (SM80 compatible).

    Bit-identical to ``x.to(tl.float8e4nv)`` (RNE, saturating) but uses only
    integer ops, so it compiles where Triton refuses the fp8 dtype.
    """
    bits = x.to(tl.int32, bitcast=True)
    sign = (bits >> 31) & 1
    abs_bits = bits & 0x7FFFFFFF

    fp32_exp = (abs_bits >> 23) & 0xFF
    fp32_mant = abs_bits & 0x7FFFFF

    is_zero = abs_bits == 0
    fp8_exp_normal = fp32_exp - 120
    is_subnorm = (fp8_exp_normal <= 0) & (~is_zero)

    # ---- Normal path: top 3 mantissa bits with RNE rounding. ----
    truncated = fp32_mant & 0xFFFFF
    halfway = 0x80000
    lsb = (fp32_mant >> 20) & 1
    round_up_n = (truncated > halfway) | ((truncated == halfway) & (lsb == 1))
    normal_mant = (fp32_mant >> 20) + round_up_n.to(tl.int32)
    mant_ovf = normal_mant >= 8
    normal_exp_out = fp8_exp_normal + mant_ovf.to(tl.int32)
    normal_mant_out = tl.where(mant_ovf, 0, normal_mant)

    # ---- Subnormal path: shift the implicit-1 mantissa into the fp8 denormal. ----
    full_mant = 0x800000 | fp32_mant
    sub_shift = 141 - fp32_exp
    sub_shift_safe = tl.minimum(tl.maximum(sub_shift, 1), 31)
    one_i32 = tl.full((), 1, tl.int32)
    round_pos = sub_shift_safe - 1
    round_mask = one_i32 << round_pos
    sticky_mask = round_mask - 1
    sub_round_bit = (full_mant & round_mask) != 0
    sub_sticky = (full_mant & sticky_mask) != 0
    sub_truncated = full_mant >> sub_shift_safe
    sub_lsb = sub_truncated & 1
    sub_round_up = sub_round_bit & (sub_sticky | (sub_lsb == 1))
    sub_mant_out = sub_truncated + sub_round_up.to(tl.int32)
    sub_carry_to_normal = sub_mant_out >= 8
    sub_exp_final = tl.where(sub_carry_to_normal, 1, 0)
    sub_mant_final = tl.where(sub_carry_to_normal, 0, sub_mant_out & 0x7)

    out_exp = tl.where(is_subnorm, sub_exp_final, normal_exp_out)
    out_mant = tl.where(is_subnorm, sub_mant_final, normal_mant_out)
    out_exp = tl.where(is_zero, 0, out_exp)
    out_mant = tl.where(is_zero, 0, out_mant)
    out_exp = tl.minimum(tl.maximum(out_exp, 0), 15)
    out_mant = out_mant & 0x7

    return ((sign << 7) | (out_exp << 3) | out_mant).to(tl.uint8)


@triton.jit
def _quant_to_fp8(
    y,
    FP8_MAX: tl.constexpr,
    USE_FNUZ: tl.constexpr = _USE_FNUZ,
    SW_FP8: tl.constexpr = _SW_FP8_CAST,
):
    """Scaled fp32 -> fp8 elements, saturating at +-FP8_MAX. On SW_FP8
    targets the result is e4m3fn *bytes* (uint8) instead of an fp8 value."""
    y = tl.minimum(tl.maximum(y, -FP8_MAX), FP8_MAX)
    if SW_FP8:
        return _encode_e4m3fn(y)
    fp8_dtype = tl.float8e4b8 if USE_FNUZ else tl.float8e4nv
    return y.to(fp8_dtype)


@triton.jit
def _quant_to_int8(y, INT8_MAX: tl.constexpr = _INT8_MAX):
    """Scaled fp32 -> nearest int8, saturating at +-INT8_MAX (round-half-up,
    within 1 LSB of the torch reference used in the stage-A check)."""
    y = tl.minimum(tl.maximum(y, -INT8_MAX), INT8_MAX)
    return tl.math.floor(y + 0.5).to(tl.int8)


# ---------------------------------------------------------------------------
# Kernels: upstream signatures/bodies kept verbatim; only the quant branches
# differ (constexpr-defaulted knobs, so call sites match upstream exactly).
# ---------------------------------------------------------------------------


@triton.jit
def _index_k_insert(
    tok,
    k_ptr,
    norm_w_ptr,
    norm_b_ptr,
    norm_eps,
    slot_mapping_ptr,
    num_slots,
    cache_ptr,
    cache_scale_ptr,
    cache_block_size,
    cache_block_stride,
    HEAD_DIM: tl.constexpr,
    MX_SCALE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    USE_FNUZ: tl.constexpr = _USE_FNUZ,
    USE_INT8: tl.constexpr = _USE_INT8_INDEXER,
    INT8_MAX: tl.constexpr = _INT8_MAX,
):
    # Padded rows and (under DCP) tokens owned by another rank have no slot.
    slot = tl.load(slot_mapping_ptr + tok, mask=tok < num_slots, other=-1)
    if slot >= 0:
        offs = tl.arange(0, HEAD_DIM)
        x = tl.load(k_ptr + offs).to(tl.float32)
        mean = tl.sum(x, axis=0) / HEAD_DIM
        diff = x - mean
        rstd = tl.rsqrt(tl.sum(diff * diff, axis=0) / HEAD_DIM + norm_eps)
        w = tl.load(norm_w_ptr + offs).to(tl.float32)
        b = tl.load(norm_b_ptr + offs).to(tl.float32)
        # Round to bf16 like the unfused path so the cached bytes match.
        y = (diff * rstd * w + b).to(k_ptr.dtype.element_ty).to(tl.float32)

        # Same math as the CUDA indexer_k_quant_and_cache kernel.
        # PPU MODIFICATION: int8 quant (ue8m0 scale, raw byte cache) when the
        # target has no FP8 MMA; SW e4m3 encoder below capability 8.9.
        if USE_INT8:
            scale = _ue8m0_scale(tl.max(tl.abs(y), axis=0), 1e-4, INT8_MAX)
            qv = _quant_to_int8(tl.div_rn(y, scale), INT8_MAX)
            qv = qv.to(tl.uint8, bitcast=True)
        else:
            scale = _ue8m0_scale(tl.max(tl.abs(y), axis=0), 1e-4, FP8_MAX)
            qv = _quant_to_fp8(tl.div_rn(y, scale), FP8_MAX, USE_FNUZ)

        # Block layout: [block_size x HEAD_DIM fp8 values | block_size x 4B scale].
        block_start = (slot // cache_block_size) * cache_block_stride
        block_off = slot % cache_block_size
        tl.store(cache_ptr + block_start + block_off * HEAD_DIM + offs, qv)
        scale_byte = block_start + cache_block_size * HEAD_DIM + block_off * 4
        if MX_SCALE:
            scale = pack_ue8m0_x4(scale)
        tl.store(cache_scale_ptr + scale_byte // 4, scale)


@triton.jit
def _indexer_k_norm_quant_insert_kernel(
    k_ptr,
    k_stride,
    norm_w_ptr,
    norm_b_ptr,
    norm_eps,
    slot_mapping_ptr,
    num_slots,
    cache_ptr,
    cache_scale_ptr,
    cache_block_size,
    cache_block_stride,
    topk_ptr,
    topk_stride,
    HEAD_DIM: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_BLOCK: tl.constexpr,
    MX_SCALE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    USE_FNUZ: tl.constexpr = _USE_FNUZ,
):
    tok = tl.program_id(0).to(tl.int64)
    if tl.program_id(1) == 1:
        # Padding rows included.
        _fill_topk(topk_ptr + tok * topk_stride, TOPK, TOPK_BLOCK)
    else:
        _index_k_insert(
            tok,
            k_ptr + tok * k_stride,
            norm_w_ptr,
            norm_b_ptr,
            norm_eps,
            slot_mapping_ptr,
            num_slots,
            cache_ptr,
            cache_scale_ptr,
            cache_block_size,
            cache_block_stride,
            HEAD_DIM,
            MX_SCALE,
            FP8_MAX,
            USE_FNUZ,
        )


@triton.jit
def _indexer_q_quant_fold_weights_kernel(
    q_ptr,
    q_stride0,
    q_stride1,
    w_ptr,
    w_stride,
    q_fp8_ptr,
    q_fp8_stride0,
    q_fp8_stride1,
    w_out_ptr,
    w_out_stride,
    weight_scale,
    HEAD_DIM: tl.constexpr,
    FP8_MAX: tl.constexpr,
    USE_FNUZ: tl.constexpr = _USE_FNUZ,
    USE_INT8: tl.constexpr = _USE_INT8_INDEXER,
    INT8_MAX: tl.constexpr = _INT8_MAX,
):
    tok = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    offs = tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + tok * q_stride0 + head * q_stride1 + offs).to(tl.float32)
    # Same math as per_token_group_quant_fp8(use_ue8m0=True) (eps=1e-10).
    # PPU MODIFICATION: int8 quant when the target has no FP8 MMA.
    if USE_INT8:
        scale = _ue8m0_scale(tl.max(tl.abs(q), axis=0), 1e-10, INT8_MAX)
        q_fp8 = _quant_to_int8(tl.div_rn(q, scale), INT8_MAX)
    else:
        scale = _ue8m0_scale(tl.max(tl.abs(q), axis=0), 1e-10, FP8_MAX)
        q_fp8 = _quant_to_fp8(tl.div_rn(q, scale), FP8_MAX, USE_FNUZ)
    tl.store(q_fp8_ptr + tok * q_fp8_stride0 + head * q_fp8_stride1 + offs, q_fp8)
    w = tl.load(w_ptr + tok * w_stride + head).to(tl.float32)
    tl.store(w_out_ptr + tok * w_out_stride + head, w * scale * weight_scale)


@triton.jit
def _dsa_fused_prologue_kernel(
    q_c_ptr,
    q_c_stride,
    q_norm_w_ptr,
    q_out_ptr,
    kv_c_ptr,
    kv_c_stride,
    kv_norm_w_ptr,
    kv_out_ptr,
    k_pe_ptr,
    k_pe_stride,
    index_k_ptr,
    index_k_stride,
    k_norm_w_ptr,
    k_norm_b_ptr,
    eps,
    mla_slot_ptr,
    num_mla_slots,
    mla_cache_ptr,
    mla_scale_ptr,
    mla_rope_ptr,
    mla_block_size,
    mla_block_stride,
    mla_entry_stride,
    index_slot_ptr,
    num_index_slots,
    index_cache_ptr,
    index_scale_ptr,
    index_block_size,
    index_block_stride,
    topk_ptr,
    topk_stride,
    Q_DIM: tl.constexpr,
    Q_BLOCK: tl.constexpr,
    KV_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    INDEX_DIM: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_BLOCK: tl.constexpr,
    MX_SCALE: tl.constexpr,
    USE_PDL: tl.constexpr,
    FP8_MAX: tl.constexpr,
    USE_FNUZ: tl.constexpr = _USE_FNUZ,
):
    tok = tl.program_id(0).to(tl.int64)
    pid = tl.program_id(1)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    if pid == 0:
        _index_k_insert(
            tok,
            index_k_ptr + tok * index_k_stride,
            k_norm_w_ptr,
            k_norm_b_ptr,
            eps,
            index_slot_ptr,
            num_index_slots,
            index_cache_ptr,
            index_scale_ptr,
            index_block_size,
            index_block_stride,
            INDEX_DIM,
            MX_SCALE,
            FP8_MAX,
            USE_FNUZ,
        )
    elif pid == 1:
        # The normed row is written for every token (the prefill path reads it).
        offs = tl.arange(0, KV_DIM)
        y = _rms_norm_row(
            kv_c_ptr + tok * kv_c_stride, kv_norm_w_ptr, eps, KV_DIM, KV_DIM
        )
        y = y.to(kv_out_ptr.dtype.element_ty)
        tl.store(kv_out_ptr + tok * KV_DIM + offs, y)
        slot = tl.load(mla_slot_ptr + tok, mask=tok < num_mla_slots, other=-1)
        if slot >= 0:
            # fp8_ds_mla, same math as concat_and_cache_ds_mla: KV_DIM fp8 values
            # with power-of-two per-128 tile scales (fp32), then k_pe in bf16.
            base = (slot // mla_block_size) * mla_block_stride + (
                slot % mla_block_size
            ) * mla_entry_stride
            NUM_TILES: tl.constexpr = KV_DIM // 128
            tiles = tl.reshape(y.to(tl.float32), (NUM_TILES, 128))
            amax = tl.max(tl.abs(tiles), axis=1, keep_dims=True)
            scale = tl.math.exp2(
                tl.math.ceil(tl.math.log2(tl.maximum(amax / FP8_MAX, 1e-4)))
            )
            # PPU MODIFICATION: SW e4m3 encoder below capability 8.9.
            tl.store(
                mla_cache_ptr + base + offs,
                tl.reshape(_quant_to_fp8(tiles / scale, FP8_MAX, USE_FNUZ), (KV_DIM,)),
            )
            tl.store(
                mla_scale_ptr + base // 4 + KV_DIM // 4 + tl.arange(0, NUM_TILES),
                tl.reshape(scale, (NUM_TILES,)),
            )
            rope = tl.arange(0, ROPE_DIM)
            k_pe = tl.load(k_pe_ptr + tok * k_pe_stride + rope)
            tl.store(
                mla_rope_ptr + base // 2 + KV_DIM // 2 + 2 * NUM_TILES + rope, k_pe
            )
    elif pid == 2:
        offs = tl.arange(0, Q_BLOCK)
        y = _rms_norm_row(q_c_ptr + tok * q_c_stride, q_norm_w_ptr, eps, Q_DIM, Q_BLOCK)
        tl.store(
            q_out_ptr + tok * Q_DIM + offs,
            y.to(q_out_ptr.dtype.element_ty),
            mask=offs < Q_DIM,
        )
    else:
        # Padding rows included.
        _fill_topk(topk_ptr + tok * topk_stride, TOPK, TOPK_BLOCK)


# ---------------------------------------------------------------------------
# Replacement entry points: verbatim upstream launches; only the cache dtype
# / Q allocation lines differ (marked PPU MODIFICATION).
# ---------------------------------------------------------------------------


@patch(
    _DSA_MODULE,
    "indexer_k_norm_quant_insert",
    reason=(
        "Capability (8,0) targets (810E) cannot compile tl.float8e4nv, and "
        "targets without FP8 MMA need an int8 indexer cache, so the K "
        "LayerNorm+quant+insert launch must pass the cache as raw uint8. "
        "constexpr-gated; targets with FP8 support delegate to upstream."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream dsa_indexer selects the quant encoder through a platform hook.",
)
@functools.wraps(_upstream_k_insert)
def indexer_k_norm_quant_insert(
    k, norm_weight, norm_bias, norm_eps, slot_mapping, kv_cache,
    topk_indices_buffer, num_tokens,
):
    if not (_SW_FP8_CAST or _USE_INT8_INDEXER):
        return _upstream_k_insert(
            k, norm_weight, norm_bias, norm_eps, slot_mapping, kv_cache,
            topk_indices_buffer, num_tokens,
        )
    # Verbatim upstream body; only the marked line differs.
    if num_tokens == 0:
        return
    head_dim = k.shape[-1]
    assert k.stride(-1) == 1
    assert kv_cache.dim() == 3 and kv_cache.shape[-1] == head_dim + 4
    assert triton.next_power_of_2(head_dim) == head_dim
    mx_scale = _dsa_indexer.mqa_logits_use_mx_scales()
    u8_cache = kv_cache.view(torch.uint8)
    topk = topk_indices_buffer.shape[-1]
    _indexer_k_norm_quant_insert_kernel[(num_tokens, 2)](
        k,
        k.stride(0),
        norm_weight,
        norm_bias,
        norm_eps,
        slot_mapping,
        slot_mapping.shape[0],
        u8_cache,  # PPU MODIFICATION: raw bytes (SW encoder / int8), no fp8 view
        u8_cache.view(torch.int32 if mx_scale else torch.float32),
        u8_cache.shape[1],
        u8_cache.stride(0),
        topk_indices_buffer,
        topk_indices_buffer.stride(0),
        HEAD_DIM=head_dim,
        TOPK=topk,
        TOPK_BLOCK=min(1024, triton.next_power_of_2(topk)),
        MX_SCALE=mx_scale,
        FP8_MAX=_FP8_MAX,
        USE_FNUZ=_USE_FNUZ,
        num_warps=1,
    )


@patch(
    _DSA_MODULE,
    "indexer_q_quant_fold_weights",
    reason=(
        "Capability (8,0) targets (810E) cannot compile tl.float8e4nv, and "
        "targets without FP8 MMA must quantize the indexer Q to int8 (the "
        "downstream sparse_attn_indexer scoring expects it). constexpr-gated; "
        "targets with FP8 support delegate to upstream."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream dsa_indexer selects the quant encoder through a platform hook.",
)
@functools.wraps(_upstream_q_quant)
def indexer_q_quant_fold_weights(
    index_q, index_weights, weight_scale, padded_num_heads,
):
    if not (_SW_FP8_CAST or _USE_INT8_INDEXER):
        return _upstream_q_quant(index_q, index_weights, weight_scale, padded_num_heads)
    # Verbatim upstream body; only the marked lines differ.
    num_tokens, num_heads, head_dim = index_q.shape
    assert index_q.stride(-1) == 1
    assert index_weights.shape == (num_tokens, num_heads)
    assert triton.next_power_of_2(head_dim) == head_dim
    alloc = torch.empty if padded_num_heads == num_heads else torch.zeros
    q_fp8 = alloc(
        (num_tokens, padded_num_heads, head_dim),
        # PPU MODIFICATION: int8 Q when the target has no FP8 MMA.
        dtype=torch.int8 if _USE_INT8_INDEXER else _FP8_DTYPE,
        device=index_q.device,
    )
    weights = alloc(
        (num_tokens, padded_num_heads), dtype=torch.float32, device=index_q.device
    )
    if num_tokens == 0:
        return q_fp8, weights
    _indexer_q_quant_fold_weights_kernel[(num_tokens, num_heads)](
        index_q,
        index_q.stride(0),
        index_q.stride(1),
        index_weights,
        index_weights.stride(0),
        # PPU MODIFICATION: the SW encoder writes raw e4m3fn bytes.
        q_fp8 if (_USE_INT8_INDEXER or not _SW_FP8_CAST) else q_fp8.view(torch.uint8),
        q_fp8.stride(0),
        q_fp8.stride(1),
        weights,
        weights.stride(0),
        weight_scale,
        HEAD_DIM=head_dim,
        FP8_MAX=_FP8_MAX,
        USE_FNUZ=_USE_FNUZ,
        num_warps=1,
    )
    return q_fp8, weights


@patch(
    _DSA_MODULE,
    "dsa_fused_prologue",
    reason=(
        "Capability (8,0) targets (810E) cannot compile tl.float8e4nv, so the "
        "fused prologue's fp8_ds_mla and indexer-key inserts must quantize "
        "through the software e4m3 encoder (int8 for the key when FP8 MMA is "
        "missing) over raw uint8 cache views. constexpr-gated; targets with "
        "FP8 support delegate to upstream."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream dsa_indexer selects the quant encoder through a platform hook.",
)
@functools.wraps(_upstream_prologue)
def dsa_fused_prologue(
    q_c, kv_c, k_pe, index_k, q_norm_weight, kv_norm_weight, k_norm_weight,
    k_norm_bias, eps, topk_indices_buffer, mla_cache, mla_slot_mapping,
    index_cache, index_slot_mapping,
):
    if not (_SW_FP8_CAST or _USE_INT8_INDEXER):
        return _upstream_prologue(
            q_c, kv_c, k_pe, index_k, q_norm_weight, kv_norm_weight,
            k_norm_weight, k_norm_bias, eps, topk_indices_buffer, mla_cache,
            mla_slot_mapping, index_cache, index_slot_mapping,
        )
    # Verbatim upstream body; only the marked lines differ.
    num_tokens, q_dim = q_c.shape
    kv_dim, rope_dim, index_dim = kv_c.shape[1], k_pe.shape[1], index_k.shape[1]
    assert all(t.stride(-1) == 1 for t in (q_c, kv_c, k_pe, index_k))
    q_out = torch.empty((num_tokens, q_dim), dtype=q_c.dtype, device=q_c.device)
    kv_out = torch.empty((num_tokens, kv_dim), dtype=kv_c.dtype, device=kv_c.device)
    if num_tokens == 0:
        return q_out, kv_out

    # No cache (profile runs): an empty slot mapping disables the insert.
    def placeholder() -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.empty(1, 1, 4, dtype=torch.uint8, device=q_c.device),
            torch.empty(0, dtype=torch.int64, device=q_c.device),
        )

    if mla_cache is None or mla_slot_mapping is None:
        mla_cache, mla_slot_mapping = placeholder()
    if index_cache is None or index_slot_mapping is None:
        index_cache, index_slot_mapping = placeholder()
    mla = mla_cache.view(torch.uint8)
    index = index_cache.view(torch.uint8)
    mx_scale = _dsa_indexer.mqa_logits_use_mx_scales()
    topk = topk_indices_buffer.shape[-1]
    use_pdl = current_platform.is_arch_support_pdl()
    _dsa_fused_prologue_kernel[(num_tokens, 4)](
        q_c,
        q_c.stride(0),
        q_norm_weight,
        q_out,
        kv_c,
        kv_c.stride(0),
        kv_norm_weight,
        kv_out,
        k_pe,
        k_pe.stride(0),
        index_k,
        index_k.stride(0),
        k_norm_weight,
        k_norm_bias,
        eps,
        mla_slot_mapping,
        mla_slot_mapping.shape[0],
        # PPU MODIFICATION: raw bytes (SW encoder), no fp8 view
        mla if _SW_FP8_CAST else mla.view(_FP8_DTYPE),
        mla.view(torch.float32),
        mla.view(torch.bfloat16),
        mla.shape[1],
        mla.stride(0),
        mla.stride(1),
        index_slot_mapping,
        index_slot_mapping.shape[0],
        index,  # PPU MODIFICATION: raw bytes (SW encoder / int8), no fp8 view
        index.view(torch.int32 if mx_scale else torch.float32),
        index.shape[1],
        index.stride(0),
        topk_indices_buffer,
        topk_indices_buffer.stride(0),
        Q_DIM=q_dim,
        Q_BLOCK=triton.next_power_of_2(q_dim),
        KV_DIM=kv_dim,
        ROPE_DIM=rope_dim,
        INDEX_DIM=index_dim,
        TOPK=topk,
        TOPK_BLOCK=min(1024, triton.next_power_of_2(topk)),
        MX_SCALE=mx_scale,
        USE_PDL=use_pdl,
        FP8_MAX=_FP8_MAX,
        USE_FNUZ=_USE_FNUZ,
        launch_pdl=use_pdl,
    )
    return q_out, kv_out


# ---------------------------------------------------------------------------
# int8 scoring redirection in vllm.model_executor.layers.sparse_attn_indexer
# ---------------------------------------------------------------------------


def _install_int8_scoring() -> None:
    """Rebind the sparse_attn_indexer scoring helpers for the int8 path.

    The fork branches inside ``sparse_attn_indexer`` on
    ``q.dtype == torch.int8``; instead of copying that large function, rebind
    its three module-level helpers. ``mqa_logits_mx_inputs`` becomes identity
    (the int8 API takes the raw folded weights and ignores the Q scale), and
    the ``fp8_fp4_*`` shims translate to the deep_gemm int8 APIs, falling back
    to the real fp8 kernels when the caller still hands over FP8 tensors.
    """
    _orig_mx_inputs = _sparse.mqa_logits_mx_inputs
    _orig_fp8_mqa = _sparse.fp8_fp4_mqa_logits
    _orig_fp8_paged = _sparse.fp8_fp4_paged_mqa_logits

    @patch(
        _SPARSE_MODULE,
        "mqa_logits_mx_inputs",
        reason=(
            "On targets without FP8 MMA the int8 indexer path scores through "
            "deep_gemm.int8_mqa_logits, which takes the raw folded weights and "
            "no Q scale, so the mx-input transform would corrupt both. "
            "Installed only on those targets; other targets keep upstream."
        ),
        affected_versions=_AFFECTED,
        remove_when="upstream's int8 indexer path skips the mx transform itself.",
    )
    def mqa_logits_mx_inputs(q_quant, q_scale, weights):
        return q_scale, weights

    @patch(
        _SPARSE_MODULE,
        "fp8_fp4_mqa_logits",
        reason=(
            "Targets without FP8 MMA have no fp8 deep_gemm mqa_logits kernel; "
            "the int8 indexer's Q/K (int8 values, ue8m0 scales) score through "
            "deep_gemm.int8_mqa_logits with clean_logits=False, matching the "
            "fp8_fp4 contract of compressed rows with an uncleaned tail. "
            "FP8 tensors still fall back to the real fp8 kernel."
        ),
        affected_versions=_AFFECTED,
        remove_when="deep_gemm exposes a platform-dispatched mqa_logits entry point.",
    )
    def fp8_fp4_mqa_logits(
        q, k, weights, cu_seqlen_ks, cu_seqlen_ke, max_seqlen_k=None, **kwargs
    ):
        q_values = q[0] if isinstance(q, tuple) else q
        if q_values.dtype != torch.int8:
            return _orig_fp8_mqa(
                q, k, weights, cu_seqlen_ks, cu_seqlen_ke, max_seqlen_k, **kwargs
            )
        k_values, k_scales = (k[0], k[1]) if isinstance(k, tuple) else (k, None)
        return _int8_mqa_logits(
            q_values,
            (k_values.view(torch.int8), k_scales),
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            clean_logits=False,
            max_seqlen_k=max_seqlen_k,
        )

    @patch(
        _SPARSE_MODULE,
        "fp8_fp4_paged_mqa_logits",
        reason=(
            "Targets without FP8 MMA have no fp8 deep_gemm paged mqa_logits "
            "kernel; the int8 indexer's padded Q scores through "
            "deep_gemm.int8_paged_mqa_logits over the same uint8 cache view "
            "(int8 values + fp32 ue8m0 scale per token), preserving the "
            "[B*next_n, max_model_len] output contract with uncleaned tails. "
            "FP8 tensors still fall back to the real fp8 kernel."
        ),
        affected_versions=_AFFECTED,
        remove_when="deep_gemm exposes a platform-dispatched paged mqa_logits entry point.",
    )
    def fp8_fp4_paged_mqa_logits(
        q,
        kv_cache,
        weights,
        seq_lens,
        block_table,
        schedule_metadata,
        max_model_len=None,
        indices=None,
        **kwargs,
    ):
        q_values = q[0] if isinstance(q, tuple) else q
        if q_values.dtype != torch.int8:
            return _orig_fp8_paged(
                q,
                kv_cache,
                weights,
                seq_lens,
                block_table,
                schedule_metadata,
                max_model_len=max_model_len,
                indices=indices,
                **kwargs,
            )
        if block_table.dim() >= 2 and block_table.stride(-1) != 1:
            block_table = block_table.clone(memory_format=torch.contiguous_format)
        return _int8_paged_mqa_logits(
            q_values,
            kv_cache,
            weights,
            seq_lens,
            block_table,
            schedule_metadata,
            max_model_len,
            clean_logits=False,
        )


if _USE_INT8_INDEXER:
    _install_int8_scoring()
