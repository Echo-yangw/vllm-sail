# SPDX-License-Identifier: Apache-2.0
"""PPU patch: reorder the loads in ``_fused_qk_rmsnorm_rope_gate_kernel``.

Upstream runs the Qwen3.5/Qwen3.6 attention head as
``split -> RMSNorm -> RoPE -> gate chunk`` in one Triton launch. The reduction
(``tl.sum(x * x)``) is the long dependency at the head of that kernel, and
upstream issues the position loads, the cos/sin cache loads and the gate load
*after* it, so their memory latency is exposed serially. This patch hoists them
above the reduction, which lets the scheduler overlap them with the reduction
instead of stalling behind it.

Nothing else changes: same kernel name, same parameter list, same launcher, and
the arithmetic is untouched (the gate is still read from the ``[q|gate]`` pair
and stored through the same mask, the cos/sin values are converted to float32 at
first use instead of at load). The launcher and the kernel live in the same
module and the launcher resolves the kernel through module globals at launch
time, so rebinding the kernel attribute is enough -- the launcher is not
patched.

The reorder is platform-agnostic, so it is applied on every platform, not only
on PPU. ``@patch`` sits **outside** ``@triton.jit`` so the fully-decorated
Triton object is what gets installed; see ``vllm_sail/patch/README.md``.
"""

from __future__ import annotations

from vllm.triton_utils import tl, triton

from vllm_sail.patch.utils import patch

_AFFECTED = ">=0.32.0,<0.33.0"


@patch(
    "vllm.model_executor.layers.fused_qk_norm_rope",
    "_fused_qk_rmsnorm_rope_gate_kernel",
    reason=(
        "Upstream loads positions, the cos/sin cache and the gate after the "
        "RMSNorm reduction, so their latency is exposed behind the tl.sum "
        "dependency chain. Issue those loads above the reduction instead. "
        "Same signature, launcher and numerics."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "the pinned upstream kernel issues the position, cos/sin and gate "
        "loads before the RMSNorm reduction."
    ),
)
@triton.jit
def _fused_qk_rmsnorm_rope_gate_kernel(
    q_gate_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    gate_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    q_gate_stride_t,
    k_stride_t,
    q_out_stride_t,
    k_out_stride_t,
    gate_out_stride_t,
    cache_stride_p,
    positions_stride_m,
    positions_stride_t,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    half_rotary: tl.constexpr,
    eps: tl.constexpr,
    norm_beta: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    ROT_HALF_BLOCK: tl.constexpr,
    HAS_PASS: tl.constexpr,
    HAS_MROPE: tl.constexpr,
    MROPE_SECTION_H: tl.constexpr,
    MROPE_SECTION_W: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    is_k = head >= num_q_heads
    local_head = tl.where(is_k, head - num_q_heads, head)

    # PPU MODIFICATION: begin
    head_offs = tl.arange(0, HEAD_BLOCK)
    head_mask = head_offs < head_dim
    rot_offs = tl.arange(0, ROT_HALF_BLOCK)
    rot_mask = rot_offs < half_rotary

    # Start the position-dependent cache lookup before consuming the head input.
    pos_t = tl.load(positions_ptr + token * positions_stride_t)
    if HAS_MROPE:
        pos_h = tl.load(positions_ptr + positions_stride_m + token * positions_stride_t)
        pos_w = tl.load(
            positions_ptr + 2 * positions_stride_m + token * positions_stride_t
        )
    # PPU MODIFICATION: end

    if is_k:
        in_base = k_ptr + token * k_stride_t + local_head * head_dim
        w_ptr = k_weight_ptr
        out_base = k_out_ptr + token * k_out_stride_t + local_head * head_dim
    else:
        in_base = q_gate_ptr + token * q_gate_stride_t + local_head * 2 * head_dim
        w_ptr = q_weight_ptr
        out_base = q_out_ptr + token * q_out_stride_t + local_head * head_dim

    # PPU MODIFICATION: begin
    x_in = tl.load(in_base + head_offs, mask=head_mask, other=0.0)
    w_in = tl.load(w_ptr + head_offs, mask=head_mask, other=0.0)
    gate_mask = head_mask & ~is_k
    gate_in_base = (
        q_gate_ptr + token * q_gate_stride_t + local_head * 2 * head_dim + head_dim
    )
    g = tl.load(gate_in_base + head_offs, mask=gate_mask, other=0.0)

    # Convert positions before multiplying to avoid address overflow.
    pos_t = pos_t.to(tl.int64)
    if HAS_MROPE:
        pos_h = pos_h.to(tl.int64)
        pos_w = pos_w.to(tl.int64)
        is_h = (rot_offs % 3 == 1) & (rot_offs < 3 * MROPE_SECTION_H)
        is_w = (rot_offs % 3 == 2) & (rot_offs < 3 * MROPE_SECTION_W)
        pos = tl.where(is_h, pos_h, tl.where(is_w, pos_w, pos_t))
    else:
        pos = pos_t
    cache_offset = pos * cache_stride_p
    cos = tl.load(cos_sin_cache_ptr + cache_offset + rot_offs, mask=rot_mask, other=0.0)
    sin = tl.load(
        cos_sin_cache_ptr + cache_offset + half_rotary + rot_offs,
        mask=rot_mask,
        other=0.0,
    )

    # Compute RMSNorm after prefetching the RoPE coefficients.
    x = x_in.to(tl.float32)
    w = w_in.to(tl.float32) + norm_beta
    # PPU MODIFICATION: end
    var = tl.sum(x * x, axis=0) / head_dim
    inv_rms = tl.rsqrt(var + eps)
    # Round-trip through INPUT_DTYPE so the RoPE input matches the bf16-storage
    # behavior of the unfused (qk_rmsnorm -> memory -> apply_rope) reference path.
    x_norm = (x * inv_rms * w).to(INPUT_DTYPE).to(tl.float32)

    # --- Pass-through tail [rotary_dim, head_dim): RMSNorm-only, no rotation ---
    # The rotary head [0, rotary_dim) will be overwritten by the RoPE store below.
    if HAS_PASS:
        pass_mask = head_mask & (head_offs >= rotary_dim)
        tl.store(out_base + head_offs, x_norm, mask=pass_mask)

    # --- Partial RoPE on the first rotary_dim elements ---
    # PPU MODIFICATION: begin -- upstream's three-line recompute note, and its
    # rot_offs/rot_mask definition, moved above the reduction.
    # Reload the rotary halves on the smaller ROT_HALF_BLOCK layout.
    # PPU MODIFICATION: end
    x_rot1 = tl.load(in_base + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
    x_rot2 = tl.load(in_base + half_rotary + rot_offs, mask=rot_mask, other=0.0).to(
        tl.float32
    )
    w_rot1 = (
        tl.load(w_ptr + rot_offs, mask=rot_mask, other=0.0).to(tl.float32) + norm_beta
    )
    w_rot2 = (
        tl.load(w_ptr + half_rotary + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
        + norm_beta
    )
    x_rot1 = (x_rot1 * inv_rms * w_rot1).to(INPUT_DTYPE).to(tl.float32)
    x_rot2 = (x_rot2 * inv_rms * w_rot2).to(INPUT_DTYPE).to(tl.float32)

    # PPU MODIFICATION: begin
    cos = cos.to(tl.float32)
    sin = sin.to(tl.float32)
    # PPU MODIFICATION: end
    o1 = x_rot1 * cos - x_rot2 * sin
    o2 = x_rot2 * cos + x_rot1 * sin
    tl.store(out_base + rot_offs, o1, mask=rot_mask)
    tl.store(out_base + half_rotary + rot_offs, o2, mask=rot_mask)

    # PPU MODIFICATION: begin
    gate_out_base = gate_out_ptr + token * gate_out_stride_t + local_head * head_dim
    tl.store(gate_out_base + head_offs, g, mask=gate_mask)
    # PPU MODIFICATION: end
