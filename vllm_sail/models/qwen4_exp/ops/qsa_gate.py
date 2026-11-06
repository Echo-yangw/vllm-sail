# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Apply the QSA output gate in place, matching the Triton kernel bit for bit.

The Qwen4Exp QSA owner no longer applies the gate itself: it folds
``output_gate`` into ``qsa_sparse_paged_attention``, whose epilogue rounds in a
specific order (``ops/qsa.py``, both the single-split store and the split-KV
merge)::

    normalized_output = normalized_output.to(output_ptr.dtype.element_ty)  # bf16
    output_gate = tl.load(...).to(tl.float32)
    normalized_output = normalized_output.to(tl.float32) * tl.sigmoid(output_gate)
    tl.store(..., normalized_output, ...)

FA3's bf16 store of the attention output *is* the first rounding, so reloading
it here and gating in fp32 reproduces the same value. Gating in bf16, or using
``torch.sigmoid`` instead of ``tl.sigmoid``, would not.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton


@triton.jit
def _qsa_apply_gate_kernel(
    out_ptr,
    gate_ptr,
    stride_out_row,
    stride_out_head,
    stride_gate_row,
    stride_gate_head,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
) -> None:
    """Gate one token row's whole attention output in place."""
    row = tl.program_id(0)
    dims = tl.arange(0, HEAD_DIM)
    for head in tl.static_range(0, NUM_HEADS):
        out_off = row * stride_out_row + head * stride_out_head + dims
        gate_off = row * stride_gate_row + head * stride_gate_head + dims
        attn = tl.load(out_ptr + out_off).to(tl.float32)
        gate = tl.load(gate_ptr + gate_off).to(tl.float32)
        tl.store(out_ptr + out_off, attn * tl.sigmoid(gate))


def qsa_apply_output_gate(output: torch.Tensor, output_gate: torch.Tensor) -> None:
    """Multiply ``output`` by ``sigmoid(output_gate)`` in place.

    Args:
        output: ``(num_tokens, num_heads, head_dim)`` bf16, the attention result
            FA3 already stored. Modified in place.
        output_gate: bf16, pre-sigmoid, either already ``(num_tokens, num_heads,
            head_dim)`` or the flat ``(num_tokens, num_heads * head_dim)`` the
            unfused projection emits. Reshaped with ``view_as(output)``, as
            upstream's Triton launcher does.
    """
    if not HAS_TRITON:
        raise RuntimeError("QSA output gating requires Triton")
    num_tokens, num_heads, head_dim = output.shape
    if num_tokens == 0:
        return
    assert output.dtype == torch.bfloat16, output.dtype
    assert output_gate.dtype == torch.bfloat16, output_gate.dtype
    assert output_gate.is_contiguous(), "QSA output gate must be contiguous"
    assert head_dim & (head_dim - 1) == 0, f"head_dim must be a power of 2: {head_dim}"
    gate = output_gate.view_as(output)

    _qsa_apply_gate_kernel[(num_tokens,)](
        output,
        gate,
        output.stride(0),
        output.stride(1),
        gate.stride(0),
        gate.stride(1),
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        num_warps=4,
    )


__all__ = ["qsa_apply_output_gate"]
