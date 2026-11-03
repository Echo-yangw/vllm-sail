# SPDX-License-Identifier: Apache-2.0
"""Numerical conversion contract, independent of vLLM and device kernels."""

import torch


def dequantize_fp8(weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Decode 128x128 E4M3 blocks with FP32 multiply and BF16 rounding.

    One row block at a time bounds FP32 scratch space. In particular, fused
    gate/up tensors must call this separately for their independent grids.
    """
    if weight.ndim != 2 or weight.dtype != torch.float8_e4m3fn:
        raise ValueError("Expected a two-dimensional E4M3 weight")
    rows, cols = weight.shape
    if tuple(scales.shape) != ((rows + 127) // 128, (cols + 127) // 128):
        raise ValueError("FP8 scale grid does not match weight shape")
    if not bool(torch.isfinite(scales).all()) or not bool((scales > 0).all()):
        raise ValueError("FP8 scales must be loaded, finite and positive")
    result = torch.empty_like(weight, dtype=torch.bfloat16)
    for start in range(0, rows, 128):
        block = weight[start : start + 128].float()
        scale = scales[start // 128].float().repeat_interleave(128)[:cols]
        result[start : start + 128] = (block * scale).to(torch.bfloat16)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("FP8 conversion produced non-finite weights")
    return result


def quantize_int8(
    weight: torch.Tensor, *, tp_group=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row INT8; optionally reduce full-row maxima across TP."""
    if weight.ndim != 2 or weight.dtype != torch.bfloat16:
        raise ValueError("INT8 conversion expects a BF16 matrix")
    values = weight.float()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Cannot quantize non-finite weights")
    maxima = values.abs().amax(dim=1, keepdim=True)
    if tp_group is not None:
        groups = tp_group if isinstance(tp_group, tuple) else (tp_group,)
        for group in groups:
            torch.distributed.all_reduce(
                maxima, op=torch.distributed.ReduceOp.MAX, group=group
            )
    scales = (maxima / 127.0).clamp_min(torch.finfo(torch.float32).eps)
    quantized = (values / scales).round().clamp(-127, 127).to(torch.int8)
    return quantized, scales
