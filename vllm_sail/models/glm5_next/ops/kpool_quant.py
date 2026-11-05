# SPDX-License-Identifier: Apache-2.0
"""INT8 Hadamard-128 query quantization for the glm5_next kpool indexer.

The CUDA model rotates every indexer query with a Hadamard-128 transform and
stores it as FP8 (``fwht128_quant_fp8`` in
``vllm.models.glm5next.nvidia.ops.kpool_compress``). PPU 1.0 Triton cannot
compile the ``fp8e4nv`` store, and PPU DeepGEMM's indexer consumes INT8 Q/K, so
this module provides the INT8 counterpart with the same rotation:

* :func:`hadamard128` builds the exact ``1 / sqrt(128)`` Hadamard matrix in
  fp32. The CUDA butterfly computes the same transform in fp32 and rounds to
  bf16 before quantizing; :func:`rotate128` reproduces that round-off.
* :func:`fwht128_quant_int8` feeds the rotated rows through
  ``per_token_group_quant_int8`` -- the same quantizer PPU's K-cache path uses
  (``vllm_sail.attention.ops.mla_sparse.indexer_k_quant_and_cache_int8``).

Queries and pooled keys must use the same matrix on a host. H is orthogonal,
so the rotation leaves their dot product unchanged; only the quantization
error of the two sides differs from the CUDA fp8 path.
"""

from __future__ import annotations

import torch
from vllm.model_executor.layers.quantization.utils.int8_utils import (
    per_token_group_quant_int8,
)

HEAD_DIM = 128

# One matrix per device; entries are exact powers of two, so the cache is
# invariant across processes and dtypes (always fp32).
_HADAMARD: dict[str, torch.Tensor] = {}


def hadamard128(device: torch.device | str) -> torch.Tensor:
    """Return the ``[128, 128]`` Hadamard matrix scaled by ``1/sqrt(128)``."""
    key = str(device)
    matrix = _HADAMARD.get(key)
    if matrix is None:
        matrix = torch.ones((1, 1), dtype=torch.float32, device=device)
        while matrix.shape[0] < HEAD_DIM:
            matrix = torch.cat(
                [
                    torch.cat([matrix, matrix], dim=1),
                    torch.cat([matrix, -matrix], dim=1),
                ],
                dim=0,
            )
        matrix = matrix * (HEAD_DIM**-0.5)
        _HADAMARD[key] = matrix
    return matrix


def rotate128(x: torch.Tensor) -> torch.Tensor:
    """Rotate ``[rows, 128]`` rows into the Hadamard basis (fp32, bf16-rounded).

    Mirrors the CUDA kernel's precision contract: fp32 butterfly, then a bf16
    materialization, then back to fp32 for the scale reduction.
    """
    assert x.ndim == 2 and x.shape[1] == HEAD_DIM, x.shape
    rotated = x.to(torch.float32) @ hadamard128(x.device)
    return rotated.to(torch.bfloat16).to(torch.float32)


def fwht128_quant_int8(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate and INT8-quantize a ``[rows, 128]`` bf16 query block.

    Args:
        q: ``[rows, 128]`` bf16, one head vector per row (already contiguous).

    Returns:
        ``(q_int8 [rows, 128] int8, q_scale [rows, 1] float32)``. The scale is
        the one produced by ``per_token_group_quant_int8`` (``max(absmax,
        eps) / 127``) and is folded into the logits weights by the model.
    """
    assert q.ndim == 2 and q.shape[1] == HEAD_DIM, q.shape
    assert q.dtype == torch.bfloat16, q.dtype
    return per_token_group_quant_int8(rotate128(q.contiguous()), HEAD_DIM)
