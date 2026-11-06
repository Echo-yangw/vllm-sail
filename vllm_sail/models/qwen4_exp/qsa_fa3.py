# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp QSA kernel policy; the wheel and device are queried only by callers.

Mirrors :mod:`vllm_sail.attention.gdn_decode`: the decision itself is pure and
torch-free so ``tests/ut`` covers it on a bare interpreter, while the probes that
need a platform, a ``flash_attn_3`` wheel or an allocation live in the caller
(:mod:`vllm_sail.patch.enhancement.models.qwen4_exp_qsa`) and hand their verdict
in as ``cuda_reason``.
"""

from __future__ import annotations

from dataclasses import dataclass

from vllm_sail import envs

#: FA3 instantiates QSA for head dim 256 only.
QSA_FA3_HEAD_DIM = 256
#: ... and rejects more than 32 query heads per KV head.
QSA_FA3_MAX_QHEADS_PER_KVHEAD = 32
#: FA3 requires K and V to share Q's dtype, and Q is BF16. An FP8 cache reaches
#: that check as a C++ abort, so it is rejected here instead. The Triton kernel
#: does dequantize e4m3 on load; FA3's QSA path does not.
QSA_FA3_KV_CACHE_DTYPES = ("auto", "bfloat16")


@dataclass(frozen=True)
class QSAFA3Kernel:
    """Which QSA kernel a layer shape resolved to, and why."""

    requested: str
    active: str
    reason: str | None = None

    @property
    def use_fa3(self) -> bool:
        return self.active == "cuda"


def qsa_fa3_selection_width(text_config) -> int | None:
    """Selection columns per row, or ``None`` if the config does not say.

    Mirrors ``QSAIndexer.output_width``: the indexer expands each of its
    ``indexer_budget`` compressed selections into ``indexer_compress_ratio``
    tokens, overlapping by one at each boundary. The owner's packed buffer adds a
    trailing count column on top of this.
    """
    budget = getattr(text_config, "indexer_budget", None)
    ratio = getattr(text_config, "indexer_compress_ratio", None)
    if not isinstance(budget, int) or not isinstance(ratio, int):
        return None
    if budget <= 0 or ratio <= 0:
        return None
    return budget + ratio - 1


def qsa_fa3_shape_reason(
    head_size: int, num_heads: int, num_kv_heads: int
) -> str | None:
    """Why FA3's QSA kernel rejects these head shapes, or ``None`` if it accepts."""
    if head_size != QSA_FA3_HEAD_DIM:
        return f"FA3 QSA takes head dim {QSA_FA3_HEAD_DIM} only, got {head_size}"
    if num_kv_heads <= 0 or num_heads % num_kv_heads:
        return f"invalid GQA ratio: {num_heads} query over {num_kv_heads} KV heads"
    group = num_heads // num_kv_heads
    if group > QSA_FA3_MAX_QHEADS_PER_KVHEAD:
        return (
            f"FA3 QSA supports at most {QSA_FA3_MAX_QHEADS_PER_KVHEAD} query heads "
            f"per KV head, got {group}"
        )
    return None


def qsa_fa3_kv_dtype_reason(kv_cache_dtype: str) -> str | None:
    """Why FA3's QSA cannot read this main KV cache, or ``None`` if it can."""
    if kv_cache_dtype in QSA_FA3_KV_CACHE_DTYPES:
        return None
    return (
        "FA3 QSA needs a BF16 main KV cache (K and V must share Q's dtype), got "
        f"kv_cache_dtype={kv_cache_dtype!r}"
    )


def resolve_qsa_fa3_kernel(requested: str, *, cuda_reason: str | None) -> QSAFA3Kernel:
    """Resolve a request against FA3's verdict for one layer shape.

    ``auto`` takes FA3 whenever it can run this config and Triton otherwise;
    ``cuda`` and ``triton`` pin either side for A/B measurement. An explicit
    ``cuda`` that cannot run raises rather than silently falling back, matching
    :func:`vllm_sail.attention.gdn_decode.resolve_gdn_decode_backend`: a pinned
    choice is a measurement, and a quiet fallback would make it a lie.

    Args:
        requested: ``envs.VLLM_SAIL_QSA_KERNEL``.
        cuda_reason: why FA3 cannot run this shape, or ``None`` if it can. The
            caller supplies it because answering needs the platform, the wheel
            and -- for the workspace -- an allocation.
    """
    if requested not in envs.QSA_KERNELS:
        raise ValueError(f"Invalid QSA kernel: {requested}")
    if requested == "triton":
        return QSAFA3Kernel(requested, "triton", "VLLM_SAIL_QSA_KERNEL=triton")
    if cuda_reason is not None and requested == "cuda":
        raise ValueError(f"VLLM_SAIL_QSA_KERNEL=cuda is not supported: {cuda_reason}")
    return QSAFA3Kernel(
        requested, "cuda" if cuda_reason is None else "triton", cuda_reason
    )


__all__ = [
    "QSA_FA3_HEAD_DIM",
    "QSA_FA3_KV_CACHE_DTYPES",
    "QSA_FA3_MAX_QHEADS_PER_KVHEAD",
    "QSAFA3Kernel",
    "qsa_fa3_kv_dtype_reason",
    "qsa_fa3_selection_width",
    "qsa_fa3_shape_reason",
    "resolve_qsa_fa3_kernel",
]
