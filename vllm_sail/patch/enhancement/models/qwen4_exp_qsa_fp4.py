# SPDX-License-Identifier: Apache-2.0
"""Serve the Qwen4Exp QSA indexer's compressed key cache as MXFP4 on PPU.

DeepGEMM's FP4 average MQA logits kernels score the QSA indexer's compressed
keys at roughly half the cache bandwidth of upstream's Triton kernels, but they
only read MXFP4: E2M1 values two per byte plus one UE8M0 scale per 32 logical
dims, stored planar within a 64-row page. Upstream stores those rows as bf16 or
plain e4m3 and scores them with Triton, so four things have to move together:

* ``QSACompressedKeyCache`` describes its rows as ``[68] uint8`` instead of
  ``[128] bf16``. ``_QSAIndexerCache.bind_kv_cache`` already reads
  ``self.dtype`` / ``self.head_size``, so the subclass only needs those two
  attributes set before it registers itself.
* ``QSAIndexerBackend`` reports a kernel block size that is a multiple of
  ``64 * compress_ratio``, so one framework block holds a whole number of FP4
  pages. This is what pushes a 1600-token budget to 1792 (7 * 256).
* The fused pre-indexer kernel packs MXFP4 into the planar page instead of
  writing a flat row. ``vllm_sail.models.qwen4_exp.ops.qsa_prepare_fp4`` holds
  the copy; only the compressed-row store differs.
* Decode and prefill scoring delegate to DeepGEMM — the paged kernel for a
  uniform decode batch, the non-paged kernel for prefill, which is the only
  entry that takes a compressed ``max_seqlen_k``.

Queries stay bf16 and are packed inside the scoring functions, so
``QSAIndexer.forward`` needs no PPU-specific arguments.

The raw key cache is untouched: it feeds group pooling and is never scored.

FP8 is not offered as a fallback. A build without DeepGEMM's FP4 average entry
points fails at model load rather than silently serving a slower path.
"""

from __future__ import annotations

import torch
from vllm.models.qwen4_exp.common.qsa_indexer import _QSAIndexerCache
from vllm.models.qwen4_exp.nvidia.ops import qsa_indexer as _qsa_indexer_ops
from vllm.models.qwen4_exp.nvidia.ops import qsa_prepare as _qsa_prepare_ops
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

from vllm_sail.models.qwen4_exp.ops.fp4_select import (
    qsa_select_paged_decode as _fp4_select_paged_decode,
)
from vllm_sail.models.qwen4_exp.ops.fp4_select import (
    qsa_select_paged_prefill as _fp4_select_paged_prefill,
)
from vllm_sail.models.qwen4_exp.ops.fp4_select import (
    require_fp4_mqa_avg_logits,
)
from vllm_sail.models.qwen4_exp.ops.fp4_select import (
    warmup_qsa_mqa_paged_decode as _fp4_warmup_paged_decode,
)
from vllm_sail.models.qwen4_exp.ops.mxfp4 import MXFP4_PAGE_ROWS, MXFP4_ROW_BYTES
from vllm_sail.models.qwen4_exp.ops.qsa_prepare_fp4 import (
    qsa_prepare as _fp4_qsa_prepare,
)
from vllm_sail.patch.utils import patch

_upstream_qsa_prepare = _qsa_prepare_ops.qsa_prepare
_upstream_select_paged_decode = _qsa_indexer_ops.qsa_select_paged_decode
_upstream_select_paged_prefill = _qsa_indexer_ops.qsa_select_paged_prefill
_upstream_warmup_paged_decode = _qsa_indexer_ops.warmup_qsa_mqa_paged_decode

_AFFECTED = ">=0.32.0,<0.33.0"

_FP8_REJECTED = (
    "The Qwen4Exp QSA indexer stores its compressed key cache as MXFP4 on PPU, "
    "so --indexer-kv-dtype fp8 has no effect here. Leave it at its bf16 "
    "default; only the cache encoding changes, not the indexer's arithmetic."
)


@patch(
    "vllm.models.qwen4_exp.common.qsa_indexer",
    "QSACompressedKeyCache.__init__",
    reason=(
        "DeepGEMM's FP4 average MQA logits kernels read the QSA compressed key "
        "cache as MXFP4: 64 packed value bytes plus one int32 of UE8M0 scales "
        "per 128-dim row. Upstream describes those rows as [128] bf16 or plain "
        "e4m3, which no FP4 kernel can score. _QSAIndexerCache.bind_kv_cache "
        "already validates against self.dtype and self.head_size, so setting "
        "them here is enough to bind a [blocks, rows, 1, 68] uint8 cache."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream QSACompressedKeyCache accepts a packed sub-byte row width "
        "(an MXFP4/UE8M0 entry in its indexer_kv_dtype resolution), so the "
        "subclass can describe the FP4 page without a plugin __init__."
    ),
    allow_missing=True,
)
def __init__(
    self,
    *,
    head_size: int,
    dtype: torch.dtype,
    cache_config,
    prefix: str,
    vllm_config,
    compress_ratio: int = 1,
) -> None:
    from vllm.platforms import current_platform

    if not current_platform.is_ppu():
        # Upstream defines no subclass __init__, so this is plain inheritance.
        _QSAIndexerCache.__init__(
            self,
            head_size=head_size,
            dtype=dtype,
            cache_config=cache_config,
            prefix=prefix,
            vllm_config=vllm_config,
            compress_ratio=compress_ratio,
        )
        return
    if dtype == torch.float8_e4m3fn:
        raise NotImplementedError(_FP8_REJECTED)
    require_fp4_mqa_avg_logits()
    _QSAIndexerCache.__init__(
        self,
        head_size=MXFP4_ROW_BYTES,
        dtype=torch.uint8,
        cache_config=cache_config,
        prefix=prefix,
        vllm_config=vllm_config,
        compress_ratio=compress_ratio,
    )


@patch(
    "vllm.models.qwen4_exp.common.qsa_indexer",
    "QSAIndexerBackend.get_supported_kernel_block_sizes",
    reason=(
        "The FP4 paged kernel scores whole 64-row pages, so one framework block "
        "of the compressed cache must hold a whole number of them. With "
        "compress_ratio groups per row that makes the block size a multiple of "
        "64 * compress_ratio; upstream declares no constraint for this backend "
        "and inherits MultipleOf(1)."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream's FP4-capable indexer backends declare their page-derived "
        "kernel block size through get_supported_kernel_block_sizes."
    ),
    allow_missing=True,
)
# An additive class patch installs the replacement verbatim, so upstream's
# staticmethod descriptor has to be redeclared here.
@staticmethod
def get_supported_kernel_block_sizes(
    kv_cache_spec: KVCacheSpec | None = None,
) -> list[int | MultipleOf]:
    from vllm.platforms import current_platform

    if not current_platform.is_ppu():
        return [MultipleOf(1)]
    if not isinstance(kv_cache_spec, MLAAttentionSpec):
        # The raw key ring is bf16 and never reaches an FP4 kernel.
        return [MultipleOf(1)]
    return [MultipleOf(MXFP4_PAGE_ROWS * int(kv_cache_spec.tokens_per_state))]


@patch(
    "vllm.models.qwen4_exp.nvidia.ops.qsa_prepare",
    "qsa_prepare",
    reason=(
        "The fused pre-indexer kernel writes each compressed key row flat at "
        "row * stride, which cannot address DeepGEMM's planar FP4 page (all of "
        "a page's packed values, then all of its UE8M0 scales). The plugin copy "
        "packs MXFP4 at the store and is otherwise verbatim."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream's _qsa_prepare_kernel takes the compressed cache's packed row "
        "geometry as a parameter, so a planar MXFP4 store needs no copied body."
    ),
)
@patch(
    "vllm.models.qwen4_exp.nvidia.qsa_indexer",
    "qsa_prepare",
    reason=(
        "Rebind the alias QSAIndexer imported at module scope, so the indexer "
        "calls the MXFP4-aware prepare instead of the upstream one."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream's _qsa_prepare_kernel takes the compressed cache's packed row "
        "geometry as a parameter, so a planar MXFP4 store needs no copied body."
    ),
)
def qsa_prepare(*args, **kwargs):
    from vllm.platforms import current_platform

    if not current_platform.is_ppu():
        return _upstream_qsa_prepare(*args, **kwargs)
    return _fp4_qsa_prepare(*args, **kwargs)


@patch(
    "vllm.models.qwen4_exp.nvidia.ops.qsa_indexer",
    "qsa_select_paged_decode",
    reason=(
        "Score the MXFP4 compressed cache with DeepGEMM's paged FP4 average MQA "
        "logits kernel instead of upstream's Triton uniform-decode kernel, which "
        "cannot read packed FP4 rows."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream's qsa_select_paged_decode dispatches on the compressed cache "
        "dtype, so a packed FP4 cache reaches an FP4 scorer without a patch."
    ),
)
def qsa_select_paged_decode(*args, **kwargs):
    from vllm.platforms import current_platform

    if not current_platform.is_ppu():
        return _upstream_select_paged_decode(*args, **kwargs)
    return _fp4_select_paged_decode(*args, **kwargs)


@patch(
    "vllm.models.qwen4_exp.nvidia.ops.qsa_indexer",
    "qsa_select_paged_prefill",
    reason=(
        "Score the MXFP4 compressed cache with DeepGEMM's non-paged FP4 average "
        "MQA logits kernel. It is the only entry that takes a compressed "
        "max_seqlen_k, which bounds the transient logits by the longest context "
        "instead of by the whole packed key space."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream's qsa_select_paged_prefill dispatches on the compressed cache "
        "dtype, so a packed FP4 cache reaches an FP4 scorer without a patch."
    ),
)
def qsa_select_paged_prefill(*args, **kwargs):
    from vllm.platforms import current_platform

    if not current_platform.is_ppu():
        return _upstream_select_paged_prefill(*args, **kwargs)
    return _fp4_select_paged_prefill(*args, **kwargs)


@patch(
    "vllm.models.qwen4_exp.nvidia.ops.qsa_indexer",
    "warmup_qsa_mqa_paged_decode",
    reason=(
        "Upstream compiles its Triton decode specializations here. On PPU the "
        "scoring kernels come from DeepGEMM and JIT at first call, which CUDA "
        "graph capture cannot tolerate, so warm the FP4 paged, non-paged and "
        "MXFP4 packing kernels for every capturable decode query length."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream's QSA warmup dispatches on the compressed cache dtype, so an "
        "FP4 cache warms its own kernels without a patch."
    ),
)
def warmup_qsa_mqa_paged_decode(*args, **kwargs):
    from vllm.platforms import current_platform

    if not current_platform.is_ppu():
        return _upstream_warmup_paged_decode(*args, **kwargs)
    return _fp4_warmup_paged_decode(*args, **kwargs)
