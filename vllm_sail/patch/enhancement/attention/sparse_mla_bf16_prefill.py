# SPDX-License-Identifier: Apache-2.0
"""PPU patch: force the sparse MLA prefill onto the BF16 sparse kernel.

The sparse FlashMLA backend normally keeps prefill and decode in one
mixed-batch call (``flash_mla_with_kvcache``) whenever the per-rank query head
count is below ``MIN_HEADS_FOR_BF16_PREFILL``. Setting
``VLLM_SAIL_SPARSE_MLA_BF16_PREFILL=1`` moves prefill onto the dedicated BF16
sparse kernel (``flash_mla_sparse_fwd``) instead, for accuracy checks and for
A/B benchmarking the two kernels.

With the switch unset (the default) every patch here is a pure delegation to
upstream, so behaviour is identical to the unpatched backend.

Three patches, all on ``vllm.v1.attention.backends.mla.flashmla_sparse``:

* ``use_fp8_mixed_batch`` - module-level predicate consulted by both the
  metadata builder and the impl, so patching the module attribute covers both
  call sites.
* ``FlashMLASparseImpl.__init__`` - clears ``use_shardq`` afterwards: shardq
  only exists on the mixed-batch fp8 decode kernel, and
  ``MLAAttention.__init__`` reads ``impl.use_shardq`` right after construction.
  The DCP configuration the separate path cannot serve is already rejected by
  upstream's metadata builder, which calls ``use_fp8_mixed_batch`` as well.
* ``FlashMLASparseImpl.prefill_padding`` - a data descriptor that narrows the
  head padding from the flat 64 to PPU's actual prefill head tile
  (kBlockM = 16/32/64). It has to be a descriptor rather than an assignment
  after ``__init__``: the value sizes the prefill workspace inside ``__init__``
  (the padded query/out tensors), so a late change would leave the workspace
  wider than the kernel call and trip the contiguous/shape guards.
"""

from __future__ import annotations

import torch
from vllm.v1.attention.backends.mla import flashmla_sparse as _flashmla_sparse_backend

from vllm_sail import envs as sail_envs
from vllm_sail.patch.utils import patch

_AFFECTED = ">=0.32.0,<0.33.0"
_MODULE = "vllm.v1.attention.backends.mla.flashmla_sparse"

#: Upstream pads the prefill query heads to 64 rows (``self.prefill_padding =
#: 64``), the cute kernel's widest query-head tile.
_UPSTREAM_PREFILL_PADDING = 64

_upstream_use_fp8_mixed_batch = _flashmla_sparse_backend.use_fp8_mixed_batch
_upstream_impl_init = _flashmla_sparse_backend.FlashMLASparseImpl.__init__


def _prefill_head_align(num_heads: int, upstream_align: int) -> int:
    """PPU prefill head tile for ``num_heads`` (upstream's flat 64 when off)."""
    if not sail_envs.VLLM_SAIL_SPARSE_MLA_BF16_PREFILL or num_heads <= 0:
        return upstream_align
    if num_heads <= 16:
        return 16
    if num_heads <= 32:
        return 32
    return upstream_align


@patch(
    _MODULE,
    "use_fp8_mixed_batch",
    reason=(
        "Optional forced-BF16 sparse prefill (VLLM_SAIL_SPARSE_MLA_BF16_PREFILL). "
        "Returning False routes prefill to the BF16 sparse kernel instead of the "
        "mixed-batch fp8 decode kernel. Both the metadata builder and the impl "
        "call this module-level predicate, so one patch covers both."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream grows a first-class switch for forcing the BF16 sparse prefill "
        "kernel, or PPU stops padding prefill heads to a flat 64."
    ),
)
def use_fp8_mixed_batch(
    num_heads: int, use_hisparse: bool, pcp_dcp_kv_gather: bool
) -> bool:
    if sail_envs.VLLM_SAIL_SPARSE_MLA_BF16_PREFILL:
        return False
    return _upstream_use_fp8_mixed_batch(num_heads, use_hisparse, pcp_dcp_kv_gather)


@patch(
    _MODULE,
    "FlashMLASparseImpl.__init__",
    reason=(
        "Pair the forced-BF16 sparse prefill switch with the shardq gate: shardq "
        "only exists on the mixed-batch fp8 decode kernel, so it must stay off "
        "when the switch moves prefill elsewhere."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream grows a first-class switch for forcing the BF16 sparse prefill "
        "kernel, at which point its own shardq gating applies."
    ),
)
def impl_init(
    self,
    num_heads: int,
    head_size: int,
    scale: float,
    num_kv_heads: int,
    alibi_slopes,
    sliding_window: int,
    kv_cache_dtype,
    logits_soft_cap,
    attn_type,
    kv_sharing_target_layer_name,
    topk_indices_buffer: torch.Tensor | None = None,
    indexer=None,
    **mla_args,
) -> None:
    _upstream_impl_init(
        self,
        num_heads,
        head_size,
        scale,
        num_kv_heads,
        alibi_slopes,
        sliding_window,
        kv_cache_dtype,
        logits_soft_cap,
        attn_type,
        kv_sharing_target_layer_name,
        topk_indices_buffer=topk_indices_buffer,
        indexer=indexer,
        **mla_args,
    )
    if sail_envs.VLLM_SAIL_SPARSE_MLA_BF16_PREFILL:
        # MLAAttention.__init__ reads impl.use_shardq right after this returns,
        # so clearing it here is equivalent to gating it in upstream's __init__.
        self.use_shardq = False


def _get_prefill_padding(self) -> int:
    return self.__dict__.get("_sail_prefill_padding", _UPSTREAM_PREFILL_PADDING)


def _set_prefill_padding(self, value: int) -> None:
    self.__dict__["_sail_prefill_padding"] = _prefill_head_align(
        getattr(self, "num_heads", 0), value
    )


patch(
    _MODULE,
    "FlashMLASparseImpl.prefill_padding",
    reason=(
        "Narrow the prefill head padding from the flat 64 to PPU's actual query "
        "head tile (kBlockM = 16/32/64) when VLLM_SAIL_SPARSE_MLA_BF16_PREFILL is "
        "on. Upstream's 64 is the cute kernel's widest tile, so with 16 or 32 "
        "heads the padded rows are pure waste. A data descriptor is required "
        "because the value sizes the prefill workspace inside __init__."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream derives the prefill head padding from the actual head count "
        "instead of hard-coding 64, or PPU's kernel stops tiling by head count."
    ),
    allow_missing=True,
)(
    property(
        _get_prefill_padding,
        _set_prefill_padding,
        doc=(
            "Prefill query-head padding: upstream's 64 unless the forced-BF16 "
            "prefill switch is on, in which case it follows PPU's head tile."
        ),
    )
)
