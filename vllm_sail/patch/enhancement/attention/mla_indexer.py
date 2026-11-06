# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F821
"""PPU patches for ``vllm.v1.attention.backends.mla.indexer`` (DSA indexer).

The fork's changes, translated to plugin patches. Everything else is left to
upstream: ``__init__`` delegates to the original method and re-applies only the
fork's PPU state, ``build`` delegates untouched, so upstream drift (new
helpers, new parameters, refactors) cannot break this plugin.

* Two module-level constants added by the fork (``PPU_DEEP_GEMM_TB_PER_CU``,
  ``PPU_FP4_INDEXER_ELEMENT_SIZE``) — installed with ``patch_value``.
* The fork's import switch (``vllm.utils.ppu_deep_gemm`` instead of
  ``vllm.utils.deep_gemm`` under ``is_ppu()``): ``has_deep_gemm`` is an
  identity (same object); ``get_paged_mqa_logits_metadata`` is rebound on the
  target module to the PPU implementation. PPU declares
  ``PlatformEnum.CUDA``, so upstream's decode build DOES enter its
  ``is_cuda() and has_deep_gemm()`` block; the rebound helper derives the
  fork's ``metadata_extra`` from the builder currently inside ``build``, so
  upstream's own buffer fill already uses the fp4 layout.
* ``_split_indexer_prefill_chunks`` derives a logits-dtype-aware budget
  from the builder's fp4 capability (PPU-only delta: both upstreams budget
  a fixed 4 bytes/elem), so the fp4 indexer (bf16 logits) chunks against
  the correct logits budget.
* ``DeepseekV32IndexerMetadataBuilder.__init__`` delegates to upstream, then
  adds the fork's PPU-only state: pins ``supports_varlen`` off (PPU declares
  ``PlatformEnum.CUDA``, so the fork's fixed-length paged-logits metadata
  layout must not be bypassed), ``indexer_n_head``/``indexer_q_head_dim``
  from the KV-cache spec, the SM count from PPU DeepGEMM, and the enlarged
  fp4 scheduler-metadata buffer.
* ``DeepseekV32IndexerMetadataBuilder.build`` delegates to upstream, keeping
  the builder on ``_current_builder`` for the duration of the call so the
  static chunk splitter and the rebound paged-metadata helper can read the
  fork's fp4 capability. Upstream owns the whole step build.
* ``DeepseekV32IndexerMetadataBuilder.update_draft_decode_metadata`` (the MTP
  draft refresh) is the other call site that reaches the rebound helper, so it
  keeps the builder on ``_current_builder`` the same way; without it the fp4
  scheduler metadata would be rebuilt with the fp8 layout and fail upstream's
  shape assertion.
"""

from __future__ import annotations

import torch
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla import indexer as _indexer_module
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadataBuilder,
    DeepseekV32IndexerMetadata,
)

import vllm_sail.utils.deep_gemm as _ppu_deep_gemm
from vllm_sail.patch.utils import patch, patch_value

_AFFECTED = ">=0.32.0,<0.33.0"
_MODULE = "vllm.v1.attention.backends.mla.indexer"

#: Builder currently inside ``build``. The static chunk splitter and the
#: paged-MQA-logits metadata helper read the fork's fp4 capability from it
#: (neither upstream passes those values explicitly).
_current_builder: list = []


patch_value(
    _MODULE,
    "PPU_DEEP_GEMM_TB_PER_CU",
    8,
    allow_missing=True,
    reason=(
        "Fork constant: thread blocks per compute unit used by PPU DeepGEMM "
        "when sizing the fp4 indexer scheduler-metadata buffer."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream defines the constant itself (patch_value becomes a no-op).",
)

patch_value(
    _MODULE,
    "PPU_FP4_INDEXER_ELEMENT_SIZE",
    1,
    allow_missing=True,
    reason=(
        "Fork constant: element size (bytes) of the PPU fp4 indexer cache, "
        "passed to the PPU paged MQA logits metadata builder via metadata_extra."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream defines the constant itself (patch_value becomes a no-op).",
)

# Fork's import switch: on PPU the indexer takes its DeepGEMM helpers from the
# PPU DeepGEMM wrapper. Only get_paged_mqa_logits_metadata needs rebinding —
# the plugin's wrapper re-exports the very same has_deep_gemm object upstream
# already binds, so that half of the switch is an identity. The patched build()
# and update_draft_decode_metadata() below keep their builder on
# _current_builder, so upstream's calls resolve the rebound name at call time
# with the fork's fp4 geometry available.
_upstream_paged_metadata = _indexer_module.get_paged_mqa_logits_metadata


def _ppu_paged_metadata(
    context_lens, block_size, num_sms, metadata_extra=None, indices=None
):
    from vllm.platforms import current_platform

    if not current_platform.is_ppu():
        return _upstream_paged_metadata(
            context_lens, block_size, num_sms, indices=indices
        )
    if indices is not None:
        raise NotImplementedError(
            "SAIL DeepGEMM's paged MQA logits metadata is fixed-length layout only; "
            "disable decode flattening (e.g. fewer speculative tokens) on PPU"
        )
    if metadata_extra is None:
        # PPU declares PlatformEnum.CUDA, so upstream's decode build does reach
        # its is_cuda() block and calls this helper without metadata_extra.
        # Derive the fork's tuple from the builder inside build, so upstream's
        # own fill already uses the fp4 layout (instead of a redundant second
        # pass over the decode scheduler metadata). Every PPU-reachable call
        # site (build and update_draft_decode_metadata) runs behind a wrapper
        # that puts its builder on the stack.
        builder = _current_builder[-1] if _current_builder else None
        n_head = getattr(builder, "indexer_n_head", 0) if builder else 0
        if n_head > 0:
            heads_per_token = (
                context_lens.shape[1] if context_lens.dim() == 2 else 1
            )
            metadata_extra = (
                heads_per_token,
                n_head,
                builder.indexer_q_head_dim,
                _indexer_module.PPU_FP4_INDEXER_ELEMENT_SIZE,
            )
    return _ppu_deep_gemm.get_paged_mqa_logits_metadata(
        context_lens, block_size, num_sms, metadata_extra
    )


patch(
    _MODULE,
    "get_paged_mqa_logits_metadata",
    reason=(
        "Fork's is_ppu() import switch selects the PPU DeepGEMM wrapper's "
        "get_paged_mqa_logits_metadata, which accepts the PPU metadata_extra "
        "tuple. On PPU the rebound helper also derives metadata_extra from the "
        "builder inside build (PPU declares PlatformEnum.CUDA, so upstream's "
        "decode build reaches the same call), keeping the fp4 scheduler "
        "metadata correct in upstream's single fill. The CUDA helper has no "
        "such parameter."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "vllm.utils.deep_gemm dispatches on current_platform internally, or "
        "upstream accepts a DeepGEMM backend registry."
    ),
)(_ppu_paged_metadata)


_upstream_supports_flattened_device_query_lens = (
    _indexer_module._supports_flattened_device_query_lens
)


def _supports_flattened_device_query_lens() -> bool:
    from vllm.platforms import current_platform

    if current_platform.is_ppu():
        # SAIL fixed-length paged logits consumes one expanded query per row.
        return _ppu_deep_gemm.is_deep_gemm_supported()
    return _upstream_supports_flattened_device_query_lens()


patch(
    _MODULE,
    "_supports_flattened_device_query_lens",
    reason="Use device query lengths for adaptive V4.1 verification with SAIL fixed-length paged logits.",
    affected_versions=_AFFECTED,
    remove_when="The indexer delegates flattened device-query-length support to the platform.",
)(_supports_flattened_device_query_lens)


def _split_indexer_prefill_chunks_body(
    compressed_seq_lens_cpu: torch.Tensor,
    prefill_query_lens_cpu: torch.Tensor,
    workspace_size: int,
    max_logits_bytes: int,
    request_offset: int = 0,
) -> list[tuple[slice, slice]]:
    """Upstream chunker with the fork's logits_dtype budget.

    PPU-only delta: neither upstream passes ``logits_dtype`` (both budget a
    fixed 4 bytes/elem), so the dtype is derived from the builder currently
    inside ``build`` (see ``_current_builder``). bf16 (fp4 indexer) halves
    max_logits_elems; fp32 matches upstream. Falls back to fp32 outside
    ``build``.
    """
    chunks: list[tuple[slice, slice]] = []
    n = len(compressed_seq_lens_cpu)
    # PPU MODIFICATION: begin
    _builder = _current_builder[-1] if _current_builder else None
    logits_dtype = (
        torch.bfloat16
        if _builder is not None and getattr(_builder, "indexer_uses_fp4", False)
        else torch.float32
    )
    bytes_per_elem = torch.empty((), dtype=logits_dtype).element_size()
    max_logits_elems = max(1, max_logits_bytes // bytes_per_elem)
    # PPU MODIFICATION: end
    end = 0

    while end < n:
        start, chunk_m, chunk_n = end, 0, 0

        while end < n:
            q, s = (
                prefill_query_lens_cpu[end].item(),
                compressed_seq_lens_cpu[end].item(),
            )
            new_m, new_n = chunk_m + q, chunk_n + s
            if new_n <= workspace_size and new_m * new_n <= max_logits_elems:
                chunk_m, chunk_n = new_m, new_n
                end += 1
            else:
                break

        # A single request can exceed the budget, requiring sub-chunking
        # on the query dimension.
        if end == start:
            chunk_m, chunk_n = (
                prefill_query_lens_cpu[end].item(),
                compressed_seq_lens_cpu[end].item(),
            )
            end += 1

        req_slice = slice(start + request_offset, end + request_offset)
        max_q = max(1, max_logits_elems // chunk_n) if chunk_n > 0 else max(1, chunk_m)
        for q_off in range(0, chunk_m, max_q):
            sub_m = min(max_q, chunk_m - q_off)
            chunks.append((req_slice, slice(q_off, q_off + sub_m)))

    return chunks


patch(
    _MODULE,
    "DeepseekV32IndexerMetadataBuilder._split_indexer_prefill_chunks",
    reason=(
        "Fork chunks the fp4 indexer cache (bf16 logits) against the correct "
        "logits budget. Neither upstream passes logits_dtype (both budget a "
        "fixed 4 bytes/elem), so the dtype derives from the builder currently "
        "inside build (_current_builder). The max(1, ...) clamp differs from "
        "upstream, which can yield 0."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream grows a logits-dtype-aware chunk budget.",
)(staticmethod(_split_indexer_prefill_chunks_body))


_upstream_uses_fp4 = _indexer_module.dsa_indexer_uses_fp4


@patch(
    _MODULE,
    "dsa_indexer_uses_fp4",
    reason=(
        "PPU MXFP4 indexer support is supplied by SAIL DeepGEMM, not CUDA "
        "SM100, and sm_89 defaults indexer_kv_dtype to mxfp4."
    ),
    affected_versions=_AFFECTED,
    remove_when="The indexer exposes a platform capability hook.",
)
def dsa_indexer_uses_fp4(vllm_config):
    from vllm.platforms import current_platform

    if not current_platform.is_ppu():
        return _upstream_uses_fp4(vllm_config)
    # PPU MODIFICATION: begin -- sm_89 (ZW-890P) defaults to the MXFP4 indexer
    # cache. It halves the per-token K cache (132B -> 68B) and keeps the whole
    # indexer on the upstream Triton MXFP4 route instead of the INT8 quantising
    # kernels that only exist for PPU 1.0. Upstream's fp8 default is kept on
    # every other capability, where mxfp4 is rejected just below.
    default_kv_dtype = "mxfp4" if current_platform.is_device_capability(89) else "fp8"
    # PPU MODIFICATION: end
    kv_dtype = vllm_config.attention_config.resolve_indexer_kv_dtype(default_kv_dtype)
    if kv_dtype not in _indexer_module.DSA_INDEXER_KV_DTYPES:
        raise ValueError(f"Unsupported PPU indexer_kv_dtype={kv_dtype!r}")
    if kv_dtype == "mxfp4" and not current_platform.is_device_capability(89):
        raise ValueError("PPU indexer_kv_dtype='mxfp4' requires capability sm_89")
    return kv_dtype == "mxfp4"


# Model modules can capture the capability helper before general plugins load.
import sys

for _consumer_name in (
    "vllm.model_executor.models.deepseek_v2",
    "vllm.models.deepseek_v4.attention",
):
    _consumer = sys.modules.get(_consumer_name)
    if (
        _consumer is not None
        and getattr(_consumer, "dsa_indexer_uses_fp4", None) is _upstream_uses_fp4
    ):
        patch(
            _consumer_name,
            "dsa_indexer_uses_fp4",
            reason="Preloaded indexer consumers must use SAIL's MXFP4 capability.",
            affected_versions=_AFFECTED,
            remove_when="Consumers resolve the helper through its provider module.",
        )(dsa_indexer_uses_fp4)


_upstream_builder_init = DeepseekV32IndexerMetadataBuilder.__init__


def _builder_init_wrapper(self, *args, **kwargs) -> None:
    """Delegate to upstream, then add the fork's PPU-only builder state."""
    _upstream_builder_init(self, *args, **kwargs)
    # PPU MODIFICATION: begin -- fork-only builder state for the MXFP4
    # indexer (sm_89): head geometry from the KV-cache spec, SM count from
    # PPU DeepGEMM, and the enlarged fp4 scheduler-metadata buffer.
    from vllm.platforms import current_platform

    if not current_platform.is_ppu():
        return
    # PPU declares PlatformEnum.CUDA, so upstream's varlen probe would pass
    # its is_cuda() gate and only fail on the capability check. Pin it off:
    # SAIL's paged logits API uses the existing fixed-length metadata layout.
    self.supports_varlen = False
    self.indexer_n_head = (
        getattr(self.kv_cache_spec, "indexer_n_head", 0) or 0
        if self.indexer_uses_fp4
        else 0
    )
    self.indexer_q_head_dim = (
        getattr(self.kv_cache_spec, "indexer_q_head_dim", 0) or 0
        if self.indexer_uses_fp4
        else 0
    )
    from vllm_sail.utils.deep_gemm import get_num_sms

    self.num_sms = get_num_sms()
    if self.indexer_uses_fp4:
        # NOTE(kai): fp4 paged uses extra metadata
        # (num_sms * tb_per_cu + 1, 2)
        self.scheduler_metadata_buffer = torch.empty(
            (self.num_sms * _indexer_module.PPU_DEEP_GEMM_TB_PER_CU + 1, 2),
            dtype=torch.int32,
            device=self.device,
        )
    # PPU MODIFICATION: end


patch(
    _MODULE,
    "DeepseekV32IndexerMetadataBuilder.__init__",
    reason=(
        "PPU-only builder state from the fork: pin supports_varlen off (PPU "
        "declares PlatformEnum.CUDA, so the upstream varlen probe passes its "
        "is_cuda() gate), indexer_n_head/indexer_q_head_dim "
        "read from the KV-cache spec, SM count from PPU DeepGEMM's get_num_sms, "
        "and the enlarged num_sms * PPU_DEEP_GEMM_TB_PER_CU scheduler-metadata "
        "buffer for fp4 paged logits. Delegates to the upstream __init__ so "
        "upstream buffer/refactor drift cannot break this plugin."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream grows PPU branches in the DSA indexer metadata builder.",
)(_builder_init_wrapper)


_upstream_build = DeepseekV32IndexerMetadataBuilder.build


def _build_wrapper(
    self,
    common_prefix_len: int,
    common_attn_metadata: CommonAttentionMetadata,
    fast_build: bool = False,
) -> DeepseekV32IndexerMetadata:
    """Delegate to upstream with the builder visible to the static helpers.

    Upstream owns the whole step build, including the fp4 paged-MQA-logits
    scheduler metadata: PPU declares ``PlatformEnum.CUDA``, so its decode
    block runs here and the rebound ``get_paged_mqa_logits_metadata`` adds the
    fork's ``metadata_extra`` itself.
    """
    _current_builder.append(self)
    try:
        return _upstream_build(
            self, common_prefix_len, common_attn_metadata, fast_build
        )
    finally:
        _current_builder.pop()


patch(
    _MODULE,
    "DeepseekV32IndexerMetadataBuilder.build",
    reason=(
        "Keeps the builder on _current_builder while upstream builds the step, "
        "so the static chunk splitter and the rebound paged-MQA-logits "
        "metadata helper can read the fork's fp4 capability. Upstream keeps "
        "ownership of the step build (including TP prefill row sharding), so "
        "upstream drift cannot break this plugin."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream exposes the builder to the static chunk splitter and paged "
        "metadata helper without module-level context."
    ),
)(_build_wrapper)


_upstream_update_draft_decode = (
    DeepseekV32IndexerMetadataBuilder.update_draft_decode_metadata
)


def _update_draft_decode_wrapper(self, metadata) -> None:
    """Delegate to upstream with the builder visible to the static helpers."""
    _current_builder.append(self)
    try:
        return _upstream_update_draft_decode(self, metadata)
    finally:
        _current_builder.pop()


patch(
    _MODULE,
    "DeepseekV32IndexerMetadataBuilder.update_draft_decode_metadata",
    reason=(
        "Second PPU-reachable caller of the rebound paged-MQA-logits metadata "
        "helper (the MTP draft refresh). Keeps the builder on "
        "_current_builder so the helper re-derives the fork's fp4 "
        "metadata_extra; otherwise the fp4 scheduler metadata would be "
        "refreshed in the fp8 layout and upstream's shape assertion would fail."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream exposes the builder to the paged metadata helper without "
        "module-level context."
    ),
)(_update_draft_decode_wrapper)
