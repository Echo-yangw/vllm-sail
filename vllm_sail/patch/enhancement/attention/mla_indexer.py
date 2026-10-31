# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F821
# Copied bodies resolve globals in their upstream module through bind_body.
"""PPU patches for ``vllm.v1.attention.backends.mla.indexer`` (DSA indexer).

The fork's changes, translated to plugin patches:

* Two module-level constants added by the fork (``PPU_DEEP_GEMM_TB_PER_CU``,
  ``PPU_FP4_INDEXER_ELEMENT_SIZE``) — installed with ``patch_value``.
* The fork's import switch (``vllm.utils.ppu_deep_gemm`` instead of
  ``vllm.utils.deep_gemm`` under ``is_ppu()``) affects two names.
  ``has_deep_gemm`` needs no patch: ``vllm_sail.utils.deep_gemm`` re-exports the
  very same object upstream already binds, so that half of the switch is an
  identity. ``get_paged_mqa_logits_metadata`` IS rebound on the target module
  at the plugin's PPU implementation (which accepts the PPU ``metadata_extra``
  tuple), the same shape as ``enhancement/deep_gemm_redirect.py``. On PPU the
  patched ``build`` below always takes the PPU branch, so the rebound name
  cannot reach a CUDA-only call signature.
* ``split_indexer_prefill_chunks`` gains a ``logits_dtype`` parameter. NOTE:
  this is UNCONDITIONAL in the fork (it also changes the stock default by
  clamping ``max_logits_elems`` to >= 1); reproduced as-in-fork and flagged in
  the report. Verbatim body with the fork's lines marked.
* ``DeepseekV32IndexerMetadataBuilder.__init__`` gains PPU branches for the fp4
  capability assert (sm_89), ``indexer_n_head``/``indexer_q_head_dim`` from the
  KV-cache spec, the SM count (``get_num_sms``), and the enlarged fp4
  scheduler-metadata buffer. Delegation cannot intercept the upstream assert,
  hence the verbatim body.
* ``DeepseekV32IndexerMetadataBuilder.build`` passes ``logits_dtype`` to the
  chunk splitter and builds PPU paged-MQA-logits scheduler metadata with the
  fork's ``metadata_extra`` tuple. Verbatim body with the fork's lines marked.

The copied bodies are rebased onto the target module's ``__dict__`` before
installation so every free name resolves exactly as it does upstream.
"""

from __future__ import annotations

import torch
from vllm.v1.attention.backends.mla import indexer as _indexer_module
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadataBuilder

import vllm_sail.utils.deep_gemm as _ppu_deep_gemm
from vllm_sail.patch.bodies import bind_body
from vllm_sail.patch.utils import patch, patch_value

_AFFECTED = ">=0.32.0,<0.33.0"
_MODULE = "vllm.v1.attention.backends.mla.indexer"


def _with_target_globals(fn):
    return bind_body(fn, _indexer_module)


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
# below resolves the rebound name through module globals at call time.
_upstream_paged_metadata = _indexer_module.get_paged_mqa_logits_metadata


def _ppu_paged_metadata(
    context_lens, block_size, num_sms, metadata_extra=None, indices=None
):
    from vllm.platforms import current_platform

    if current_platform.is_ppu():
        if indices is not None:
            raise NotImplementedError(
                "SAIL DeepGEMM uses fixed-length indexer metadata"
            )
        return _ppu_deep_gemm.get_paged_mqa_logits_metadata(
            context_lens, block_size, num_sms, metadata_extra
        )
    return _upstream_paged_metadata(context_lens, block_size, num_sms, indices=indices)


patch(
    _MODULE,
    "get_paged_mqa_logits_metadata",
    reason=(
        "Fork's is_ppu() import switch selects the PPU DeepGEMM wrapper's "
        "get_paged_mqa_logits_metadata, which accepts the PPU metadata_extra "
        "tuple. The CUDA helper has no such parameter."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "vllm.utils.deep_gemm dispatches on current_platform internally, or "
        "upstream accepts a DeepGEMM backend registry."
    ),
)(_ppu_paged_metadata)


def _split_indexer_prefill_chunks_body(
    compressed_seq_lens_cpu: torch.Tensor,
    prefill_query_lens_cpu: torch.Tensor,
    workspace_size: int,
    max_logits_bytes: int,
    request_offset: int = 0,
    # PPU MODIFICATION: begin
    logits_dtype: torch.dtype = torch.float32,
    # PPU MODIFICATION: end
) -> list[tuple[slice, slice]]:
    """Split this step's prefill requests into chunks, respecting:
    - N constraint: total_seq_lens <= workspace_size (existing O(N)
      workspace)
    - Logits constraint: M * N * 4 <= max_logits_bytes

    When a single request-level chunk still exceeds the logits budget,
    sub-chunks on the query dimension (M) to bound peak memory.

    Returns list of (req_slice, query_slice) tuples.
    """
    chunks: list[tuple[slice, slice]] = []
    n = len(compressed_seq_lens_cpu)
    # PPU MODIFICATION: begin
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
        "Fork adds a logits_dtype parameter so the fp4 indexer cache (bf16 "
        "logits) chunks against the correct logits budget. UNCONDITIONAL in the "
        "fork: the default path also changes (max_logits_elems clamped to >= 1); "
        "reproduced as-in-fork and flagged in the Phase-4 report."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream adds the logits_dtype parameter itself.",
)(_with_target_globals(_split_indexer_prefill_chunks_body))


_upstream_uses_fp4 = _indexer_module.dsa_indexer_uses_fp4


@patch(
    _MODULE,
    "dsa_indexer_uses_fp4",
    reason="PPU MXFP4 indexer support is supplied by SAIL DeepGEMM, not CUDA SM100.",
    affected_versions=_AFFECTED,
    remove_when="The indexer exposes a platform capability hook.",
)
def dsa_indexer_uses_fp4(vllm_config):
    from vllm.platforms import current_platform

    if not current_platform.is_ppu():
        return _upstream_uses_fp4(vllm_config)
    kv_dtype = vllm_config.attention_config.resolve_indexer_kv_dtype("fp8")
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


def _builder_init_body(self, *args, block_table_width: int, **kwargs) -> None:
    # PPU MODIFICATION: begin
    # This copied body is defined outside a class, so it has no __class__
    # closure for zero-argument super(). Resolve the owner in target globals.
    super(DeepseekV32IndexerMetadataBuilder, self).__init__(*args, **kwargs)
    # PPU MODIFICATION: end
    scheduler_config = self.vllm_config.scheduler_config
    parallel_config = self.vllm_config.parallel_config
    self.dcp_world_size = parallel_config.decode_context_parallel_size
    self.tp_size = parallel_config.tensor_parallel_size
    self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
    self.pcp_world_size = parallel_config.prefill_context_parallel_size
    self.use_pcp = self.pcp_world_size > 1
    self.pcp_rank = get_pcp_group().rank_in_group if self.use_pcp else 0
    self.cp_kv_cache_interleave_size = parallel_config.cp_kv_cache_interleave_size
    # KV compression (DeepseekV4). Default to 1 for no compression.
    self.compress_ratio = 1
    if isinstance(self.kv_cache_spec, MLAAttentionSpec):
        assert isinstance(self.kv_cache_spec.tokens_per_state, int)
        self.compress_ratio = self.kv_cache_spec.tokens_per_state
    # NOTE(Chen):an estimated max size of flattened_kv. Need to double check.
    # Counted in compressed rows, like the chunker's seq_lens and the
    # workspace.
    self.max_prefill_buffer_size = (
        get_max_prefill_buffer_size(self.vllm_config) // self.compress_ratio
    )
    self.num_speculative_tokens = (
        self.vllm_config.speculative_config.num_speculative_tokens
        if self.vllm_config.speculative_config
        else 0
    )
    self.use_adaptive_verification = bool(
        self.vllm_config.speculative_config
        and self.vllm_config.speculative_config.enable_adaptive_verification
    )
    self.indexer_uses_fp4 = dsa_indexer_uses_fp4(self.vllm_config)
    # PPU MODIFICATION: begin
    self.indexer_n_head = (
        self.kv_cache_spec.indexer_n_head or 0 if self.indexer_uses_fp4 else 0
    )
    self.indexer_q_head_dim = (
        self.kv_cache_spec.indexer_q_head_dim or 0 if self.indexer_uses_fp4 else 0
    )
    # PPU MODIFICATION: end

    next_n = self.num_speculative_tokens + 1
    self.decode_threshold = next_n
    self.reorder_batch_threshold = None
    self.use_flattening = _use_flattening(self.vllm_config)
    # PPU MODIFICATION: begin
    # SAIL's paged logits API uses the existing fixed-length metadata layout.
    self.supports_varlen = (
        False if current_platform.is_ppu() else _supports_varlen_paged_mqa_logits()
    )
    # PPU MODIFICATION: end
    logger.info_once(
        "DSA indexer decode path: use_flattening=%s supports_varlen=%s "
        "(next_n=%d, use_fp4_cache=%s)",
        self.use_flattening,
        self.supports_varlen,
        next_n,
        self.indexer_uses_fp4,
    )

    # PPU MODIFICATION: begin
    if current_platform.is_ppu():
        from vllm_sail.utils.deep_gemm import get_num_sms

        sm_count = get_num_sms()
    else:
        sm_count = num_compute_units(self.device.index)
    # PPU MODIFICATION: end
    self.num_sms = sm_count

    self.offsets_buffer = torch.arange(next_n, device=self.device, dtype=torch.int32)
    self.decode_lens_buffer = torch.zeros(
        (scheduler_config.max_num_batched_tokens,),
        dtype=torch.int32,
        device=self.device,
    )
    self.per_req_decode_lens_buffer = torch.zeros(
        (scheduler_config.max_num_batched_tokens,),
        dtype=torch.int32,
        device=self.device,
    )
    # Shared workspace for decode seq_lens. Native MTP views this as
    # (B, max_decode_len) at runtime, keeping context_lens contiguous even
    # when max_decode_len is smaller than next_n.
    self.decode_seq_lens_buffer = torch.zeros(
        (scheduler_config.max_num_batched_tokens,),
        dtype=torch.int32,
        device=self.device,
    )
    self.global_decode_seq_lens_buffer = torch.zeros(
        (scheduler_config.max_num_batched_tokens,),
        dtype=torch.int32,
        device=self.device,
    )
    self.decode_indices_buffer = torch.zeros(
        (scheduler_config.max_num_batched_tokens,),
        dtype=torch.int32,
        device=self.device,
    )
    self.arange_buffer = torch.arange(
        max(
            scheduler_config.max_num_seqs * next_n,
            scheduler_config.max_num_batched_tokens,
        ),
        dtype=torch.int32,
        device=self.device,
    )
    # Materialize the rank on device during builder initialization. Creating
    # this scalar in build() would introduce a GPU<->CPU sync in the decode
    # hot path.
    self.dcp_rank_tensor = torch.tensor(
        self.dcp_rank, dtype=torch.int32, device=self.device
    )
    max_decode_tokens = min(
        scheduler_config.max_num_seqs * next_n,
        scheduler_config.max_num_batched_tokens,
    )
    self.expanded_block_table_buffer = torch.zeros(
        (max_decode_tokens, block_table_width),
        dtype=torch.int32,
        device=self.device,
    )

    # See: DeepGEMM/csrc/apis/attention.hpp
    # PPU MODIFICATION: begin
    if current_platform.is_ppu() and self.indexer_uses_fp4:
        # NOTE(kai): fp4 paged using extra metadata
        # (num_sms * tb_per_cu + 1, 2)
        self.scheduler_metadata_buffer = torch.empty(
            (self.num_sms * PPU_DEEP_GEMM_TB_PER_CU + 1, 2),
            dtype=torch.int32,
            device=self.device,
        )
    else:
        self.scheduler_metadata_buffer = torch.empty(
            (self.num_sms + 1, 2), dtype=torch.int32, device=self.device
        )
    # PPU MODIFICATION: end

    # Each compressed state must live on a single rank, so the DCP
    # interleave has to cover whole states.
    if (
        self.dcp_world_size > 1
        and self.compress_ratio > 1
        and (
            (
                isinstance(self.kv_cache_spec, MLAAttentionSpec)
                and self.kv_cache_spec.model_version == "deepseek_v4"
            )
            or self.cp_kv_cache_interleave_size % self.compress_ratio != 0
        )
    ):
        raise NotImplementedError(
            "DCP with sparse indexer KV compression "
            f"(compress_ratio={self.compress_ratio}) requires "
            "--cp-kv-cache-interleave-size to be a multiple of it; "
            "DeepSeek-V4 is not supported."
        )
    # DCP interleave in units of compressed states.
    self.compressed_cp_interleave_size = (
        self.cp_kv_cache_interleave_size // self.compress_ratio
        if self.dcp_world_size > 1 and self.compress_ratio > 1
        else self.cp_kv_cache_interleave_size
    )

    # Pre-allocate buffers for CUDA graph compatibility when
    if self.compress_ratio > 1:
        # compress_ratio > 1 (DeepseekV4)
        # Compressed slot mapping output buffer
        self.compressed_slot_mapping_buffer = torch.zeros(
            (scheduler_config.max_num_batched_tokens,),
            dtype=torch.int64,
            device=self.device,
        )
        # Buffer for compressed seq_lens in decode path
        self.expanded_seq_lens_buffer = torch.zeros(
            (scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=self.device,
        )
    # DCP not supported yet
    self.supports_draft_decode_metadata_update = self.dcp_world_size == 1


patch(
    _MODULE,
    "DeepseekV32IndexerMetadataBuilder.__init__",
    reason=(
        "PPU branches from the fork: fp4 indexer cache allowed on sm_89, "
        "indexer_n_head/indexer_q_head_dim read from the KV-cache spec on PPU, "
        "SM count from PPU DeepGEMM's get_num_sms, and the enlarged "
        "num_sms * PPU_DEEP_GEMM_TB_PER_CU scheduler-metadata buffer for fp4 "
        "paged logits. Delegation cannot intercept the upstream capability "
        "assert, hence the verbatim body with only the fork's changes marked."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream grows PPU branches in the DSA indexer metadata builder.",
)(_with_target_globals(_builder_init_body))


def _build_body(
    self,
    common_prefix_len: int,
    common_attn_metadata: CommonAttentionMetadata,
    fast_build: bool = False,
) -> DeepseekV32IndexerMetadata:
    num_reqs = common_attn_metadata.num_reqs
    num_tokens = common_attn_metadata.num_actual_tokens
    query_start_loc = common_attn_metadata.query_start_loc
    query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
    seq_lens = common_attn_metadata.seq_lens
    slot_mapping = common_attn_metadata.slot_mapping
    block_table = common_attn_metadata.block_table_tensor
    dcp_local_seq_lens = common_attn_metadata.dcp_local_seq_lens

    compressed_slot_mapping = slot_mapping
    if self.compress_ratio > 1:
        padded_num_tokens = num_tokens
        local_slot_mapping = slot_mapping
        if self.use_pcp:
            # The gathered layout holds each rank's local tokens, padded, in
            # rank order, so this rank's segment lines up with query_start_loc.
            padded_num_tokens = slot_mapping.shape[0] // self.pcp_world_size
            local_slot_mapping = slot_mapping[
                self.pcp_rank * padded_num_tokens : (self.pcp_rank + 1)
                * padded_num_tokens
            ]
        compressed_slot_mapping = get_compressed_slot_mapping(
            num_tokens,
            local_slot_mapping,
            query_start_loc,
            seq_lens,
            block_table,
            self.kv_cache_spec.num_states,
            self.compress_ratio,
            out=self.compressed_slot_mapping_buffer,
            dcp_world_size=self.dcp_world_size,
            cp_interleave=self.cp_kv_cache_interleave_size,
        )
        if self.pcp_world_size > 1:
            compressed_slot_mapping = get_pcp_group().all_gather(
                self.compressed_slot_mapping_buffer[:padded_num_tokens],
                dim=0,
            )

    # PCP decode sharding keeps a zero-token placeholder on ranks that own
    # no request in a step so collectives retain a uniform rank shape. Do
    # not turn that placeholder into a zero-length indexer decode request.
    if num_tokens == 0:
        return DeepseekV32IndexerMetadata(
            seq_lens=seq_lens,
            max_seq_len=common_attn_metadata.max_seq_len,
            slot_mapping=compressed_slot_mapping,
            num_decodes=0,
            num_decode_tokens=0,
            num_prefills=0,
            num_prefill_tokens=0,
            prefill=None,
            decode=None,
        )

    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=self.decode_threshold,
            require_uniform=not (self.use_flattening or self.supports_varlen),
            treat_short_extends_as_decodes=not self.use_pcp,
        )
    )

    assert num_decodes + num_prefills == num_reqs
    assert num_decode_tokens + num_prefill_tokens == num_tokens

    prefill_metadata = None
    if num_prefills > 0:
        compressed_seq_lens = (
            seq_lens // self.compress_ratio if self.compress_ratio > 1 else seq_lens
        )
        # This CPU value is an upper bound for async-spec extend rows.  It
        # is safe for chunking/allocation because CUDA metadata below is
        # built from exact device seq_lens and gather ignores the tail.
        assert common_attn_metadata.seq_lens_cpu_upper_bound is not None
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
        compressed_seq_lens_cpu = (
            seq_lens_cpu // self.compress_ratio
            if self.compress_ratio > 1
            else seq_lens_cpu
        )
        prefill_query_lens_cpu = torch.diff(
            query_start_loc_cpu[num_decodes : num_decodes + num_prefills + 1]
        )
        max_logits_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        # Upper bound is exact for prefill rows (the `[num_decodes:]`
        # slice below).
        assert common_attn_metadata.seq_lens_cpu_upper_bound is not None
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
        req_idx = None
        shard_rows = None
        if self.use_pcp and self.dcp_world_size > 1:
            # The gathered KV must be packed identically on every PCP rank:
            # chunk by request from its DCP shard rows, which every rank
            # holds. A dummy batch bypasses the PCP manager and has one
            # row per request, so its own extent is the request's.
            req_idx = common_attn_metadata.req_idx
            if req_idx is None:
                req_idx = np.arange(num_reqs)
            shard_rows_cpu = common_attn_metadata.dcp_local_seq_lens_cpu_upper_bound
            if shard_rows_cpu is None:
                shard_rows_cpu = get_dcp_local_seq_lens(
                    seq_lens_cpu,
                    self.dcp_world_size,
                    0,
                    self.cp_kv_cache_interleave_size,
                )
            shard_rows = shard_rows_cpu.numpy()
            chunk_specs = self._split_pcp_dcp_prefill_chunks(
                req_idx[num_decodes:],
                shard_rows[num_decodes:],
                prefill_query_lens_cpu,
                max_logits_bytes,
                request_offset=num_decodes,
            )
        else:
            chunk_specs = self._split_indexer_prefill_chunks(
                self._prefill_split_seq_lens(compressed_seq_lens_cpu[num_decodes:]),
                prefill_query_lens_cpu,
                self.max_prefill_buffer_size,
                max_logits_bytes,
                request_offset=num_decodes,
                # PPU MODIFICATION: begin
                logits_dtype=torch.bfloat16 if self.indexer_uses_fp4 else torch.float32,
                # PPU MODIFICATION: end
            )

        chunks = []
        for req_slice, query_slice in chunk_specs:
            pcp_plan = None
            if req_idx is not None:
                assert shard_rows is not None
                pcp_plan = build_pcp_global_chunk_plan(
                    req_idx[req_slice],
                    shard_rows[req_slice],
                    self.dcp_world_size,
                    self.device,
                    self.cp_kv_cache_interleave_size,
                )
            metadata = build_prefill_chunk_metadata(
                req_slice.start,
                req_slice.stop,
                query_start_loc,
                query_start_loc_cpu,
                seq_lens,
                compressed_seq_lens,
                compressed_seq_lens_cpu,
                block_table,
                self.compress_ratio,
                query_slice=query_slice,
                skip_kv_gather=query_slice.start > 0,
                dcp_rank=self.dcp_rank,
                dcp_world_size=self.dcp_world_size,
                cp_kv_cache_interleave_size=self.compressed_cp_interleave_size,
                pcp_plan=pcp_plan,
            )
            # Skip when total_seq_lens is 0 (i.e., no compressed token).
            if metadata is not None:
                chunks.append(metadata)
        row_shard_sizes = None
        if tp_prefill_row_sharding_supported(
            self.vllm_config, self.dcp_world_size, self.use_pcp, self.tp_size
        ):
            row_shard_sizes = balanced_prefill_row_shard(
                seq_lens_cpu[num_decodes:],
                prefill_query_lens_cpu,
                self.compress_ratio,
                self.tp_size,
                self.vllm_config.attention_config.indexer_tp_shard_min_rows,
            )
        prefill_metadata = DeepseekV32IndexerPrefillMetadata(
            chunks,
            max_prefill_seq_len=(
                int(seq_lens_cpu[num_decodes:].max().item()) if num_prefills > 0 else 0
            ),
            row_shard_sizes=row_shard_sizes,
        )

    decode_metadata = None
    if num_decodes > 0:
        assert num_decode_tokens <= self.expanded_block_table_buffer.shape[0]
        if not self.supports_varlen:
            torch.diff(
                common_attn_metadata.query_start_loc[: num_decodes + 1],
                out=self.decode_lens_buffer[:num_decodes],
            )
            self.per_req_decode_lens_buffer[:num_decodes].copy_(
                self.decode_lens_buffer[:num_decodes]
            )
        decode_lens = self.decode_lens_buffer[:num_decodes]
        decode_lens_cpu = torch.diff(
            common_attn_metadata.query_start_loc_cpu[: num_decodes + 1]
        )

        # Under DCP the per-token decode bounds must be localized AFTER the
        # per-token expansion below, not before. Expanding from a
        # request-level localized length subtracts decode offsets in local
        # space and yields too-short bounds (e.g. world=2, rank=1, global
        # per-token bounds [8, 9, 10] -> [3, 4, 5] instead of [4, 4, 5]), so
        # the first decode token would run top-k against too short a local KV
        # range and miss valid tokens. Keep the global seq_lens here and
        # localize the expanded bounds further down.
        global_seq_lens_for_decode: torch.Tensor | None = None
        if dcp_local_seq_lens is not None:
            global_seq_lens_for_decode = common_attn_metadata.seq_lens[:num_decodes]
        seq_lens = common_attn_metadata.seq_lens[:num_decodes]
        block_table = common_attn_metadata.block_table_tensor[:num_decodes, ...]

        max_decode_len = int(decode_lens_cpu.max().item())
        min_decode_len = int(decode_lens_cpu.min().item())
        write_is_uniform = min_decode_len == max_decode_len
        next_n = 1 + self.num_speculative_tokens
        if self.use_adaptive_verification:
            # Graph capture must reserve the largest runtime draft budget,
            # irrespective of the capture-time dummy request partition.
            max_decode_len = next_n
            write_is_uniform = False
        # The kernel sees max_decode_len Q rows, not the configured next_n,
        # so legality is per-step: on SM90 a uniformly 3-deep batch has no
        # native kernel. max_decode_len <= 1 always has one.
        step_next_n_ok = max_decode_len <= 1 or _supports_native_decode(max_decode_len)
        use_native = (
            not (self.use_flattening or self.supports_varlen)
            and max_decode_len <= next_n
            and step_next_n_ok
        )

        if not self.supports_varlen:
            global_seq_lens_for_decode = self._prepare_global_decode_seq_lens(
                global_seq_lens=global_seq_lens_for_decode,
                decode_lens=decode_lens,
                decode_lens_cpu=decode_lens_cpu,
                query_start_loc=common_attn_metadata.query_start_loc[:num_decodes],
                num_decode_tokens=num_decode_tokens,
                use_native=use_native,
                max_decode_len=max_decode_len,
            )

        decode_indices = None
        if self.supports_varlen:
            from vllm.v1.attention.ops.metadata import (
                _indexer_decode_metadata_kernel,
            )

            capacity = self.decode_seq_lens_buffer.numel()
            grid = max(
                num_decodes,
                num_decode_tokens + triton.cdiv(capacity - num_decode_tokens, 256),
            )
            _indexer_decode_metadata_kernel[(grid,)](
                query_start_loc,
                seq_lens,
                block_table,
                self.decode_seq_lens_buffer,
                self.expanded_block_table_buffer,
                self.decode_lens_buffer,
                self.decode_indices_buffer,
                self.per_req_decode_lens_buffer,
                num_decodes,
                num_decode_tokens,
                capacity,
                block_table.stride(0),
                self.expanded_block_table_buffer.stride(0),
                BLOCK_COLS=block_table.shape[1],
                num_warps=4,
            )
            seq_lens = self.decode_seq_lens_buffer[:num_decode_tokens]
            block_table = self.expanded_block_table_buffer[:num_decode_tokens]
            decode_lens = self.decode_lens_buffer[:num_decode_tokens]
            decode_indices = self.decode_indices_buffer[:num_decode_tokens]
            requires_padding = False
            if global_seq_lens_for_decode is not None and max_decode_len > 1:
                self.global_decode_seq_lens_buffer[:num_decode_tokens].copy_(seq_lens)
                global_seq_lens_for_decode = self.global_decode_seq_lens_buffer[
                    :num_decode_tokens
                ]
        else:
            seq_lens, block_table, decode_lens, batch_size, requires_padding = (
                self._prepare_decode_tensors(
                    seq_lens=seq_lens,
                    block_table=block_table,
                    decode_lens=decode_lens,
                    decode_lens_cpu=decode_lens_cpu,
                    query_start_loc=common_attn_metadata.query_start_loc[:num_decodes],
                    num_decodes=num_decodes,
                    num_decode_tokens=num_decode_tokens,
                    use_native=use_native,
                    next_n=next_n,
                    max_decode_len=max_decode_len,
                )
            )

        # Flattening always returns a buffer view, including single-token
        # batches. Keep its address stable across varlen graph replays.
        seq_lens_is_buffer_view = not use_native or next_n > 1

        # DCP: localize the now-expanded per-token global bounds to this
        # rank's owned KV. Done here (after expansion) so each token's global
        # causal length is localized individually; see the comment above.
        if dcp_local_seq_lens is not None:
            seq_lens = self._dcp_localize_decode_seq_lens(
                seq_lens, num_decodes, seq_lens_is_buffer_view
            )

        # For DeepseekV4 (compress_ratio > 1), the indexer KV cache stores
        # compressed tokens. Convert uncompressed seq_lens to compressed.
        if self.compress_ratio > 1:
            if seq_lens_is_buffer_view:
                seq_lens //= self.compress_ratio
            else:
                # Copy to avoid mutating shared state; keeps CG address stable.
                self.expanded_seq_lens_buffer[:num_decodes] = (
                    seq_lens // self.compress_ratio
                )
                self.expanded_seq_lens_buffer[num_decodes:num_decode_tokens] = 0
                seq_lens = self.expanded_seq_lens_buffer[:num_decode_tokens]

        # Non-MTP: deep_gemm paged MQA logits requires 2D context_lens
        # (csrc/apis/attention.hpp). Unsqueeze to (B, 1) so downstream
        # kernels see the same (B, next_n) layout as the MTP path.
        if seq_lens.dim() == 1:
            seq_lens = seq_lens.unsqueeze(-1)

        # PPU MODIFICATION: begin
        metadata_extra = None
        if current_platform.is_ppu() and self.indexer_n_head > 0:
            metadata_extra = (
                seq_lens.shape[1],
                self.indexer_n_head,
                self.indexer_q_head_dim,
                PPU_FP4_INDEXER_ELEMENT_SIZE,
            )
        scheduler_metadata = None
        n = self.scheduler_metadata_buffer.shape[0]
        if current_platform.is_ppu() and has_deep_gemm():
            scheduler_metadata = get_paged_mqa_logits_metadata(
                seq_lens,
                self.kv_cache_spec.num_states,
                self.num_sms,
                metadata_extra,
            )
            n = scheduler_metadata.shape[0]
            self.scheduler_metadata_buffer[:n] = scheduler_metadata
        # PPU MODIFICATION: end
        # DeepGEMM is required for the paged MQA logits on CUDA devices
        # PPU MODIFICATION: begin
        schedule_metadata = self.scheduler_metadata_buffer[:n]
        if (
            not current_platform.is_ppu()
            and current_platform.is_cuda()
            and has_deep_gemm()
        ):
            metadata = get_paged_mqa_logits_metadata(
                # PPU MODIFICATION: end
                seq_lens,
                self.kv_cache_spec.num_states,
                self.num_sms,
                indices=decode_indices,
            )
            schedule_metadata = self.scheduler_metadata_buffer[: metadata.shape[0]]
            schedule_metadata[:] = metadata

        decode_metadata = DeepSeekV32IndexerDecodeMetadata(
            block_table=block_table,
            seq_lens=seq_lens,
            decode_lens=decode_lens,
            requires_padding=requires_padding,
            schedule_metadata=schedule_metadata,
            indices=decode_indices,
            global_seq_lens=global_seq_lens_for_decode,
            per_req_decode_lens=self.per_req_decode_lens_buffer[:num_decodes],
            decode_is_uniform=write_is_uniform,
            write_max_decode_len=max_decode_len,
        )

    attn_metadata = DeepseekV32IndexerMetadata(
        seq_lens=common_attn_metadata.seq_lens,
        max_seq_len=common_attn_metadata.max_seq_len,
        slot_mapping=compressed_slot_mapping,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefill_tokens,
        prefill=prefill_metadata,
        decode=decode_metadata,
        block_table=common_attn_metadata.block_table_tensor,
    )

    return attn_metadata


patch(
    _MODULE,
    "DeepseekV32IndexerMetadataBuilder.build",
    reason=(
        "Fork's DSA indexer build path for PPU: pass logits_dtype to the "
        "prefill chunk splitter, and build PPU paged-MQA-logits scheduler "
        "metadata with the metadata_extra tuple (next_n, indexer_n_head, "
        "indexer_q_head_dim, PPU_FP4_INDEXER_ELEMENT_SIZE), slicing the "
        "scheduler buffer to the returned size. Verbatim body with only the "
        "fork's changes marked."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream grows PPU branches in the DSA indexer build path.",
)(_with_target_globals(_build_body))
