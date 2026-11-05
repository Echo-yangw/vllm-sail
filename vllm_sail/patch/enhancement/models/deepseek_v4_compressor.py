# ruff: noqa: E731, W291, W293, UP037
# ruff: noqa: F821
# Copied bodies resolve globals in their upstream module through bind_body.
# SPDX-License-Identifier: Apache-2.0
"""Keep DeepSeek V4 compression on Triton on PPU (both architectures)."""

from __future__ import annotations

from vllm.models.deepseek_v4 import compressor as _compressor

from vllm_sail.patch.bodies import bind_body
from vllm_sail.patch.utils import patch

_AFFECTED = ">=0.32.0,<0.33.0"
_MODULE = "vllm.models.deepseek_v4.compressor"


@patch(
    _MODULE,
    "CompressorBackend.supports_pcp",
    allow_missing=True,
    reason=(
        "DeepSeek V4 PCP replicates compressor state and compressed KV on PPU, "
        "so its internal attention backend participates in PCP metadata builds."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream PR #65305 lands with CompressorBackend.supports_pcp.",
)
@classmethod
def compressor_backend_supports_pcp(cls) -> bool:
    return _compressor.current_platform.is_ppu()


_upstream_metadata_init = _compressor.CompressorMetadataBuilder.__init__
_upstream_metadata_build = _compressor.CompressorMetadataBuilder.build


def _compressor_metadata_init(self, *args, **kwargs) -> None:
    _upstream_metadata_init(self, *args, **kwargs)
    self._ppu_pcp_size = self.vllm_config.parallel_config.prefill_context_parallel_size
    if not _compressor.current_platform.is_ppu() or self._ppu_pcp_size <= 1:
        return
    max_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
    max_reqs = 2 * self.vllm_config.scheduler_config.max_num_seqs
    self._ppu_pcp_local_req_indices = _compressor.torch.empty(
        max_tokens, dtype=_compressor.torch.int32, device=self.device
    )
    self._ppu_pcp_token_to_req_indices = _compressor.torch.empty(
        max_tokens * self._ppu_pcp_size,
        dtype=_compressor.torch.int32,
        device=self.device,
    )
    self._ppu_pcp_positions = _compressor.torch.empty(
        max_tokens * self._ppu_pcp_size,
        dtype=_compressor.torch.int64,
        device=self.device,
    )
    self._ppu_pcp_local_block_table = None
    self._ppu_pcp_block_table = None
    self._ppu_pcp_max_reqs = max_reqs


def _gather_compressor_metadata(
    self, metadata, common_attn_metadata, pcp_shared_metadata=None
) -> None:
    from vllm.distributed import get_pcp_group

    group = get_pcp_group()
    pcp_size = self._ppu_pcp_size
    local_tokens = metadata.slot_mapping.numel() // pcp_size
    if pcp_shared_metadata is None:
        local_num_reqs = _compressor.torch.tensor(
            [common_attn_metadata.num_reqs],
            dtype=_compressor.torch.int32,
            device=self.device,
        )
        gather_num_reqs = max(
            1, int(group.all_gather(local_num_reqs, dim=0).max().item())
        )
    else:
        gather_num_reqs = pcp_shared_metadata.get_gather_num_reqs(
            common_attn_metadata, self.device
        )
    assert gather_num_reqs <= self._ppu_pcp_max_reqs

    block_table_width = metadata.block_table.shape[1]
    if self._ppu_pcp_local_block_table is None:
        self._ppu_pcp_local_block_table = _compressor.torch.empty(
            (self._ppu_pcp_max_reqs, block_table_width),
            dtype=_compressor.torch.int32,
            device=self.device,
        )
        self._ppu_pcp_block_table = _compressor.torch.empty(
            (self._ppu_pcp_max_reqs * pcp_size, block_table_width),
            dtype=_compressor.torch.int32,
            device=self.device,
        )
    local_block_table = self._ppu_pcp_local_block_table[:gather_num_reqs]
    local_block_table.zero_()
    local_block_table[: common_attn_metadata.num_reqs].copy_(
        metadata.block_table[: common_attn_metadata.num_reqs]
    )
    gathered_block_table = group.all_gather(local_block_table, dim=0)
    pcp_block_table = self._ppu_pcp_block_table[: gather_num_reqs * pcp_size]
    pcp_block_table.copy_(gathered_block_table).clamp_(min=0)

    assert metadata.token_to_req_indices is not None
    if pcp_shared_metadata is None:
        local_req_indices = self._ppu_pcp_local_req_indices[:local_tokens]
        local_req_indices.zero_()
        local_req_indices[: common_attn_metadata.num_actual_tokens].copy_(
            metadata.token_to_req_indices
        )
        local_req_indices += group.rank_in_group * gather_num_reqs
        num_tokens = local_tokens * pcp_size
        pcp_req_indices = self._ppu_pcp_token_to_req_indices[:num_tokens]
        pcp_req_indices.copy_(group.all_gather(local_req_indices, dim=0))

        assert common_attn_metadata.positions is not None
        pcp_positions = self._ppu_pcp_positions[:num_tokens]
        pcp_positions.copy_(
            group.all_gather(common_attn_metadata.positions.contiguous(), dim=0)
        )
    else:
        pcp_req_indices = pcp_shared_metadata.get_token_to_req_indices(
            common_attn_metadata,
            local_tokens,
            metadata.token_to_req_indices,
            self._ppu_pcp_local_req_indices,
            self._ppu_pcp_token_to_req_indices,
        )
        pcp_positions = pcp_shared_metadata.get_cache_positions(common_attn_metadata)
    metadata.block_table = pcp_block_table
    metadata.token_to_req_indices = pcp_req_indices
    metadata.cache_positions = pcp_positions
    metadata.num_decode_tokens = None
    metadata.c128_boundary = None


@patch(
    _MODULE,
    "CompressorMetadataBuilder.__init__",
    reason=(
        "PPU DeepSeek V4 PCP needs persistent buffers for rank-major compressor "
        "request, position, and block-table metadata."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream PR #65305 lands with PCP compressor metadata buffers.",
)
def compressor_metadata_init(self, *args, **kwargs) -> None:
    _compressor_metadata_init(self, *args, **kwargs)


@patch(
    _MODULE,
    "CompressorMetadataBuilder.build",
    reason=(
        "PPU DeepSeek V4 PCP pools C4/C128 states over rank-major gathered rows, "
        "which requires uniform request tables and global cache positions."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream PR #65305 lands with PCP compressor metadata gathering.",
)
def compressor_metadata_build(
    self, common_prefix_len, common_attn_metadata, fast_build=False,
    pcp_shared_metadata=None,
):
    metadata = _upstream_metadata_build(
        self, common_prefix_len, common_attn_metadata, fast_build
    )
    if _compressor.current_platform.is_ppu() and self._ppu_pcp_size > 1:
        _gather_compressor_metadata(
            self, metadata, common_attn_metadata, pcp_shared_metadata
        )
    return metadata


_upstream_forward = _compressor.DeepseekCompressor.forward


@patch(
    _MODULE,
    "DeepseekCompressor.forward",
    reason=(
        "PPU DeepSeek V4 PCP must gather raw KV and score rows before writing "
        "partial states and pooling across C4/C128 shard boundaries."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream PR #65305 lands with PCP compressor input gathering.",
)
def compressor_forward(self, kv_score, positions, rotary_emb) -> None:
    attn_metadata = _compressor.get_forward_context().attn_metadata
    if isinstance(attn_metadata, dict):
        state_metadata = attn_metadata[self.state_cache.prefix]
        cache_positions = getattr(state_metadata, "cache_positions", None)
        if cache_positions is not None:
            from vllm.distributed import get_pcp_group

            kv_score = get_pcp_group().all_gather(kv_score.contiguous(), dim=0)
            positions = cache_positions
    _upstream_forward(self, kv_score, positions, rotary_emb)


# fmt: off
def __init__(
    self,
    vllm_config: VllmConfig,
    compress_ratio: int,
    hidden_size: int,
    head_dim: int,
    rotate: bool = False,
    prefix: str = "",
    k_cache_prefix="",
    use_fp4_cache: bool = False,
):
    # PPU MODIFICATION: begin - copied methods need an explicit super target.
    super(DeepseekCompressor, self).__init__()
    # PPU MODIFICATION: end
    self.compress_ratio = compress_ratio
    self.hidden_size = hidden_size
    self.head_dim = head_dim
    self.rotate = rotate
    self.prefix = prefix
    self.k_cache_prefix = k_cache_prefix
    self.use_fp4_cache = use_fp4_cache

    config = vllm_config.model_config.hf_config
    self.rope_head_dim = config.qk_rope_head_dim
    self.nope_head_dim = self.head_dim - self.rope_head_dim
    self.rms_norm_eps = config.rms_norm_eps
    self.device = current_platform.device_type
    self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
    self.max_model_len = vllm_config.model_config.max_model_len

    self.overlap = compress_ratio == 4
    self.coff = 1 + self.overlap

    # The head=512 cr>=128 no-overlap deep gather uses the two-stage
    # compressor, which needs an fp32 scratch [max_batched, 512] for
    # the intermediate compressed_kv.
    # Currently only tested on ROCm
    self._use_two_stage_fused_compressor = (
        _prefer_two_stage_compressor() and head_dim == 512 and not self.overlap
    )
    self.max_num_batched_tokens = (
        vllm_config.scheduler_config.max_num_batched_tokens
    )
    self._compress_scratch: torch.Tensor | None = None
    if self._use_two_stage_fused_compressor:
        self._compress_scratch = torch.empty(
            self.max_num_batched_tokens,
            self.head_dim,
            dtype=torch.float32,
            device=self.device,
        )

    state_dtype = torch.float32
    self.ape = nn.Parameter(
        torch.empty(
            (compress_ratio, self.coff * self.head_dim),
            dtype=state_dtype,
            device=self.device,
        ),
        requires_grad=False,
    )

    self.fused_wkv_wgate = MergedColumnParallelLinear(
        self.hidden_size,
        [self.coff * self.head_dim, self.coff * self.head_dim],
        bias=False,
        return_bias=False,
        quant_config=None,
        disable_tp=True,
        prefix=f"{prefix}.fused_wkv_wgate",
    )
    self.norm = RMSNorm(self.head_dim, self.rms_norm_eps)

    self.state_cache = CompressorStateCache(
        state_dim=2 * self.coff * self.head_dim,  # kv_state + score_state
        dtype=state_dtype,
        compress_ratio=compress_ratio,
        prefix=f"{prefix}.state_cache",
    )

    # Save reference to static_forward_context for forward-time KV cache lookup.
    # get_current_vllm_config() is only available during __init__, not forward.
    self._static_forward_context = (
        vllm_config.compilation_config.static_forward_context
    )

    if self.head_dim == 512:
        assert not use_fp4_cache, (
            "MXFP4 cache is only supported for indexer (head=128)"
        )
        self._quant_block = 64
        self._token_stride = self.nope_head_dim + self.rope_head_dim * 2
        self._scale_dim = self.nope_head_dim // 64 + 1  # 7 real + 1 pad
    elif self.head_dim == 128:
        if use_fp4_cache:
            self._quant_block = MXFP4_BLOCK_SIZE
            self._token_stride = self.head_dim // 2
            self._scale_dim = self.head_dim // MXFP4_BLOCK_SIZE
        else:
            self._quant_block = 128
            self._token_stride = self.head_dim
            self._scale_dim = 4  # single float32 scale
    else:
        raise ValueError(
            f"Unsupported head_dim for fused quant+cache: {self.head_dim}"
        )

    if vllm_config.kernel_config.enable_jit_warmup:
        _SAVE_PARTIAL_STATES_KERNEL.register_warmup(
            head_dim=self.head_dim,
            compress_ratio=self.compress_ratio,
        )
        # PPU MODIFICATION: begin - CUDA identity does not imply CuTeDSL support.
        if (
            current_platform.is_cuda()
            and not current_platform.is_ppu()
            and self.head_dim == 512
        ):
            # PPU MODIFICATION: end
            from vllm.models.deepseek_v4.nvidia.ops.sparse_attn_compress_cutedsl import (  # noqa: E501
                _SPARSE_ATTN_COMPRESS_C128_BLOCK8_KERNEL,
                _SPARSE_ATTN_COMPRESS_NORM_ROPE_STORE_C4_KERNEL,
                _SPARSE_ATTN_COMPRESS_NORM_ROPE_STORE_FULL_C4_KERNEL,
                _SPARSE_ATTN_NORM_ROPE_STORE_FULL_KERNEL,
                _SPARSE_ATTN_NORM_ROPE_STORE_KERNEL,
            )

            store_full_kv = vllm_config.cache_config.cache_dtype != "fp8_ds_mla"
            if self.compress_ratio == 4:
                (
                    _SPARSE_ATTN_COMPRESS_NORM_ROPE_STORE_FULL_C4_KERNEL
                    if store_full_kv
                    else _SPARSE_ATTN_COMPRESS_NORM_ROPE_STORE_C4_KERNEL
                ).register_warmup()
            else:
                _SPARSE_ATTN_COMPRESS_C128_BLOCK8_KERNEL.register_warmup()
                if store_full_kv:
                    _SPARSE_ATTN_NORM_ROPE_STORE_FULL_KERNEL.register_warmup()
                else:
                    _SPARSE_ATTN_NORM_ROPE_STORE_KERNEL.register_warmup(
                        vllm_config,
                        k_cache_prefix=self.k_cache_prefix,
                        compress_ratio=self.compress_ratio,
                    )
        else:
            from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (  # noqa: E501
                _FUSED_KV_COMPRESS_NORM_ROPE_INSERT_INDEXER_TRITON_KERNEL,
            )

            _FUSED_KV_COMPRESS_NORM_ROPE_INSERT_INDEXER_TRITON_KERNEL.register_warmup()
# fmt: on


def _compress_norm_rope_store(
    self,
    state_metadata: CompressorMetadata,
    state_cache: torch.Tensor,
    state_width: int,
    positions: torch.Tensor,
    rotary_emb,
    attn_metadata: dict[str, Any],
    pdl_kwargs: dict,
) -> None:
    """Compress the saved states and write K to the cache; a platform
    subclass overrides it to write its own cache layout."""
    token_to_req_indices = state_metadata.token_to_req_indices
    slot_mapping = state_metadata.slot_mapping
    num_actual = slot_mapping.shape[0]
    block_table = state_metadata.block_table
    block_size = state_metadata.block_size

    # Fused: compress → RMSNorm → RoPE → FP8 quant → KV cache write.
    # RoPE requirements (kernel applies forward GPT-J style rotation):
    # - is_neox_style=False (interleaved pairs, NOT split-half)
    # - cos_sin_cache layout: [max_pos, rope_head_dim] with first half cos,
    #   second half sin (per-pair, length rope_head_dim // 2 each)
    # - applied to LAST rope_head_dim elements of head_dim
    # - position used: (positions // compress_ratio) * compress_ratio
    cos_sin_cache = rotary_emb.cos_sin_cache
    k_cache_metadata = cast(Any, attn_metadata[self.k_cache_prefix])
    k_cache_layer = self._static_forward_context[self.k_cache_prefix]
    kv_cache = k_cache_layer.kv_cache

    # Plain-row V4 reads a contiguous bf16 / per-tensor fp8 cache row; the
    # fp8_ds_mla path uses the UE8M0 paged uint8 layout.
    store_full_kv = self.head_dim == 512 and kv_cache.dtype != torch.uint8
    store_full_fp8 = kv_cache.dtype == torch.float8_e4m3fn
    fp8_scale = (
        getattr(k_cache_layer, "_flashinfer_fp8_kv_scale", None)
        if store_full_fp8
        else None
    )

    # cutedsl (head=512) accepts the full-cache flags; triton (indexer/AMD)
    # does not, so the two callables have different signatures.
    compress_norm_rope_store_fn: Any
    # PPU MODIFICATION: begin
    if (
        current_platform.is_cuda()
        and not current_platform.is_ppu()
        and self.head_dim == 512
    ):
        # PPU MODIFICATION: end
        from .nvidia.ops.sparse_attn_compress_cutedsl import (
            _SPARSE_ATTN_COMPRESSOR_CUTEDSL_KERNEL,
        )

        # head=512 on CUDA always uses cutedsl, for both the fp8_ds_mla
        # layout and the plain full-cache layout. The full-cache flags
        # are consumed only here.
        compress_norm_rope_store_fn = _SPARSE_ATTN_COMPRESSOR_CUTEDSL_KERNEL
        extra_kwargs: dict[str, Any] = dict(
            store_full_kv=store_full_kv,
            store_full_fp8=store_full_fp8,
            fp8_scale=fp8_scale,
        )
    elif self._use_two_stage_fused_compressor:
        # head=512 cr>=128 (no overlap): two-pass split compressor on the
        # prefill suffix, single-pass on the decode prefix.
        assert state_metadata.num_decode_tokens is not None
        compress_norm_rope_store_fn = compress_norm_rope_store_two_stage_triton
        extra_kwargs = {
            "num_decode_tokens": state_metadata.num_decode_tokens,
            "compress_scratch": self._compress_scratch,
        }
    else:
        # Indexer path (head_dim == 128) or non-CUDA GPUs (AMD, XPU, etc.).
        compress_norm_rope_store_fn = compress_norm_rope_store_triton
        extra_kwargs = {}

    compress_norm_rope_store_fn(
        state_cache=state_cache,
        num_actual=num_actual,
        token_to_req_indices=token_to_req_indices,
        positions=positions,
        slot_mapping=slot_mapping,
        block_table=block_table,
        block_size=block_size,
        state_width=state_width,
        cos_sin_cache=cos_sin_cache,
        kv_cache=kv_cache,
        k_cache_metadata=k_cache_metadata,
        pdl_kwargs=pdl_kwargs,
        head_dim=self.head_dim,
        rope_head_dim=self.rope_head_dim,
        compress_ratio=self.compress_ratio,
        overlap=self.overlap,
        use_fp4_cache=self.use_fp4_cache,
        rms_norm_weight=self.norm.weight,
        rms_norm_eps=self.rms_norm_eps,
        quant_block=self._quant_block,
        token_stride=self._token_stride,
        scale_dim=self._scale_dim,
        **extra_kwargs,
    )


patch(
    "vllm.models.deepseek_v4.compressor",
    "DeepseekCompressor._compress_norm_rope_store",
    reason=(
        "The compression helper selects head-512 CuTeDSL by CUDA identity; "
        "PPU requires its Triton implementation. The common attention and "
        "indexer constructors do not share a compressor factory hook."
    ),
    affected_versions=">=0.32.0,<0.33.0",
    remove_when="The compression helper selects kernels through platform capabilities.",
)(bind_body(_compress_norm_rope_store, _compressor))


patch(
    "vllm.models.deepseek_v4.compressor",
    "DeepseekCompressor.__init__",
    reason=(
        "The constructor registers head-512 CuTeDSL warmup based only on CUDA "
        "identity, bypassing has_cutedsl and the PPU forward guard. PPU must "
        "register the common Triton warmup instead."
    ),
    affected_versions=">=0.32.0,<0.33.0",
    remove_when="DeepseekCompressor gates warmup registration on backend capabilities.",
)(bind_body(__init__, _compressor))
