# SPDX-License-Identifier: Apache-2.0
"""The PPU platform.

PPU hardware is CUDA-compatible, so this class inherits
:class:`vllm.platforms.cuda.NvmlCudaPlatform` and reuses upstream vLLM's CUDA
attention backends, distributed communication, compilation and kernel
infrastructure wholesale. Only genuine PPU differences are overridden.

Deliberately inherited from ``NvmlCudaPlatform`` (do not re-implement):
``set_device``, ``get_current_memory_usage``, ``get_vit_attn_backend``,
``get_punica_wrapper``, ``get_device_communicator_cls``, ``supports_fp8``,
``opaque_attention_op``, ``get_static_graph_wrapper_cls``,
``stateless_init_device_torch_dist_pg``, ``device_count``,
``check_if_supports_dtype``, ``insert_blocks_to_device``,
``swap_out_blocks_to_host``, ``support_hybrid_kv_cache``,
``support_static_graph_mode``, ``num_compute_units``,
``use_custom_op_collectives``.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, NamedTuple

import torch
from vllm.logger import init_logger
from vllm.platforms.cuda import NvmlCudaPlatform
from vllm.platforms.interface import DeviceCapability, PlatformEnum
from vllm.v1.attention.backends.registry import AttentionBackendEnum

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig, VllmConfig
    from vllm.v1.attention.backend import AttentionBackend
    from vllm.v1.attention.selector import AttentionSelectorConfig

logger = init_logger(__name__)

# PyTorch >= 2.5 enables the cuDNN SDPA backend by default, which crashes on
# some models. See https://github.com/huggingface/diffusers/issues/9704.
torch.backends.cuda.enable_cudnn_sdp(False)


def _get_attn_backend_class(backend: AttentionBackendEnum) -> type[AttentionBackend]:
    return backend.get_class()


def _retarget_dense_draft_backend(
    speculative_config: SpeculativeConfig,
) -> None:
    """Point a dense MLA draft at PPU's fp8 MLA decoder.

    Upstream's Kimi K3 config hook (``configure_dense_draft``) pins a dense
    EAGLE3/DSpark MLA draft to ``FLASHINFER_MLA`` on an fp8 KV cache: the draft
    holds no cache of its own, it keeps plain fp8 rows at the head of the
    target's merged pages. PPU can honour neither half of that pin -- FlashInfer
    is absent, and the dense FlashMLA has no FP8 decode path -- so the draft
    dies in the selector with "compute capability not supported".

    ``TRITON_MLA`` reads the same fp8 rows, dequantizes in-kernel from the
    layer's ``_k_scale``, takes the real page stride of the cross-layer view and
    flattens the draft's non-causal block to one decode row per query token.
    Only the decode backend changes: the KV format, the page binding and
    ``kv_cache_dtype="fp8"`` stay exactly as upstream set them.
    """
    if (
        speculative_config.attention_backend is AttentionBackendEnum.FLASHINFER_MLA
        and speculative_config.kv_cache_dtype in ("fp8", "fp8_e4m3")
    ):
        logger.info(
            "Retargeting the dense %s draft from FLASHINFER_MLA (unavailable on "
            "PPU) to TRITON_MLA; the fp8 KV rows and page binding are unchanged.",
            speculative_config.method,
        )
        speculative_config.attention_backend = AttentionBackendEnum.TRITON_MLA


class _BackendCandidate(NamedTuple):
    backend_class: type[AttentionBackend]
    backend: AttentionBackendEnum
    priority: int


@cache
def _get_backend_priorities(
    use_mla: bool,
    device_capability: DeviceCapability,
    num_heads: int | None = None,
) -> list[AttentionBackendEnum]:
    """PPU attention backend preference order.

    Lazily resolved to avoid a circular import at module load time.
    """
    if use_mla:
        return [
            AttentionBackendEnum.FLASHMLA,
            AttentionBackendEnum.TRITON_MLA,
            AttentionBackendEnum.FLASHMLA_SPARSE,
        ]
    return [
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.TRITON_ATTN,
        AttentionBackendEnum.FLEX_ATTENTION,
    ]


class PPUPlatform(NvmlCudaPlatform):
    """vLLM platform for PPU accelerators.

    ``_enum`` is deliberately ``PlatformEnum.CUDA`` rather than
    ``PlatformEnum.OOT``. This is a *compatibility declaration*, not an
    oversight, and it is the single most consequential decision in this plugin:

    * It makes ``current_platform.is_cuda()`` and ``is_cuda_alike()`` true, so
      the ~207 upstream call sites that gate CUDA code paths behave correctly on
      PPU. Declaring ``OOT`` would flip all of them onto non-CUDA paths, which is
      exactly why vLLM-MetaX has to vendor its own copies of the attention
      backends, MLA layers and fused-MoE stack.
    * It makes ``is_out_of_tree()`` false, so the four upstream ``is_out_of_tree``
      short-circuits that fire *before* the CUDA logic
      (``model_executor/custom_op.py``, ``fused_moe/oracle/unquantized.py``,
      ``distributed/parallel_state.py``, ``distributed/stateless_coordinator.py``)
      correctly fall through to the CUDA behaviour PPU wants.

    PPU is distinguished from real CUDA by ``current_platform.is_ppu()``,
    available here before general plugins run. The platform-predicate patch
    later adds the method to the upstream base for non-PPU platforms.

    See ``docs/developer_guide/architecture.md``.
    """

    _enum = PlatformEnum.CUDA
    device_name: str = "ppu"
    device_type: str = "cuda"
    dispatch_key: str = "CUDA"
    ray_device_key: str = "GPU"
    dist_backend: str = "nccl"
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"
    ray_noset_device_env_vars: list[str] = [
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
    ]

    @classmethod
    def is_ppu(cls) -> bool:
        """Expose PPU identity even before general-plugin patches install.

        Direct imports of registered kernels can precede that lifecycle. Relying
        on its additive base-class patch makes upstream's __getattr__ return
        None, turning is_ppu() into a NoneType call at import time.
        """
        return True

    @classmethod
    def get_hisparse_backend_cls(cls) -> str | None:
        """HiSparse compute kernels are excluded from the PPU native corpus."""
        return None

    @classmethod
    def import_kernels(cls) -> None:
        """Load PPU's own compiled kernels, then let upstream try its own.

        vLLM funnels all kernel imports through this hook, which makes it the
        seam for the namespace handshake: a ``VLLM_TARGET_DEVICE=empty`` vLLM
        leaves ``torch.ops._C`` / ``_moe_C`` unclaimed, and PPU's extensions
        register into them so every ``vllm/_custom_ops.py`` wrapper resolves
        with no patching. See ``docs/developer_guide/kernels.md``.
        """
        from vllm_sail import native

        native.install()
        super().import_kernels()

    @classmethod
    def get_valid_backends(
        cls,
        device_capability: DeviceCapability,
        attn_selector_config: AttentionSelectorConfig,
        num_heads: int | None = None,
    ) -> tuple[
        list[_BackendCandidate],
        dict[AttentionBackendEnum, tuple[int, list[str]]],
    ]:
        """Validate PPU's backend preference order against the model config.

        Overridden rather than inherited because PPU's viable backend set is a
        strict subset of CUDA's: FlashInfer, CUTLASS-MLA and the cuDNN paths are
        not available on PPU, and offering them produces confusing "no valid
        backend" diagnostics.
        """
        valid_backends_priorities: list[_BackendCandidate] = []
        invalid_reasons: dict[AttentionBackendEnum, tuple[int, list[str]]] = {}

        backend_priorities = _get_backend_priorities(
            attn_selector_config.use_mla,
            device_capability,
            num_heads,
        )
        for priority, backend in enumerate(backend_priorities):
            try:
                backend_class = _get_attn_backend_class(backend)
                invalid_reasons_i = backend_class.validate_configuration(
                    device_capability=device_capability,
                    **attn_selector_config._asdict(),
                )
            except ImportError:
                invalid_reasons_i = ["ImportError"]
            if invalid_reasons_i:
                invalid_reasons[backend] = (priority, invalid_reasons_i)
            else:
                valid_backends_priorities.append(
                    _BackendCandidate(backend_class, backend, priority)
                )

        return valid_backends_priorities, invalid_reasons

    @classmethod
    def get_supported_vit_attn_backends(cls) -> list[AttentionBackendEnum]:
        """Vision-encoder attention backends available on PPU."""
        if cls.has_device_capability(80):
            return [
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.TRITON_ATTN,
                AttentionBackendEnum.TORCH_SDPA,
            ]
        return [
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.TORCH_SDPA,
            AttentionBackendEnum.TRITON_ATTN,
        ]

    @classmethod
    def apply_config_platform_defaults(cls, vllm_config: VllmConfig) -> None:
        """Inject PPU defaults into the resolved vLLM config.

        This is the sanctioned hook for platform-specific config defaults, and
        it is why none of these settings needs a patch.
        """
        # Registering PPU custom ops here (rather than at plugin import) keeps
        # the registration after torch and the platform are both initialised.
        from vllm_sail import ops  # noqa: F401

        from vllm_sail.models.deepseek_v41.config import normalize_expert_dtype

        normalize_expert_dtype(vllm_config)

        if vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe":
            raise ValueError(
                "PPU DeepGEMM does not support MegaMoE; select ppu_deep_gemm or auto."
            )

        compilation_config = vllm_config.compilation_config
        attention_config = vllm_config.attention_config

        # 0 disables CUDA-graph-time split-KV heuristics, which measurably helps
        # FA3 on PPU.
        attention_config.flash_attn_max_num_splits_for_cuda_graph = 0

        # Dispatch to PPU's sparse_attn_indexer implementation (registered as the
        # `ppu_sparse_attn_indexer` custom op).
        if "+sparse_attn_indexer" not in compilation_config.custom_ops:
            compilation_config.custom_ops.append("+sparse_attn_indexer")

        # Same for the glm5_next kpool indexer (additionally registers the
        # `ppu_sparse_attn_indexer_kpool` custom op via vllm_sail.ops).
        if "+sparse_attn_indexer_kpool" not in compilation_config.custom_ops:
            compilation_config.custom_ops.append("+sparse_attn_indexer_kpool")

        # Prefer the compiled dynamic_per_token_scaled_fp8_quant kernel over the
        # fused Triton kernel torch.compile generates from forward_native: on PPU
        # the compiled path is faster. Respect an explicit user opt-out.
        if (
            "-quant_fp8" not in compilation_config.custom_ops
            and "+quant_fp8" not in compilation_config.custom_ops
        ):
            compilation_config.custom_ops.append("+quant_fp8")

        # Model config hooks have already run, so a dense MLA draft on a hybrid
        # (KDA + MLA) DSA target has been pinned to a backend PPU cannot select.
        if vllm_config.speculative_config is not None:
            _retarget_dense_draft_backend(vllm_config.speculative_config)

    @classmethod
    def is_arch_support_pdl(cls) -> bool:
        return False

    @classmethod
    def use_custom_allreduce(cls) -> bool:
        """PPU has no custom all-reduce kernel; use NCCL."""
        return False
