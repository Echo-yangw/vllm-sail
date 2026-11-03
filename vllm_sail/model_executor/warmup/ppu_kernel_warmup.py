# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The upstream warmup body with its NVIDIA-only MSA import gated on PPU.

Bind this body to the real upstream module globals before calling it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# These globals are supplied by bind_body; keep imports static-only.
if TYPE_CHECKING:
    from vllm.model_executor.warmup.kernel_warmup import (
        _warmup_bf16x3_router_gemm,
        _warmup_gemm_rs_ar,
        _warmup_ll_bf16_router_gemm,
        b12x_warmup,
        current_platform,
        cutedsl_warmup,
        deep_gemm_warmup,
        deepseek_v4_sparse_mla_attention_warmup,
        envs,
        flashinfer_autotune,
        flashinfer_sparse_mla_decode_autotune_warmup,
        has_flashinfer,
        is_deep_gemm_supported,
        kimi_k3_triton_warmup,
        logger,
        mamba_triton_warmup,
        mixed_batch_attention_warmup,
        qwen4_exp_qsa_triton_warmup,
        qwen_triton_warmup,
        qwen_vl_triton_warmup,
        time,
        watermark_sample_warmup,
    )
    from vllm.v1.worker.gpu_worker import Worker


def kernel_warmup(worker: Worker, *, process_local_only: bool = False):
    from vllm.model_executor.warmup.step5_dsa_warmup import step5_dsa_warmup

    if not worker.use_v2_model_runner:
        # The KV-block zeroing kernel is driven by the scheduler's
        # `new_block_ids_to_zero`, so no dummy run ever reaches it.
        zeroer = getattr(worker.model_runner, "_kv_block_zeroer", None)
        if zeroer is not None:
            zeroer.warmup(worker.model_runner.kv_cache_config.num_blocks)

    enable_jit_warmup = worker.vllm_config.kernel_config.enable_jit_warmup
    if enable_jit_warmup:
        logger.info("JIT kernel warmup starting.")
        jit_warmup_start = time.perf_counter()
        try:
            registry = (
                worker.model_runner.jit_warmup_registry  # type: ignore[attr-defined]
            )
            registry.warmup()
        except Exception:
            logger.exception(
                "JIT kernel warmup failed after %.2fs.",
                time.perf_counter() - jit_warmup_start,
            )
            raise
        logger.info(
            "JIT kernel warmup finished in %.2fs.",
            time.perf_counter() - jit_warmup_start,
        )

    qwen_triton_warmup(worker.model_runner, worker.vllm_config.model_config)
    qwen_vl_triton_warmup(worker.model_runner)
    mamba_triton_warmup(worker.model_runner)

    compilation_config = worker.vllm_config.compilation_config
    cudagraph_capture_sizes = list(compilation_config.cudagraph_capture_sizes or [])

    # Run next so input-prep kernels JIT against pristine runner state.
    if enable_jit_warmup:
        kimi_k3_triton_warmup(worker)
        watermark_sample_warmup(worker)
        qwen4_exp_qsa_triton_warmup(worker)

    if enable_jit_warmup and current_platform.is_device_capability_family(100):
        _warmup_bf16x3_router_gemm(
            worker.get_model(),
            worker.scheduler_config.max_num_batched_tokens,
        )

    if current_platform.has_device_capability(90):
        _warmup_ll_bf16_router_gemm(worker.get_model())

    _warmup_gemm_rs_ar()

    if worker.vllm_config.kernel_config.enable_cutedsl_warmup:
        # TODO(roberto): Remove after registered CuTeDSL warmups are migrated
        # to the shared JIT warmup infrastructure.
        # https://github.com/vllm-project/vllm/pull/47451
        cutedsl_warmup()

    if process_local_only:
        return

    if current_platform.is_rocm():
        from vllm.model_executor.warmup.rocm_segmented_attn_autotune_warmup import (
            rocm_segmented_attn_autotune_warmup,
        )

        rocm_segmented_attn_autotune_warmup(worker)

    flashinfer_sparse_mla_decode_autotune_warmup(worker)
    deepseek_v4_sparse_mla_attention_warmup(worker)
    step5_dsa_warmup(worker)

    # Deep GEMM warmup
    do_deep_gemm_warmup = (
        is_deep_gemm_supported() and envs.VLLM_DEEP_GEMM_WARMUP != "skip"
    )
    if do_deep_gemm_warmup:
        model = worker.get_model()
        max_tokens = worker.scheduler_config.max_num_batched_tokens
        deep_gemm_warmup(model, max_tokens)

    b12x_warmup(worker, cudagraph_capture_sizes)

    if current_platform.is_cuda() and not current_platform.is_ppu():
        # The MSA kernels, and the CuTe DSL their modules import, are CUDA-only.
        from vllm.model_executor.warmup.minimax_m3_msa_warmup import (
            minimax_m3_msa_warmup,
        )

        minimax_m3_msa_warmup(worker)

    # Allocate the exact decode-sized workspace, autotune cache misses, and
    # resolve every CUDA Graph bucket before capture begins.
    # Lazy import: flashinfer_pcie_ipc_all_reduce imports flashinfer.comm,
    # which initializes CUDA at import time and must not run at module scope.
    if envs.VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC:
        from vllm.distributed.device_communicators import (
            flashinfer_pcie_ipc_all_reduce,
        )

        flashinfer_pcie_ipc_all_reduce.warmup_flashinfer_pcie_ipc_allreduce(worker)

    enable_flashinfer_autotune = (
        worker.vllm_config.kernel_config.enable_flashinfer_autotune
    )
    # FlashInfer autotune for Hopper (SM 9.0) and Blackwell (SM 10.0) GPUs
    if enable_flashinfer_autotune is False:
        logger.info_once("Skipping FlashInfer autotune because it is disabled.")
    elif has_flashinfer() and current_platform.has_device_capability(90):
        flashinfer_autotune(worker.model_runner)

    # FlashInfer attention warmup
    # Only warmup if the model has FlashInfer attention groups
    # and is not a pooling model
    def _is_flashinfer_backend(backend):
        try:
            return backend.get_name() == "FLASHINFER"
        except NotImplementedError:
            return False

    if (
        not worker.model_runner.is_pooling_model
        and worker.model_runner.attn_groups
        # NOTE: This should be `any` instead of `all` but other hybrid attention
        # backends don't support this dummy run. Once we remove
        # `build_for_cudagraph_capture`, we can change it to `any`.
        and all(
            _is_flashinfer_backend(group.backend)
            for groups in worker.model_runner.attn_groups
            for group in groups
        )
    ):
        logger.info_once("Warming up FlashInfer attention.")
        mixed_batch_attention_warmup(worker)
