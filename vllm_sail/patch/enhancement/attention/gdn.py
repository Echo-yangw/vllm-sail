# ruff: noqa: E731, W291, W293, UP037
# ruff: noqa: F821
# Copied bodies resolve globals in their upstream module through bind_body.
# SPDX-License-Identifier: Apache-2.0
"""PPU PLA GDN decode, MTP and prefill routing with Triton fallback."""

from __future__ import annotations

import sys
from importlib import import_module

from vllm_sail.patch.bodies import bind_body
from vllm_sail.patch.utils import PATCH_MARKER, patch

_META = dict(
    reason="PPU GDN uses PLA for supported decode/prefill layouts and Triton for other shapes.",
    affected_versions=">=0.32.0,<0.33.0",
    remove_when="GDN exposes platform backend registration for PLA prefill and recurrent kernels.",
)


def fused_recurrent_gated_delta_rule_packed_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # PPU MODIFICATION: begin
    from vllm_sail.attention.pla_decode import get_sail_cuda_pla_k_last_packed

    # PPU MODIFICATION: end
    if mixed_qkv.ndim != 2:
        raise ValueError(
            f"`mixed_qkv` must be a 2D tensor (got ndim={mixed_qkv.ndim})."
        )
    if mixed_qkv.stride(-1) != 1:
        raise ValueError("`mixed_qkv` must be contiguous in the last dim.")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"`a` and `b` must be 2D tensors (got a.ndim={a.ndim}, b.ndim={b.ndim})."
        )
    if a.stride(-1) != 1 or b.stride(-1) != 1:
        raise ValueError("`a`/`b` must be contiguous in the last dim.")
    if A_log.ndim != 1 or dt_bias.ndim != 1:
        raise ValueError("`A_log`/`dt_bias` must be 1D tensors.")
    if A_log.stride(0) != 1 or dt_bias.stride(0) != 1:
        raise ValueError("`A_log`/`dt_bias` must be contiguous.")
    if ssm_state_indices.ndim != 1:
        raise ValueError(
            f"`ssm_state_indices` must be 1D for packed decode (got ndim={ssm_state_indices.ndim})."
        )
    if not out.is_contiguous():
        raise ValueError("`out` must be contiguous.")

    dev = mixed_qkv.device
    if (
        a.device != dev
        or b.device != dev
        or A_log.device != dev
        or dt_bias.device != dev
        or initial_state.device != dev
        or out.device != dev
        or ssm_state_indices.device != dev
    ):
        raise ValueError("All inputs must be on the same device.")

    B = mixed_qkv.shape[0]
    if a.shape[0] != B or b.shape[0] != B:
        raise ValueError(
            "Mismatched batch sizes: "
            f"mixed_qkv.shape[0]={B}, a.shape[0]={a.shape[0]}, b.shape[0]={b.shape[0]}."
        )
    if ssm_state_indices.shape[0] != B:
        raise ValueError(
            f"`ssm_state_indices` must have shape [B] (got {tuple(ssm_state_indices.shape)}; expected ({B},))."
        )

    if initial_state.ndim != 4:
        raise ValueError(
            f"`initial_state` must be a 4D tensor (got ndim={initial_state.ndim})."
        )
    if initial_state.stride(-1) != 1:
        raise ValueError("`initial_state` must be contiguous in the last dim.")
    HV, V, K = initial_state.shape[-3:]
    if a.shape[1] != HV or b.shape[1] != HV:
        raise ValueError(
            f"`a`/`b` must have shape [B, HV] with HV={HV} (got a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)})."
        )
    if A_log.numel() != HV or dt_bias.numel() != HV:
        raise ValueError(
            f"`A_log` and `dt_bias` must have {HV} elements (got A_log.numel()={A_log.numel()}, dt_bias.numel()={dt_bias.numel()})."
        )
    if out.shape != (B, 1, HV, V):
        raise ValueError(
            f"`out` must have shape {(B, 1, HV, V)} (got out.shape={tuple(out.shape)})."
        )

    qkv_dim = mixed_qkv.shape[1]
    qk_dim = qkv_dim - HV * V
    if qk_dim <= 0 or qk_dim % 2 != 0:
        raise ValueError(
            f"Invalid packed `mixed_qkv` last dim={qkv_dim} for HV={HV}, V={V}."
        )
    q_dim = qk_dim // 2
    if q_dim % K != 0:
        raise ValueError(f"Invalid packed Q size {q_dim}: must be divisible by K={K}.")
    H = q_dim // K
    if H <= 0 or HV % H != 0:
        raise ValueError(
            f"Invalid head config inferred from mixed_qkv: H={H}, HV={HV}."
        )

    BK = triton.next_power_of_2(K)
    if triton.cdiv(K, BK) != 1:
        raise ValueError(
            f"Packed decode kernel only supports NK=1 (got K={K}, BK={BK})."
        )
    BV = min(triton.next_power_of_2(V), 32)
    if (
        K == V == 128
        and not current_platform.is_ppu()
        and current_platform.is_cuda()
        and current_platform.is_device_capability_family(100, device_id=dev.index)
    ):
        # Tuned on GB300; validated on GB200 and with 148 SMs.
        # MRv2 decode warmup/graph capture compiles the reachable BV variants.
        if B * HV <= 64:
            BV = 4
        elif initial_state.dtype == torch.float32:
            BV = 8
    elif (
        K == V == 128
        and initial_state.dtype == torch.float32
        and not current_platform.is_ppu()
        and current_platform.is_cuda()
        and current_platform.is_device_capability(90, device_id=dev.index)
    ):
        # Upstream's H200 tuning applies to CUDA; PPU keeps its own tile.
        BV = 4
    num_stages = 3
    num_warps = 1

    stride_mixed_qkv_tok = mixed_qkv.stride(0)
    stride_a_tok = a.stride(0)
    stride_b_tok = b.stride(0)
    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = initial_state.stride(0)
    stride_indices_seq = ssm_state_indices.stride(0)

    NV = triton.cdiv(V, BV)
    # CUDA limits grid Y/Z dimensions to 65535.
    split_batch_head_grid = B * HV > 65535
    grid = (NV, HV, B) if split_batch_head_grid else (NV, B * HV)
    # PPU MODIFICATION: begin

    # PPU SAIL CUDA PLA fast path (VLLM_SAIL_USE_PLA); see sail_cuda_pla.py
    # for the gating conditions. The packed CUDA kernel requires a float32
    # ssm state pool, int32 state indices and K == V == 128; unsatisfied
    # calls fall back to Triton.
    cuda_k_last_packed = get_sail_cuda_pla_k_last_packed()
    if (
        cuda_k_last_packed is not None
        and initial_state.dtype == torch.float32
        and ssm_state_indices.dtype == torch.int32
        and B > 0
        and K == 128
        and V == 128
    ):
        out = cuda_k_last_packed(
            mixed_qkv,
            a,
            b,
            A_log,
            dt_bias,
            1.0,  # softplus_beta, matching the Triton kernel
            20.0,  # softplus_threshold, matching SOFTPLUS_THRESHOLD
            scale,
            initial_state,
            ssm_state_indices,
            out,
            use_qk_l2norm_in_kernel,
            torch.arange(B + 1, device=a.device, dtype=torch.int32),
            False,  # is_kda
            False,  # is_sglang: vLLM reserves slot 0 (NULL_BLOCK_ID), unlike
            # sglang whose PAD_SLOT_ID is -1 with slot 0 a valid row.
        )
        return out, initial_state

    # PPU MODIFICATION: end
    fused_recurrent_gated_delta_rule_packed_decode_kernel[grid](
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        o=out,
        h0=initial_state,
        ht=initial_state,
        ssm_state_indices=ssm_state_indices,
        scale=scale,
        stride_mixed_qkv_tok=stride_mixed_qkv_tok,
        stride_a_tok=stride_a_tok,
        stride_b_tok=stride_b_tok,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        stride_indices_seq=stride_indices_seq,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        SOFTPLUS_THRESHOLD=20.0,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        SPLIT_BATCH_HEAD_GRID=split_batch_head_grid,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out, initial_state


fused_recurrent_gated_delta_rule_packed_decode = patch(
    "vllm.third_party.flash_linear_attention.ops.fused_recurrent",
    "fused_recurrent_gated_delta_rule_packed_decode",
    allow_missing=False,
    **_META,
)(
    bind_body(
        fused_recurrent_gated_delta_rule_packed_decode,
        import_module("vllm.third_party.flash_linear_attention.ops.fused_recurrent"),
    )
)

for _consumer in (
    "vllm.third_party.flash_linear_attention.ops",
    "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
):
    _module = sys.modules.get(_consumer)
    _original = getattr(fused_recurrent_gated_delta_rule_packed_decode, PATCH_MARKER)[
        "vllm.third_party.flash_linear_attention.ops.fused_recurrent.fused_recurrent_gated_delta_rule_packed_decode"
    ]
    if (
        _module is not None
        and getattr(_module, "fused_recurrent_gated_delta_rule_packed_decode", None)
        is _original
    ):
        patch(_consumer, "fused_recurrent_gated_delta_rule_packed_decode", **_META)(
            fused_recurrent_gated_delta_rule_packed_decode
        )


def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
    scale: float = None,
    initial_state: torch.Tensor = None,
    inplace_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    is_kda: bool = False,
):
    """
    Fused triton implementation of sigmoid gating delta rule update.
    This function uses a single fused kernel that combines both sigmoid gating
    computation and the recurrent delta rule update for better performance.
    """
    # PPU MODIFICATION: begin
    from vllm_sail.attention.pla_decode import get_sail_cuda_pla_k_last

    # PPU MODIFICATION: end
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 4

    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]}"
            f" when using `cu_seqlens`. Please flatten variable-length"
            f" inputs before processing."
        )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"
    # PPU MODIFICATION: begin

    # PPU SAIL CUDA PLA fast path (VLLM_SAIL_USE_PLA); see sail_cuda_pla.py
    # for the gating conditions. The CUDA kernel requires a float32 ssm
    # state pool, int32 state indices and K == V == 128.
    #
    # A single branch routes both decode modes to PLA:
    # - Non-spec decode (1D indices, num_accepted_tokens is None): uses the
    #   fast decode path (T=1) with direct state write-back
    #   (disable_state_update=False).
    # - Spec decode / MTP (2D indices, num_accepted_tokens is a tensor):
    #   uses the MTP_VERIFY path with per-timestep state writes to the 2D
    #   ssm_state_indices slots and disable_state_update=True to skip the
    #   redundant final write-back (per-timestep writes already cover all
    #   slots, matching the Triton kernel's INPLACE_FINAL_STATE behavior).
    # Unsatisfied calls fall back to Triton.
    cuda_k_last = get_sail_cuda_pla_k_last()
    is_spec = num_accepted_tokens is not None
    if (
        cuda_k_last is not None
        and inplace_final_state
        and initial_state is not None
        and initial_state.dtype == torch.float32
        and ssm_state_indices is not None
        and ssm_state_indices.dtype == torch.int32
        and (
            not is_spec
            or (
                ssm_state_indices.ndim == 2 and num_accepted_tokens.dtype == torch.int32
            )
        )
        and (cu_seqlens is None or cu_seqlens.dtype == torch.int32)
        and N > 0
        and K == 128
        and V == 128
    ):
        o_cuda = cuda_k_last(
            A_log,
            a.contiguous(),
            dt_bias,
            beta,
            threshold,
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            b.contiguous(),
            initial_state,
            ssm_state_indices,
            scale,
            use_qk_l2norm_in_kernel,
            cu_seqlens,
            is_kda,
            is_spec,  # disable_state_update: True for spec decode (per-timestep
            # writes to 2D slots already cover all state updates; skip
            # redundant final write-back), False for non-spec decode
            # (fast decode path writes state directly).
            None,  # intermediate_states_buffer (vLLM uses main state pool)
            None,  # intermediate_state_indices
            num_accepted_tokens,  # cache_steps_or_num_accept: None for non-spec
            # (no accepted tokens), tensor for spec decode
            None,  # retrieve_parent_token (vLLM does not use eagle tree)
            None,  # lower_bound (KDA-only)
            False,  # is_sglang: vLLM reserves slot 0 (NULL_BLOCK_ID), unlike
            # sglang whose PAD_SLOT_ID is -1 with slot 0 a valid row.
        )
        return o_cuda, initial_state
    # PPU MODIFICATION: end

    o = q.new_empty(NK, *v.shape)
    if inplace_final_state:
        final_state = initial_state
    else:
        final_state = q.new_empty(T, HV, V, K, dtype=initial_state.dtype)

    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = final_state.stride(0)

    if ssm_state_indices is None:
        stride_indices_seq, stride_indices_tok = 1, 1
    elif ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    else:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    grid = (NK, NV, N * HV)
    fused_sigmoid_gating_delta_rule_update_kernel[grid](
        A_log=A_log,
        a=a.contiguous(),
        b=b.contiguous(),
        dt_bias=dt_bias,
        beta=beta,
        threshold=threshold,
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        scale=scale,
        N=N,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        INPLACE_FINAL_STATE=inplace_final_state,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_KDA=is_kda,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = o.squeeze(0)
    return o, final_state


fused_sigmoid_gating_delta_rule_update = patch(
    "vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating",
    "fused_sigmoid_gating_delta_rule_update",
    allow_missing=False,
    **_META,
)(
    bind_body(
        fused_sigmoid_gating_delta_rule_update,
        import_module(
            "vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating"
        ),
    )
)

for _consumer in (
    "vllm.third_party.flash_linear_attention.ops",
    "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
):
    _module = sys.modules.get(_consumer)
    _original = getattr(fused_sigmoid_gating_delta_rule_update, PATCH_MARKER)[
        "vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating.fused_sigmoid_gating_delta_rule_update"
    ]
    if (
        _module is not None
        and getattr(_module, "fused_sigmoid_gating_delta_rule_update", None)
        is _original
    ):
        patch(_consumer, "fused_sigmoid_gating_delta_rule_update", **_META)(
            fused_sigmoid_gating_delta_rule_update
        )


# PPU MODIFICATION: begin
def _pla_prefill_supported(vllm_config: VllmConfig) -> bool:
    """Check whether the pla FlashQLA prefill kernel can run here.

    The resolver applies the ``VLLM_SAIL_USE_PLA`` and PPU platform gates and
    returns ``None`` when the kernel is unusable. FlashQLA additionally
    requires head dims of 128 and instantiates a fixed set of per-rank
    (num_v_heads, num_k_heads) configs, so validate the TP-sharded head counts
    against its whitelist before selecting the backend.
    """
    # PPU MODIFICATION: begin
    from vllm_sail.attention.pla_prefill import (
        get_sail_cuda_pla_prefill_fwd,
        get_sail_cuda_pla_prefill_head_configs,
    )

    # PPU MODIFICATION: end
    if get_sail_cuda_pla_prefill_fwd() is None:
        return False
    hf_cfg = vllm_config.model_config.hf_text_config
    head_k_dim = getattr(hf_cfg, "linear_key_head_dim", None)
    head_v_dim = getattr(hf_cfg, "linear_value_head_dim", None)
    num_k_heads = getattr(hf_cfg, "linear_num_key_heads", None)
    num_v_heads = getattr(hf_cfg, "linear_num_value_heads", None)
    if head_k_dim != 128 or head_v_dim != 128:
        return False
    tp_size = vllm_config.parallel_config.tensor_parallel_size
    if num_k_heads is None or num_v_heads is None:
        return False

    return (
        num_v_heads // tp_size,
        num_k_heads // tp_size,
    ) in get_sail_cuda_pla_prefill_head_configs()


# PPU MODIFICATION: end

_pla_prefill_supported = patch(
    "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
    "_pla_prefill_supported",
    allow_missing=True,
    **_META,
)(
    bind_body(
        _pla_prefill_supported,
        import_module("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"),
    )
)


def _resolve_gdn_prefill_backend(
    vllm_config: VllmConfig,
    # PPU MODIFICATION: begin
) -> tuple[
    str, Literal["triton", "flashinfer", "cutedsl", "aiter_flydsl", "sycl", "pla"]
]:
    # PPU MODIFICATION: end
    """Resolve GDN prefill backend.

    The SYCL kernel is chosen when ``requested in ["sycl", "auto"]``, the
    platform is XPU and ``_xpu_C::gdn_attention`` is built. Unlike the other
    backends it is a fused whole-layer op, so selecting it also replaces the
    conv1d and decode kernels, not just prefill.

    FlashInfer's GDN prefill kernel is chosen when:
    * ``requested in ["flashinfer", "auto"]``;
    * ``platform == cuda``;
    * one of the following:
      - Hopper (SM90) - no further constraints;
      - Blackwell (SM10.x) with ``head_k_dim == 128``, ``cuda_runtime >= 13``;
      - Blackwell (SM12.x) with ``head_k_dim == 128``, ``cuda_runtime >= 13``.

    In-tree CuteDSL GDN prefill kernel is chosen when:
    * "cutedsl" is requested; (opt-in only)
    * Blackwell (SM10.x) with ``head_k_dim == 128``;

    AITER FlyDSL GDN prefill kernels are chosen when:
    * "aiter_flydsl" is requested, or "auto" is requested on gfx942/gfx950;
    * ROCm AITER exposes the optimized VK prefill API and FlyDSL kernels;
    * the model uses BF16 activations with ``head_k_dim == head_v_dim == 128``.
    # PPU MODIFICATION: begin

    The PPU pla (FlashQLA) CUDA kernel is chosen when:
    * ``requested in ["pla", "auto"]``;
    * ``platform == ppu``;
    * ``VLLM_SAIL_USE_PLA`` is enabled (the default) and the ``pla`` package is
      importable;
    * head dims are 128 and the TP-sharded head counts are supported.
    # PPU MODIFICATION: end
    """
    additional_config = vllm_config.additional_config
    backend_cfg = (
        additional_config.get("gdn_prefill_backend", "auto")
        if isinstance(additional_config, dict)
        else "auto"
    )
    backend = str(backend_cfg).strip().lower()

    if backend == "sycl" and not current_platform.is_xpu():
        raise ValueError("GDN prefill backend 'sycl' is only supported on XPU.")

    if current_platform.is_xpu():
        if backend in ("auto", "sycl") and xpu_gdn_kernel_available():
            return backend, "sycl"
        if backend == "sycl":
            raise ValueError(
                "GDN prefill backend 'sycl' was requested but "
                "torch.ops._xpu_C.gdn_attention is not available. Rebuild "
                "vllm-xpu-kernels with VLLM_GDN_ENABLED."
            )
        return backend, "triton"

    head_k_dim = getattr(
        vllm_config.model_config.hf_text_config, "linear_key_head_dim", None
    )
    head_v_dim = getattr(
        vllm_config.model_config.hf_text_config, "linear_value_head_dim", None
    )

    if current_platform.is_rocm():
        if backend == "aiter_flydsl":
            if not rocm_aiter_ops.is_gdn_flydsl_prefill_available():
                raise RuntimeError(
                    "GDN prefill backend 'aiter_flydsl' was requested but the "
                    "AITER FlyDSL prefill kernels are not importable. Check "
                    "that VLLM_ROCM_USE_AITER=1, that this is a CDNA 3 or "
                    "newer GPU, and that the installed AITER exports them."
                )
            reasons = _aiter_flydsl_unsupported_reasons(
                vllm_config, head_k_dim, head_v_dim
            )
            if reasons:
                logger.warning_once(
                    "GDN prefill backend 'aiter_flydsl' was requested but %s. "
                    "Falling back to Triton/FLA.",
                    "; ".join(reasons),
                )
                return backend, "triton"
            return backend, "aiter_flydsl"
        if backend == "auto":
            from vllm.platforms.rocm import on_mi3xx

            if (
                on_mi3xx()
                and not _aiter_flydsl_unsupported_reasons(
                    vllm_config, head_k_dim, head_v_dim
                )
                and rocm_aiter_ops.is_gdn_flydsl_prefill_available()
            ):
                return backend, "aiter_flydsl"
        return backend, "triton"

    if backend == "aiter_flydsl":
        raise RuntimeError(
            "GDN prefill backend 'aiter_flydsl' was requested but it is a ROCm "
            f"AITER backend and the current platform is {current_platform.device_name}."
        )

    if not current_platform.is_cuda():
        # PPU MODIFICATION: begin
        return backend, "triton"

    if current_platform.is_ppu():
        if backend in ("pla", "auto") and _pla_prefill_supported(vllm_config):
            return backend, "pla"
        # PPU MODIFICATION: end
        return backend, "triton"

    supports_flashinfer = False
    supports_cutedsl = False

    if current_platform.is_device_capability(90):
        supports_flashinfer = True
    elif (
        current_platform.is_device_capability_family(100)
        and head_k_dim == 128
        and current_platform.get_cuda_runtime_major() >= 13
    ):
        supports_flashinfer = True
        supports_cutedsl = True
    elif (
        current_platform.is_device_capability_family(120)
        and head_k_dim == 128
        and current_platform.get_cuda_runtime_major() >= 13
    ):
        # The in-tree CuteDSL kernel targets SM100 only, so it stays off here.
        supports_flashinfer = True

    if backend in ["flashinfer", "auto"] and supports_flashinfer:
        return backend, "flashinfer"
    if backend == "cutedsl" and supports_cutedsl:
        return backend, "cutedsl"
    return backend, "triton"


_resolve_gdn_prefill_backend = patch(
    "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
    "_resolve_gdn_prefill_backend",
    allow_missing=False,
    **_META,
)(
    bind_body(
        _resolve_gdn_prefill_backend,
        import_module("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"),
    )
)


def _log_gdn_backend_decision(
    vllm_config: VllmConfig,
    requested_backend: str,
    active_backend: str,
) -> None:
    """Log the GDN prefill backend choice in the attention-selector style."""
    head_k_dim = getattr(
        vllm_config.model_config.hf_text_config, "linear_key_head_dim", None
    )

    if current_platform.is_cpu():
        logger.info_once(
            "Using %s GDN prefill kernel (head_k_dim=%s).",
            "CPU",
            head_k_dim,
        )
        return
    elif current_platform.is_xpu():
        return

    chosen = {
        "flashinfer": "FlashInfer",
        "cutedsl": "CuteDSL",
        "aiter_flydsl": "AITER FlyDSL",
        # PPU MODIFICATION: begin
        "pla": "PLA FlashQLA",
        # PPU MODIFICATION: end
        "triton": "Triton/FLA",
        "sycl": "SYCL",
    }[active_backend]
    logger.info_once(
        "Using %s GDN prefill kernel (requested=%s, head_k_dim=%s).",
        chosen,
        requested_backend,
        head_k_dim,
    )
    if active_backend == "flashinfer" and current_platform.is_device_capability(90):
        logger.warning_once(
            "FlashInfer GDN prefill is JIT-compiled; first run may take a "
            "while. Set --gdn-prefill-backend triton to skip JIT.",
        )


_log_gdn_backend_decision = patch(
    "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
    "_log_gdn_backend_decision",
    allow_missing=False,
    **_META,
)(
    bind_body(
        _log_gdn_backend_decision,
        import_module("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"),
    )
)


# PPU MODIFICATION: begin
def forward_pla(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    core_attn_out: torch.Tensor | None = None,
    aiter_prefill_metadata: object | None = None,
):
    # PPU MODIFICATION: begin
    from vllm_sail.attention.pla_prefill import get_sail_cuda_pla_prefill_fwd

    # PPU MODIFICATION: end
    assert not use_qk_l2norm_in_kernel, (
        "The pla prefill backend expects l2-normalized q/k; run "
        "fused_post_conv_prep with apply_l2norm=True."
    )

    q = q.squeeze(0).contiguous()
    k = k.squeeze(0).contiguous()
    v = v.squeeze(0).contiguous()
    g = g.squeeze(0).to(torch.float32).contiguous()
    beta = beta.squeeze(0).to(torch.float32).contiguous()
    # pla expects [N, H, DK, DV] fp32 while vLLM stores [N, H, DV, DK].
    # A cold start passes None, which pla accepts and zero-initializes
    # internally, so it must not be transposed.
    state = (
        initial_state.transpose(-1, -2)
        .contiguous()
        .to(torch.float32, non_blocking=True)
        if initial_state is not None
        else None
    )

    pla_chunk_gated_delta_rule_fwd = get_sail_cuda_pla_prefill_fwd()
    _, _, o, _, final_state = pla_chunk_gated_delta_rule_fwd(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        g.unsqueeze(0),
        beta.unsqueeze(0),
        scale=None,
        initial_state=state,
        cu_seqlens=cu_seqlens,
        output_final_state=output_final_state,
    )
    if final_state is not None:
        final_state = final_state.transpose(-1, -2)
    if core_attn_out is not None:
        o_flat = o.reshape(-1)
        co_flat = core_attn_out.reshape(-1)
        co_flat[: o_flat.numel()].copy_(o_flat)
    return o, final_state


# PPU MODIFICATION: end

forward_pla = patch(
    "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
    "ChunkGatedDeltaRule.forward_pla",
    allow_missing=True,
    **_META,
)(
    bind_body(
        forward_pla,
        import_module("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"),
    )
)


def __init__(self) -> None:
    # PPU MODIFICATION: begin
    super(ChunkGatedDeltaRule, self).__init__()
    # PPU MODIFICATION: end
    vllm_config = get_current_vllm_config()
    backend, active_backend = _resolve_gdn_prefill_backend(vllm_config)
    self.gdn_prefill_backend = active_backend

    # PPU MODIFICATION: begin
    if backend in ("flashinfer", "cutedsl", "pla") and (active_backend != backend):
        # PPU MODIFICATION: end
        logger.warning_once(
            "GDN prefill backend '%s' is selected but cannot use this "
            "kernel on the current platform. Falling back to Triton/FLA.",
            backend,
        )
    _log_gdn_backend_decision(vllm_config, backend, active_backend)

    if active_backend == "flashinfer":
        self._forward_method = self.forward_cuda
    elif active_backend == "cutedsl":
        self._forward_method = self.forward_cutedsl
    elif active_backend == "aiter_flydsl":
        self._forward_method = self.forward_aiter_flydsl
    # PPU MODIFICATION: begin
    elif active_backend == "pla":
        self._forward_method = self.forward_pla
    # PPU MODIFICATION: end
    else:
        self._forward_method = self.forward_native


__init__ = patch(
    "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
    "ChunkGatedDeltaRule.__init__",
    allow_missing=False,
    **_META,
)(
    bind_body(
        __init__,
        import_module("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"),
    )
)
