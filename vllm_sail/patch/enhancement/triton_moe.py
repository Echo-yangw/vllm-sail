# ruff: noqa: E731, W291, W293, UP037
# ruff: noqa: F821
# Copied bodies resolve globals in their upstream module through bind_body.
# SPDX-License-Identifier: Apache-2.0
"""Consume PPU UP/DOWN/VALU tuning through upstream MoE dispatch seams."""

from __future__ import annotations

import functools
import sys

from vllm.model_executor.layers.fused_moe import fused_moe as _moe
from vllm.model_executor.layers.fused_moe.experts import triton_moe as _experts
from vllm.platforms import current_platform

from vllm_sail.patch.bodies import bind_body
from vllm_sail.patch.utils import patch

_META = dict(
    reason="PPU tuned configurations require per-projection launch parameters, VALU and PPU Triton kernels.",
    affected_versions=">=0.32.0,<0.33.0",
    remove_when="Upstream supports registration of the complete Triton MoE launch strategy.",
)


def invoke_fused_moe_triton_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: torch.Tensor | None,
    B_scale: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    block_shape: list[int] | None = None,
    B_bias: torch.Tensor | None = None,
    # PPU MODIFICATION: begin
    use_valu: bool = False,
    # PPU MODIFICATION: end
):
    # PPU MODIFICATION: begin
    from vllm_sail.model_executor.layers.fused_moe.triton_kernels import (
        fused_moe_kernel,
    )

    # PPU MODIFICATION: end
    assert topk_weights is not None or not mul_routed_weight
    assert topk_weights is None or topk_weights.stride(1) == 1
    assert sorted_token_ids is None or sorted_token_ids.stride(0) == 1

    if use_fp8_w8a8:
        SWAP_AB = enable_swap_ab(config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"])
    else:
        SWAP_AB = False

    # Quantized weights always carry a B_scale (see the asserts below); key off
    # that rather than enumerating quant flags, which misses w8a16-fp8/nvfp4/etc.
    is_quantized = B_scale is not None
    warn_if_moe_use_td_ineffective("TRITON", is_quantized=is_quantized)

    # TD path is unvalidated under quantization; fall back to the pointer path.
    use_td = resolve_moe_use_td() and not is_quantized
    if use_td:
        # The TD path builds a tensor descriptor inside the kernel, which
        # requires a PyTorch-backed scratch allocator to be registered
        # (Triton raises "no allocator was set" otherwise on CUDA).
        set_triton_allocator(A.device)

    if use_fp8_w8a8 or use_int8_w8a8:
        assert B_scale is not None
        assert block_shape is None or triton.cdiv(
            B.size(-2), block_shape[0]
        ) == B_scale.size(-2)
        assert block_shape is None or triton.cdiv(
            B.size(-1), block_shape[1]
        ) == B_scale.size(-1)
    elif use_int8_w8a16 or use_int4_w4a16:
        assert B_scale is not None
        assert block_shape is None or block_shape[0] == 0
    else:
        assert A_scale is None
        assert B_scale is None

    M = A.size(0)
    num_tokens = M * top_k
    if sorted_token_ids is not None:
        EM = sorted_token_ids.size(0)
        if A.size(0) < config["BLOCK_SIZE_M"]:
            # optimize for small batch_size.
            # We assume that top_ids of each token is unique,
            # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
            # and we can skip some invalid blocks.
            EM = min(
                sorted_token_ids.size(0), A.size(0) * top_k * config["BLOCK_SIZE_M"]
            )
    else:
        EM = num_tokens * config["BLOCK_SIZE_M"]
    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(B.size(1), META["BLOCK_SIZE_N"]),
    )
    HAS_BIAS = B_bias is not None

    config = config.copy()
    config["SPLIT_K"] = 1
    BLOCK_SIZE_K = config.pop("BLOCK_SIZE_K")
    if block_shape is not None:
        BLOCK_SIZE_K = min(BLOCK_SIZE_K, min(block_shape[0], block_shape[1]))
    if use_td and A.size(1) % BLOCK_SIZE_K != 0:
        # TD gather/load feeding tl.dot with a non-block-aligned K
        # miscompiles (~74% of output elements wrong) on real HW;
        # this is a compiler-codegen issue, not a Python-maskable
        # boundary gap. Fall back to the pointer-arith path.
        logger.warning_once(
            "Disabling VLLM_TRITON_USE_TD for this MoE launch: K=%d is not "
            "a multiple of BLOCK_SIZE_K=%d, which triggers a known "
            "Triton tensor-descriptor + tl.dot miscompilation.",
            A.size(1),
            BLOCK_SIZE_K,
        )
        use_td = False

    # Triton treats 0-D tensor arguments as scalar values, but the kernel
    # loads tensor-wise activation scales through a pointer.
    if A_scale is not None and A_scale.ndim == 0:
        A_scale = A_scale.reshape(1)

    # PPU MODIFICATION: begin

    # `tl.aiu_load` below is a PPU Triton-fork builtin, so the block-ptr path it
    # gates must never be compiled on NVIDIA. has_device_capability() is a >=
    # test and would match sm_90+, so is_ppu() is required on top of it.
    even_Ks = (
        B.size(2) % BLOCK_SIZE_K == 0
        and current_platform.is_ppu()
        and current_platform.has_device_capability((8, 9))
    )

    # PPU MODIFICATION: end
    fused_moe_kernel[grid](
        A,
        B,
        C,
        B_bias,
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.size(1),
        B.size(2),
        EM,
        num_tokens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        A_scale.stride(0) if A_scale is not None and A_scale.ndim == 2 else 0,
        A_scale.stride(1) if A_scale is not None and A_scale.ndim == 2 else 0,
        B_scale.stride(0) if B_scale is not None else 0,
        B_scale.stride(2) if B_scale is not None and B_scale.ndim == 3 else 0,
        B_scale.stride(1)
        if B_scale is not None and B_scale.ndim >= 2 and B_scale.size(1) > 1
        else 0,
        B_bias.stride(0) if B_bias is not None else 0,
        B_bias.stride(1) if B_bias is not None else 0,
        0 if block_shape is None else block_shape[0],
        0 if block_shape is None else block_shape[1],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        per_channel_quant=per_channel_quant,
        naive_block_assignment=(sorted_token_ids is None),
        HAS_BIAS=HAS_BIAS,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        SWAP_AB=SWAP_AB,
        USE_TD=use_td,
        # PPU MODIFICATION: begin
        use_valu=use_valu,
        even_Ks=even_Ks,
        # PPU MODIFICATION: end
        **config,
    )
    # PPU MODIFICATION: begin

    # PPU MODIFICATION: end


def invoke_fused_moe_wna16_triton_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    B_scale: torch.Tensor | None,
    B_zp: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    block_shape: list[int] | None,
):
    assert B_scale is not None and B_scale.ndim == 3
    assert B_zp is None or B_zp.ndim == 3
    assert block_shape is not None and block_shape[0] == 0

    M = A.size(0)
    num_tokens = M * top_k

    EM = sorted_token_ids.size(0)
    if A.size(0) < config["BLOCK_SIZE_M"]:
        # optimize for small batch_size.
        # We assume that top_ids of each token is unique,
        # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
        # and we can skip some invalid blocks.
        EM = min(sorted_token_ids.size(0), A.size(0) * top_k * config["BLOCK_SIZE_M"])
    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(B.size(1), META["BLOCK_SIZE_N"]),
    )
    config = config.copy()
    # PPU MODIFICATION: begin
    config["SPLIT_K"] = 1
    # PPU MODIFICATION: end
    config.update(
        get_moe_wna16_block_config(
            config=config,
            use_moe_wna16_cuda=False,
            num_valid_tokens=num_tokens,
            size_k=A.size(1),
            size_n=B.size(1),
            num_experts=B.size(1),
            group_size=block_shape[1],
            real_top_k=top_k,
            block_size_m=config["BLOCK_SIZE_M"],
        )
    )

    fused_moe_kernel_gptq_awq[grid](
        A,
        B,
        C,
        B_scale,
        B_zp,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.size(1),
        A.size(1),
        EM,
        num_tokens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        B_scale.stride(0),
        B_scale.stride(2),
        B_scale.stride(1),
        B_zp.stride(0) if B_zp is not None else 0,
        B_zp.stride(2) if B_zp is not None else 0,
        B_zp.stride(1) if B_zp is not None else 0,
        block_k_diviable=A.size(1) % config["BLOCK_SIZE_K"] == 0,
        group_size=block_shape[1],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        has_zp=B_zp is not None,
        use_int4_w4a16=use_int4_w4a16,
        use_int8_w8a16=use_int8_w8a16,
        **config,
    )


def dispatch_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: torch.Tensor | None,
    B_scale: torch.Tensor | None,
    B_zp: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    block_shape: list[int] | None = None,
    B_bias: torch.Tensor | None = None,
    # PPU MODIFICATION: begin
    use_valu: bool | None = None,
    # PPU MODIFICATION: end
) -> None:
    assert topk_weights is not None or not mul_routed_weight
    assert topk_weights is None or topk_weights.stride(1) == 1
    assert sorted_token_ids is None or sorted_token_ids.stride(0) == 1

    M = A.size(0)
    num_tokens = M * top_k

    if (use_int8_w8a16 or use_int4_w4a16) and (
        block_shape is not None and block_shape[1] > 0
    ):
        assert B_bias is None

        use_moe_wna16_cuda = should_moe_wna16_use_cuda(
            num_valid_tokens=num_tokens,
            group_size=block_shape[1],
            num_experts=B.size(0),
            bit=4 if use_int4_w4a16 else 8,
        )

        if use_moe_wna16_cuda:
            invoke_fused_moe_wna16_cuda_kernel(
                A,
                B,
                C,
                B_scale,
                B_zp,
                topk_weights,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                mul_routed_weight,
                top_k,
                config,
                block_shape,
            )
            return
        invoke_fused_moe_wna16_triton_kernel(
            A,
            B,
            C,
            B_scale,
            B_zp,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight,
            top_k,
            config,
            compute_type,
            use_int8_w8a16,
            use_int4_w4a16,
            block_shape,
        )

    else:
        invoke_fused_moe_triton_kernel(
            A,
            B,
            C,
            A_scale,
            B_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight,
            top_k,
            config,
            compute_type,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            use_int4_w4a16,
            per_channel_quant,
            block_shape,
            B_bias,
            # PPU MODIFICATION: begin
            use_valu,
            # PPU MODIFICATION: end
        )


def get_config_file_name(
    # PPU MODIFICATION: begin
    E: int,
    N: int,
    dtype: str | None,
    block_shape: list[int] | None = None,
    use_moe_wna16_cuda: bool = False,
    # PPU MODIFICATION: end
) -> str:
    device_name = get_device_name_as_file_name()
    # Set device_name to H200 if a device from the H200 family is detected
    if "H200" in device_name.split("_"):
        device_name = "NVIDIA_H200"
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    block_shape_selector = (
        "" if not block_shape or not all(block_shape) else f",block_shape={block_shape}"
    ).replace(" ", "")
    # PPU MODIFICATION: begin
    use_cuda_selector = "" if not use_moe_wna16_cuda else ",use_cuda=True"
    return f"E={E},N={N},device_name={device_name}{dtype_selector}{block_shape_selector}{use_cuda_selector}.json"  # noqa: E501
    # PPU MODIFICATION: end


def get_moe_configs(
    E: int,
    N: int,
    dtype: str | None,
    block_n: int | None = None,
    block_k: int | None = None,
    # PPU MODIFICATION: begin
    use_moe_wna16_cuda: bool = False,
    # PPU MODIFICATION: end
) -> dict[int, Any] | None:
    """
    Return optimized configurations for the fused MoE kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the fused_moe kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """

    # Avoid optimizing for the batch invariant case. Use default config
    if envs.VLLM_BATCH_INVARIANT:
        return None

    # First look up if an optimized configuration is available in the configs
    # directory
    block_shape = [block_n, block_k] if block_n and block_k else None
    # PPU MODIFICATION: begin
    json_file_name = get_config_file_name(E, N, dtype, block_shape, use_moe_wna16_cuda)
    # PPU MODIFICATION: end

    config_file_paths = []

    # note that we prioritize user defined config
    user_defined_config_folder = envs.VLLM_TUNED_CONFIG_FOLDER
    if user_defined_config_folder is not None:
        user_defined_config_file_path = os.path.join(
            user_defined_config_folder, json_file_name
        )
        config_file_paths.append(user_defined_config_file_path)

    default_config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name
    )
    config_file_paths.append(default_config_file_path)

    for config_file_path in config_file_paths:
        if os.path.exists(config_file_path):
            with open(config_file_path) as f:
                logger.info_once(
                    "Using configuration from %s for MoE layer.",
                    config_file_path,
                    scope="global",
                )
                # If a configuration has been found, return it
                tuned_config = json.load(f)
                # Delete triton_version from tuned_config
                tuned_config.pop("triton_version", None)
                return {int(key): val for key, val in tuned_config.items()}

    # If no optimized configuration is available, we will use the default
    # configuration
    logger.warning_once(
        "Using default MoE config. Performance might be sub-optimal! "
        "Config file not found at %s",
        ", ".join(config_file_paths),
    )
    return None


def get_moe_wna16_block_config(
    config: dict[str, int],
    use_moe_wna16_cuda: bool,
    num_valid_tokens: int,
    size_k: int,
    size_n: int,
    num_experts: int,
    group_size: int,
    real_top_k: int,
    block_size_m: int,
):
    if "BLOCK_SIZE_N" in config and "BLOCK_SIZE_K" in config:
        # optimal block config is set
        return {}
    if not use_moe_wna16_cuda:
        # triton moe wna16 kernel
        if num_valid_tokens // real_top_k == 1:
            # if bs=1, use a smaller BLOCK_SIZE_N
            return {"BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64}
        else:
            return {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}
    else:
        # cuda moe wna16 kernel
        # set default block_size 128, and increase them when num_blocks
        # is too large.
        block_size_n = 128
        block_size_k = 128
        if block_size_k <= group_size:
            block_size_k = group_size

        num_n_blocks = size_k // block_size_k
        num_k_blocks = size_n // block_size_k
        num_m_blocks = (
            num_valid_tokens + block_size_m - 1
        ) / block_size_m + num_experts
        if num_valid_tokens // real_top_k <= block_size_m:
            num_m_blocks = min(num_m_blocks, num_valid_tokens)
        num_blocks = num_m_blocks * num_n_blocks * num_k_blocks

        if size_k % 256 == 0 and num_blocks >= 256 and block_size_k < 256:
            block_size_k = 256
            num_blocks = num_blocks // (256 // block_size_k)

        if (
            num_m_blocks <= 16
            and size_k % (block_size_k * 2) == 0
            and size_k % (block_size_k * 2) == 0
            and block_size_k <= 512
            and num_blocks >= 512
        ):
            block_size_k = block_size_k * 2
            num_blocks = num_blocks // 2

        if num_blocks > 1024:
            block_size_n = 256
            num_n_blocks = num_n_blocks // 2
            num_blocks = num_blocks // 2

        # PPU MODIFICATION: begin
        if current_platform.is_ppu():
            # size_k must divisible by BLOCK_SIZE_K
            # BLOCK_SIZE_K must divisible by group_size
            if size_k % block_size_k or block_size_k % group_size:
                block_size_k = group_size

            # BLOCK_SIZE_K // group_size must be one of [1, 2, 4, 8]
            if block_size_k // group_size > 8:
                block_size_k = group_size * 8
        else:
            if size_n <= 1024 and num_blocks >= 1024:
                # The kernel performance got much better with BLOCK_SIZE_N=1024
                # when num_blocks is large, event when N is small.
                # Not sure why, maybe it force the CUDA SM process only one block
                # at the same time.
                block_size_n = 1024
        # PPU MODIFICATION: end

        # Ensure BLOCK_SIZE_K is a divisor of size_k for CUDA kernel compatibility
        block_size_k = _ensure_block_size_k_divisible(size_k, block_size_k, group_size)

        return {"BLOCK_SIZE_N": block_size_n, "BLOCK_SIZE_K": block_size_k}


def try_get_optimal_moe_config(
    w1_shape: tuple[int, ...],
    w2_shape: tuple[int, ...],
    top_k: int,
    dtype: str | None,
    M: int,
    block_shape: list[int] | None = None,
) -> dict[str, int]:
    from vllm.model_executor.layers.fused_moe import get_config

    override_config = get_config()
    if override_config:
        config = override_config
    else:
        # First try to load optimal config from the file
        E, _, N = w2_shape
        if dtype == "int4_w4a16":
            N = N * 2
        block_n = block_shape[0] if block_shape else 0
        block_k = block_shape[1] if block_shape else 0
        # PPU MODIFICATION: begin

        if (
            current_platform.is_ppu()
            and dtype in ["int4_w4a16", "int8_w8a16"]
            and block_shape is not None
            and should_moe_wna16_use_cuda(
                M * top_k, block_shape[1], E, 4 if dtype == "int4_w4a16" else 8
            )
        ):
            configs = get_moe_configs(
                E, N, dtype, block_n, block_k, use_moe_wna16_cuda=True
            )
        else:
            configs = get_moe_configs(E, N, dtype, block_n, block_k)
        # PPU MODIFICATION: end

        if configs:
            # If an optimal configuration map has been found, look up the
            # optimal config
            config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
        else:
            # Else use the default config
            config = get_default_config(M, E, N, w1_shape[2], top_k, dtype, block_shape)
    return config


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    ocp_mx_scheme: str | None = None,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    w1_zp: torch.Tensor | None = None,
    w2_zp: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    block_shape: list[int] | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if ocp_mx_scheme is not None:
        raise NotImplementedError(
            f"Using ocp_mx_scheme={ocp_mx_scheme} in functional fused_experts call is "
            "deprecated. Please use OCP_MXQuantizationEmulationTritonExperts."
        )

    # Convert string activation to enum for internal use
    activation_enum = MoEActivation.from_str(activation)

    # PPU MODIFICATION: begin
    from vllm_sail.model_executor.layers.fused_moe.triton_kernels import (
        moe_sum_reduce_triton,
    )

    # PPU MODIFICATION: end
    # Check constraints.
    if use_int4_w4a16:
        assert hidden_states.size(1) // 2 == w1.size(2), "Hidden size mismatch"
    else:
        assert hidden_states.size(1) == w1.size(2), (
            f"Hidden size mismatch {hidden_states.size(1)} != {w1.size(2)}"
        )

    assert topk_weights.size() == topk_ids.size(), "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    num_tokens = hidden_states.size(0)
    E, N, _ = w1.size()
    K = w2.size(1)
    if global_num_experts == -1:
        global_num_experts = E
    top_k_num = topk_ids.size(1)

    M = num_tokens

    # PPU MODIFICATION: begin
    from vllm.model_executor.layers.fused_moe.config import _get_config_dtype_str

    # PPU MODIFICATION: end
    config_dtype = _get_config_dtype_str(
        use_fp8_w8a8=use_fp8_w8a8,
        # PPU MODIFICATION: begin
        use_int8_w8a8=use_int8_w8a8,
        # PPU MODIFICATION: end
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        dtype=hidden_states.dtype,
    )

    # Note: for use_int8_w8a16 or use_int4_w4a16, the activations are
    # quantized prior to calling fused_experts.
    quant_dtype = _get_config_quant_dtype(
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
    )

    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.size(),
        w2.size(),
        top_k_num,
        config_dtype,
        block_shape=block_shape,
    )

    config = get_config_func(M)

    # We can reuse the memory between these because by the time we need
    # cache3, we're done with cache1
    cache13 = torch.empty(
        M * top_k_num * max(N, K),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache1 = cache13[: M * top_k_num * N].view(M, top_k_num, N)
    intermediate_cache3 = cache13[: M * top_k_num * K].view(M, top_k_num, K)

    # This needs separate memory since it's used concurrently with cache1
    activation_out_dim = mk.FusedMoEExpertsModular.adjust_N_for_activation(
        N, activation_enum
    )
    intermediate_cache2 = torch.empty(
        (M * top_k_num, activation_out_dim),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    if hidden_states.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif hidden_states.dtype == torch.float16:
        compute_type = tl.float16
    elif hidden_states.dtype == torch.float32:
        compute_type = tl.float32
    else:
        raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

    out_hidden_states = torch.empty_like(hidden_states)

    qhidden_states, a1q_scale = moe_kernel_quantize_input(
        A=hidden_states,
        A_scale=a1_scale,
        quant_dtype=quant_dtype,
        per_act_token_quant=per_channel_quant,
        block_shape=block_shape,
    )

    sorted_token_ids, expert_ids, num_tokens_post_padded = _prepare_expert_assignment(
        topk_ids,
        config,
        num_tokens,
        top_k_num,
        global_num_experts,
        expert_map,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        block_shape=block_shape,
        ignore_invalid_experts=True,
        weights_host_resident=_weights_host_resident(w1, w2),
    )

    dispatch_fused_moe_kernel(
        qhidden_states,
        w1,
        intermediate_cache1,
        a1q_scale,
        w1_scale,
        w1_zp,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        apply_router_weight_on_input,
        top_k_num,
        # PPU MODIFICATION: begin
        config.get("UP", config),
        # PPU MODIFICATION: end
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        block_shape=block_shape,
        B_bias=w1_bias,
        # PPU MODIFICATION: begin
        use_valu=config.get("USE_VALU", False),
        # PPU MODIFICATION: end
    )

    apply_moe_activation(
        activation_enum, intermediate_cache2, intermediate_cache1.view(-1, N)
    )

    qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(
        A=intermediate_cache2,
        A_scale=a2_scale,
        quant_dtype=quant_dtype,
        per_act_token_quant=per_channel_quant,
        block_shape=block_shape,
    )

    if expert_map is not None:
        intermediate_cache3.zero_()

    dispatch_fused_moe_kernel(
        qintermediate_cache2,
        w2,
        intermediate_cache3,
        a2q_scale,
        w2_scale,
        w2_zp,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        not apply_router_weight_on_input,
        1,
        # PPU MODIFICATION: begin
        config.get("DOWN", config),
        # PPU MODIFICATION: end
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        block_shape=block_shape,
        B_bias=w2_bias,
        # PPU MODIFICATION: begin
        use_valu=config.get("USE_VALU", False),
        # PPU MODIFICATION: end
    )

    # PPU MODIFICATION: begin
    if current_platform.is_ppu() and M > 1024:
        moe_sum_reduce_triton(
            intermediate_cache3.view(*intermediate_cache3.size()),
            out_hidden_states,
        )
    else:
        ops.moe_sum(
            intermediate_cache3.view(*intermediate_cache3.size()),
            out_hidden_states,
        )
    # PPU MODIFICATION: end

    return out_hidden_states


@functools.lru_cache(maxsize=1)
def _has_native_wna16():
    from vllm_sail.native.stubs import unsupported_ops

    return "moe_wna16_gemm" not in {op.rsplit("::", 1)[-1] for op in unsupported_ops()}


def should_moe_wna16_use_cuda(num_valid_tokens, group_size, num_experts, bit):
    from vllm_sail import envs as ppu_envs

    if ppu_envs.VLLM_SAIL_DISABLE_MOE_WNA16_CUDA:
        return False
    # CUDA-free wheels currently supply the numerically equivalent Triton path.
    # A stub's schema must never be mistaken for an available native kernel.
    if not _has_native_wna16():
        if ppu_envs.VLLM_SAIL_FORCE_MOE_WNA16_CUDA:
            raise NotImplementedError(
                "PPU moe_wna16_gemm is unavailable; unset VLLM_SAIL_FORCE_MOE_WNA16_CUDA to use Triton."
            )
        return False
    if ppu_envs.VLLM_SAIL_FORCE_MOE_WNA16_CUDA:
        return True
    return group_size in (32, 64, 128) and num_valid_tokens / num_experts <= (
        32 if bit == 4 else 3
    )


def _install(module, name, body, *, target_name=None):
    target_name = target_name or name
    owner = module
    parts = target_name.split(".")
    for part in parts[:-1]:
        owner = getattr(owner, part)
    original = getattr(owner, parts[-1])
    replacement = bind_body(body, module)
    if name == "get_moe_configs":
        replacement = functools.lru_cache()(replacement)
    # This helper needs plugin globals, while copied bodies need target globals.
    if name == "should_moe_wna16_use_cuda":
        replacement = body

    @functools.wraps(original)
    def dispatch(*args, **kwargs):
        if current_platform.is_ppu():
            if name == "fused_experts_impl":
                from vllm_sail import envs as ppu_envs

                if ppu_envs.VLLM_SAIL_NVTX_PROFILE:
                    # Registered fused_experts_op resolves this provider at call time.
                    import inspect

                    from vllm_sail.profiling.moe import moe_range

                    arguments = (
                        inspect.signature(replacement).bind(*args, **kwargs).arguments
                    )
                    with moe_range(
                        arguments["hidden_states"],
                        arguments["w1"],
                        arguments["topk_ids"],
                    ):
                        return replacement(*args, **kwargs)
            return replacement(*args, **kwargs)
        return original(*args, **kwargs)

    patch(module.__name__, target_name, **_META)(dispatch)
    if len(parts) == 1:
        for consumer_name in _CONSUMERS.get(name, ()):
            consumer = sys.modules.get(consumer_name)
            if consumer is not None and getattr(consumer, name, None) is original:
                patch(consumer_name, name, **_META)(dispatch)


_CONSUMERS = {
    "get_config_file_name": ["vllm.model_executor.layers.fused_moe.__init__"],
    "try_get_optimal_moe_config": [
        "vllm.model_executor.layers.fused_moe.experts.fused_batched_moe",
        "vllm.model_executor.layers.fused_moe.experts.triton_moe",
        "vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe",
        "vllm.lora.layers.utils",
    ],
    "invoke_fused_moe_triton_kernel": [
        "vllm.model_executor.layers.fused_moe.experts.triton_moe"
    ],
    "invoke_fused_moe_wna16_triton_kernel": [
        "vllm.model_executor.layers.fused_moe.experts.triton_moe"
    ],
}

_install(_moe, "invoke_fused_moe_triton_kernel", invoke_fused_moe_triton_kernel)
_install(
    _moe, "invoke_fused_moe_wna16_triton_kernel", invoke_fused_moe_wna16_triton_kernel
)
_install(_moe, "dispatch_fused_moe_kernel", dispatch_fused_moe_kernel)
_install(_moe, "get_config_file_name", get_config_file_name)
_install(_moe, "get_moe_configs", get_moe_configs)
_install(_moe, "get_moe_wna16_block_config", get_moe_wna16_block_config)
_install(_moe, "try_get_optimal_moe_config", try_get_optimal_moe_config)
_install(_moe, "fused_experts_impl", fused_experts_impl)
_install(_moe, "should_moe_wna16_use_cuda", should_moe_wna16_use_cuda)


def apply(
    self,
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: MoEActivation,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    a1q_scale: torch.Tensor | None,
    a2_scale: torch.Tensor | None,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
    expert_tokens_meta: mk.ExpertTokensMetadata | None,
    apply_router_weight_on_input: bool,
):
    # Check constraints.
    if self.quant_config.use_int4_w4a16:
        assert hidden_states.size(-1) // 2 == w1.size(2), "Hidden size mismatch"
    else:
        assert hidden_states.size(-1) == w1.size(2), (
            f"Hidden size mismatch {hidden_states.size(-1)} != {w1.size(2)}"
        )

    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert hidden_states.dim() == 2
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.int8,
    ]

    # Empty EP batches have no expert rows. Avoid zero-grid SDK launches,
    # which can leave an error for the next checked device operation.
    if hidden_states.size(0) == 0:
        return

    # We declared expects_unquantized_inputs (LoRA + DP/EP all2all), so the
    # prepare step deferred activation quantization to this kernel:
    # `hidden_states` arrives unquantized. Keep the unquantized tensor for
    # the LoRA shrink input and quantize a copy here for the base GEMM
    # (mirrors what the prepare step would have done, but after the
    # all-gather so the layout matches the gathered topk_ids / token map).
    lora_unquantized_hidden_states: torch.Tensor | None = None
    if self.expects_unquantized_inputs:
        assert a1q_scale is None
        lora_unquantized_hidden_states = hidden_states
        hidden_states, a1q_scale = moe_kernel_quantize_input(
            hidden_states,
            self.a1_scale,
            self.quant_dtype,
            self.per_act_token_quant,
            self.block_shape,
            quantization_emulation=self.quantization_emulation,
        )

    E, num_tokens, N, K, top_k_num = self.moe_problem_size(
        hidden_states, w1, w2, topk_ids
    )

    if global_num_experts == -1:
        global_num_experts = E

    config = try_get_optimal_moe_config(
        w1.size(),
        w2.size(),
        top_k_num,
        self.quant_config.config_name(hidden_states.dtype),
        num_tokens,
        block_shape=self.block_shape,
    )

    if hidden_states.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif hidden_states.dtype == torch.float16:
        compute_type = tl.float16
    elif hidden_states.dtype == torch.float32:
        compute_type = tl.float32
    elif (
        hidden_states.dtype == torch.float8_e4m3fn
        or hidden_states.dtype == torch.float8_e4m3fnuz
        or hidden_states.dtype == torch.int8
    ):
        compute_type = tl.bfloat16
    else:
        raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

    # Note that the output tensor might be in workspace1
    intermediate_cache1 = _resize_cache(workspace2, (num_tokens, top_k_num, N))
    cache2_dim = self.adjust_N_for_activation(N, activation)
    intermediate_cache2 = _resize_cache(
        workspace13, (num_tokens * top_k_num, cache2_dim)
    )
    intermediate_cache3 = _resize_cache(workspace2, (num_tokens, top_k_num, K))

    hooked = self._uses_post_gemm2_hook(num_tokens)
    if hooked and (apply_router_weight_on_input or self._lora_context is not None):
        raise NotImplementedError(
            "post_gemm2_hook needs unweighted GEMM2 rows; it supports neither "
            "router weights on the input nor LoRA."
        )

    # Under EP, drop the top-k slots routed to remote experts at alignment
    # time instead of launching `off_experts == -1` GEMM blocks that only
    # write zeros; the pad-aware `moe_sum` below skips the same slots.
    # Their rows in the workspace caches are then never written, so every
    # consumer of those caches must be row-local: a dynamic per-tensor
    # scale for the second GEMM's activation would take its amax over the
    # untouched rows, and LoRA aligns the full `topk_ids` itself.
    a2_scale_is_global = (
        self.quant_dtype is not None
        and a2_scale is None
        and self.block_shape is None
        and (not self.per_act_token_quant or self.quantization_emulation)
    )
    skip_invalid = (
        expert_map is not None and self._lora_context is None and not a2_scale_is_global
    )

    # Include fused shared-expert rows while preserving EP remapping.
    num_align_experts = w1.shape[0] if expert_map is None else global_num_experts
    sorted_token_ids, expert_ids, num_tokens_post_padded = _prepare_expert_assignment(
        topk_ids,
        config,
        num_tokens,
        top_k_num,
        num_align_experts,
        expert_map,
        use_int8_w8a16=self.quant_config.use_int8_w8a16,
        use_int4_w4a16=self.quant_config.use_int4_w4a16,
        block_shape=self.block_shape,
        ignore_invalid_experts=skip_invalid,
        weights_host_resident=_weights_host_resident(w1, w2),
    )

    # LoRA w13: applied to intermediate_cache1 before activation. When
    # the LoRA layer requested a dual-stream schedule, we run base w13
    # GEMM on the default stream and the LoRA fast-path on aux_stream;
    # the LoRA writes its delta into a fresh zero buffer (add_inputs=
    # False) and we sum it into intermediate_cache1 after both finish.
    #
    # The LoRA shrink kernel needs unquantized, gathered-layout
    # activations. When activation quant was deferred to this kernel
    # (expects_unquantized_inputs), the input we quantized above is exactly
    # that, so use it directly. Otherwise fall back to the context stash
    # (e.g. weight-only quant), guarding on a row-count match so a
    # DP-gathered layout never indexes a local stash out of bounds.
    sorted_token_ids_lora = None
    expert_ids_lora = None
    num_tokens_post_padded_lora = None
    token_lora_mapping = None
    lora_context = self._lora_context
    if lora_unquantized_hidden_states is not None:
        lora_x = lora_unquantized_hidden_states
    elif (
        lora_context is not None
        and lora_context.original_hidden_states is not None
        and lora_context.original_hidden_states.shape[0] == hidden_states.shape[0]
    ):
        lora_x = lora_context.original_hidden_states
    else:
        lora_x = hidden_states

    # TODO: The fallback to self.a1_scale was added for deferred static
    # activation quantization in https://github.com/vllm-project/vllm/pull/40857.
    # Activation emulation relies solely on `a1q_scale` output of
    # `moe_kernel_quantize_input` - this should be adapted to
    # always solely rely on `a1q_scale`.
    input_scale = (
        a1q_scale
        if self.quantization_emulation
        else (a1q_scale if a1q_scale is not None else self.a1_scale)
    )

    def _base_w13_fn():
        invoke_fused_moe_triton_kernel(
            hidden_states,
            w1,
            intermediate_cache1,
            input_scale,
            self.w1_scale,
            None,  # topk_weights
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            False,  # mul_routed_weights
            top_k_num,
            # PPU MODIFICATION: begin
            config.get("UP", config),
            # PPU MODIFICATION: end
            compute_type=compute_type,
            use_fp8_w8a8=self.quant_config.use_fp8_w8a8,
            use_int8_w8a8=self.quant_config.use_int8_w8a8,
            use_int8_w8a16=self.quant_config.use_int8_w8a16,
            use_int4_w4a16=self.quant_config.use_int4_w4a16,
            per_channel_quant=self.per_act_token_quant,
            block_shape=self.block_shape,
            B_bias=self.w1_bias,
            # PPU MODIFICATION: begin
            use_valu=config.get("USE_VALU", False),
            # PPU MODIFICATION: end
        )

    if lora_context is not None and lora_context.aux_stream is not None:
        # add_inputs=False: kernel overwrites lora_delta_w13. zeros (not
        # empty) so untouched rows -- e.g. blocks where every program
        # early-exits because lora_id<0 -- stay at zero and the trailing
        # add_() is a no-op there.
        lora_delta_w13 = torch.zeros_like(intermediate_cache1)

        def _lora_w13_fn():
            return self.apply_w13_lora(
                lora_context,
                y=lora_delta_w13,
                x=lora_x,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                expert_map=expert_map,
                w1=w1,
                w2=w2,
                num_tokens=num_tokens,
                top_k_num=top_k_num,
                add_inputs=False,
            )

        assert lora_context.events is not None
        _, lora_meta = maybe_execute_in_parallel(
            _base_w13_fn,
            _lora_w13_fn,
            lora_context.events[0],
            lora_context.events[1],
            lora_context.aux_stream,
        )
        (
            sorted_token_ids_lora,
            expert_ids_lora,
            num_tokens_post_padded_lora,
            token_lora_mapping,
        ) = lora_meta
        intermediate_cache1.add_(lora_delta_w13)
    else:
        _base_w13_fn()
        if lora_context is not None:
            (
                sorted_token_ids_lora,
                expert_ids_lora,
                num_tokens_post_padded_lora,
                token_lora_mapping,
            ) = self.apply_w13_lora(
                lora_context,
                y=intermediate_cache1,
                x=lora_x,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                expert_map=expert_map,
                w1=w1,
                w2=w2,
                num_tokens=num_tokens,
                top_k_num=top_k_num,
            )

    a2q_scale: torch.Tensor | None = None

    # Fuse SiLU+Mul + FP8 block quantize into a single kernel
    # when conditions permit (gated SiLU, fp8 block quant with
    # group_size=128, no LoRA requiring the BF16 intermediate).
    # The fused kernel has no clamp parameter, so a configured
    # SwiGLU clamp limit falls through to the unfused path.
    if (
        activation == MoEActivation.SILU
        and self.activation_config.clamp_limit is None
        and self.quant_config.use_fp8_w8a8
        and self.block_shape == [128, 128]
        and lora_context is None
        and not is_deep_gemm_e8m0_used()
    ):
        qintermediate_cache2, a2q_scale = ops.silu_and_mul_per_block_quant(
            intermediate_cache1.view(-1, N),
            group_size=128,
            quant_dtype=current_platform.fp8_dtype(),
        )
    else:
        self.activation(
            activation, intermediate_cache2, intermediate_cache1.view(-1, N)
        )

        qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(
            intermediate_cache2,
            a2_scale,
            self.quant_dtype,
            self.per_act_token_quant,
            self.block_shape,
            quantization_emulation=self.quantization_emulation,
        )

    # LoRA w2: applied to intermediate_cache3 before moe_sum, using the
    # unquantized intermediate_cache2 as the lora_a input.  Reuses the
    # sorted_token_ids_lora computed above. Same dual-stream pattern as
    # the w13 pair: base GEMM on default stream, LoRA delta on aux,
    # join via .add_() into intermediate_cache3.
    def _base_w2_fn():
        invoke_fused_moe_triton_kernel(
            qintermediate_cache2,
            w2,
            intermediate_cache3,
            a2q_scale,
            self.w2_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not hooked
            and self._apply_router_weight_in_gemm2(apply_router_weight_on_input),
            1,
            # PPU MODIFICATION: begin
            config.get("DOWN", config),
            # PPU MODIFICATION: end
            compute_type=compute_type,
            use_fp8_w8a8=self.quant_config.use_fp8_w8a8,
            use_int8_w8a8=self.quant_config.use_int8_w8a8,
            use_int8_w8a16=self.quant_config.use_int8_w8a16,
            use_int4_w4a16=self.quant_config.use_int4_w4a16,
            per_channel_quant=self.per_act_token_quant,
            block_shape=self.block_shape,
            B_bias=self.w2_bias,
            # PPU MODIFICATION: begin
            use_valu=config.get("USE_VALU", False),
            # PPU MODIFICATION: end
        )

    if lora_context is not None and lora_context.aux_stream is not None:
        lora_delta_w2 = torch.zeros_like(intermediate_cache3)

        def _lora_w2_fn():
            self.apply_w2_lora(
                lora_context,
                y=lora_delta_w2,
                x=intermediate_cache2,
                topk_weights=topk_weights,
                sorted_token_ids_lora=sorted_token_ids_lora,
                expert_ids_lora=expert_ids_lora,
                num_tokens_post_padded_lora=num_tokens_post_padded_lora,
                token_lora_mapping=token_lora_mapping,
                num_tokens=num_tokens,
                w1=w1,
                w2=w2,
                top_k_num=top_k_num,
                add_inputs=False,
            )

        assert lora_context.events is not None
        maybe_execute_in_parallel(
            _base_w2_fn,
            _lora_w2_fn,
            lora_context.events[2],
            lora_context.events[3],
            lora_context.aux_stream,
        )
        intermediate_cache3.add_(lora_delta_w2)
    else:
        _base_w2_fn()
        if lora_context is not None:
            self.apply_w2_lora(
                lora_context,
                y=intermediate_cache3,
                x=intermediate_cache2,
                topk_weights=topk_weights,
                sorted_token_ids_lora=sorted_token_ids_lora,
                expert_ids_lora=expert_ids_lora,
                num_tokens_post_padded_lora=num_tokens_post_padded_lora,
                token_lora_mapping=token_lora_mapping,
                num_tokens=num_tokens,
                w1=w1,
                w2=w2,
                top_k_num=top_k_num,
            )

    if hooked:
        self._run_post_gemm2_hook(
            route_major_moe_output(
                intermediate_cache3, topk_weights, topk_ids, expert_map
            ),
            output,
        )
    else:
        self._finalize_expert_output(
            intermediate_cache3, output, topk_ids, expert_map
        )


_install(_experts, "apply", apply, target_name="TritonExperts.apply")
