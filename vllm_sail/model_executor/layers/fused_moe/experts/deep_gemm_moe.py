# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import (
    ApplyMoEActivationConfig,
    MoEActivation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache,
    swiglu_limit_func,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
    per_token_group_quant_fp8_packed_for_deepgemm,
)
from vllm.model_executor.layers.quantization.utils.int8_utils import (
    per_token_quant_int8,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kInt4Static32,
    kInt8DynamicTokenSym,
    kInt8StaticChannelSym,
    kMxfp4Dynamic,
    kMxfp4Static,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_deep_gemm

import vllm_sail.envs as ppu_envs
from vllm_sail.model_executor.layers.fused_moe.deep_gemm_utils import (
    compute_aligned_M,
    deepgemm_moe_permute,
    deepgemm_unpermute_and_reduce,
)
from vllm_sail.model_executor.layers.quantization.utils.mxfp4_utils import (
    downcast_to_mxfp4,
)
from vllm_sail.utils.deep_gemm import (
    DeepGemmQuantScaleFMT,
    activation_and_mul,
    get_activation_mode,
    get_deep_gemm_config,
    get_mk_alignment_for_contiguous_layout,
    get_quant_mode,
    has_fused_act_quant,
    is_deep_gemm_supported,
    is_ppu_15,
    m_grouped_bf16_gemm_nt_nopad,
    m_grouped_fp4_gemm_nt_fused,
    m_grouped_fp4_gemm_nt_nopad,
    m_grouped_fp8_fp4_gemm_nt_fused,
    m_grouped_fp8_fp4_gemm_nt_nopad,
    m_grouped_fp8_gemm_nt_nopad,
    m_grouped_int8_gemm_nt_nopad,
    m_grouped_w4a16_gemm_nt_fused,
    m_grouped_w4a16_gemm_nt_nopad,
    moe_align_block_size,
)

logger = init_logger(__name__)


# Add for nvtx profiling
NVTX_PROFILE = ppu_envs.VLLM_SAIL_NVTX_PROFILE
if NVTX_PROFILE:
    try:
        from torch.cuda.nvtx import range_pop as th_nvtx_range_pop
        from torch.cuda.nvtx import range_push as th_nvtx_range_push
    except ImportError:
        NVTX_PROFILE = False

if not NVTX_PROFILE:

    def th_nvtx_range_push(label):
        pass

    def th_nvtx_range_pop():
        pass


def _valid_deep_gemm_shape(M: int, N: int, K: int) -> bool:
    align, align_k = get_mk_alignment_for_contiguous_layout()
    return align <= M and N % align == 0 and K % align_k == 0


def _valid_deep_gemm(
    hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> bool:
    """
    Check if the given problem size is supported by the DeepGemm grouped
    gemm kernel.  All of M, N, K and the quantization block_shape must be
    aligned by `dg.get_m_alignment_for_contiguous_layout()`.
    """
    if not has_deep_gemm():
        logger.debug_once("DeepGemm disabled: deep_gemm not available.")
        return False

    _M = hidden_states.size(0)
    _, K, N = w2.size()

    if (current_platform.is_device_capability((8, 0))) and (
        w1.dtype in [torch.float8_e4m3fn] or w2.dtype in [torch.float8_e4m3fn]
    ):
        logger.debug_once(
            "DeepGemm disabled: invalid weight dtype(s). w1.dtype: %s, w2.dtype: %s on sm80",
            w1.dtype,
            w2.dtype,
        )
        return False

    if w1.dtype not in [
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.int8,
        torch.uint8,
        torch.float8_e4m3fn,
    ] or w2.dtype not in [
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.int8,
        torch.uint8,
        torch.float8_e4m3fn,
    ]:
        logger.debug_once(
            "DeepGemm disabled: invalid weight dtype(s). w1.dtype: %s, w2.dtype: %s",
            w1.dtype,
            w2.dtype,
        )
        return False

    if (
        not hidden_states.is_contiguous()
        or not w1.is_contiguous()
        or not w2.is_contiguous()
    ):
        logger.debug_once(
            "DeepGemm disabled: weights or activations not contiguous. "
            "hidden_states.is_contiguous(): %s, w1.is_contiguous(): %s, "
            "w2.is_contiguous(): %s",
            hidden_states.is_contiguous(),
            w1.is_contiguous(),
            w2.is_contiguous(),
        )
        return False

    return True


def _deepgemm_act_mul(
    input: torch.Tensor,
    activation: MoEActivation,
    quant_mode: Any,
    activation_config: ApplyMoEActivationConfig,
) -> tuple[torch.Tensor, ...] | torch.Tensor | None:
    """Run DeepGEMM's `activation_and_mul` with the epilogue `quant_mode`.

    The kernel folds the activation, the gate*up multiply and the quantization
    selected by `quant_mode` into a single launch. The quantizing epilogues
    return `(values, scales)`; `QuantMode.BF16` returns the activated values
    alone. Returns ``None`` when the activation (or its parameterization) is
    not implemented by that kernel, so the caller can fall back to
    `self.activation` followed by a separate quantize.

    The returned FP8 row scales are the plain per-token ``amax / 448``: a weight
    preprocessing inverse scale is *not* folded in here, the consuming W4A8
    GEMM applies it through its own ``inv_scale`` instead.
    """
    activation_mode = get_activation_mode()
    if activation == MoEActivation.SILU:
        limit = activation_config.clamp_limit
        if limit is not None and (
            activation_config.alpha != 1.0 or activation_config.beta != 0.0
        ):
            # alpha/beta only feed the SWIGLUOAI epilogue of the fallback; the
            # fused kernel implements the plain SwiGLU(-with-clamp) math.
            return None
        return activation_and_mul(
            activation_mode.SILU,
            quant_mode,
            input,
            swiglu_limit=None if limit is None else float(limit),
        )
    if activation == MoEActivation.SWIGLUSTEP:
        # `swiglustep_and_mul` clamps SiLU(gate) and up at a fixed 7.0, which is
        # exactly DeepGEMM's SILU_STEP with the same limit.
        return activation_and_mul(
            activation_mode.SILU_STEP, quant_mode, input, swiglu_limit=7.0
        )
    if activation == MoEActivation.SITU:
        beta = activation_config.activation_situ_beta
        if beta is None:
            # Mirror the upstream fallback, which refuses to guess a beta.
            return None
        linear_beta = activation_config.activation_situ_linear_beta
        if linear_beta is not None and linear_beta <= 0:
            # <= 0 means "unset" to the vLLM kernel; DeepGEMM wants None.
            linear_beta = None
        return activation_and_mul(
            activation_mode.SITU,
            quant_mode,
            input,
            beta=float(beta),
            linear_beta=None if linear_beta is None else float(linear_beta),
        )
    return None


def _fused_act_quant_available() -> bool:
    """Whether DeepGEMM's fused activation kernels may be used at all.

    `activation_and_mul` asserts `is_ppu1v5_device()` inside the C++ entry
    point, so the gate has to cover the device and not just the symbol
    availability of the installed DeepGEMM build.
    """
    return (
        ppu_envs.VLLM_SAIL_DEEPGEMM_MOE_FUSED_ACT_QUANT
        and has_fused_act_quant()
        and is_ppu_15()
    )


class PPUDeepGemmExperts(mk.FusedMoEExpertsModular):
    """DeepGemm-based fused MoE expert implementation."""

    def __init__(self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        self.block_wise = quant_config.block_shape is not None

        if self.block_wise and not quant_config.use_int4_w4a16:
            assert (
                quant_config.block_shape[1]
                == get_mk_alignment_for_contiguous_layout(is_blockwise=self.block_wise)[
                    1
                ]
            )

        # Fused PPU kernels and the upstream fallback share the configuration
        # built by FusedMoEExperts.__init__ via from_configs(moe_config, quant_config).
        self.gemm1_clamp_limit = self.activation_config.clamp_limit
        self.gemm1_alpha = self.activation_config.alpha
        self.gemm1_beta = self.activation_config.beta

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return is_deep_gemm_supported()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (None, None),
            (kFp8Static128BlockSym, kFp8Dynamic128Sym),
            (kFp8StaticChannelSym, kFp8DynamicTokenSym),
            (kInt8StaticChannelSym, kInt8DynamicTokenSym),
            (kInt4Static32, None),
            (kMxfp4Static, None),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        # silu/swigluoai go through the fused alpha/beta kernel; swiglustep
        # uses the unfused activation path. The fused kernel reads packed w13
        # (gate = first half, up = second half), so it implements the
        # *uninterleaved* SwiGLU-OAI variant.
        return activation in [
            MoEActivation.SILU,
            MoEActivation.SWIGLUSTEP,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            MoEActivation.SITU,
        ]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        # NOTE(rob): discovered an IMA with this combination. Needs investigation.
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

    @staticmethod
    def _supports_bias() -> bool:
        return False

    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        E, M, N, K, topk = super().moe_problem_size(a1, w1, w2, topk_ids)
        if self.quant_config.use_mxfp4_w4a16 or self.quant_config.use_int4_w4a16:
            N = w2.size(1) * 32
        return E, M, N, K, topk

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        block_m = get_mk_alignment_for_contiguous_layout(is_blockwise=self.block_wise)[
            0
        ]
        M_sum = compute_aligned_M(
            M, topk, local_num_experts, block_m, expert_tokens_meta
        )
        assert M_sum % block_m == 0

        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace1 = (M_sum, max(activation_out_dim, K))
        workspace2 = (M_sum, max(N, K))
        output = (M, K)
        return (workspace1, workspace2, output)

    def _deepgemm_act_quant(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        activation: MoEActivation,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | None:
        """Run DeepGEMM's fused activation (+ A2 quant) when it fits `output`.

        `activation_and_mul` writes the activated (`[M, 2N] -> [M, N]`) result
        straight into the layout consumed by the second GEMM, replacing
        `self.activation` plus a separate quantize pass. FP8 destinations use
        the channel/block quantizing epilogue; BF16 destinations (w4a16 or an
        unquantized second GEMM) use the values-only epilogue and report no
        scales. Int8 has no equivalent epilogue, and the UE8M0-packed scale
        layout has no fused producer either.

        Returns ``None`` when the switch is off, the build/device lacks the
        kernels, or the activation/shape is not supported, so the caller keeps
        the two-kernel fallback.
        """
        if not _fused_act_quant_available():
            return None
        if input.dtype != torch.bfloat16 or not input.is_contiguous():
            return None
        # Every epilogue derives one row from both gate halves, so the
        # destination width pins the expected input width.
        if output.size(-1) * 2 != input.size(-1):
            return None

        quant_mode_enum = get_quant_mode()
        if output.dtype == torch.bfloat16:
            # Values-only epilogue: the second GEMM stays BF16, which is the
            # w4a16 and unquantized path.
            mode = getattr(quant_mode_enum, "BF16", None)
        elif output.dtype == torch.float8_e4m3fn:
            if DeepGemmQuantScaleFMT.from_oracle() == DeepGemmQuantScaleFMT.UE8M0:
                # Scales are packed 4-per-int32 for the GEMM; the fused
                # kernels can only emit float32 (or MX) scales.
                return None
            if self.block_shape is None:
                # Channel-wise A2 quant: one float32 scale per row.
                mode = getattr(quant_mode_enum, "FP8_CHANNEL", None)
            elif self.block_shape[1] != 128 or output.size(-1) % 128 != 0:
                # Block-wise A2 quant only covers 128-element groups.
                return None
            else:
                mode = getattr(quant_mode_enum, "FP8_BLOCK", None)
        else:
            # Int8 has no equivalent epilogue.
            return None

        if mode is None:
            return None
        fused = _deepgemm_act_mul(input, activation, mode, self.activation_config)
        if fused is None:
            return None
        # The epilogue decides: the quantizing modes append their scales, the
        # BF16 mode returns the values alone.
        return fused if isinstance(fused, tuple) else (fused, None)

    def _act_mul_quant(
        self, input: torch.Tensor, output: torch.Tensor, activation: MoEActivation
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale_fmt = DeepGemmQuantScaleFMT.from_oracle()

        M_sum, N = input.size()
        activation_out_dim = self.adjust_N_for_activation(N, activation)

        block_k = self.block_shape[1] if self.block_shape else activation_out_dim

        # 1. DeepGemm UE8M0: use packed per-token-group quant. The packed
        # scales only exist for an FP8 A2, so BF16 destinations skip this.
        if scale_fmt == DeepGemmQuantScaleFMT.UE8M0 and (
            output.dtype == torch.float8_e4m3fn
        ):
            act_out = torch.empty(
                (M_sum, activation_out_dim), dtype=input.dtype, device=input.device
            )
            self.activation(
                activation,
                act_out,
                input,
            )
            a2q, a2q_scale = per_token_group_quant_fp8_packed_for_deepgemm(
                act_out,
                block_k,
                out_q=output,
            )
            return a2q, a2q_scale

        # 2. Hopper / non-E8M0: prefer the fused activation+mul+quant kernel
        fused = self._deepgemm_act_quant(input, output, activation)
        if fused is not None:
            return fused

        # 3. fallback path for activations the fused kernel does not implement.
        act_out = torch.empty(
            (M_sum, activation_out_dim), dtype=input.dtype, device=input.device
        )

        if self.gemm1_clamp_limit is not None and activation == MoEActivation.SILU:
            swiglu_limit_func(
                act_out,
                input,
                self.gemm1_clamp_limit,
            )
        else:
            # Assign act path
            self.activation(
                activation,
                act_out,
                input,
            )
        if output.dtype == torch.float8_e4m3fn:
            block_k = self.block_shape[1] if self.block_shape else activation_out_dim
            return per_token_group_quant_fp8(
                act_out, block_k, column_major_scales=True, out_q=output
            )
        elif output.dtype == torch.int8:
            return per_token_quant_int8(act_out)
        else:
            return act_out, None

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
        if self.quant_config.quant_dtype:
            assert a1q_scale is not None
            assert a2_scale is None
            assert self.w1_scale is not None
            assert self.w2_scale is not None
            quant_dtype = self.quant_config.quant_dtype
        else:
            quant_dtype = torch.bfloat16

        a1q = hidden_states
        if self.quant_config.use_mxfp4_w4a16 or self.quant_config.use_int4_w4a16:
            E = w1.size(0)
            K = hidden_states.size(-1)
            N = w2.size(1) * 32
        else:
            E, N, K = w1.size()
            assert w2.size(1) == K

        local_num_experts = w1.size(0)
        if global_num_experts == -1:
            global_num_experts = local_num_experts

        M_sum = compute_aligned_M(
            M=topk_ids.size(0),
            num_topk=topk_ids.size(1),
            local_num_experts=local_num_experts,
            alignment=get_mk_alignment_for_contiguous_layout(
                is_blockwise=self.block_wise
            )[0],
            expert_tokens_meta=expert_tokens_meta,
        )

        a1q_perm = _resize_cache(workspace13.view(dtype=quant_dtype), (M_sum, K))
        is_block_wise_quant = is_channel_wise_quant = False
        if self.quant_config.use_fp8_w8a8 or self.quant_config.use_int8_w8a8:
            is_block_wise_quant = self.block_shape is not None
            is_channel_wise_quant = not is_block_wise_quant

        a1q, a1q_scale, expert_ids, inv_perm, expert_num_tokens = deepgemm_moe_permute(
            aq=a1q,
            aq_scale=a1q_scale,
            topk_ids=topk_ids,
            local_num_experts=local_num_experts,
            expert_map=expert_map,
            expert_tokens_meta=expert_tokens_meta,
            aq_out=a1q_perm,
            is_block_wise_quant=is_block_wise_quant,
            is_channel_wise_quant=is_channel_wise_quant,
        )
        assert a1q.size(0) == M_sum

        mm1_out = _resize_cache(workspace2, (M_sum, N))

        nvtx_pushed = False
        if NVTX_PROFILE:
            if (
                ppu_envs.VLLM_SAIL_NVTX_DUMP_TOPK
                and not torch.cuda.is_current_stream_capturing()
            ):
                num_activated_experts = (expert_num_tokens > 0).sum().item()
                th_nvtx_range_push(
                    f"MoE,M_{hidden_states.shape[0]}_E_{w1.shape[0]}_H_{w1.shape[2]}_In_{w1.shape[1]}_topk_{topk_ids.shape[1]}_topkids{expert_num_tokens.flatten().cpu().tolist()}_unique_{num_activated_experts}"
                )
                nvtx_pushed = True

        # calculate expert_first_token_offset for deepgemm
        experts_for_rows = torch.zeros(
            local_num_experts, dtype=torch.int32, device="cuda"
        )
        counts = expert_num_tokens
        min_n = min(counts.size(0), local_num_experts)
        if min_n > 0:
            experts_for_rows[:min_n] = counts[:min_n]

        if quant_dtype == torch.float8_e4m3fn:
            m_grouped_fp8_gemm_nt_nopad(
                (a1q, a1q_scale),
                (w1, self.w1_scale),
                mm1_out,
                expert_ids,
                experts_for_rows,
            )
            activation_out_dim = self.adjust_N_for_activation(N, activation)
            quant_out = _resize_cache(
                workspace13.view(dtype=torch.float8_e4m3fn),
                (M_sum, activation_out_dim),
            )
            a2q, a2q_scale = self._act_mul_quant(
                input=mm1_out.view(-1, N), output=quant_out, activation=activation
            )

            mm2_out = _resize_cache(workspace2, (M_sum, K))
            m_grouped_fp8_gemm_nt_nopad(
                (a2q, a2q_scale),
                (w2, self.w2_scale),
                mm2_out,
                expert_ids,
                experts_for_rows,
            )
        elif quant_dtype == torch.int8:
            best_config = get_deep_gemm_config(M_sum, N, K, num_groups=E)
            m_grouped_int8_gemm_nt_nopad(
                (a1q, a1q_scale),
                (w1, self.w1_scale),
                mm1_out,
                expert_ids,
                experts_for_rows,
                configs=best_config,
            )
            activation_out_dim = self.adjust_N_for_activation(N, activation)
            quant_out = _resize_cache(
                workspace13.view(dtype=torch.int8), (M_sum, activation_out_dim)
            )

            a2q, a2q_scale = self._act_mul_quant(
                input=mm1_out.view(-1, N), output=quant_out, activation=activation
            )

            mm2_out = _resize_cache(workspace2, (M_sum, K))
            best_config = get_deep_gemm_config(
                a2q.shape[0],
                w2.shape[1],
                a2q.shape[-1],
                num_groups=E,
            )
            m_grouped_int8_gemm_nt_nopad(
                (a2q, a2q_scale),
                (w2, self.w2_scale),
                mm2_out,
                expert_ids,
                experts_for_rows,
                configs=best_config,
            )
        elif self.quant_config.use_mxfp4_w4a16 or self.quant_config.use_int4_w4a16:
            if self.quant_config.use_mxfp4_w4a16:
                w1_scale = self.w1_scale.view(torch.uint8)
                w2_scale = self.w2_scale.view(torch.uint8)
            else:
                w1_scale, w2_scale = self.w1_scale, self.w2_scale
            m_grouped_w4a16_gemm_nt_nopad(
                a1q,
                (w1, w1_scale),
                mm1_out,
                expert_ids,
                experts_for_rows,
            )
            activation_out_dim = self.adjust_N_for_activation(N, activation)
            quant_out = _resize_cache(workspace13, (M_sum, activation_out_dim))
            a2q, _ = self._act_mul_quant(
                input=mm1_out.view(-1, N), output=quant_out, activation=activation
            )
            mm2_out = _resize_cache(workspace2, (M_sum, K))
            m_grouped_w4a16_gemm_nt_nopad(
                a2q,
                (w2, w2_scale),
                mm2_out,
                expert_ids,
                experts_for_rows,
            )
        else:
            m_grouped_bf16_gemm_nt_nopad(a1q, w1, mm1_out, expert_ids, experts_for_rows)
            activation_out_dim = self.adjust_N_for_activation(N, activation)
            quant_out = _resize_cache(workspace13, (M_sum, activation_out_dim))
            a2q, a2q_scale = self._act_mul_quant(
                input=mm1_out.view(-1, N), output=quant_out, activation=activation
            )

            mm2_out = _resize_cache(workspace2, (M_sum, K))
            m_grouped_bf16_gemm_nt_nopad(a2q, w2, mm2_out, expert_ids, experts_for_rows)

        if apply_router_weight_on_input:
            assert topk_weights is not None
            # topk_weights = torch.ones_like(topk_weights)

        if nvtx_pushed:
            th_nvtx_range_pop()

        deepgemm_unpermute_and_reduce(
            a=mm2_out,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            inv_perm=inv_perm,
            expert_map=expert_map,
            output=output,
        )


class PPUDeepGemmExpertsW4FA16MMA(PPUDeepGemmExperts):
    @staticmethod
    def _supports_current_device() -> bool:
        # w4fa16_mma requires sm90+ on PPU; disable on sm80
        if current_platform.is_device_capability((8, 0)):
            return False
        return is_deep_gemm_supported()

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kMxfp4Static, None),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        return mk.FusedMoEExpertsModular.moe_problem_size(self, a1, w1, w2, topk_ids)

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
        assert self.quant_config.use_mxfp4_w4a16

        a1q = hidden_states
        _, N, K = w1.size()
        K = K * 2

        local_num_experts = w1.size(0)
        if global_num_experts == -1:
            global_num_experts = local_num_experts

        assert w2.size(1) == K

        M_sum = compute_aligned_M(
            M=topk_ids.size(0),
            num_topk=topk_ids.size(1),
            local_num_experts=local_num_experts,
            alignment=get_mk_alignment_for_contiguous_layout()[0],
            expert_tokens_meta=expert_tokens_meta,
        )
        mm1_out = _resize_cache(workspace2, (M_sum, N))

        use_fused_path = ppu_envs.VLLM_SAIL_DEEPGEMM_MOE_TP_FUSED and expert_map is None

        if use_fused_path:
            # ── fused path: moe_align + GEMM1 fused (gather read) ──
            logger.info_once("Using PPU DeepGEMM fused moe.")
            (
                configs_list,
                m_rows,
                expert_ids_and_cumsum,
                sorted_token_ids,
                aligned_num_m_blocks,
                inv_perm,
                expert_ids,
            ) = moe_align_block_size(
                a1q,
                w1,
                topk_ids,
                perchannel_quant=False,
            )

            m_grouped_w4a16_gemm_nt_fused(
                a1q,
                (w1, self.w1_scale),
                mm1_out,
                m_rows,
                expert_ids_and_cumsum,
                sorted_token_ids,
                aligned_num_m_blocks,
                configs_list,
            )
            experts_for_rows = m_rows
            inv_perm = inv_perm.view(topk_ids.size(0), topk_ids.size(1))

        else:
            # ── nopad path: deepgemm_moe_permute + GEMM1 nopad ──
            a1q_perm = _resize_cache(workspace13.view(dtype=torch.bfloat16), (M_sum, K))
            is_block_wise_quant = is_channel_wise_quant = False
            a1q, _, expert_ids, inv_perm, expert_num_tokens = deepgemm_moe_permute(
                aq=a1q,
                aq_scale=a1q_scale,
                topk_ids=topk_ids,
                local_num_experts=local_num_experts,
                expert_map=expert_map,
                expert_tokens_meta=expert_tokens_meta,
                aq_out=a1q_perm,
                is_block_wise_quant=is_block_wise_quant,
                is_channel_wise_quant=is_channel_wise_quant,
            )
            assert a1q.size(0) == M_sum

            # calculate expert_first_token_offset for deepgemm
            experts_for_rows = torch.zeros(
                local_num_experts, dtype=torch.int32, device=a1q.device
            )
            counts = expert_num_tokens
            min_n = min(counts.size(0), local_num_experts)
            if min_n > 0:
                experts_for_rows[:min_n] = counts[:min_n]

            m_grouped_w4a16_gemm_nt_nopad(
                a1q,
                (w1, self.w1_scale),
                mm1_out,
                expert_ids,
                experts_for_rows,
            )

        activation_out_dim = self.adjust_N_for_activation(N, activation)
        quant_out = _resize_cache(workspace13, (M_sum, activation_out_dim))
        a2q, _ = self._act_mul_quant(
            input=mm1_out.view(-1, N), output=quant_out, activation=activation
        )
        mm2_out = _resize_cache(workspace2, (M_sum, K))
        m_grouped_w4a16_gemm_nt_nopad(
            a2q,
            (w2, self.w2_scale),
            mm2_out,
            expert_ids,
            experts_for_rows,
        )

        if apply_router_weight_on_input:
            assert topk_weights is not None

        deepgemm_unpermute_and_reduce(
            a=mm2_out,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            inv_perm=inv_perm,
            expert_map=expert_map,
            output=output,
        )


# PPU MXFP4 DeepGEMM Experts
class PPUDeepGemmExpertsMXFP4(mk.FusedMoEExpertsModular):
    """
    PPU DeepGEMM-based fused MoE expert implementation for MXFP4.

    This class implements MXFP4 x MXFP4 (both activations and weights are mxfp4)
    on PPU platform using DeepGEMM.
    """

    def __init__(self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        # Fused PPU kernels and the upstream fallback share the configuration
        # built by FusedMoEExperts.__init__ via from_configs(moe_config, quant_config).
        self.gemm1_clamp_limit = self.activation_config.clamp_limit
        self.gemm1_alpha = self.activation_config.alpha
        self.gemm1_beta = self.activation_config.beta

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        # MXFP4 (w4a4) requires sm90+ on PPU; disable on sm80
        if current_platform.is_device_capability((8, 0)):
            return False
        return is_deep_gemm_supported()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kMxfp4Static, kMxfp4Dynamic),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [
            MoEActivation.SILU,
            MoEActivation.SWIGLUSTEP,
            MoEActivation.SWIGLUOAI,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            # Kimi-K3 latent MoE experts use the SiTU gated activation; the
            # betas flow via moe_config -> base activation() -> situ_and_mul.
            MoEActivation.SITU,
        ]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        block_m = get_mk_alignment_for_contiguous_layout(is_mxfp4=True)[0]
        M_sum = compute_aligned_M(
            M, topk, local_num_experts, block_m, expert_tokens_meta
        )
        assert M_sum % block_m == 0

        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace1 = (M_sum, max(activation_out_dim, K))
        workspace2 = (M_sum, max(N, K))
        output = (M, K)
        return (workspace1, workspace2, output)

    def _deepgemm_act_quant(
        self,
        input: torch.Tensor,
        activation: MoEActivation,
        activation_out_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Run DeepGEMM's fused activation + MXFP4 quant when applicable.

        `activation_and_mul` folds the activation, the gate*up multiply and the
        MXFP4 downcast into a single kernel, replacing `self.activation` +
        `downcast_to_mxfp4`. It returns a `(uint8 (M, dim / 2), uint16 M-major
        scales)` pair (`QuantMode.MXFP4` defaults to `reorder_scale=True`),
        which is exactly what the nopad MXFP4 GEMM consumes.

        Returns ``None`` when the switch is off, the build/device lacks these
        entry points, or the activation/shape is not supported, so the caller
        falls back.
        """
        if not _fused_act_quant_available():
            return None
        if input.dtype != torch.bfloat16 or not input.is_contiguous():
            return None
        # The kernel packs two MXFP4 values per byte.
        if activation_out_dim % 2 != 0:
            return None

        return _deepgemm_act_mul(
            input,
            activation,
            get_quant_mode().MXFP4,
            self.activation_config,
        )

    def _act_mul_quant(
        self, input: torch.Tensor, output: torch.Tensor, activation: MoEActivation
    ) -> tuple[torch.Tensor, torch.Tensor]:
        M_sum, N = input.size()
        activation_out_dim = self.adjust_N_for_activation(N, activation)

        fused = self._deepgemm_act_quant(input, activation, activation_out_dim)
        if fused is not None:
            return fused

        act_out = torch.empty(
            (M_sum, activation_out_dim), dtype=input.dtype, device=input.device
        )
        self.activation(
            activation,
            act_out,
            input,
        )
        a_q, a_scale = downcast_to_mxfp4(act_out, axis=1)
        return a_q, a_scale

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
        assert self.quant_config.quant_dtype == "mxfp4"
        quant_dtype = torch.uint8

        a1q = hidden_states
        E, N, K = w1.size()

        local_num_experts = w1.size(0)
        if global_num_experts == -1:
            global_num_experts = local_num_experts

        # for mxfp4, pack 2 e2m1 qweight into 1 uint8
        # w1.size(2) is hidden_size // 2 and w2.size(1) is hidden_size
        assert w2.size(1) == K * 2

        M_sum = compute_aligned_M(
            M=topk_ids.size(0),
            num_topk=topk_ids.size(1),
            local_num_experts=local_num_experts,
            alignment=get_mk_alignment_for_contiguous_layout(is_mxfp4=True)[0],
            expert_tokens_meta=expert_tokens_meta,
        )

        # The fused path replaces `deepgemm_moe_permute` + nopad GEMM1 by a single
        # kernel pass (`moe_align_block_size` + gather-read fused GEMM1).
        use_fused_path = (
            ppu_envs.VLLM_SAIL_DEEPGEMM_MOE_TP_FUSED
            and expert_map is None
            # The fused fp4 MoE kernel only implements the bias-free epilogue.
            and self.quant_config._w1.bias is None
            and self.quant_config._w2.bias is None
            # Fused kernel constraints: K % 16 == 0 and N % 2 == 0.
            and K % 16 == 0
            and N % 2 == 0
            and a1q.is_contiguous()
        )

        if use_fused_path:
            # ── fused path: moe_align + GEMM1 fused (gather read) ──
            # The kernel writes the compact rows [0, numel), and M_sum equals
            # numel for mxfp4 (M alignment 1) when `expert_map` is None.
            assert M_sum == topk_ids.numel(), (
                f"fused mxfp4 MoE expects M_sum == numel, got M_sum={M_sum} "
                f"and numel={topk_ids.numel()}"
            )
            logger.info_once("Using PPU DeepGEMM fused moe (mxfp4).")
            (
                configs_list,
                m_rows,
                expert_ids_and_cumsum,
                sorted_token_ids,
                aligned_num_m_blocks,
                inv_perm,
                expert_ids,
            ) = moe_align_block_size(
                a1q,
                w1,
                topk_ids,
                perchannel_quant=False,
            )
            experts_for_rows = m_rows
            expert_num_tokens = m_rows
            inv_perm = inv_perm.view(topk_ids.size(0), topk_ids.size(1))
            # The fused kernel reads a plain contiguous (K-major) uint16 SFA,
            # while the activation quant emits a column-major scale view.
            if not a1q_scale.is_contiguous():
                a1q_scale = a1q_scale.contiguous()
        else:
            # ── nopad path: deepgemm_moe_permute + GEMM1 nopad ──
            a1q_perm = _resize_cache(workspace13.view(dtype=quant_dtype), (M_sum, K))
            is_block_wise_quant = self.block_shape is not None
            is_channel_wise_quant = not is_block_wise_quant

            a1q, a1q_scale, expert_ids, inv_perm, expert_num_tokens = (
                deepgemm_moe_permute(
                    aq=a1q,
                    aq_scale=a1q_scale,
                    topk_ids=topk_ids,
                    local_num_experts=local_num_experts,
                    expert_map=expert_map,
                    expert_tokens_meta=expert_tokens_meta,
                    aq_out=a1q_perm,
                    is_block_wise_quant=is_block_wise_quant,
                    is_channel_wise_quant=is_channel_wise_quant,
                )
            )
            assert a1q.size(0) == M_sum

            # calculate expert_first_token_offset for deepgemm
            experts_for_rows = torch.zeros(
                local_num_experts, dtype=torch.int32, device=a1q.device
            )
            counts = expert_num_tokens
            min_n = min(counts.size(0), local_num_experts)
            if min_n > 0:
                experts_for_rows[:min_n] = counts[:min_n]

        mm1_out = _resize_cache(workspace2, (M_sum, N))

        nvtx_pushed = False
        if NVTX_PROFILE:
            if (
                ppu_envs.VLLM_SAIL_NVTX_DUMP_TOPK
                and not torch.cuda.is_current_stream_capturing()
            ):
                num_activated_experts = (expert_num_tokens > 0).sum().item()
                th_nvtx_range_push(
                    f"MoE,M_{hidden_states.shape[0]}_E_{w1.shape[0]}_H_{w1.shape[2]}_In_{w1.shape[1]}_topk_{topk_ids.shape[1]}_topkids{expert_num_tokens.flatten().cpu().tolist()}_unique_{num_activated_experts}"
                )
                nvtx_pushed = True

        if use_fused_path:
            m_grouped_fp4_gemm_nt_fused(
                (a1q, a1q_scale),
                (w1, self.w1_scale),
                mm1_out,
                m_rows,
                expert_ids_and_cumsum,
                sorted_token_ids,
                aligned_num_m_blocks,
                configs_list,
            )
        else:
            m_grouped_fp4_gemm_nt_nopad(
                (a1q, a1q_scale),
                (w1, self.w1_scale),
                self.quant_config._w1.bias,
                mm1_out,
                expert_ids,
                experts_for_rows,
            )

        activation_out_dim = self.adjust_N_for_activation(N, activation)
        if self.gemm1_clamp_limit is not None and activation == MoEActivation.SILU:
            # deepseek_v4
            act_out = torch.empty(
                (M_sum, activation_out_dim), dtype=mm1_out.dtype, device=mm1_out.device
            )
            swiglu_limit_func(
                output=act_out,
                input=mm1_out.view(-1, N),
                swiglu_limit=self.gemm1_clamp_limit,
            )
            a2q, a2q_scale = downcast_to_mxfp4(act_out, axis=1)
        else:
            quant_out = _resize_cache(
                workspace13.view(dtype=torch.uint8),
                (M_sum, activation_out_dim),
            )
            a2q, a2q_scale = self._act_mul_quant(
                input=mm1_out.view(-1, N), output=quant_out, activation=activation
            )

        # for mxfp4, K is hidden_size // 2, need to set `K * 2` here
        mm2_out = _resize_cache(workspace2, (M_sum, K * 2))
        m_grouped_fp4_gemm_nt_nopad(
            (a2q, a2q_scale),
            (w2, self.w2_scale),
            self.quant_config._w2.bias,
            mm2_out,
            expert_ids,
            experts_for_rows,
        )

        if apply_router_weight_on_input:
            assert topk_weights is not None
            # topk_weights = torch.ones_like(topk_weights)

        if nvtx_pushed:
            th_nvtx_range_pop()

        deepgemm_unpermute_and_reduce(
            a=mm2_out,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            inv_perm=inv_perm,
            expert_map=expert_map,
            output=output,
        )

class PPUDeepGemmExpertsMXFP4W4A8(mk.FusedMoEExpertsModular):
    """PPU DeepGEMM fused MoE experts for MXFP4 weights with FP8 activations.

    W4A8: weights are packed MXFP4 (uint8, [E, N, K/2]) preprocessed by
    `deep_gemm.preprocess_weights_and_scales(W4A8_MXFP4_MMA, ...)` at load
    time; activations are per-token FP8. The weight-preprocessing inverse
    scales (`w13_inv_scale` / `w2_inv_scale`) are handed to the W4A8 GEMMs as
    their `inv_scale`, which applies each exactly once inside the kernel and
    keeps the activation row scales plain (`amax / 448`).
    """

    def __init__(self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        self.gemm1_clamp_limit = self.activation_config.clamp_limit
        self.gemm1_alpha = self.activation_config.alpha
        self.gemm1_beta = self.activation_config.beta
        # Attached by the patched make_mxfp4_moe_quant_config; default to 1.0
        # when the quant config predates the w4a8 plumbing.
        self.w13_inv_scale = float(getattr(quant_config, "w13_inv_scale", 1.0))
        self.w2_inv_scale = float(getattr(quant_config, "w2_inv_scale", 1.0))

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        # W4A8-MXFP4-MMA requires sm90+ on PPU; disable on sm80. The fused
        # activation+quant kernel that A2 depends on only exists on PPU 1.5.
        if current_platform.is_device_capability((8, 0)):
            return False
        return is_deep_gemm_supported() and has_fused_act_quant() and is_ppu_15()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kMxfp4Static, kFp8DynamicTokenSym),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        # Only the activations DeepGEMM's fused activation+quant kernel
        # implements: there is no unfused A2 path for W4A8.
        return activation in [
            MoEActivation.SILU,
            MoEActivation.SWIGLUSTEP,
            MoEActivation.SITU,
        ]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

    @staticmethod
    def _supports_bias() -> bool:
        return False

    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        # w1 is [E, N, K/2] packed MXFP4; report the logical K so the
        # upstream pipeline sizes the A1 buffers correctly.
        E, N, K_half = w1.size()
        return E, a1.size(0), N, K_half * 2, topk_ids.size(1)

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # Per-token activations: no M-block alignment requirement.
        M_sum = compute_aligned_M(
            M, topk, local_num_experts, 1, expert_tokens_meta
        )
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace1 = (M_sum, max(activation_out_dim, K))
        workspace2 = (M_sum, max(N, K))
        output = (M, K)
        return (workspace1, workspace2, output)

    def _act_mul_quant_fp8_channel(
        self, input: torch.Tensor, activation: MoEActivation
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Activation+mul with fused per-token FP8 quant (A2 of W4A8).

        Returns ``(a2q, a2q_scale)`` with plain `amax / 448` row-major [M, 1]
        scales; GEMM2 applies `w2_inv_scale` through its own `inv_scale`.

        There is deliberately no unfused fallback for W4A8: the activation, the
        gate*up multiply and the quantization all have to come from DeepGEMM, so
        an unsupported build/activation/device is a hard error instead of a
        silent (much slower and differently rounded) two-step path.
        """
        if not (has_fused_act_quant() and is_ppu_15()):
            raise RuntimeError(
                "PPUDeepGemmExpertsMXFP4W4A8 requires DeepGEMM's fused "
                "activation+quant kernels on PPU 1.5, but the installed "
                "deep_gemm build does not expose them"
            )
        if input.dtype != torch.bfloat16 or not input.is_contiguous():
            raise RuntimeError(
                "PPUDeepGemmExpertsMXFP4W4A8 fused activation+quant needs a "
                f"contiguous BF16 gate/up buffer, got dtype={input.dtype}, "
                f"shape={tuple(input.shape)}, contiguous={input.is_contiguous()}"
            )

        fused = _deepgemm_act_mul(
            input,
            activation,
            get_quant_mode().FP8_CHANNEL,
            self.activation_config,
        )
        if fused is None:
            raise ValueError(
                f"Activation {activation} is not supported by DeepGEMM's fused "
                "activation+quant kernel; PPUDeepGemmExpertsMXFP4W4A8 has no "
                "unfused fallback"
            )
        return fused

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
        assert self.quant_config.quant_dtype == current_platform.fp8_dtype()
        assert a1q_scale is not None

        a1q = hidden_states
        E, N, K_half = w1.size()
        K = K_half * 2
        assert w2.size(1) == K

        local_num_experts = w1.size(0)
        if global_num_experts == -1:
            global_num_experts = local_num_experts

        M_sum = compute_aligned_M(
            M=topk_ids.size(0),
            num_topk=topk_ids.size(1),
            local_num_experts=local_num_experts,
            alignment=1,
            expert_tokens_meta=expert_tokens_meta,
        )

        mm1_out = _resize_cache(workspace2, (M_sum, N))

        use_fused_path = (
            ppu_envs.VLLM_SAIL_DEEPGEMM_MOE_TP_FUSED
            and expert_map is None
            and self.quant_config._w1.bias is None
            and self.quant_config._w2.bias is None
            and K % 16 == 0
            and N % 2 == 0
            and a1q.is_contiguous()
        )

        if use_fused_path:
            # ── fused path: moe_align + GEMM1 fused (gather read) ──
            logger.info_once("Using PPU DeepGEMM fused moe (w4a8).")
            (
                configs_list,
                m_rows,
                expert_ids_and_cumsum,
                sorted_token_ids,
                aligned_num_m_blocks,
                inv_perm,
                expert_ids,
            ) = moe_align_block_size(
                a1q,
                w1,
                topk_ids,
                perchannel_quant=False,
            )
            experts_for_rows = m_rows
            expert_num_tokens = m_rows
            inv_perm = inv_perm.view(topk_ids.size(0), topk_ids.size(1))
            a1q_scale = a1q_scale.contiguous()

            m_grouped_fp8_fp4_gemm_nt_fused(
                (a1q, a1q_scale),
                (w1, self.w1_scale),
                mm1_out,
                m_rows,
                expert_ids_and_cumsum,
                sorted_token_ids,
                aligned_num_m_blocks,
                configs_list,
                inv_scale=self.w13_inv_scale,
            )
        else:
            # ── nopad path: deepgemm_moe_permute + GEMM1 nopad ──
            a1q_perm = _resize_cache(
                workspace13.view(dtype=a1q.dtype), (M_sum, K)
            )
            a1q, a1q_scale, expert_ids, inv_perm, expert_num_tokens = (
                deepgemm_moe_permute(
                    aq=a1q,
                    aq_scale=a1q_scale,
                    topk_ids=topk_ids,
                    local_num_experts=local_num_experts,
                    expert_map=expert_map,
                    expert_tokens_meta=expert_tokens_meta,
                    aq_out=a1q_perm,
                    is_block_wise_quant=False,
                    is_channel_wise_quant=True,
                )
            )
            assert a1q.size(0) == M_sum

            # calculate expert_first_token_offset for deepgemm
            experts_for_rows = torch.zeros(
                local_num_experts, dtype=torch.int32, device=a1q.device
            )
            counts = expert_num_tokens
            min_n = min(counts.size(0), local_num_experts)
            if min_n > 0:
                experts_for_rows[:min_n] = counts[:min_n]

            m_grouped_fp8_fp4_gemm_nt_nopad(
                (a1q, a1q_scale),
                (w1, self.w1_scale),
                mm1_out,
                expert_ids,
                experts_for_rows,
                inv_scale=self.w13_inv_scale,
            )

        a2q, a2q_scale = self._act_mul_quant_fp8_channel(
            input=mm1_out.view(-1, N), activation=activation
        )

        # GEMM2 contracts over the (possibly 128-padded) packed K; pad the
        # activation with zero codes (zero * anything = 0 contribution).
        down_k = w2.size(2) * 2
        if a2q.size(-1) != down_k:
            a2q = torch.nn.functional.pad(a2q, (0, down_k - a2q.size(-1)))

        mm2_out = _resize_cache(workspace2, (M_sum, K))
        m_grouped_fp8_fp4_gemm_nt_nopad(
            (a2q, a2q_scale),
            (w2, self.w2_scale),
            mm2_out,
            expert_ids,
            experts_for_rows,
            inv_scale=self.w2_inv_scale,
        )

        if apply_router_weight_on_input:
            assert topk_weights is not None

        deepgemm_unpermute_and_reduce(
            a=mm2_out,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            inv_perm=inv_perm,
            expert_map=expert_map,
            output=output,
        )
