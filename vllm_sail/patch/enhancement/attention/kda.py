# SPDX-License-Identifier: Apache-2.0
"""Use PLA KDA kernels on PPU, delegating everything else to upstream.

Every patch here is a thin wrapper: it inspects the platform, and either runs
the PPU-specific PLA kernel or forwards the call unchanged to the upstream
implementation captured at import time. Upstream therefore stays the single
source of truth for the KDA dataflow, so a vLLM upgrade only needs the PPU
branches re-checked instead of the whole layer re-copied.
"""

from __future__ import annotations

from dataclasses import replace
from types import ModuleType

import torch
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.models.kimi_k3.nvidia import kda as _kda
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

from vllm_sail import envs as ppu_envs
from vllm_sail.patch.bodies import bind_body
from vllm_sail.patch.utils import patch

_MODULE = "vllm.models.kimi_k3.nvidia.kda"
_META = dict(
    reason="Kimi KDA must select PPU PLA kernels instead of NVIDIA FlashKDA/native decode.",
    affected_versions=">=0.32.0,<0.33.0",
    remove_when="Kimi KDA supports platform kernel registration for prefill and decode.",
)

logger = _kda.logger
"""Upstream's logger, so PLA messages appear next to the layer's own."""

is_conv_state_dim_first = _kda.is_conv_state_dim_first

# Captured before any patch is installed: the wrappers below delegate to these,
# and resolving them afterwards would call the wrappers themselves.
_upstream_is_flashkda_supported = _kda.is_flashkda_supported
_upstream_is_fused_kda_decode_supported = _kda.is_fused_kda_decode_supported
_upstream_flashkda_prefill = _kda._flashkda_prefill
_upstream_forward = _kda.KimiK3DeltaAttention._forward
_upstream_init = _kda.KimiK3DeltaAttention.__init__
_upstream_get_kv_cache_spec = _kda.KimiK3DeltaAttention.get_kv_cache_spec


def is_fused_kda_decode_supported(
    num_heads: int,
    head_dim: int,
    conv_width: int,
    num_spec: int,
    input_dtype: torch.dtype,
    conv_state_dtype: torch.dtype,
    recurrent_state_dtype: torch.dtype,
) -> bool:
    if not current_platform.is_ppu():
        return _upstream_is_fused_kda_decode_supported(
            num_heads,
            head_dim,
            conv_width,
            num_spec,
            input_dtype,
            conv_state_dtype,
            recurrent_state_dtype,
        )
    # PPU has neither ``torch.ops._C.fused_kda_decode`` nor an SM90/SM10x/SM12x
    # capability to gate on; PLA's decode kernel replaces both, and it wants a
    # dim-first convolution state.
    from vllm_sail.attention.pla_kda import get_pla_kda_kernel

    return (
        get_pla_kda_kernel("decode") is not None
        and not is_conv_state_dim_first()
        and num_heads in (12, 24, 48, 96)
        and head_dim == 128
        and conv_width == 4
        and num_spec == 0
        and input_dtype == torch.bfloat16
        and conv_state_dtype == torch.bfloat16
        and recurrent_state_dtype == torch.float32
    )


patch(_MODULE, "is_fused_kda_decode_supported", **_META)(is_fused_kda_decode_supported)


def is_flashkda_supported(
    head_dim: int,
    input_dtype: torch.dtype,
    recurrent_state_dtype: torch.dtype,
    lower_bound: float | None,
) -> bool:
    if not current_platform.is_ppu():
        return _upstream_is_flashkda_supported(
            head_dim, input_dtype, recurrent_state_dtype, lower_bound
        )
    # PPU has no NVIDIA FlashKDA binary, and no compute capability to gate on.
    # PLA's prefill kernel takes over and supports the same input contract.
    from vllm_sail.attention.pla_kda import get_pla_kda_kernel

    return (
        get_pla_kda_kernel("prefill") is not None
        and head_dim == 128
        and input_dtype == torch.bfloat16
        and recurrent_state_dtype in (torch.bfloat16, torch.float32)
        and lower_bound is not None
    )


patch(_MODULE, "is_flashkda_supported", **_META)(is_flashkda_supported)


def _flashkda_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor,
    final_state: torch.Tensor,
    workspace: torch.Tensor,
    checkpoint_state: torch.Tensor | None = None,
    checkpoint_offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not (current_platform.is_ppu() and ppu_envs.VLLM_SAIL_USE_PLA):
        return _upstream_flashkda_prefill(
            q,
            k,
            v,
            g,
            beta,
            A_log,
            dt_bias,
            lower_bound,
            initial_state,
            cu_seqlens,
            out,
            final_state,
            workspace,
            checkpoint_state,
            checkpoint_offsets,
        )

    from pla.prefill.flashkdapro import flashkda_fwd

    logger.info_once("Using PLA KDA kernel: flashkda_fwd")
    if checkpoint_state is not None or checkpoint_offsets is not None:
        raise NotImplementedError(
            "PLA KDA does not expose intermediate checkpoint states"
        )
    B, T, H, D = q.shape
    flashkda_fwd(
        q=q.view(B * T, H, D),
        k=k.view(B * T, H, D),
        v=v.view(B * T, H, D),
        g=g.view(B * T, H, D),
        beta=beta.view(B * T, H),
        A_log=A_log,
        dt_bias=dt_bias.view(H, D),
        out=out.view(B * T, H, D),
        scale=q.shape[-1] ** -0.5,
        lower_bound=lower_bound,
        initial_state=initial_state,
        final_state=final_state,
        cu_seqlens=cu_seqlens,
    )
    return out, final_state


patch(_MODULE, "_flashkda_prefill", **_META)(_flashkda_prefill)


def _forward(
    self,
    mixed_qkv: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
    beta: torch.Tensor,
    core_attn_out: torch.Tensor,
) -> None:
    if current_platform.is_ppu() and ppu_envs.VLLM_SAIL_USE_PLA:
        metadata = get_forward_context().attn_metadata
        m = metadata.get(self.prefix) if isinstance(metadata, dict) else None
        # Exactly upstream's fully fused decode branch: that is the only case
        # PLA replaces, and every other batch shape stays with upstream.
        if (
            m is not None
            and m.non_spec_state_indices_tensor is not None
            and self.kda_decode_backend != "triton"
            and self.decode_conv1d_weight is not None
            and self.decode_norm_weight is not None
            and m.num_spec_decodes == 0
            and m.num_prefills == 0
            and m.num_decodes > 0
        ):
            from pla.decode.kda import fused_kda_decode_mega_forward

            logger.info_once("Using PLA KDA kernel: fused_kda_decode_mega_forward")
            num_actual_tokens = m.num_actual_tokens
            conv_state, recurrent_state = self.kv_cache[:2]
            if not is_conv_state_dim_first():
                conv_state = conv_state.transpose(-1, -2)
            fused_kda_decode_mega_forward(
                x=mixed_qkv[:num_actual_tokens],
                weight=self.decode_conv1d_weight,
                bias=self.conv1d.bias,
                conv_state=conv_state,
                raw_g=g1[:, :num_actual_tokens],
                raw_beta=beta[:, :num_actual_tokens],
                a_log=self.A_log,
                dt_bias=self.dt_bias,
                state_indices=m.non_spec_state_indices_tensor[:num_actual_tokens],
                state=recurrent_state,
                out=core_attn_out[:, :num_actual_tokens],
                lower_bound=self.gate_lower_bound,
                output_gate=g2[:num_actual_tokens],
                norm_weight=self.decode_norm_weight,
                norm_eps=self.o_norm.eps,
            )
            return
    _upstream_forward(self, mixed_qkv, g1, g2, beta, core_attn_out)


patch(_MODULE, "KimiK3DeltaAttention._forward", **_META)(
    _kda.eager_break_during_capture(_forward)
)


def _init(
    self,
    config: KimiLinearConfig,
    vllm_config: VllmConfig,
    prefix: str = "",
    aux_stream: torch.cuda.Stream | None = None,
    run_gemm_rs_ar: bool = False,
) -> None:
    if not current_platform.is_ppu():
        _upstream_init(self, config, vllm_config, prefix, aux_stream, run_gemm_rs_ar)
        return

    # Upstream's FlashKDA prefill branch exists to size an ``vllm._flashkda_C``
    # workspace, which PPU cannot import. Build the layer as if upstream had
    # picked Triton, then restore the FlashKDA prefill contract that the PLA
    # kernel serves: the same buffers, no workspace and no checkpoint output.
    upstream_resolve = _kda.resolve_kda_prefill_backend
    demoted_specs = []

    def resolve_without_nvidia_workspace(*args, **kwargs) -> str:
        backend = upstream_resolve(*args, **kwargs)
        if backend == "flashkda":
            T = vllm_config.scheduler_config.max_num_batched_tokens
            N = vllm_config.scheduler_config.max_num_seqs
            H, D = self.local_num_heads, self.head_dim
            specs = (
                ((1, T, H, D), self.model_config.dtype),
                ((N, H, D, D), self.get_state_dtype()[1]),
                ((N, H, D, D), self.get_state_dtype()[1]),
                ((0,), torch.uint8),
            )
            # The demoted upstream branch skips its reservation. Reserve the
            # whole simultaneous layout here, before norm and KV sizing.
            if _kda.is_workspace_manager_initialized():
                _kda.current_workspace_manager().get_simultaneous(*specs)
            demoted_specs.append(specs)
            return "triton"
        return backend

    # Bind only this constructor call to a private provider namespace. Other
    # constructors, including nested/non-PPU calls, keep the live resolver.
    namespace = ModuleType(_upstream_init.__module__)
    namespace.__dict__.update(_upstream_init.__globals__)
    namespace.resolve_kda_prefill_backend = resolve_without_nvidia_workspace
    bind_body(_upstream_init, namespace)(
        self, config, vllm_config, prefix, aux_stream, run_gemm_rs_ar
    )
    if not demoted_specs:
        return
    logger.info_once("PPU serves the FlashKDA prefill backend with the PLA kernel.")
    self.kda_prefill_backend = "flashkda"
    self._flashkda_buffer_specs = demoted_specs[-1]
    self._checkpoint_exporter = _kda.FlashKDAPrefillCheckpointExporter()


patch(_MODULE, "KimiK3DeltaAttention.__init__", **_META)(_init)


@patch(_MODULE, "KimiK3DeltaAttention.get_kv_cache_spec", **_META)
def _get_kv_cache_spec(self, vllm_config):
    spec = _upstream_get_kv_cache_spec(self, vllm_config)
    if current_platform.is_ppu() and self.kda_prefill_backend == "flashkda":
        # PLA exposes the final state, but not FlashKDA's intermediate
        # checkpoint output. Keep the same cache contract as Triton prefill.
        return replace(
            spec, num_prefill_checkpoint_blocks=0, prefill_checkpoint_alignment=None
        )
    return spec
