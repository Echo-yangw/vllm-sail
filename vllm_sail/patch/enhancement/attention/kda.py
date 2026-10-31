# ruff: noqa: E731, W291, W293, UP037
# ruff: noqa: F821
# Copied bodies resolve globals in their upstream module through bind_body.
# SPDX-License-Identifier: Apache-2.0
"""Use PLA KDA kernels on PPU, retaining Triton when PLA is disabled."""

from __future__ import annotations

from vllm.models.kimi_k3.nvidia import kda as _kda

from vllm_sail.patch.bodies import bind_body
from vllm_sail.patch.utils import patch

_MODULE = "vllm.models.kimi_k3.nvidia.kda"
_META = dict(
    reason="Kimi KDA must select PPU PLA kernels instead of NVIDIA FlashKDA/native decode.",
    affected_versions=">=0.32.0,<0.33.0",
    remove_when="Kimi KDA supports platform kernel registration for prefill and decode.",
)


def is_fused_kda_decode_supported(
    num_heads: int,
    head_dim: int,
    conv_width: int,
    num_spec: int,
    input_dtype: torch.dtype,
    conv_state_dtype: torch.dtype,
    recurrent_state_dtype: torch.dtype,
) -> bool:
    # The fused kernel handles both conv-state cache layouts (SD and DS); the
    # inner strides are selected from the tensor at launch time.
    # PPU MODIFICATION: begin
    from vllm_sail import envs as ppu_envs
    from vllm_sail.attention.pla_kda import get_pla_kda_kernel

    # PPU MODIFICATION: end
    if (
        num_heads not in (12, 24, 48, 96)
        or head_dim != 128
        or conv_width != 4
        or num_spec != 0
        or input_dtype != torch.bfloat16
        or conv_state_dtype != torch.bfloat16
        or recurrent_state_dtype != torch.float32
        # PPU MODIFICATION: begin
        or (current_platform.is_ppu() and is_conv_state_dim_first())
        or (
            not current_platform.is_ppu()
            and not hasattr(torch.ops._C, "fused_kda_decode")
        )
        # PPU MODIFICATION: end
    ):
        return False
    # SM90 is architecture-specific; SM10x and SM12x use family binaries.
    # PPU MODIFICATION: begin
    # PPU also supports fused_kda_decode.
    # PPU MODIFICATION: end
    return (
        # PPU MODIFICATION: begin
        get_pla_kda_kernel("decode") is not None
        if current_platform.is_ppu()
        else current_platform.is_device_capability(90)
        # PPU MODIFICATION: end
        or current_platform.is_device_capability_family(100)
        or current_platform.is_device_capability_family(120)
    )


is_fused_kda_decode_supported = patch(
    _MODULE, "is_fused_kda_decode_supported", **_META
)(bind_body(is_fused_kda_decode_supported, _kda))


def is_flashkda_supported(
    head_dim: int,
    input_dtype: torch.dtype,
    recurrent_state_dtype: torch.dtype,
    lower_bound: float | None,
) -> bool:
    # PPU MODIFICATION: begin
    from vllm_sail import envs as ppu_envs
    from vllm_sail.attention.pla_kda import get_pla_kda_kernel

    # PPU MODIFICATION: end
    if not current_platform.is_cuda():
        return False
    capability = current_platform.get_device_capability()
    return (
        # PPU MODIFICATION: begin
        (
            get_pla_kda_kernel("prefill") is not None
            if current_platform.is_ppu()
            else (capability is not None and capability.major in (9, 10, 12))
        )
        # PPU MODIFICATION: end
        and head_dim == 128
        and input_dtype == torch.bfloat16
        and recurrent_state_dtype in (torch.bfloat16, torch.float32)
        and lower_bound is not None
    )


is_flashkda_supported = patch(_MODULE, "is_flashkda_supported", **_META)(
    bind_body(is_flashkda_supported, _kda)
)


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
    # PPU MODIFICATION: begin
    from vllm_sail import envs as ppu_envs

    if current_platform.is_ppu() and ppu_envs.VLLM_SAIL_USE_PLA:
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

    # PPU MODIFICATION: end
    import vllm._flashkda_C  # noqa: F401

    # FlashKDA hardcodes dense Q/K/V/G strides. Beta may be row-strided because
    # FlashKDA materializes its transposed [H, T] layout internally.
    # TODO: Teach FlashKDA to consume beta in [T, H] layout directly instead
    # of transposing it to contiguous [H, T] storage internally.
    torch.ops._flashkda_C.fwd(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g.contiguous(),
        beta,
        q.shape[-1] ** -0.5,
        out,
        workspace,
        A_log.contiguous(),
        dt_bias.view(-1, q.shape[-1]).contiguous(),
        lower_bound,
        initial_state.contiguous(),
        final_state,
        cu_seqlens.contiguous(),
        checkpoint_state,
        checkpoint_offsets.contiguous() if checkpoint_offsets is not None else None,
    )
    return out, final_state


_flashkda_prefill = patch(_MODULE, "_flashkda_prefill", **_META)(
    bind_body(_flashkda_prefill, _kda)
)


def _forward(
    self,
    mixed_qkv: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
    beta: torch.Tensor,
    core_attn_out: torch.Tensor,
) -> None:
    # PPU MODIFICATION: begin
    from vllm_sail import envs as ppu_envs

    # PPU MODIFICATION: end
    forward_context = get_forward_context()
    attn_metadata_raw = forward_context.attn_metadata
    if attn_metadata_raw is None:
        return

    from vllm.models.kimi_k3.nvidia.ops.third_party.kda import (
        chunk_kda_with_fused_gate,
        fused_recurrent_kda,
        fused_recurrent_kda_packed_decode,
    )

    assert isinstance(attn_metadata_raw, dict)
    attn_metadata_narrowed = attn_metadata_raw.get(self.prefix)
    if attn_metadata_narrowed is None:
        return
    assert isinstance(attn_metadata_narrowed, KimiK3KDAMetadata)
    m = attn_metadata_narrowed
    has_initial_state = m.has_initial_state
    non_spec_query_start_loc = m.non_spec_query_start_loc
    non_spec_state_indices_tensor = m.non_spec_state_indices_tensor
    spec_token_indx = m.spec_token_indx
    non_spec_token_indx = m.non_spec_token_indx
    spec_token_start = m.spec_token_start
    non_spec_token_start = m.non_spec_token_start
    spec_state_indices_tensor = m.spec_state_indices_tensor
    spec_query_start_loc = m.spec_query_start_loc
    num_accepted_tokens = m.num_accepted_tokens
    num_actual_tokens = m.num_actual_tokens
    checkpoint = m.checkpoint
    has_spec_decode = m.num_spec_decodes > 0
    mixed_qkv = mixed_qkv[:num_actual_tokens]
    g1 = g1[:, :num_actual_tokens]
    beta = beta[:, :num_actual_tokens]

    conv_state, recurrent_state = self.kv_cache[:2]
    # The convolution kernels consume (..., dim, width - 1).
    if not is_conv_state_dim_first():
        conv_state = conv_state.transpose(-1, -2)

    if (
        self.kda_spec_decode_backend == "flashinfer"
        and has_spec_decode
        and m.num_prefills == 0
        and m.num_decodes == 0
    ):
        assert self.decode_conv1d_weight is not None
        assert self.decode_norm_weight is not None
        assert spec_state_indices_tensor is not None
        assert spec_query_start_loc is not None
        assert num_accepted_tokens is not None
        assert self.gate_lower_bound is not None
        # "Packed" is FlashInfer's name for the varlen T>1 speculative path.
        flashinfer_packed_fused_kda_decode(
            x=mixed_qkv,
            weight=self.decode_conv1d_weight,
            conv_state=conv_state,
            raw_gate=g1,
            raw_beta=beta,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            state_indices=spec_state_indices_tensor,
            query_start_loc=spec_query_start_loc,
            num_accepted_tokens=num_accepted_tokens,
            state=recurrent_state,
            output_gate=g2[:num_actual_tokens],
            norm_weight=self.decode_norm_weight,
            lower_bound=self.gate_lower_bound,
            norm_eps=self.o_norm.eps,
            output=core_attn_out[:, :num_actual_tokens],
        )
        return

    if (
        self.kda_decode_backend != "triton"
        and self.decode_conv1d_weight is not None
        and self.decode_norm_weight is not None
        and not has_spec_decode
        and m.num_prefills == 0
        and m.num_decodes > 0
    ):
        assert non_spec_state_indices_tensor is not None
        # PPU MODIFICATION: begin

        if current_platform.is_ppu() and ppu_envs.VLLM_SAIL_USE_PLA:
            from pla.decode.kda import fused_kda_decode_mega_forward

            logger.info_once("Using PLA KDA kernel: fused_kda_decode_mega_forward")
            fused_kda_decode_mega_forward(
                x=mixed_qkv,
                weight=self.decode_conv1d_weight,
                bias=self.conv1d.bias,
                conv_state=conv_state,
                raw_g=g1,
                raw_beta=beta,
                a_log=self.A_log,
                dt_bias=self.dt_bias,
                state_indices=non_spec_state_indices_tensor[:num_actual_tokens],
                state=recurrent_state,
                out=core_attn_out[:, :num_actual_tokens],
                lower_bound=self.gate_lower_bound,
                output_gate=g2[:num_actual_tokens],
                norm_weight=self.decode_norm_weight,
                norm_eps=self.o_norm.eps,
            )
            return

        # PPU MODIFICATION: end
        state_indices = non_spec_state_indices_tensor[:num_actual_tokens]
        if self.kda_decode_backend == "flashinfer":
            flashinfer_fused_kda_decode(
                x=mixed_qkv,
                weight=self.decode_conv1d_weight,
                conv_state=conv_state,
                raw_gate=g1,
                raw_beta=beta,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                state_indices=state_indices,
                state=recurrent_state,
                output_gate=g2[:num_actual_tokens],
                norm_weight=self.decode_norm_weight,
                lower_bound=self.gate_lower_bound,
                norm_eps=self.o_norm.eps,
                output=core_attn_out[:, :num_actual_tokens],
            )
        else:
            ops.fused_kda_decode(
                x=mixed_qkv,
                weight=self.decode_conv1d_weight,
                bias=self.conv1d.bias,
                conv_state=conv_state,
                raw_g=g1,
                raw_beta=beta,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                state_indices=state_indices,
                state=recurrent_state,
                out=core_attn_out[:, :num_actual_tokens],
                lower_bound=self.gate_lower_bound,
                output_gate=g2[:num_actual_tokens],
                norm_weight=self.decode_norm_weight,
                norm_eps=self.o_norm.eps,
            )
        return

    conv_weights = self.conv1d.weight.view(
        self.conv1d.weight.size(0), self.conv1d.weight.size(2)
    )
    q_conv_weight, k_conv_weight, v_conv_weight = conv_weights.split(
        self.local_projection_size, dim=0
    )
    q_conv_state, k_conv_state, v_conv_state = conv_state.split(
        self.local_projection_size, dim=-2
    )

    # Separate multi-query speculative tokens from prefill/plain decode.
    if has_spec_decode:
        if m.num_prefills == 0 and m.num_decodes == 0:
            mixed_qkv_spec = mixed_qkv
            g1_spec, beta_spec = g1, beta
            mixed_qkv_ns = g1_ns = beta_ns = None
        elif spec_token_start is not None:
            assert non_spec_token_start is not None
            spec_end = spec_token_start + m.num_spec_decode_tokens
            non_spec_end = (
                non_spec_token_start + m.num_prefill_tokens + m.num_decode_tokens
            )
            spec_slice = slice(spec_token_start, spec_end)
            non_spec_slice = slice(non_spec_token_start, non_spec_end)
            mixed_qkv_spec = mixed_qkv[spec_slice]
            g1_spec, beta_spec = g1[:, spec_slice], beta[:, spec_slice]
            mixed_qkv_ns = mixed_qkv[non_spec_slice]
            g1_ns, beta_ns = g1[:, non_spec_slice], beta[:, non_spec_slice]
        else:
            assert spec_token_indx is not None
            assert non_spec_token_indx is not None
            mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
            g1_spec = g1.index_select(1, spec_token_indx)
            beta_spec = beta.index_select(1, spec_token_indx)
            mixed_qkv_ns = mixed_qkv.index_select(0, non_spec_token_indx)
            g1_ns = g1.index_select(1, non_spec_token_indx)
            beta_ns = beta.index_select(1, non_spec_token_indx)
    else:
        mixed_qkv_spec = g1_spec = beta_spec = None
        mixed_qkv_ns, g1_ns, beta_ns = mixed_qkv, g1, beta

    # Spec-decode multi-query path.
    core_attn_out_spec = None
    if has_spec_decode:
        assert spec_state_indices_tensor is not None
        assert spec_query_start_loc is not None
        spec_cu_seqlens = spec_query_start_loc[: m.num_spec_decodes + 1]
        spec_out = (
            core_attn_out[:, : mixed_qkv_spec.shape[0]]
            if m.num_prefills == 0 and m.num_decodes == 0
            else None
        )
        if spec_token_start is not None:
            spec_out = core_attn_out[:, spec_slice]
        if self.use_recoverssm:
            core_attn_out_spec = self._recoverssm_verify(
                m,
                mixed_qkv_spec,
                g1_spec,
                beta_spec,
                conv_weights,
                spec_cu_seqlens,
                spec_out,
            )
        else:
            mixed_qkv_spec = causal_conv1d_update(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                activation="silu",
                conv_state_indices=spec_state_indices_tensor[:, 0][
                    : m.num_spec_decodes
                ],
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
                validate_data=False,
                out=torch.empty_like(mixed_qkv_spec),
            )
            q_spec, k_spec, v_spec = (
                rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim)
                for x in mixed_qkv_spec.split(self.local_projection_size, dim=-1)
            )
            core_attn_out_spec, _ = fused_recurrent_kda(
                q=q_spec,
                k=k_spec,
                v=v_spec,
                raw_g=g1_spec,
                raw_beta=beta_spec,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                lower_bound=self.gate_lower_bound,
                initial_state=recurrent_state,
                cu_seqlens=spec_cu_seqlens,
                ssm_state_indices=spec_state_indices_tensor,
                num_accepted_tokens=num_accepted_tokens,
                out=spec_out,
            )

    # Prefill or plain-decode path.
    core_attn_out_non_spec = None
    if mixed_qkv_ns is not None:
        assert g1_ns is not None and beta_ns is not None
        non_spec_out = None
        if non_spec_token_start is not None:
            non_spec_out = core_attn_out[:, non_spec_slice]
        if m.num_prefills > 0:
            q_ns, k_ns, v_ns = mixed_qkv_ns.split(self.local_projection_size, dim=-1)

            # Separate convolution calls accept row-strided packed inputs
            # and produce dense Q/K/V without an additional V copy.
            def _prefill_conv(
                x: torch.Tensor,
                state: torch.Tensor,
                weight: torch.Tensor,
            ) -> torch.Tensor:
                return causal_conv1d_fn(
                    x.transpose(0, 1),
                    weight,
                    None,
                    activation="silu",
                    conv_states=state,
                    has_initial_state=has_initial_state,
                    cache_indices=non_spec_state_indices_tensor,
                    query_start_loc=non_spec_query_start_loc,
                    metadata=m,
                ).transpose(0, 1)

            q_ns = _prefill_conv(q_ns, q_conv_state, q_conv_weight)
            k_ns = _prefill_conv(k_ns, k_conv_state, k_conv_weight)
            v_ns = _prefill_conv(v_ns, v_conv_state, v_conv_weight)
            q_ns, k_ns, v_ns = (
                rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim)
                for x in (q_ns, k_ns, v_ns)
            )

            assert non_spec_state_indices_tensor is not None
            assert has_initial_state is not None
            initial_state = gather_initial_states(
                recurrent_state,
                non_spec_state_indices_tensor,
                has_initial_state,
            )
            if self.kda_prefill_backend == "flashkda":
                assert self.gate_lower_bound is not None
                assert self._flashkda_buffer_specs is not None
                workspace_out, final_state, checkpoint_state, workspace = (
                    current_workspace_manager().get_simultaneous(
                        *self._flashkda_buffer_specs
                    )
                )
                flashkda_out = (workspace_out if has_spec_decode else core_attn_out)[
                    :, : q_ns.shape[1]
                ]
                if non_spec_out is not None:
                    flashkda_out = non_spec_out
                if checkpoint is not None:
                    assert non_spec_query_start_loc is not None
                    num_sequences = initial_state.shape[0]
                    assert checkpoint.checkpoint_offsets.shape == (num_sequences,)
                    final_state = final_state[:num_sequences]
                    checkpoint_state = checkpoint_state[:num_sequences]
                    checkpoint_offsets = checkpoint.checkpoint_offsets
                    _flashkda_prefill(
                        q=q_ns,
                        k=k_ns,
                        v=v_ns,
                        g=g1_ns,
                        beta=beta_ns,
                        A_log=self.A_log,
                        dt_bias=self.dt_bias,
                        lower_bound=self.gate_lower_bound,
                        initial_state=initial_state,
                        cu_seqlens=non_spec_query_start_loc,
                        out=flashkda_out,
                        final_state=final_state,
                        workspace=workspace,
                        checkpoint_state=checkpoint_state,
                        checkpoint_offsets=checkpoint_offsets,
                    )
                    core_attn_out_non_spec = flashkda_out
                    last_recurrent_state = final_state
                    assert self._checkpoint_exporter is not None
                    self._checkpoint_exporter.export(
                        checkpoint,
                        raw_qkv=mixed_qkv_ns,
                        conv_state=conv_state,
                        recurrent_checkpoint=checkpoint_state,
                        recurrent_state=recurrent_state,
                        cu_seqlens=non_spec_query_start_loc,
                    )
                else:
                    (
                        core_attn_out_non_spec,
                        last_recurrent_state,
                    ) = _flashkda_prefill(
                        q=q_ns,
                        k=k_ns,
                        v=v_ns,
                        g=g1_ns,
                        beta=beta_ns,
                        A_log=self.A_log,
                        dt_bias=self.dt_bias,
                        lower_bound=self.gate_lower_bound,
                        initial_state=initial_state,
                        cu_seqlens=non_spec_query_start_loc,
                        out=flashkda_out,
                        final_state=final_state[: initial_state.shape[0]],
                        workspace=workspace,
                    )
            elif self.kda_prefill_backend == "flashinfer":
                assert self.gate_lower_bound is not None
                assert m.flashinfer_prefill_query_start_loc is not None
                if q_ns.shape[1] > initial_state.shape[0]:
                    assert m.flashinfer_prefill_seq_order is not None
                flashinfer_out = core_attn_out[:, : q_ns.shape[1]]
                if non_spec_out is not None:
                    flashinfer_out = non_spec_out
                elif has_spec_decode:
                    assert self._flashinfer_kda_output_spec is not None
                    (workspace_out,) = current_workspace_manager().get_simultaneous(
                        self._flashinfer_kda_output_spec
                    )
                    flashinfer_out = workspace_out[:, : q_ns.shape[1]]
                (
                    core_attn_out_non_spec,
                    last_recurrent_state,
                ) = _flashinfer_kda_prefill(
                    q=q_ns,
                    k=k_ns,
                    v=v_ns,
                    raw_g=g1_ns,
                    raw_beta=beta_ns,
                    A_log=self.A_log,
                    dt_bias=self.dt_bias,
                    lower_bound=self.gate_lower_bound,
                    initial_state=initial_state,
                    cu_seqlens=m.flashinfer_prefill_query_start_loc,
                    out=flashinfer_out,
                    seq_order=m.flashinfer_prefill_seq_order,
                )
            else:
                (
                    core_attn_out_non_spec,
                    last_recurrent_state,
                ) = chunk_kda_with_fused_gate(
                    q=q_ns,
                    k=k_ns,
                    v=v_ns,
                    raw_g=g1_ns,
                    raw_beta=beta_ns,
                    A_log=self.A_log,
                    g_bias=self.dt_bias,
                    lower_bound=self.gate_lower_bound,
                    initial_state=initial_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                    cu_seqlens=non_spec_query_start_loc,
                    out=non_spec_out,
                )
            recurrent_state[non_spec_state_indices_tensor] = last_recurrent_state.to(
                recurrent_state.dtype
            )
        else:
            # Pure non-speculative decode.
            assert non_spec_state_indices_tensor is not None
            decode_conv_indices = non_spec_state_indices_tensor[: mixed_qkv_ns.size(0)]
            packed_conv_out = torch.empty_like(mixed_qkv_ns)
            mixed_qkv_ns = causal_conv1d_update(
                mixed_qkv_ns,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                activation="silu",
                conv_state_indices=decode_conv_indices,
                validate_data=True,
                out=packed_conv_out,
            )
            (
                core_attn_out_non_spec,
                _,
            ) = fused_recurrent_kda_packed_decode(
                mixed_qkv=mixed_qkv_ns,
                raw_g=g1_ns,
                raw_beta=beta_ns,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                lower_bound=self.gate_lower_bound,
                initial_state=recurrent_state,
                state_indices=decode_conv_indices,
                out=non_spec_out,
            )

    # Restore the scheduler's original token order for mixed batches.
    if core_attn_out_spec is not None and core_attn_out_non_spec is not None:
        if spec_token_start is None:
            assert spec_token_indx is not None
            assert non_spec_token_indx is not None
            core_attn_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            core_attn_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
    elif core_attn_out_non_spec is not None:
        if (
            self.kda_prefill_backend not in ("flashkda", "flashinfer")
            or m.num_prefills == 0
        ):
            # TODO: decode kernels write directly to core_attn_out
            core_attn_out[0, :num_actual_tokens] = core_attn_out_non_spec[
                0, :num_actual_tokens
            ]
    else:
        assert core_attn_out_spec is not None
    # Triton normalizes in place, so this is a self-copy with no device
    # work. Keep it for the out-of-place native implementation.
    core_attn_out.copy_(self.o_norm(core_attn_out, g2))


_forward = patch(_MODULE, "KimiK3DeltaAttention._forward", **_META)(
    _kda.eager_break_during_capture(bind_body(_forward, _kda))
)


def _init(
    self,
    config: KimiLinearConfig,
    vllm_config: VllmConfig,
    prefix: str = "",
    aux_stream: torch.cuda.Stream | None = None,
    run_gemm_rs_ar: bool = False,
) -> None:
    # PPU MODIFICATION: begin
    # An out-of-class copied method has no implicit __class__ cell.
    super(KimiK3DeltaAttention, self).__init__(config, vllm_config, prefix)
    # PPU MODIFICATION: end
    self.use_recoverssm = self.cache_config.use_kda_recoverssm
    if self.cache_config.use_replayssm and not self.use_recoverssm:
        raise ValueError(
            "Kimi-K3 supports --use-replayssm only with speculative decoding"
        )
    self.spec_query_len = 1 + self.num_spec

    kda_config = config.linear_attn_config  # type: ignore[attr-defined]
    assert kda_config is not None, "linear_attn_config must be set"
    self.head_dim = kda_config["head_dim"]
    self.num_heads = kda_config["num_heads"]
    assert self.num_heads % self.tp_size == 0
    self.local_num_heads = divide(self.num_heads, self.tp_size)
    self.projection_size = self.head_dim * self.num_heads
    self.local_projection_size = divide(self.projection_size, self.tp_size)
    self.conv_size = kda_config["short_conv_kernel_size"]
    self.gate_lower_bound: float | None = kda_config.get("gate_lower_bound", None)
    if self.gate_lower_bound is not None:
        assert _KDA_GATE_LOGBOUND_MIN <= self.gate_lower_bound < 0, (
            "KDA gate lower bound must be in "
            f"[{_KDA_GATE_LOGBOUND_MIN}, 0). "
            f"Got {self.gate_lower_bound}."
        )
    assert kda_config.get("use_full_rank_gate", False), (
        "KimiK3DeltaAttention requires a full-rank gate"
    )
    self._projection_aux_stream = aux_stream
    self._projection_events = (
        (torch.cuda.Event(), torch.cuda.Event()) if aux_stream is not None else None
    )
    self._projection_overlap_max_tokens = 0

    # Keep f_a before the narrow beta shard, then align each TP-local row.
    qkvg_output_sizes = [self.projection_size] * 4
    in_proj_output_sizes = qkvg_output_sizes + [
        self.head_dim,
        self.num_heads,
    ]
    local_output_size = (
        4 * self.local_projection_size + self.head_dim + self.local_num_heads
    )
    in_proj_prefix = f"{prefix}.in_proj_qkvgfab"
    alignment = (
        128
        if isinstance(self.quant_config, ModelOptMixedPrecisionConfig)
        and self.quant_config._resolve_quant_algo(in_proj_prefix) == "FP8_PB_WO"
        else 16
    )
    self.in_proj_padding = -local_output_size % alignment
    if self.in_proj_padding:
        in_proj_output_sizes.append(self.in_proj_padding * self.tp_size)
    self.in_proj_qkvgfab = _KimiGDNMergedColumnParallelLinear(
        self.hidden_size,
        in_proj_output_sizes,
        replicated_shard_id=4,
        tp_size=self.tp_size,
        padding_size=self.in_proj_padding,
        bias=False,
        quant_config=self.quant_config,
        prefix=in_proj_prefix,
    )

    self.f_b_proj = ColumnParallelLinear(
        self.head_dim,
        self.projection_size,
        bias=False,
        quant_config=self.quant_config,
        prefix=f"{prefix}.f_b_proj",
    )
    self.dt_bias = nn.Parameter(
        torch.empty(self.local_projection_size, dtype=torch.float32)
    )
    set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

    # One packed parameter and cache let decode run a single conv update.
    # Prefill slices them back into Q/K/V to obtain dense outputs cheaply.
    self.conv1d = ColumnParallelLinear(
        input_size=self.conv_size,
        output_size=3 * self.projection_size,
        bias=False,
        params_dtype=torch.float32,
        prefix=f"{prefix}.conv1d",
    )
    self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)
    # Keep a width-major copy for fused decode without changing the layout
    # consumed by the prefill and fallback decode kernels.
    conv_state_dtype, recurrent_state_dtype = self.get_state_dtype()[:2]
    additional_config = vllm_config.additional_config
    decode_backend = (
        additional_config.get("kda_decode_backend", "auto")
        if isinstance(additional_config, dict)
        else "auto"
    )
    self.kda_decode_backend = resolve_kda_decode_backend(
        decode_backend,
        self.local_num_heads,
        self.head_dim,
        self.conv_size,
        self.num_spec,
        vllm_config.model_config.dtype,
        conv_state_dtype,
        recurrent_state_dtype,
    )
    spec_decode_backend = (
        additional_config.get("kda_spec_decode_backend", "auto")
        if isinstance(additional_config, dict)
        else "auto"
    )
    self.kda_spec_decode_backend = resolve_kda_spec_decode_backend(
        spec_decode_backend,
        self.local_num_heads,
        self.head_dim,
        self.conv_size,
        self.num_spec,
        vllm_config.model_config.dtype,
        conv_state_dtype,
        recurrent_state_dtype,
        self.gate_lower_bound,
        self.use_recoverssm,
    )
    # RecoverSSM: this layer's verify records of a speculative step, and its
    # verify and commit kernels: "cute" (SM10x, the short convolution fused
    # into the verify) where they can run, else "triton".
    self.recoverssm_backend: Literal["cute", "triton"] | None = None
    self.recoverssm_records: KDARecoverSSMRecords | None = None
    self.recoverssm_verify: CuteRecoverSSMVerify | None = None
    if self.use_recoverssm:
        self.recoverssm_records = KDARecoverSSMRecords.allocate(
            vllm_config.scheduler_config.max_num_seqs,
            self.local_num_heads,
            self.spec_query_len,
            self.head_dim,
            vllm_config.model_config.dtype,
            self.conv1d.weight.device,
        )
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        records_bytes = len(kda_config["kda_layers"]) * sum(
            t.numel() * t.element_size()
            for t in (
                self.recoverssm_records.correction,
                self.recoverssm_records.key_gate,
            )
        )
        logger.info_once(
            "Allocated the KDA RecoverSSM verify records: %.2f GiB in total "
            "for %d requests (%.2f MiB per request).",
            records_bytes / 2**30,
            max_num_seqs,
            records_bytes / max_num_seqs / 2**20,
        )
        supported, reason = can_use_cute_recoverssm(
            head_dim=self.head_dim,
            window=self.spec_query_len,
            conv_width=self.conv_size,
            has_conv_bias=self.conv1d.bias is not None,
            activation_dtype=vllm_config.model_config.dtype,
            conv_state_dtype=conv_state_dtype,
            state_dtype=recurrent_state_dtype,
            conv_weight_dtype=self.conv1d.weight.dtype,
            lower_bound=self.gate_lower_bound,
        )
        self.recoverssm_backend = "cute" if supported else "triton"
        if self.recoverssm_backend == "triton":
            logger.info_once(
                "Using the Triton KDA RecoverSSM verify and commit: the CuTe "
                "kernels cannot run (%s).",
                reason,
            )
        else:
            logger.info_once("Using the CuTe KDA RecoverSSM verify and commit.")
            self.recoverssm_verify = CuteRecoverSSMVerify(
                num_heads=self.local_num_heads,
                window=self.spec_query_len,
                conv_dim_first=is_conv_state_dim_first(),
                conv_weight_dtype=self.conv1d.weight.dtype,
                device=self.conv1d.weight.device,
            )
            self.recoverssm_verify.compile()
            compile_state_commit(
                self.local_num_heads,
                self.spec_query_len,
                self.conv1d.weight.device,
                align=self.cache_config.mamba_cache_mode == "align",
            )
    decode_conv1d_weight = None
    if (
        self.kda_decode_backend != "triton"
        or self.kda_spec_decode_backend == "flashinfer"
    ):
        decode_conv1d_weight = torch.empty(
            3,
            self.conv_size,
            self.local_projection_size,
            dtype=self.conv1d.weight.dtype,
            device=self.conv1d.weight.device,
        )
    self.register_buffer("decode_conv1d_weight", decode_conv1d_weight, persistent=False)
    delattr(self.conv1d.weight, "weight_loader")
    set_weight_attrs(
        self.conv1d.weight,
        {
            "weight_loader": _make_decode_conv1d_weight_loader(
                [self.projection_size] * 3,
                self.tp_size,
                self.tp_rank,
                decode_conv1d_weight,
            )
        },
    )

    self.A_log = nn.Parameter(torch.empty(self.local_num_heads, dtype=torch.float32))
    set_weight_attrs(self.A_log, {"weight_loader": a_log_weight_loader(0)})

    backend = (
        additional_config.get("kda_prefill_backend", "auto")
        if isinstance(additional_config, dict)
        else "auto"
    )
    self.kda_prefill_backend = resolve_kda_prefill_backend(
        backend,
        self.head_dim,
        vllm_config.model_config.dtype,
        recurrent_state_dtype,
        self.gate_lower_bound,
    )
    self._flashkda_buffer_specs: (
        tuple[tuple[tuple[int, ...], torch.dtype], ...] | None
    ) = None
    self._flashinfer_kda_output_spec: tuple[tuple[int, ...], torch.dtype] | None = None
    self._checkpoint_exporter: FlashKDAPrefillCheckpointExporter | None = None
    if self.kda_prefill_backend == "flashkda":
        T = vllm_config.scheduler_config.max_num_batched_tokens
        N = vllm_config.scheduler_config.max_num_seqs
        H, D = self.local_num_heads, self.head_dim
        # PPU MODIFICATION: begin
        if current_platform.is_ppu():
            # PLA owns its workspace; no NVIDIA extension is loaded on PPU.
            workspace_size = 0
        else:
            import vllm._flashkda_C  # noqa: F401

            workspace_size = torch.ops._flashkda_C.get_workspace_size(T, H, N)
        # PPU MODIFICATION: end
        self._flashkda_buffer_specs = (
            ((1, T, H, D), self.model_config.dtype),
            ((N, H, D, D), self.get_state_dtype()[1]),
            ((N, H, D, D), self.get_state_dtype()[1]),
            ((workspace_size,), torch.uint8),
        )
        self._checkpoint_exporter = FlashKDAPrefillCheckpointExporter()
    elif self.kda_prefill_backend == "flashinfer":
        T = vllm_config.scheduler_config.max_num_batched_tokens
        H, D = self.local_num_heads, self.head_dim
        self._flashinfer_kda_output_spec = (
            (1, T, H, D),
            self.model_config.dtype,
        )

    # Profiling skips attention, so reserve the shared runtime buffers before
    # the KV cache is sized. PLA's vendor workspace can be empty while its
    # output and recurrent-state buffers still need this reservation.
    if is_workspace_manager_initialized():
        if self._flashkda_buffer_specs is not None:
            current_workspace_manager().get_simultaneous(*self._flashkda_buffer_specs)
        elif self._flashinfer_kda_output_spec is not None:
            current_workspace_manager().get_simultaneous(
                self._flashinfer_kda_output_spec
            )

    self.o_norm = FusedRMSNormGated(self.head_dim, activation="sigmoid")
    decode_norm_weight = None
    if decode_conv1d_weight is not None:
        decode_norm_weight = torch.empty(
            self.head_dim,
            dtype=torch.float32,
            device=self.o_norm.weight.device,
        )
    self.register_buffer("decode_norm_weight", decode_norm_weight, persistent=False)
    if decode_norm_weight is not None:
        # Upcast once while loading; direct BF16 norm weights slow the
        # fully fused decode kernel.
        if hasattr(self.o_norm.weight, "weight_loader"):
            delattr(self.o_norm.weight, "weight_loader")
        set_weight_attrs(
            self.o_norm.weight,
            {"weight_loader": _make_decode_norm_weight_loader(decode_norm_weight)},
        )
    self.o_proj = RowParallelLinear(
        self.projection_size,
        self.hidden_size,
        bias=False,
        quant_config=self.quant_config,
        prefix=f"{prefix}.o_proj",
    )
    self.gemm_rs_ar = None
    if run_gemm_rs_ar:
        from vllm.model_executor.kernels.linear.cute_dsl.gemm_rs_ar import (
            get_gemm_rs_ar,
        )

        gemm_rs_ar = get_gemm_rs_ar()
        if gemm_rs_ar.can_run(self.o_proj):
            self.gemm_rs_ar = gemm_rs_ar
        else:
            gemm_rs_ar.warn_incompatible_projection()
    compilation_config = vllm_config.compilation_config
    if prefix in compilation_config.static_forward_context:
        raise ValueError(f"Duplicate layer name: {prefix}")
    compilation_config.static_forward_context[prefix] = self


patch(_MODULE, "KimiK3DeltaAttention.__init__", **_META)(bind_body(_init, _kda))


_upstream_cache_spec = _kda.KimiK3DeltaAttention.get_kv_cache_spec


@patch(_MODULE, "KimiK3DeltaAttention.get_kv_cache_spec", **_META)
def _get_kv_cache_spec(self, vllm_config):
    from dataclasses import replace

    from vllm.platforms import current_platform

    spec = _upstream_cache_spec(self, vllm_config)
    if current_platform.is_ppu() and self.kda_prefill_backend == "flashkda":
        # PLA exposes the final state, but not FlashKDA's intermediate
        # checkpoint output. Keep the same cache contract as Triton prefill.
        return replace(
            spec, num_prefill_checkpoint_blocks=0, prefill_checkpoint_alignment=None
        )
    return spec
