# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 delayed mHC: retain upstream epilogues, use PPU prenorm GEMM."""

from __future__ import annotations

import sys

import torch
from vllm.platforms import current_platform

from vllm_sail.patch.utils import PATCH_MARKER, patch

_MODULE = "vllm.model_executor.kernels.mhc.tilelang"
_AFFECTED = ">=0.32.0,<0.33.0"
_REASON = "Delayed mHC bypasses the PPU prenorm helper and its zeroed single-split SAIL accumulators."
_REMOVE_WHEN = (
    "Upstream delayed mHC launchers call a platform-dispatched prenorm helper."
)


@patch(
    _MODULE,
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def mhc_pre_delayed_tilelang(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: torch.Tensor | None = None,
    x: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run mHC pre with a carried pre-mix and return the next pre-mix.

    Args:
        residual: BF16 residual streams of shape (tokens, hc_mult, hidden_size).
        fn: FP32 projection of shape (hc_mult * (hc_mult + 2), input_size).
        hc_scale: FP32 scales of shape (3,).
        hc_base: FP32 bias of shape (hc_mult * (hc_mult + 2),).
        rms_eps: RMS normalization epsilon.
        hc_pre_eps: Pre-mix epsilon.
        hc_sinkhorn_eps: Sinkhorn epsilon.
        hc_post_mult_value: Post-mix multiplier.
        sinkhorn_repeat: Number of Sinkhorn iterations.
        pre_mix: FP32 coefficients from the previous sublayer, or None to
            select residual stream zero at model entry.
        x: Optional BF16 projection input of shape (tokens, input_size), for
            the first layer's broadcast embedding and summed projection.
        norm_weight: Optional BF16 RMSNorm weight for the collapsed input.
        norm_eps: RMSNorm epsilon for the collapsed input.

    Returns:
        Post and residual coefficients, optionally normalized BF16 layer input,
        and the next FP32 pre-mix, with shapes (tokens, hc_mult, 1),
        (tokens, hc_mult, hc_mult), (tokens, hidden_size), and (tokens, hc_mult).

    """
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        _HC_PRENORM_GEMM_TILELANG_KERNEL,
        mhc_pre_big_fuse_tilelang,
    )
    from vllm.model_executor.kernels.mhc.warmup import (
        MHC_PRE_NORM_KERNEL,
        compute_mhc_pre_num_splits,
    )
    from vllm.utils.deep_gemm import (
        is_deep_gemm_supported,
        tf32_hc_prenorm_gemm,
    )

    assert residual.ndim == 3 and residual.dtype == torch.bfloat16
    assert residual.is_contiguous()
    num_tokens, hc_mult, hidden_size = residual.shape
    if x is None:
        x = residual.view(num_tokens, hc_mult * hidden_size)
    assert x.ndim == 2 and x.dtype == torch.bfloat16 and x.is_contiguous()
    assert x.shape[0] == num_tokens
    input_size = x.shape[1]
    mix_size = hc_mult * (hc_mult + 2)
    assert fn.shape == (mix_size, input_size) and fn.dtype == torch.float32
    assert hc_scale.shape == (3,) and hc_scale.dtype == torch.float32
    assert hc_base.shape == (mix_size,) and hc_base.dtype == torch.float32
    if pre_mix is not None:
        assert pre_mix.shape == (num_tokens, hc_mult)
        assert pre_mix.dtype == torch.float32 and pre_mix.is_contiguous()

    next_pre_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    post = torch.empty_like(next_pre_mix)
    comb = torch.empty(
        num_tokens, hc_mult * hc_mult, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )
    outputs = (
        post.unsqueeze(-1),
        comb.view(num_tokens, hc_mult, hc_mult),
        layer_input,
        next_pre_mix,
    )
    if num_tokens == 0:
        return outputs

    # PPU MODIFICATION: begin
    if current_platform.is_ppu():
        from vllm_sail.patch.enhancement.attention.mhc_tilelang import (
            _hc_prenorm_gemm_outputs,
        )

        mixes, sqrsum = _hc_prenorm_gemm_outputs(
            x, fn, hidden_size=input_size, hc_mult=1
        )
        n_splits = mixes.shape[0]
    else:
        # PPU MODIFICATION: end
        use_deep_gemm = is_deep_gemm_supported()
        n_splits = (
            compute_mhc_pre_num_splits(input_size, num_tokens) if use_deep_gemm else 1
        )
        mixes = torch.empty(
            n_splits, num_tokens, mix_size, dtype=torch.float32, device=residual.device
        )
        sqrsum = torch.empty(
            n_splits, num_tokens, dtype=torch.float32, device=residual.device
        )
        if use_deep_gemm:
            tf32_hc_prenorm_gemm(x, fn, mixes, sqrsum, n_splits)
        else:
            _HC_PRENORM_GEMM_TILELANG_KERNEL(
                x,
                fn,
                mixes,
                sqrsum,
                input_size,
                1,
            )
    if norm_weight is not None:
        assert norm_weight.shape == (hidden_size,)
        assert norm_weight.dtype == torch.bfloat16 and norm_weight.is_contiguous()
        MHC_PRE_NORM_KERNEL(
            mixes,
            sqrsum,
            hc_scale,
            hc_base,
            residual,
            post,
            comb,
            layer_input,
            norm_weight,
            pre_mix if pre_mix is not None else post,
            next_pre_mix,
            layer_input,
            hidden_size=hidden_size,
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            hc_sinkhorn_eps=hc_sinkhorn_eps,
            hc_post_mult_value=hc_post_mult_value,
            sinkhorn_repeat=sinkhorn_repeat,
            norm_eps=norm_eps,
            hc_mult=hc_mult,
            use_pre_mix_in=pre_mix is not None,
            save_pre_mix=True,
            rms_numel=input_size,
        )
        return outputs
    mhc_pre_big_fuse_tilelang(
        mixes,
        sqrsum,
        hc_scale,
        hc_base,
        residual,
        post,
        comb,
        layer_input,
        pre_mix if pre_mix is not None else post,
        next_pre_mix,
        layer_input,
        hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
        hc_mult,
        use_pre_mix_in=pre_mix is not None,
        save_pre_mix=True,
        rms_numel=input_size,
    )
    return outputs


@patch(
    _MODULE,
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def mhc_fused_post_pre_delayed_tilelang(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    capture_aux: bool = False,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Run one mHC post block followed by the next delayed mHC pre block.

    Within the fused kernel's token range the post mapping is folded into the
    pre-norm GEMM, so the updated residual streams feed the projection from
    registers instead of a second pass over global memory. Above it this runs
    the same post kernel and split-k GEMM as the unfused pair.

    Args:
        x: BF16 sublayer output of shape (tokens, hidden_size).
        residual: BF16 residual streams of shape (tokens, hc_mult, hidden_size).
        post_layer_mix: FP32 post coefficients of shape (tokens, hc_mult, 1).
        comb_res_mix: FP32 residual coefficients, (tokens, hc_mult, hc_mult).
        fn: FP32 projection of shape (hc_mult * (hc_mult + 2), input_size).
        hc_scale: FP32 scales of shape (3,).
        hc_base: FP32 bias of shape (hc_mult * (hc_mult + 2),).
        rms_eps: RMS normalization epsilon.
        hc_pre_eps: Pre-mix epsilon.
        hc_sinkhorn_eps: Sinkhorn epsilon.
        hc_post_mult_value: Post-mix multiplier.
        sinkhorn_repeat: Number of Sinkhorn iterations.
        pre_mix: FP32 coefficients from the previous sublayer, or None to
            select residual stream zero.
        norm_weight: Optional BF16 RMSNorm weight for the collapsed input.
        norm_eps: RMSNorm epsilon for the collapsed input.
        capture_aux: Also return the mean over the post-mapped streams, which
            draft models consume as the target's hidden state. It is folded
            into the collapse, which already reads those streams.

    Returns:
        The post-mapped residual streams, the post and residual coefficients,
        the optionally normalized BF16 layer input, the next FP32 pre-mix, and
        the BF16 stream mean (empty unless capture_aux), with shapes
        (tokens, hc_mult, hidden_size), (tokens, hc_mult, 1),
        (tokens, hc_mult, hc_mult), (tokens, hidden_size), (tokens, hc_mult),
        and (tokens, hidden_size).

    """
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        _HC_PRENORM_GEMM_TILELANG_KERNEL,
        _MHC_FUSED_TILELANG_KERNEL,
        _MHC_POST_TILELANG_KERNEL,
        mhc_fused_post_pre_split_config,
        mhc_pre_big_fuse_tilelang,
    )
    from vllm.model_executor.kernels.mhc.warmup import (
        MHC_PRE_NORM_KERNEL,
        compute_mhc_pre_num_splits,
    )
    from vllm.utils.deep_gemm import (
        is_deep_gemm_supported,
        tf32_hc_prenorm_gemm,
    )

    assert residual.ndim == 3 and residual.dtype == torch.bfloat16
    assert residual.is_contiguous()
    num_tokens, hc_mult, hidden_size = residual.shape
    input_size = hc_mult * hidden_size
    mix_size = hc_mult * (hc_mult + 2)
    assert x.shape == (num_tokens, hidden_size) and x.dtype == torch.bfloat16
    assert x.is_contiguous()
    assert post_layer_mix.shape[:2] == (num_tokens, hc_mult)
    assert post_layer_mix.dtype == torch.float32 and post_layer_mix.is_contiguous()
    assert comb_res_mix.shape == (num_tokens, hc_mult, hc_mult)
    assert comb_res_mix.dtype == torch.float32 and comb_res_mix.is_contiguous()
    assert fn.shape == (mix_size, input_size) and fn.dtype == torch.float32
    assert hc_scale.shape == (3,) and hc_scale.dtype == torch.float32
    assert hc_base.shape == (mix_size,) and hc_base.dtype == torch.float32
    if pre_mix is not None:
        assert pre_mix.shape == (num_tokens, hc_mult)
        assert pre_mix.dtype == torch.float32 and pre_mix.is_contiguous()

    next_pre_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    post = torch.empty_like(next_pre_mix)
    comb = torch.empty(
        num_tokens, hc_mult * hc_mult, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )
    aux = torch.empty(
        num_tokens if capture_aux else 0,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )
    if num_tokens == 0:
        return (
            torch.empty_like(residual),
            post.unsqueeze(-1),
            comb.view(num_tokens, hc_mult, hc_mult),
            layer_input,
            next_pre_mix,
            aux,
        )

    fused_config = mhc_fused_post_pre_split_config(num_tokens, hidden_size, hc_mult)
    if fused_config is not None:
        mixes, sqrsum, residual_cur = _MHC_FUSED_TILELANG_KERNEL(
            comb_res_mix,
            residual,
            post_layer_mix.view(num_tokens, hc_mult),
            x,
            fn.view(mix_size, hc_mult, hidden_size),
            hc_mult,
            hidden_size,
            mix_size,
        )
    else:
        residual_cur = torch.empty_like(residual)
        _MHC_POST_TILELANG_KERNEL(
            comb_res_mix,
            residual,
            post_layer_mix.view(num_tokens, hc_mult),
            x,
            residual_cur,
            hc_mult,
            hidden_size,
        )
        # PPU MODIFICATION: begin
        if current_platform.is_ppu():
            from vllm_sail.patch.enhancement.attention.mhc_tilelang import (
                _hc_prenorm_gemm_outputs,
            )

            mixes, sqrsum = _hc_prenorm_gemm_outputs(
                residual_cur.view(num_tokens, input_size),
                fn,
                hidden_size=input_size,
                hc_mult=1,
            )
            n_splits = mixes.shape[0]
        else:
            # PPU MODIFICATION: end
            # The delayed epilogue is compiled per bucketed split count, so the
            # projection has to use the same bucket rather than the raw estimate.
            use_deep_gemm = is_deep_gemm_supported()
            n_splits = (
                compute_mhc_pre_num_splits(input_size, num_tokens)
                if use_deep_gemm
                else 1
            )
            mixes = torch.empty(
                n_splits,
                num_tokens,
                mix_size,
                dtype=torch.float32,
                device=residual.device,
            )
            sqrsum = torch.empty(
                n_splits, num_tokens, dtype=torch.float32, device=residual.device
            )
            residual_cur_2d = residual_cur.view(num_tokens, input_size)
            if use_deep_gemm:
                tf32_hc_prenorm_gemm(residual_cur_2d, fn, mixes, sqrsum, n_splits)
            else:
                _HC_PRENORM_GEMM_TILELANG_KERNEL(
                    residual_cur_2d,
                    fn,
                    mixes,
                    sqrsum,
                    input_size,
                    1,
                )

    outputs = (
        residual_cur,
        post.unsqueeze(-1),
        comb.view(num_tokens, hc_mult, hc_mult),
        layer_input,
        next_pre_mix,
        aux,
    )
    if norm_weight is not None:
        assert norm_weight.shape == (hidden_size,)
        assert norm_weight.dtype == torch.bfloat16 and norm_weight.is_contiguous()
        MHC_PRE_NORM_KERNEL(
            mixes,
            sqrsum,
            hc_scale,
            hc_base,
            residual_cur,
            post,
            comb,
            layer_input,
            norm_weight,
            pre_mix if pre_mix is not None else post,
            next_pre_mix,
            aux if capture_aux else layer_input,
            hidden_size=hidden_size,
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            hc_sinkhorn_eps=hc_sinkhorn_eps,
            hc_post_mult_value=hc_post_mult_value,
            sinkhorn_repeat=sinkhorn_repeat,
            norm_eps=norm_eps,
            hc_mult=hc_mult,
            use_pre_mix_in=pre_mix is not None,
            save_pre_mix=True,
            rms_numel=input_size,
            write_aux=capture_aux,
        )
        return outputs
    mhc_pre_big_fuse_tilelang(
        mixes,
        sqrsum,
        hc_scale,
        hc_base,
        residual_cur,
        post,
        comb,
        layer_input,
        pre_mix if pre_mix is not None else post,
        next_pre_mix,
        aux if capture_aux else layer_input,
        hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        mixes.shape[0],
        hc_mult,
        use_pre_mix_in=pre_mix is not None,
        save_pre_mix=True,
        rms_numel=input_size,
        write_aux=capture_aux,
    )
    return outputs


# V4.1 imports these functions by value, including the Mega-mHC fallback.
_ALIAS_CONSUMERS = {
    "mhc_pre_delayed_tilelang": (
        "vllm.models.deepseek_v41.nvidia.model",
        "vllm.models.deepseek_v41.nvidia.specialized_decode.boundary.boundary",
    ),
    "mhc_fused_post_pre_delayed_tilelang": (
        "vllm.models.deepseek_v41.nvidia.ops.mega_mhc",
        "vllm.models.deepseek_v41.nvidia.ops.mhc",
    ),
}


def _rebind_delayed_aliases():
    for replacement in (mhc_pre_delayed_tilelang, mhc_fused_post_pre_delayed_tilelang):
        name = replacement.__name__
        original = getattr(replacement, PATCH_MARKER)[f"{_MODULE}.{name}"]
        for consumer in _ALIAS_CONSUMERS[name]:
            module = sys.modules.get(consumer)
            if module is not None and getattr(module, name, None) is original:
                patch(
                    consumer,
                    name,
                    reason=_REASON,
                    affected_versions=_AFFECTED,
                    remove_when=_REMOVE_WHEN,
                )(replacement)


_rebind_delayed_aliases()
