# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preserve MuseSpark's Triton routing path when the CUDA op is only a stub."""

from __future__ import annotations

import functools
import sys

from vllm.platforms import current_platform

from vllm_sail.patch.bodies import bind_body
from vllm_sail.patch.utils import patch

_MODULE = 'vllm.models.muse_spark.common.ops.fused_routing'
_META = dict(
    reason='PPU CUDA identity and compute-stub presence must not replace the existing Triton MuseSpark routing path with an unimplemented native operator.',
    affected_versions='>=0.32.0,<0.33.0',
    remove_when='MuseSpark selects the native router through a platform implementation capability or PPU implements and verifies this native operator.',
)

# Import the provider first so the original callable can be captured. Importing it
# also executes ``vllm.models.muse_spark``, which reaches
# ``vllm.compilation.passes.fusion.matcher_utils`` and reads ``torch.ops._C`` at
# module scope. A ``VLLM_TARGET_DEVICE=empty`` vLLM claims that namespace only once
# the PPU kernels are registered, so the import raises ``AttributeError`` before a
# single patch lands. Skipping is correct there: without the compiled extension the
# native router this patch guards against cannot dispatch either.
try:
    from vllm.models.muse_spark.common.ops import fused_routing as _provider

    _original = _provider.fused_musespark1x_routing
except (ImportError, AttributeError) as exc:  # no compiled kernels / older vLLM
    _provider = None
    _unavailable = True
    from vllm.logger import init_logger

    init_logger(__name__).warning(
        "Skipping the MuseSpark routing patch: the upstream module cannot be "
        "imported (%s). It needs a vLLM built with its compiled kernels.",
        exc,
    )
else:
    _unavailable = False


def _routing_body(
    gating_output: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    route_scale: float,
    router_output_multiplier: float | None = None,
    router_score_normalization: str = "sigmoid",
    router_score_division_eps: float | None = None,
    topk_weights: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    *,
    is_padding: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run MuseSpark1x's router in one kernel.

    Passing both destinations keeps the call allocation-free (graph-capturable).
    """
    assert gating_output.dim() == 2, "gating_output must be 2D"
    assert bias.is_contiguous(), "bias must be contiguous"

    num_tokens, num_experts = gating_output.shape
    if is_padding is not None:
        assert is_padding.shape == (num_tokens,) and is_padding.dtype == torch.bool
    assert bias.shape[0] == num_experts, "bias size must match num_experts"

    if router_score_normalization == "sigmoid":
        normalization_mode = 0
    elif router_score_normalization == "softmax":
        normalization_mode = 1
    else:
        raise ValueError(
            f"Invalid router_score_normalization: {router_score_normalization}. "
            "Expected 'sigmoid' or 'softmax'."
        )

    if topk_weights is None:
        topk_weights = torch.empty(
            (num_tokens, topk), dtype=torch.float32, device=gating_output.device
        )
    if topk_ids is None:
        topk_ids = torch.empty(
            (num_tokens, topk), dtype=torch.int32, device=gating_output.device
        )
    assert topk_weights.shape[0] >= num_tokens and topk_weights.shape[1] >= topk, (
        "topk_weights buffer too small"
    )
    assert topk_ids.shape[0] >= num_tokens and topk_ids.shape[1] >= topk, (
        "topk_ids buffer too small"
    )
    weights_view = topk_weights[:num_tokens, :topk]
    ids_view = topk_ids[:num_tokens, :topk]


    enable_pdl = False  # PPU uses the existing ordinary Triton launch.
    _routing_kernel[(num_tokens,)](
        gating_output,
        bias if bias.dtype == torch.float32 else bias.float(),
        weights_view,
        ids_view,
        is_padding,
        is_padding.stride(0) if is_padding is not None else 0,
        num_tokens,
        num_experts,
        gating_output.stride(0),
        gating_output.stride(1),
        weights_view.stride(0),
        weights_view.stride(1),
        ids_view.stride(0),
        ids_view.stride(1),
        router_output_multiplier if router_output_multiplier is not None else 1.0,
        route_scale,
        router_score_division_eps if router_score_division_eps is not None else 0.0,
        TOPK=topk,
        TOPK_PADDED=triton.next_power_of_2(topk),
        NUM_EXPERTS_PADDED=_padded_num_experts(num_experts),
        NORMALIZATION_MODE=normalization_mode,
        HAS_MULTIPLIER=router_output_multiplier is not None,
        HAS_EPS=router_score_division_eps is not None,
        USE_PDL=enable_pdl,
        # Keep Muse's repeated 256-expert selections within one warp.
        num_warps=1 if num_experts == 256 or current_platform.is_rocm() else 4,
        **({"launch_pdl": True} if enable_pdl else {}),
    )
    return weights_view, ids_view


# Registration is explicit so an environment without the compiled kernels imports
# this module, warns once and skips it instead of aborting the whole install.
if not _unavailable:
    # The upstream function chooses its native path mid-body. Keep its allocation,
    # validation and Triton call unchanged; only PPU dispatch reaches this body.
    _ppu_routing = bind_body(_routing_body, _provider)

    @patch(_MODULE, 'fused_musespark1x_routing', **_META)
    @functools.wraps(_original)
    def fused_musespark1x_routing(*args, **kwargs):
        if current_platform.is_ppu():
            return _ppu_routing(*args, **kwargs)
        return _original(*args, **kwargs)

    # A consumer imported before this module keeps the unpatched router in its
    # namespace, so rebind the aliases that still point at the original.
    for _name in (
        'vllm.models.muse_spark.common.moe',
        'vllm.models.muse_spark.nvidia.model',
    ):
        _consumer = sys.modules.get(_name)
        if (
            _consumer is not None
            and getattr(_consumer, 'fused_musespark1x_routing', None) is _original
        ):
            patch(_name, 'fused_musespark1x_routing', **_META)(
                fused_musespark1x_routing
            )
