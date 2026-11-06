# SPDX-License-Identifier: Apache-2.0
"""Route ``SiluAndMul`` / ``SituAndMul`` through DeepGEMM's fused activation kernels.

The dense MLPs and the MoE shared experts that run on *unquantized* BF16
weights -- Kimi-K3's layer-0 MLP and shared experts (the checkpoint ``ignore``
list keeps ``.*shared_experts.*`` and ``.*mlp.(gate|up|gate_up|down)_proj.*``
out of the MXFP4 quantization), plus every SwiGLU MLP of the other models --
call upstream's standalone ``vllm::silu_and_mul`` / ``vllm::situ_and_mul``
kernel: one launch reading ``[..., 2N]`` and writing ``[..., N]``.

DeepGEMM ships the same math as a standalone kernel
(``deep_gemm.activation_and_mul``) with a *values-only* epilogue
(``QuantMode.BF16``) that emits BF16 values and no scales -- exactly the
epilogue ``PPUDeepGemmExperts._deepgemm_act_quant`` selects for its BF16
(w4a16 or unquantized) experts. Using it here keeps a single fused-activation
implementation in the plugin instead of a second hand-written kernel.

Upstream has no PPU hook on these ops, so ``forward_ppu`` is installed
additively; the ``custom_op_dispatch`` patch routes ops that upstream sent to
``forward_cuda`` and whose class overrides ``forward_ppu`` to this method.
Everything the fused kernels cannot take -- another dtype, an odd width, a
non-contiguous input, a missing/old DeepGEMM build, a non-PPU-1.5 device, or a
kernel failure -- falls back to ``forward_cuda``, i.e. the unchanged upstream
``vllm::silu_and_mul`` / ``vllm::situ_and_mul`` kernel.
"""

from __future__ import annotations

import functools
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_sail import envs as ppu_envs
from vllm_sail.patch.utils import patch
from vllm_sail.utils.deep_gemm import (
    activation_and_mul,
    get_activation_mode,
    get_quant_mode,
    has_fused_act_quant,
    is_ppu_15,
)

_AFFECTED = ">=0.32.0,<0.33.0"
_MODULE = "vllm.model_executor.layers.activation"

logger = init_logger(__name__)

# Ops whose fused kernel raised once. A build/shape that cannot run it then
# costs a single failed call instead of one per activation layer per step, and a
# kernel that breaks for one activation does not disable the other.
_DISABLED: set[str] = set()


@functools.cache
def _bf16_activation_modes() -> tuple[Any, Any, Any] | None:
    """DeepGEMM's ``(SILU, SITU, BF16)`` enums, or ``None`` when unusable.

    The decision depends on the env switch and on the installed DeepGEMM
    build/device only, so it is resolved once per process (both are fixed at
    import time). `VLLM_SAIL_DEEPGEMM_MOE_FUSED_ACT_QUANT` is reused on
    purpose: the shared experts sit inside the MoE block and the routed
    experts' BF16 path already uses this exact kernel and epilogue.
    """
    if not ppu_envs.VLLM_SAIL_DEEPGEMM_MOE_FUSED_ACT_QUANT:
        return None
    if not (has_fused_act_quant() and is_ppu_15()):
        return None
    activation_mode = get_activation_mode()
    return activation_mode.SILU, activation_mode.SITU, get_quant_mode().BF16


def _packed_shape(x: torch.Tensor) -> tuple[int, int] | None:
    """``(rows, packed width)`` for an input the fused kernels accept, else ``None``."""
    if x.dtype is not torch.bfloat16 or not x.is_contiguous():
        return None
    width = x.shape[-1]
    # Every epilogue derives one row from both gate halves, so the packed gate
    # and up projections must be even.
    if width % 2 != 0:
        return None
    return x.numel() // width, width


def _fused_act_bf16(
    name: str,
    x: torch.Tensor,
    activation_mode: Any,
    quant_mode: Any,
    **kwargs: Any,
) -> torch.Tensor | None:
    """Run DeepGEMM's ``activation_and_mul`` with the BF16 (values-only) epilogue.

    Mirrors the BF16 branch of `PPUDeepGemmExperts._deepgemm_act_quant`: the
    kernel derives one row from both gate halves, writes BF16 and reports no
    scales, so no separate activation or quantize pass is needed afterwards.

    Returns ``None`` when the fused kernel must not (or cannot) be used, so the
    caller keeps upstream's ``vllm::<name>`` kernel.
    """
    if name in _DISABLED:
        return None
    packed = _packed_shape(x)
    if packed is None:
        return None

    rows, width = packed
    try:
        fused = activation_and_mul(
            activation_mode, quant_mode, x.reshape(rows, width), **kwargs
        )
    except (AssertionError, RuntimeError) as exc:
        _DISABLED.add(name)
        logger.warning_once(
            f"{name}: DeepGEMM's fused activation kernel is unusable for "
            f"shape {tuple(x.shape)} ({exc}); falling back to the upstream "
            f"vllm::{name} kernel for the rest of the run."
        )
        return None

    if not isinstance(fused, torch.Tensor):
        # The quantizing epilogues return (values, scales); BF16 must not.
        return None
    return fused.reshape(*x.shape[:-1], width // 2)


@patch(
    _MODULE,
    "SiluAndMul.forward_ppu",
    allow_missing=True,
    reason=(
        "Unquantized BF16 SwiGLU (dense MLP and shared experts) runs on upstream's "
        "standalone vllm::silu_and_mul kernel, so it is replaced by DeepGEMM's "
        "activation_and_mul(SILU, BF16) with the values-only epilogue -- the same "
        "kernel and epilogue PPUDeepGemmExperts._deepgemm_act_quant uses for the "
        "BF16 experts. "
        "Upstream has no PPU hook, so forward_ppu is installed additively and reached "
        "through the custom_op_dispatch patch; every unsupported dtype/shape/device "
        "falls back to forward_cuda."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream adds a forward_ppu hook for SiluAndMul, or PPU stops shipping DeepGEMM's standalone activation kernels.",
)
def silu_and_mul_forward_ppu(self, x: torch.Tensor) -> torch.Tensor:
    modes = _bf16_activation_modes()
    if modes is not None:
        fused = _fused_act_bf16(
            "silu_and_mul", x, modes[0], modes[2], swiglu_limit=None
        )
        if fused is not None:
            return fused
    return self.forward_cuda(x)


@patch(
    _MODULE,
    "SituAndMul.forward_ppu",
    allow_missing=True,
    reason=(
        "Kimi's SiTU-GLU runs on unquantized BF16 gate_up/down projections, so "
        "upstream's standalone vllm::situ_and_mul kernel is replaced by DeepGEMM's "
        "activation_and_mul(SITU, BF16) with the values-only epilogue -- the same "
        "kernel and epilogue PPUDeepGemmExperts._deepgemm_act_quant uses for the "
        "BF16 experts. Upstream has no PPU hook, so forward_ppu is installed "
        "additively and reached through the custom_op_dispatch patch; every "
        "unsupported dtype/shape/device falls back to forward_cuda."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream adds a forward_ppu hook for SituAndMul, or PPU stops shipping DeepGEMM's standalone activation kernels.",
)
def situ_and_mul_forward_ppu(self, x: torch.Tensor) -> torch.Tensor:
    linear_beta = self.linear_beta
    if linear_beta is not None and linear_beta <= 0:
        # <= 0 means "unset" to the vLLM kernel; DeepGEMM wants None.
        linear_beta = None
    modes = _bf16_activation_modes()
    if modes is not None:
        fused = _fused_act_bf16(
            "situ_and_mul",
            x,
            modes[1],
            modes[2],
            beta=float(self.beta),
            linear_beta=None if linear_beta is None else float(linear_beta),
        )
        if fused is not None:
            return fused
    return self.forward_cuda(x)
