# SPDX-License-Identifier: Apache-2.0
"""Route the DSV4.1 Mega mHC kernel to the PPU DeepGEMM build.

Upstream gates the kernel on the SM 10.x (Blackwell) family and on
``vllm.utils.deep_gemm.is_deep_gemm_supported()``, which is False on PPU
because the platform does not advertise CUDA DeepGEMM support. The PPU
DeepGEMM build does ship a Mega mHC kernel, but only for ZW-M890P (SM 8.9)
and only for the 5120-wide, ``hc_mult=4`` configuration.

This module therefore replaces the two upstream symbols the V4.1 mHC module
needs, and delegates to the upstream implementations on every other platform.

``VLLM_SAIL_DISABLE_MEGA_MHC=1`` turns the PPU gate off so the upstream
TileLang mHC path takes over; that is the A/B baseline for bisecting mHC
regressions.
"""

import functools
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform

import vllm_sail.envs as ppu_envs
from vllm_sail.patch.utils import PATCH_MARKER, patch
from vllm_sail.utils import deep_gemm as ppu_deep_gemm

logger = init_logger("vllm.vllm_sail.deepseek_v41_mega_mhc")
# The logger name is deliberately under vLLM's ``vllm.*`` namespace: vLLM only
# configures that namespace, so a ``vllm_sail.*`` logger would inherit the root
# logger, which has no handler in the worker processes and therefore drops
# every INFO record. The Mega-mHC path reports which kernel it selected, which
# has to be observable in the server log.

_MODULE = "vllm.models.deepseek_v41.nvidia.ops.mega_mhc"
_MHC_MODULE = "vllm.models.deepseek_v41.nvidia.ops.mhc"
_AFFECTED = ">=0.31.0,<0.32.0"

# The PPU Mega mHC kernel is generated for exactly one shape.
_PPU_HIDDEN_SIZE = 5120
_PPU_HC_MULT = 4
_PPU_CAPABILITY = (8, 9)
_PPU_DEVICE_NAME_MARKER = "M890"


@functools.cache
def _ppu_mega_mhc_capability(hidden_size: int, hc_mult: int) -> bool:
    """Return whether the PPU Mega mHC kernel can run this configuration."""
    if hidden_size != _PPU_HIDDEN_SIZE or hc_mult != _PPU_HC_MULT:
        return False
    deep_gemm_supported = ppu_deep_gemm.is_deep_gemm_supported()
    has_mega = ppu_deep_gemm.has_mega_mhc()
    capability = current_platform.get_device_capability()
    device_name = current_platform.get_device_name()
    supported = (
        deep_gemm_supported
        and has_mega
        and capability is not None
        and (capability.major, capability.minor) == _PPU_CAPABILITY
        and _PPU_DEVICE_NAME_MARKER in device_name
    )
    logger.info_once(
        "DSV4.1 mHC: PPU Mega mHC gate (hidden_size=%d, hc_mult=%d) -> %s "
        "[deep_gemm=%s, has_mega_mhc=%s, capability=%s, device=%s]",
        hidden_size,
        hc_mult,
        supported,
        deep_gemm_supported,
        has_mega,
        capability,
        device_name,
        scope="global",
    )
    return supported


def _ppu_mega_mhc_supported(hidden_size: int, hc_mult: int) -> bool:
    """Return whether the PPU Mega mHC kernel can run this configuration.

    The kill switch is evaluated outside the cached capability probe so that
    flipping it takes effect on a running server, which is what the A/B
    comparison against the TileLang fallback needs.
    """
    if ppu_envs.VLLM_SAIL_DISABLE_MEGA_MHC:
        logger.info_once(
            "DSV4.1 mHC: PPU Mega mHC disabled by VLLM_SAIL_DISABLE_MEGA_MHC; "
            "the TileLang mHC path is used instead.",
            scope="global",
        )
        return False
    return _ppu_mega_mhc_capability(hidden_size, hc_mult)


@patch(
    _MODULE,
    "is_mega_mhc_supported",
    reason=(
        "PPU runs the DSV4.1 Mega mHC kernel from the PPU DeepGEMM build, which "
        "targets ZW-M890P (SM 8.9) instead of the upstream SM 10.x family and "
        "only supports the 5120-wide, hc_mult=4 configuration."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "The V4.1 mHC gate queries a platform-specific DeepGEMM capability "
        "contract instead of a hard-coded SM family."
    ),
)
def is_mega_mhc_supported(hidden_size: int, hc_mult: int) -> bool:
    if current_platform.is_ppu():
        return _ppu_mega_mhc_supported(hidden_size, hc_mult)
    original = getattr(is_mega_mhc_supported, PATCH_MARKER)[
        f"{_MODULE}.is_mega_mhc_supported"
    ]
    return original(hidden_size, hc_mult)


@patch(
    _MODULE,
    "mega_mhc",
    reason=(
        "The V4.1 mHC module imports the DeepGEMM entry point by value, so PPU "
        "must rebind it to the PPU-aware wrapper that resolves the kernel in "
        "the PPU DeepGEMM build."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "The V4.1 mHC module resolves the Mega mHC kernel through a "
        "platform-aware provider instead of a by-value import."
    ),
)
def mega_mhc(*args: Any, **kwargs: Any) -> None:
    if current_platform.is_ppu():
        logger.info_once(
            "DSV4.1 mHC: running the PPU Mega mHC kernel.",
            scope="global",
        )
        ppu_deep_gemm.mega_mhc(*args, **kwargs)
        return
    original = getattr(mega_mhc, PATCH_MARKER)[f"{_MODULE}.mega_mhc"]
    original(*args, **kwargs)


@patch(
    _MHC_MODULE,
    "can_use_mega_mhc",
    reason=(
        "The V4.1 mHC dispatcher picks between the Mega mHC kernel and the "
        "TileLang fallback through a chain of short-circuiting conditions. On "
        "PPU the chain must consult the PPU Mega mHC gate, and the dispatcher "
        "has to report which condition rejected the kernel, otherwise a "
        "silently-disabled Mega mHC path is indistinguishable from a kernel "
        "that was never reached."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "The V4.1 mHC dispatcher queries one platform-aware capability "
        "contract instead of evaluating the conditions inline."
    ),
)
def can_use_mega_mhc(
    x: torch.Tensor,
    residual: torch.Tensor,
    pre_mix: torch.Tensor | None,
    norm_weight: torch.Tensor | None,
    capture_aux: bool,
) -> bool:
    if current_platform.is_ppu():
        # Resolved lazily: the dispatcher module holds this symbol by value,
        # so the lookup has to happen after the gate patch is installed.
        from vllm.models.deepseek_v41.nvidia.ops.mega_mhc import (
            is_mega_mhc_supported,
        )

        usable = (
            pre_mix is not None
            and norm_weight is not None
            and not capture_aux
            and x.shape[0] <= 1 << 20
            and is_mega_mhc_supported(x.shape[1], residual.shape[1])
        )
        logger.info_once(
            "DSV4.1 mHC: Mega mHC usable=%s [pre_mix=%s, norm_weight=%s, "
            "capture_aux=%s, tokens=%d, hidden=%d, hc_mult=%d]",
            usable,
            pre_mix is not None,
            norm_weight is not None,
            capture_aux,
            x.shape[0],
            x.shape[1],
            residual.shape[1],
            scope="global",
        )
        return usable
    original = getattr(can_use_mega_mhc, PATCH_MARKER)[
        f"{_MHC_MODULE}.can_use_mega_mhc"
    ]
    return original(x, residual, pre_mix, norm_weight, capture_aux)
