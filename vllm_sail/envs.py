# SPDX-License-Identifier: Apache-2.0
"""SAIL runtime environment variables — the single source of truth.

Rules (enforced in review):

1. Every SAIL-specific runtime environment variable is declared here. Do **not**
   read ``VLLM_SAIL_*`` or legacy ``VLLM_PPU_*`` runtime names elsewhere.
2. Access is lazy, via module ``__getattr__``, so a variable changed after
   import is still observed. Use :func:`is_set` to distinguish "explicitly set
   by the user" from "left at the default" — several PPU backend-selection paths
   depend on that difference.
3. ``VLLM_SAIL_*`` names take precedence over their ``VLLM_PPU_*`` aliases,
   including explicit false or empty values. Legacy Python attributes use the
   same resolution. Defaults are unchanged from the in-tree PPU fork.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

# ---------------------------------------------------------------------------
# Declarations for static analysis and IDE completion only.
#
# These MUST stay inside `if TYPE_CHECKING`. An annotated assignment at module
# scope creates a real module attribute, and a real attribute shadows the module
# `__getattr__` below — so the lazy getters would never run and every variable
# would read back as its literal default regardless of the environment. (This is
# also why vllm's own envs.py guards its declarations the same way.)
# ---------------------------------------------------------------------------
if TYPE_CHECKING:
    PPU_SDK: str = "/usr/local/PPU_SDK"
    VLLM_SAIL_FUSED_RMSNORM_QUANT: bool = False
    VLLM_SAIL_USE_OPT_TOKEN_GROUP_QUANT: bool = False
    VLLM_SAIL_DENSE_BF16_DEEPGEMM: bool = False
    VLLM_SAIL_USE_PLA: bool = True
    VLLM_SAIL_DEEPGEMM_MOE_TP_FUSED: bool = False
    VLLM_SAIL_MOE_BACKEND: str | None = None
    VLLM_SAIL_DENSE_BACKEND: str | None = None
    VLLM_SAIL_FUSED_GDN_DECODE: bool = True
    VLLM_SAIL_DISABLE_MOE_WNA16_CUDA: bool = False
    VLLM_SAIL_FORCE_MOE_WNA16_CUDA: bool = False
    VLLM_SAIL_ENABLE_MOE_MARLIN: bool = False
    VLLM_SAIL_DISABLE_MEGA_MHC: bool = False
    VLLM_SAIL_USE_TRITON_INT8_QUANT: bool = True
    VLLM_SAIL_NVTX_PROFILE: bool = False
    VLLM_SAIL_NVTX_DUMP_TOPK: bool = False
    VLLM_SAIL_NVTX_VFA_DUMP_SEQLEN: bool = False
    VLLM_DEEPEPLL_RECV_HOOK: bool = True
    VLLM_GDN_DECODE_KERNEL: str = "auto"

    # Backward-compatible Python attributes; all reads resolve SAIL first.
    VLLM_PPU_FUSED_RMSNORM_QUANT: bool = False
    VLLM_PPU_USE_OPT_TOKEN_GROUP_QUANT: bool = False
    VLLM_PPU_DENSE_BF16_DEEPGEMM: bool = False
    VLLM_PPU_USE_PLA: bool = True
    VLLM_PPU_DEEPGEMM_MOE_TP_FUSED: bool = False
    VLLM_PPU_MOE_BACKEND: str | None = None
    VLLM_PPU_DENSE_BACKEND: str | None = None
    VLLM_PPU_FUSED_GDN_DECODE: bool = True
    VLLM_PPU_DISABLE_MOE_WNA16_CUDA: bool = False
    VLLM_PPU_FORCE_MOE_WNA16_CUDA: bool = False
    VLLM_PPU_ENABLE_MOE_MARLIN: bool = False
    VLLM_PPU_USE_TRITON_INT8_QUANT: bool = True
    VLLM_PPU_NVTX_PROFILE: bool = False
    VLLM_PPU_NVTX_DUMP_TOPK: bool = False
    VLLM_PPU_NVTX_VFA_DUMP_SEQLEN: bool = False

_TRUTHY = ("true", "1", "yes", "on")

#: Backend names accepted by both MoE and dense GEMM selection.
GEMM_BACKENDS = ("deepgemm", "acext", "triton")
GDN_DECODE_KERNELS = ("auto", "cuda", "pla", "triton")


def _env_names(name: str) -> tuple[str, ...]:
    """Canonical name followed by its aliases, in precedence order."""
    name = _canonical_names.get(name, name)
    return (name, *_aliases.get(name, ()))


def _read_env(name: str) -> tuple[str, str | None]:
    for candidate in _env_names(name):
        raw = os.getenv(candidate)
        if raw is not None:
            return candidate, raw
    return name, None


def _bool(name: str, default: bool) -> Callable[[], bool]:
    """Read a boolean env var, falling back to aliases only when unset."""

    def _get() -> bool:
        _, raw = _read_env(name)
        if raw is None:
            return default
        return raw.strip().lower() in _TRUTHY

    return _get


def _choice(
    name: str, default: str | None, choices: tuple[str, ...]
) -> Callable[[], str | None]:
    """Read a string env var constrained to ``choices`` (case-insensitive)."""

    def _get() -> str | None:
        source, raw = _read_env(name)
        if raw is None or raw.strip() == "":
            return default
        value = raw.strip().lower()
        if value not in choices:
            raise ValueError(
                f"Invalid value {raw!r} for {source}. Expected one of "
                f"{', '.join(choices)}."
            )
        return value

    return _get


environment_variables: dict[str, Callable[[], Any]] = {
    "PPU_SDK": lambda: os.environ.get("PPU_SDK", "/usr/local/PPU_SDK"),
    # Shared upstream name; PPU extends the choices through its general plugin.
    "VLLM_GDN_DECODE_KERNEL": _choice(
        "VLLM_GDN_DECODE_KERNEL", "auto", GDN_DECODE_KERNELS
    ),
    "VLLM_SAIL_FUSED_RMSNORM_QUANT": _bool("VLLM_SAIL_FUSED_RMSNORM_QUANT", False),
    "VLLM_SAIL_USE_OPT_TOKEN_GROUP_QUANT": _bool(
        "VLLM_SAIL_USE_OPT_TOKEN_GROUP_QUANT", False
    ),
    "VLLM_SAIL_DENSE_BF16_DEEPGEMM": _bool("VLLM_SAIL_DENSE_BF16_DEEPGEMM", False),
    "VLLM_SAIL_USE_PLA": _bool("VLLM_SAIL_USE_PLA", True),
    "VLLM_SAIL_DEEPGEMM_MOE_TP_FUSED": _bool("VLLM_SAIL_DEEPGEMM_MOE_TP_FUSED", False),
    # -- MoE ---------------------------------------------------------------
    # PPU MoE group-GEMM backend. Unset means "pick the best available",
    # which is why `is_set()` is meaningful here.
    "VLLM_SAIL_MOE_BACKEND": _choice("VLLM_SAIL_MOE_BACKEND", None, GEMM_BACKENDS),
    # PPU dense-GEMM backend for quantized linear layers.
    "VLLM_SAIL_DENSE_BACKEND": _choice("VLLM_SAIL_DENSE_BACKEND", None, GEMM_BACKENDS),
    # Disable / force the wna16 CUDA MoE kernel on PPU. Both exist because the
    # heuristic that picks between the CUDA and Triton kernels is shape
    # dependent and occasionally needs a manual override in either direction.
    "VLLM_SAIL_DISABLE_MOE_WNA16_CUDA": _bool(
        "VLLM_SAIL_DISABLE_MOE_WNA16_CUDA", False
    ),
    "VLLM_SAIL_FORCE_MOE_WNA16_CUDA": _bool("VLLM_SAIL_FORCE_MOE_WNA16_CUDA", False),
    # Legacy Marlin selection gate. Native Marlin execution remains excluded
    # by the CUDA-free manifest; this flag cannot supply those kernels.
    "VLLM_SAIL_ENABLE_MOE_MARLIN": _bool("VLLM_SAIL_ENABLE_MOE_MARLIN", False),
    # -- DSV4.1 mHC ----------------------------------------------------------
    # Disable the PPU Mega mHC kernel so the upstream TileLang mHC path takes
    # over. The A/B baseline for bisecting mHC regressions: the two paths
    # compute the same shifted post/pre block, so a difference in the model's
    # output points at the Mega mHC kernel.
    "VLLM_SAIL_DISABLE_MEGA_MHC": _bool("VLLM_SAIL_DISABLE_MEGA_MHC", False),
    # -- Quantization -------------------------------------------------------
    # Triton implementation of dynamic per-token int8 quant; faster than the
    # compiled kernel on PPU.
    "VLLM_SAIL_USE_TRITON_INT8_QUANT": _bool("VLLM_SAIL_USE_TRITON_INT8_QUANT", True),
    # -- Attention / linear attention --------------------------------------
    # Legacy getter only. Current PLA/Triton dispatch uses VLLM_SAIL_USE_PLA.
    "VLLM_SAIL_FUSED_GDN_DECODE": _bool("VLLM_SAIL_FUSED_GDN_DECODE", True),
    # -- Distributed MoE ---------------------------------------------------
    # DeepEP low-latency deferred recv hook. The hook only pays off on backends
    # that can overlap the deferred receive with other compute; setting this to
    # 0 selects an event-based fallback that is also a simpler debugging
    # baseline for overlap issues.
    "VLLM_DEEPEPLL_RECV_HOOK": _bool("VLLM_DEEPEPLL_RECV_HOOK", True),
    # -- Profiling (opt-in; see vllm_sail/profiling/) ------------------------
    "VLLM_SAIL_NVTX_PROFILE": _bool(
        "VLLM_SAIL_NVTX_PROFILE",
        False,
    ),
    "VLLM_SAIL_NVTX_DUMP_TOPK": _bool("VLLM_SAIL_NVTX_DUMP_TOPK", False),
    "VLLM_SAIL_NVTX_VFA_DUMP_SEQLEN": _bool(
        "VLLM_SAIL_NVTX_VFA_DUMP_SEQLEN",
        False,
    ),
}

# All plugin variables retain the corresponding PPU spelling. The shared
# upstream VLLM_DEEPEPLL_RECV_HOOK and VLLM_GDN_DECODE_KERNEL keep their names
# and have no plugin aliases.
_aliases: dict[str, tuple[str, ...]] = {
    name: (name.replace("VLLM_SAIL_", "VLLM_PPU_", 1),)
    for name in environment_variables
    if name.startswith("VLLM_SAIL_")
}
_aliases["VLLM_SAIL_NVTX_PROFILE"] += ("SAIL_NVTX_PROFILE",)
_aliases["VLLM_SAIL_DISABLE_MEGA_MHC"] += ("SAIL_DISABLE_MEGA_MHC",)
_canonical_names = {
    alias: name for name, aliases in _aliases.items() for alias in aliases
}


def __getattr__(name: str) -> Any:
    name = _canonical_names.get(name, name)
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return [*environment_variables, *_canonical_names]


def is_set(name: str) -> bool:
    """Whether ``name`` or any of its aliases was explicitly set.

    Backend selection distinguishes "user asked for X" from "user did not
    choose", so this is not the same as comparing against the default.
    """
    return any(candidate in os.environ for candidate in _env_names(name))


def snapshot() -> dict[str, Any]:
    """Effective runtime values under canonical names, for ``collect_env``."""
    return {name: getter() for name, getter in environment_variables.items()}
