# SPDX-License-Identifier: Apache-2.0
"""Register PPU backends with the MXFP4 MoE oracle."""

from __future__ import annotations

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config.kernel import MoEBackend
from vllm.model_executor.layers.fused_moe.oracle import mxfp4 as oracle
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTokenSym,
)

from vllm_sail.patch.utils import patch
from vllm_sail.registry.moe_backends._extend import extend_enum

_MODULE = "vllm.model_executor.layers.fused_moe.oracle.mxfp4"
_AFFECTED = ">=0.32.0,<0.33.0"
_REMOVE_WHEN = (
    "upstream gains a register_moe_backend() extension point mirroring "
    "register_linear_kernel(), which would delete this patch."
)

Mxfp4MoeBackend = oracle.Mxfp4MoeBackend
# PPU members of the upstream enum. Names follow the oracle's
# "<family>_MXFP4_<activation>" convention (compare AITER_MXFP4_FP8); the order
# below is the auto-candidate order, so the W4A4 pair stays first.
_PPU_BACKEND_NAMES = (
    "PPU_DEEPGEMM_MXFP4",
    "BATCHED_PPU_DEEPGEMM_MXFP4",
    "PPU_DEEPGEMM_MXFP4_FP8",
    "BATCHED_PPU_DEEPGEMM_MXFP4_FP8",
    "PPU_DEEPGEMM_MXFP4_BF16_MMA",
    "BATCHED_PPU_DEEPGEMM_MXFP4_BF16_MMA",
    "PPU_DEEPGEMM_MXFP4_BF16",
    "BATCHED_PPU_DEEPGEMM_MXFP4_BF16",
)
for _name in _PPU_BACKEND_NAMES:
    extend_enum(Mxfp4MoeBackend, _name, _name)

# W4A8 (MXFP4 weights + per-token FP8 activations) repacks the weights, so it
# stays opt-in through moe_backend="ppu_deep_gemm_w4a8" instead of an automatic
# candidate.
_PPU_W4A8_NAMES = frozenset(
    {"PPU_DEEPGEMM_MXFP4_FP8", "BATCHED_PPU_DEEPGEMM_MXFP4_FP8"}
)
_PPU_CANDIDATES = tuple(
    getattr(Mxfp4MoeBackend, name)
    for name in _PPU_BACKEND_NAMES
    if name not in _PPU_W4A8_NAMES
)

# W4A16 families: "_BF16_MMA" runs DeepGEMM's own w4a16 MMA kernel, "_BF16"
# reuses MARLIN's 16-bit packing; both resolve to PPUDeepGemmExperts[W4FA16MMA].
_PPU_W4A16_BACKENDS = tuple(
    getattr(Mxfp4MoeBackend, name)
    for name in (
        "PPU_DEEPGEMM_MXFP4_BF16",
        "BATCHED_PPU_DEEPGEMM_MXFP4_BF16",
        "PPU_DEEPGEMM_MXFP4_BF16_MMA",
        "BATCHED_PPU_DEEPGEMM_MXFP4_BF16_MMA",
    )
)

_upstream_backend_to_kernel_cls = oracle.backend_to_kernel_cls
_upstream_map_mxfp4_backend = oracle.map_mxfp4_backend
_upstream_get_priority_backends_for_gpt_oss = oracle._get_priority_backends_for_gpt_oss
_upstream_get_priority_backends = oracle._get_priority_backends
_upstream_backend_activation_key = oracle._backend_activation_key
_upstream_select_mxfp4_moe_backend = oracle.select_mxfp4_moe_backend


@patch(
    _MODULE,
    "backend_to_kernel_cls",
    reason="The MXFP4 oracle cannot resolve plugin-owned PPU expert classes.",
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def backend_to_kernel_cls(
    backend: Mxfp4MoeBackend,
) -> list[type[mk.FusedMoEExperts]]:
    if backend == Mxfp4MoeBackend.PPU_DEEPGEMM_MXFP4:
        from vllm_sail.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
            PPUDeepGemmExpertsMXFP4,
        )

        return [PPUDeepGemmExpertsMXFP4]
    if backend == Mxfp4MoeBackend.BATCHED_PPU_DEEPGEMM_MXFP4:
        from vllm_sail.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe import (
            PPUBatchedDeepGemmExpertsMXFP4,
        )

        return [PPUBatchedDeepGemmExpertsMXFP4]
    if backend == Mxfp4MoeBackend.PPU_DEEPGEMM_MXFP4_FP8:
        from vllm_sail.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
            PPUDeepGemmExpertsMXFP4W4A8,
        )

        return [PPUDeepGemmExpertsMXFP4W4A8]
    if backend == Mxfp4MoeBackend.BATCHED_PPU_DEEPGEMM_MXFP4_FP8:
        from vllm_sail.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe import (
            PPUBatchedDeepGemmExpertsMXFP4W4A8,
        )

        return [PPUBatchedDeepGemmExpertsMXFP4W4A8]
    if backend in _PPU_W4A16_BACKENDS:
        if backend.name.startswith("BATCHED_"):
            from vllm_sail.model_executor.layers.fused_moe.experts import (
                batched_deep_gemm_moe as experts,
            )

            name = "PPUBatchedDeepGemmExperts"
        else:
            from vllm_sail.model_executor.layers.fused_moe.experts import (
                deep_gemm_moe as experts,
            )

            name = "PPUDeepGemmExperts"
        if backend.name.endswith("_MMA"):
            name += "W4FA16MMA"
        return [getattr(experts, name)]
    return _upstream_backend_to_kernel_cls(backend)


@patch(
    _MODULE,
    "map_mxfp4_backend",
    reason=(
        "The public ppu_deep_gemm aliases have no upstream MXFP4 member "
        "mapping. Each standard PPU candidate is followed by its batched "
        "family so the selectors keep upstream's body: the shared "
        "is_supported_config() activation-format check rejects the standard "
        "candidate for a batched MoE and the batched one for a standard MoE."
    ),
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def map_mxfp4_backend(runner_backend: MoEBackend) -> list[Mxfp4MoeBackend]:
    if runner_backend == "ppu_deep_gemm":
        return [
            Mxfp4MoeBackend.PPU_DEEPGEMM_MXFP4,
            Mxfp4MoeBackend.BATCHED_PPU_DEEPGEMM_MXFP4,
        ]
    if runner_backend == "ppu_deep_gemm_w4a16":
        return [
            Mxfp4MoeBackend.PPU_DEEPGEMM_MXFP4_BF16_MMA,
            Mxfp4MoeBackend.BATCHED_PPU_DEEPGEMM_MXFP4_BF16_MMA,
            Mxfp4MoeBackend.PPU_DEEPGEMM_MXFP4_BF16,
            Mxfp4MoeBackend.BATCHED_PPU_DEEPGEMM_MXFP4_BF16,
        ]
    if runner_backend == "ppu_deep_gemm_w4a8":
        return [
            Mxfp4MoeBackend.PPU_DEEPGEMM_MXFP4_FP8,
            Mxfp4MoeBackend.BATCHED_PPU_DEEPGEMM_MXFP4_FP8,
        ]
    return _upstream_map_mxfp4_backend(runner_backend)


@patch(
    _MODULE,
    "_get_priority_backends_for_gpt_oss",
    reason="The GPT-OSS MXFP4 candidate list has no PPU DeepGEMM backends.",
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def _get_priority_backends_for_gpt_oss() -> list[Mxfp4MoeBackend]:
    backends = _upstream_get_priority_backends_for_gpt_oss()
    from vllm.platforms import current_platform

    if current_platform.is_ppu() and Mxfp4MoeBackend.PPU_DEEPGEMM_MXFP4 not in backends:
        backends[0:0] = list(_PPU_CANDIDATES)
    return backends


@patch(
    _MODULE,
    "_get_priority_backends",
    reason="The generic MXFP4 candidate list has no PPU DeepGEMM backends.",
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def _get_priority_backends() -> list[Mxfp4MoeBackend]:
    backends = _upstream_get_priority_backends()
    from vllm.platforms import current_platform

    if current_platform.is_ppu() and Mxfp4MoeBackend.PPU_DEEPGEMM_MXFP4 not in backends:
        insertion = backends.index(Mxfp4MoeBackend.DEEPGEMM_MXFP4) + 1
        backends[insertion:insertion] = list(_PPU_CANDIDATES)
    return backends


@patch(
    _MODULE,
    "_backend_activation_key",
    reason="PPU MXFP4 W4A4 backends need the kMxfp4Dynamic activation key.",
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def _backend_activation_key(backend: Mxfp4MoeBackend) -> QuantKey | None:
    if backend in (
        Mxfp4MoeBackend.PPU_DEEPGEMM_MXFP4,
        Mxfp4MoeBackend.BATCHED_PPU_DEEPGEMM_MXFP4,
    ):
        return oracle.kMxfp4Dynamic
    if backend in (
        Mxfp4MoeBackend.PPU_DEEPGEMM_MXFP4_FP8,
        Mxfp4MoeBackend.BATCHED_PPU_DEEPGEMM_MXFP4_FP8,
    ):
        # W4A8 uses per-token dynamic FP8 activations.
        return kFp8DynamicTokenSym
    return _upstream_backend_activation_key(backend)


# The selector itself stays upstream's. Every PPU difference lives in the
# helpers it calls: ``map_mxfp4_backend`` supplies the standard + batched PPU
# candidates, the ``_get_priority_backends*`` patches inject them into the auto
# lists, ``backend_to_kernel_cls`` resolves the PPU expert classes and
# ``_backend_activation_key`` their activation keys. Re-exported for callers
# that used the plugin-local copy.
select_mxfp4_moe_backend = _upstream_select_mxfp4_moe_backend


_upstream_select_deepseek_v4 = oracle.select_deepseek_v4_mxfp4_moe_backend


@patch(
    _MODULE,
    "select_deepseek_v4_mxfp4_moe_backend",
    reason="Preserve PPU W4A4/W4A16 selection through the 0.31 MXFP4 kernel factory.",
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def select_deepseek_v4_mxfp4_moe_backend(config):
    from vllm.platforms import current_platform

    if not current_platform.is_ppu():
        return _upstream_select_deepseek_v4(config)
    activation_key = None
    if not current_platform.is_device_capability((8, 0)):
        if config.moe_backend == "ppu_deep_gemm_w4a8":
            # W4A8: per-token dynamic FP8 activations.
            activation_key = kFp8DynamicTokenSym
        elif config.moe_backend not in (
            "marlin",
            "ppu_deep_gemm_w4a16",
        ):
            activation_key = oracle.kMxfp4Dynamic
    return oracle.select_mxfp4_moe_backend(config, activation_key=activation_key)
