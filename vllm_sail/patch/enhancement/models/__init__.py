# SPDX-License-Identifier: Apache-2.0
"""Model-layer PPU patches (Phase 5).

Ports the fork's seven modified ``vllm/model_executor/models/`` files
(+155/-44) plus the Marlin MoE gate from decision D2 (phase-1 spec 1.8(c)).
Importing this package applies every patch in it, exactly like the
``bugfix`` / ``enhancement`` / ``performance`` category packages do.

Wired into ``vllm_sail/patch/enhancement/__init__.py`` (import applies every
patch); ``tests/ut/test_model_patches.py::test_subpackage_wiring_is_present``
guards the connection.

Modules, and why each is import-on-install like the rest of the framework:

* ``deepseek_v2`` / ``deepseek_mtp`` — fp4-vs-non-fp4 indexer projection +
  weight-loading guards (the fork's +108/+19).
* ``deepseek_v4`` — selects the plugin FlashMLA subclass only on PPU; the
  quantization-specific weights mapper remains plugin-local so CUDA behavior
  is unchanged.
* ``kimi_k3`` — quantized routed up-projection support for the latent-MoE
  tail's fused tiers.
* ``llama4`` — torch.compile on the custom router (+1).
* ``minimax_m3`` — retain ordinary MoE finalize on PPU before NVIDIA imports.
* ``qwen3_moe`` — per-layer mix_layer quant overrides (+36).
* ``qwen3_next`` — NVTX profiling scaffold (+19).
* ``step3p5_mtp`` — zero-point buffers as optional checkpoint params (+12).
* ``moe_marlin_gate`` — VLLM_PPU_ENABLE_MOE_MARLIN default-off gate.
"""

from vllm_sail.patch.enhancement.models import (
    deepseek_mtp,  # noqa: F401
    deepseek_v2,  # noqa: F401
    deepseek_v4,  # noqa: F401
    deepseek_v4_cache,  # noqa: F401
    deepseek_v4_compressor,  # noqa: F401
    deepseek_v4_metadata,  # noqa: F401
    glm5_next,  # noqa: F401
    kimi_k3,  # noqa: F401
    llama4,  # noqa: F401
    minimax_m3,  # noqa: F401
    moe_marlin_gate,  # noqa: F401
    qwen3_moe,  # noqa: F401
    qwen3_next,  # noqa: F401
    step3p5_mtp,  # noqa: F401
)
