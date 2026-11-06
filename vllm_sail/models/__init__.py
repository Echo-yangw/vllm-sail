# SPDX-License-Identifier: Apache-2.0
"""PPU model implementations registered into vLLM's ModelRegistry.

The fork selects its PPU model variants inside
``vllm/models/deepseek_v4/__init__.py`` via a platform ladder; a plugin
cannot edit the installed wheel, so it uses the documented extension point
instead: ``vllm.ModelRegistry.register_model``. Registration *overwrites*
existing architecture entries (upstream logs a debug message), which is
exactly what a hardware plugin needs.

Registered architectures (cross-checked against the PPU fork):

* ``DeepseekV4ForCausalLM`` -> ``vllm_sail.models.deepseek_v4.model`` —
  the fork's ``ppu/model.py`` (A-status, ported in Phase 3).
* ``DeepSeekV4MTPModel`` -> ``vllm_sail.models.deepseek_v4.mtp`` — the
  fork's ``ppu/mtp.py``.
* ``DSparkDraftModel`` -> ``vllm_sail.models.deepseek_v4.dspark`` — injects
  the target model's quantization mappings into DSpark's fresh quant config.
* ``GlmMoeDsaForCausalLM`` / ``DeepseekV32MTPModel`` -> legacy upstream models
  retaining PPU INT8 indexer and checkpoint-loader patches.
* ``MiniMaxM3SparseForCausalLM`` / ``MiniMaxM3SparseForConditionalGeneration``
  -> ``vllm_sail.models.minimax_m3`` — upstream classes plus the
  ``SupportsQuant`` interface the fork adds (its only minimax_m3 change).

Deliberately **not** registered:

* ``DeepSeekMTPModel``, ``Qwen3DSparkModel``, ``Step3p5MTP``,
  ``Qwen3MoeForCausalLM``, ``Llama4ForCausalLM`` — upstream classes the
  fork does not replace; its changes there are the runtime patches in
  ``vllm_sail/patch/enhancement/models/``.

All imports are deferred into :func:`register_model`. Required DeepSeek
overrides register transactionally and fail closed; optional model failures
are logged without taking down the plugin hook. Registration is idempotent —
``vllm.general_plugins`` can call the hook more than once per process.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

#: Required PPU overrides. Failure must not silently retain upstream classes.
_REQUIRED_PPU_MODELS: dict[str, str] = {
    "DeepseekV4ForCausalLM": "vllm_sail.models.deepseek_v4.model:DeepseekV4ForCausalLM",
    "DeepSeekV4MTPModel": "vllm_sail.models.deepseek_v4.mtp:DeepSeekV4MTP",
    "DSparkDraftModel": (
        "vllm_sail.models.deepseek_v4.dspark:DSparkDeepseekV4ForCausalLM"
    ),
    "DSparkV41DraftModel": (
        "vllm_sail.models.deepseek_v41.dspark:DSparkDeepseekV41ForCausalLM"
    ),
    "GlmMoeDsaForCausalLM": (
        "vllm.model_executor.models.deepseek_v2:GlmMoeDsaForCausalLM"
    ),
    "DeepseekV32MTPModel": "vllm.model_executor.models.deepseek_mtp:DeepSeekMTP",
}

#: Optional overrides may fail independently without disabling DeepSeek V4.
_OPTIONAL_PPU_MODELS: dict[str, str] = {
    "MiniMaxM3SparseForCausalLM": (
        "vllm_sail.models.minimax_m3:MiniMaxM3SparseForCausalLM"
    ),
    "MiniMaxM3SparseForConditionalGeneration": (
        "vllm_sail.models.minimax_m3:MiniMaxM3SparseForConditionalGeneration"
    ),
}

#: Complete inventory used by tests and documentation tooling.
_PPU_MODELS = {**_REQUIRED_PPU_MODELS, **_OPTIONAL_PPU_MODELS}

_registered = False


def register_model() -> None:
    """Register PPU model architectures, transactionally for required models."""
    global _registered
    if _registered:
        return

    try:
        from vllm import ModelRegistry
    except ImportError as exc:
        raise RuntimeError(
            "vLLM unavailable; cannot register required PPU models"
        ) from exc

    registry_models = getattr(ModelRegistry, "models", None)
    if not isinstance(registry_models, dict):
        raise RuntimeError(
            "vLLM ModelRegistry does not expose the model map required for "
            "transactional PPU registration"
        )

    missing = object()
    previous = {
        arch: registry_models.get(arch, missing) for arch in _REQUIRED_PPU_MODELS
    }
    try:
        for arch, target in _REQUIRED_PPU_MODELS.items():
            ModelRegistry.register_model(arch, target)
            logger.info("Registered required PPU model %s -> %s", arch, target)
    except Exception as exc:
        for arch, old_model in previous.items():
            if old_model is missing:
                registry_models.pop(arch, None)
            else:
                registry_models[arch] = old_model
        raise RuntimeError(
            f"Failed to register required PPU model {arch} ({target}); "
            "restored the previous model registry"
        ) from exc

    for arch, target in _OPTIONAL_PPU_MODELS.items():
        try:
            ModelRegistry.register_model(arch, target)
        except Exception as exc:
            # One missing optional model must not break the whole hook.
            logger.warning(
                "Failed to register PPU model %s (%s): %s", arch, target, exc
            )
        else:
            logger.info("Registered PPU model %s -> %s", arch, target)

    _registered = True
