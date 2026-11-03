# SPDX-License-Identifier: Apache-2.0
"""Register PPU quantization configs with vLLM.

Upstream exposes ``register_quantization_config(name)``
(``vllm/model_executor/layers/quantization/__init__.py:60``), which adds the name
to ``QUANTIZATION_METHODS`` and makes ``get_quantization_config(name)`` resolve
it. So the in-tree fork's edit to that module needs **no patch** -- the
``mixed_precision_w4`` config registers itself here.
"""

from __future__ import annotations

from vllm.logger import init_logger

logger = init_logger(__name__)

_registered = False


def register() -> None:
    """Register PPU quantization configs. Idempotent."""
    global _registered
    if _registered:
        return

    from vllm.model_executor.layers.quantization import (
        QUANTIZATION_METHODS,
        register_quantization_config,
    )

    from vllm_sail.model_executor.layers.quantization.mixed_precision_w4 import (
        MixedPrecisionW4Config,
    )

    name = MixedPrecisionW4Config.get_name()

    if name not in QUANTIZATION_METHODS:
        register_quantization_config(name)(MixedPrecisionW4Config)
    else:
        logger.warning(
            "Quantization method %r already exists; keeping its owner.", name
        )

    from vllm.platforms import current_platform

    if current_platform.is_ppu():
        from vllm_sail.model_executor.layers.quantization.bf16_to_int8 import (
            Bf16ToInt8Config,
        )
        from vllm_sail.model_executor.layers.quantization.fp8_to_bf16 import (
            Fp8ToBf16Config,
        )
        from vllm_sail.model_executor.layers.quantization.fp8_to_int8 import (
            Fp8ToInt8Config,
        )
        from vllm_sail.models.deepseek_v4.quant_config import DeepseekV4FP8Config

        for config in (Fp8ToInt8Config, Fp8ToBf16Config, Bf16ToInt8Config):
            name = config.get_name()
            if name in QUANTIZATION_METHODS:
                raise RuntimeError(f"Quantization method {name!r} already has an owner")
            register_quantization_config(name)(config)

        # The public registry explicitly supports overriding builtin configs.
        # Preserve its public name so checkpoint and MTP config resolution agree.
        register_quantization_config("deepseek_v4_fp8")(DeepseekV4FP8Config)
    _registered = True
    logger.debug("Registered PPU quantization configurations")
