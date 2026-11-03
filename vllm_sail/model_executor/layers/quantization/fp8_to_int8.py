# SPDX-License-Identifier: Apache-2.0
"""Explicit FP8 checkpoint conversion using plugin-owned INT8 backends."""

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.platforms import current_platform

from vllm_sail.model_executor.layers.quantization.online_conversion.config import (
    select_conversion,
    validate_fp8_config,
)
from vllm_sail.model_executor.layers.quantization.online_conversion.loading import (
    CheckedUnquantizedLinearMethod,
    CheckedUnquantizedMoEMethod,
)
from vllm_sail.model_executor.layers.quantization.online_conversion.selection import (
    is_ignored,
    is_mixed_layer,
)

logger = init_logger(__name__)


class Fp8ToInt8Config(QuantizationConfig):
    target_dtype = "int8"
    source_dtype = "fp8"
    handles_source_mixed_layers = True

    def __init__(self, config):
        super().__init__()
        validate_fp8_config(config)
        self.weight_block_size = [128, 128]
        self.source_mix_layer = list(config.get("mix_layer") or [])
        self.source_mix_layers = list(config.get("mix_layers") or [])
        self._source_ignored = list(
            config.get("ignored_layers") or config.get("modules_to_not_convert") or []
        )
        self.source_ignored_layers = list(self._source_ignored)
        self.ignored_layers = list(self._source_ignored)
        self._target_ignored = list(self._source_ignored)
        self.recipe_id = "checkpoint"

    def maybe_update_config(self, model_name, hf_config=None, revision=None):
        from vllm_sail.model_executor.layers.quantization.online_conversion.recipe import (
            select_recipe,
            target_exclusions,
        )

        if self.online_quantization_config is not None:
            raise ValueError(
                "SAIL conversion cannot be composed with another online quantizer"
            )
        if self.target_dtype == "bf16":
            return
        recipe = select_recipe(model_name, hf_config, self.source_dtype)
        if recipe is None:
            self.recipe_id = "checkpoint"
            self._target_ignored = list(self._source_ignored)
        else:
            self.recipe_id = recipe["id"]
            self._target_ignored = target_exclusions(recipe)
        self.ignored_layers = list(self._target_ignored)
        logger.info(
            "SAIL online INT8: source=%s recipe=%s", self.source_dtype, self.recipe_id
        )

    @classmethod
    def get_name(cls):
        return "fp8-to-int8"

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls):
        return 80

    @classmethod
    def get_config_filenames(cls):
        return ["int8_quant_config.json"]

    @classmethod
    def from_config(cls, config):
        return cls(config)

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        # Both registrations resolve the same policy, so registration order
        # cannot override an explicit conversion or steal native FP8.
        selected = select_conversion(
            hf_quant_cfg.get("quant_method"),
            user_quant,
            is_ppu=current_platform.is_ppu(),
        )
        if selected is not None:
            logger.info_once(
                "SAIL FP8 checkpoint conversion: %s (explicit argument: %s)",
                selected,
                user_quant is not None,
            )
        return selected

    def apply_vllm_mapper(self, hf_to_vllm_mapper):
        def mapped(entries):
            plain = [p for p in entries if not p.startswith("re:")]
            regex = [p for p in entries if p.startswith("re:")]
            return hf_to_vllm_mapper.apply_list(plain) + regex

        self.source_ignored_layers = mapped(self._source_ignored)
        self.ignored_layers = mapped(self._target_ignored)

    def get_quant_method(self, layer, prefix):
        if not current_platform.is_ppu():
            raise ValueError("Online checkpoint conversion is supported only on PPU")
        if isinstance(layer, VocabParallelEmbedding):
            return None
        if not isinstance(layer, LinearBase | RoutedExperts):
            return None
        mapping = getattr(self, "packed_modules_mapping", {})
        source_float = self.source_dtype == "bf16" or is_ignored(
            prefix, self.source_ignored_layers, mapping
        )
        # Singular metadata describes whole blocks in the model constructor;
        # plural metadata historically exempts Linear modules only.
        source_float = (
            source_float
            or is_mixed_layer(prefix, getattr(self, "source_mix_layer", []))
            or (
                isinstance(layer, LinearBase)
                and is_mixed_layer(prefix, getattr(self, "source_mix_layers", []))
            )
        )
        target_float = self.target_dtype == "bf16" or is_ignored(
            prefix, self.ignored_layers, mapping
        )
        if self.recipe_id == "checkpoint":
            target_float = target_float or source_float
        source = "bf16" if source_float else "fp8"
        target = "bf16" if target_float else "int8"
        if isinstance(layer, LinearBase):
            if source_float and target_float:
                return CheckedUnquantizedLinearMethod()
            from vllm_sail.model_executor.layers.quantization.online_conversion.linear import (
                Fp8ConversionLinearMethod,
            )

            return Fp8ConversionLinearMethod(target, source_dtype=source)
        if source_float and target_float:
            return CheckedUnquantizedMoEMethod(layer.moe_config)
        from vllm_sail.model_executor.layers.quantization.online_conversion.moe import (
            Fp8ToBf16MoEMethod,
            Fp8ToInt8MoEMethod,
        )

        method = Fp8ToInt8MoEMethod if target == "int8" else Fp8ToBf16MoEMethod
        return method(layer, source_dtype=source)
