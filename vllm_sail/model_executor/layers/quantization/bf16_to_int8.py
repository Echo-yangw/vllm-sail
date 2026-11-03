# SPDX-License-Identifier: Apache-2.0
"""Recipe-selected online INT8 conversion of unquantized BF16 checkpoints."""

from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from vllm_sail.model_executor.layers.quantization.fp8_to_int8 import Fp8ToInt8Config


class Bf16ToInt8Config(Fp8ToInt8Config):
    source_dtype = "bf16"

    def __init__(self):
        QuantizationConfig.__init__(self)
        self.weight_block_size = None
        self._source_ignored = []
        self.source_ignored_layers = []
        self._target_ignored = []
        self.ignored_layers = []
        self.recipe_id = None

    @classmethod
    def get_name(cls):
        return "bf16-to-int8"

    @classmethod
    def get_config_filenames(cls):
        return []

    @classmethod
    def from_config(cls, config):
        raise ValueError("bf16-to-int8 requires an unquantized BF16 checkpoint")

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        return None

    def get_quant_method(self, layer, prefix):
        if self.recipe_id is None:
            raise ValueError(
                "BF16 conversion requires recipe resolution before layer creation"
            )
        return super().get_quant_method(layer, prefix)
