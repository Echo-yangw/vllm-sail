# SPDX-License-Identifier: Apache-2.0
"""FP8 source decoding without INT8 requantization."""

from vllm_sail.model_executor.layers.quantization.fp8_to_int8 import Fp8ToInt8Config


class Fp8ToBf16Config(Fp8ToInt8Config):
    target_dtype = "bf16"

    @classmethod
    def get_name(cls):
        return "fp8-to-bf16"
