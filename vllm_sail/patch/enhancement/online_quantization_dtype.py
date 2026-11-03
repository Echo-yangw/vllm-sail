# SPDX-License-Identifier: Apache-2.0
"""Preserve the source fork's FP8 auto-dtype default before quant validation."""

from vllm.config import model
from vllm.platforms import current_platform

from vllm_sail.patch.utils import patch

_upstream_get_dtype = model._get_and_verify_dtype


@patch(
    "vllm.config.model",
    "_get_and_verify_dtype",
    reason=(
        "PPU FP8 compatibility loading decodes to BF16; auto dtype must resolve "
        "before ModelConfig validates quantization, earlier than platform defaults."
    ),
    affected_versions=">=0.32.0,<0.33.0",
    remove_when="ModelConfig exposes a platform hook for checkpoint-aware auto dtype.",
)
def get_and_verify_dtype(model_id, config, dtype, **kwargs):
    # PPU MODIFICATION: begin
    if current_platform.is_ppu() and isinstance(dtype, str) and dtype.lower() == "auto":
        quant = getattr(config, "quantization_config", None)
        if quant is None:
            quant = getattr(getattr(config, "text_config", None), "quantization_config", None)
        if quant is not None and quant.get("quant_method") == "fp8":
            dtype = "bfloat16"
    # PPU MODIFICATION: end
    return _upstream_get_dtype(model_id, config, dtype, **kwargs)
