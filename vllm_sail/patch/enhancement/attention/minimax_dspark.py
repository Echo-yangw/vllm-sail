# SPDX-License-Identifier: Apache-2.0
"""Keep MiniMax DSpark on its base context writer when PPU has only a stub."""

from vllm.models.minimax_m3.common.ops.dspark_context_kv import DSparkContextKVWriter
from vllm.platforms import current_platform

from vllm_sail.patch.utils import patch

_original_supported = DSparkContextKVWriter.supported


@patch(
    "vllm.models.minimax_m3.common.ops.dspark_context_kv",
    "DSparkContextKVWriter.supported",
    reason="PPU advertises CUDA identity and compute stubs, which do not implement the fused DSpark context writer.",
    affected_versions=">=0.32.0,<0.33.0",
    remove_when="The context writer checks a real platform implementation or PPU implements and verifies this operator.",
)
def supported(attn_layers: list) -> bool:
    if current_platform.is_ppu():
        return False
    return _original_supported(attn_layers)
