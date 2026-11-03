# SPDX-License-Identifier: Apache-2.0
"""PPU performance patches.

Optimisations that PPU needs on top of upstream's code paths. Unlike the
`enhancement` category, removing these does not break correctness — it costs
performance.

Contains `fla` (flash-linear-attention): the fork's cache_results autotune
additions and the Qwen3-Next fused GDN decode path; and `qsa_select_config`: a
PPU-measured launch-config table for the Qwen4Exp/Qwen3.8-Flash-Next QSA
sparse-attention split-K kernel.
"""

from vllm_sail.patch.performance import fla  # noqa: F401
from vllm_sail.patch.performance import qsa_select_config

# Leaf-style module (like enhancement/mxfp4): defines its replacement without
# vLLM imports at module scope, so installation is explicit.
qsa_select_config.install()
