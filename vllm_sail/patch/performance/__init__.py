# SPDX-License-Identifier: Apache-2.0
"""PPU performance patches.

Optimisations that PPU needs on top of upstream's code paths. Unlike the
`enhancement` category, removing these does not break correctness — it costs
performance.

Currently contains `fla` (flash-linear-attention): the fork's cache_results
autotune additions and the Qwen3-Next fused GDN decode path; and
`fused_qk_norm_rope`: the load reordering in the Qwen3.5 attention QK-RMSNorm +
RoPE + gate kernel.
"""

from vllm_sail.patch.performance import fla, fused_qk_norm_rope  # noqa: F401
