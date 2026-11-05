# SPDX-License-Identifier: Apache-2.0
"""Upstream defects that break PPU. See ../README.md.

Importing this package must not require vLLM: the CPU-only CI lane imports it on
a bare runner, and ``patch/__init__.install()`` applies a category by importing
it. The patch modules below import their vLLM targets at import time, so guard
exactly one case -- vLLM is not installed at all -- and let every other error
surface unchanged.
"""

from __future__ import annotations

try:
    from vllm_sail.patch.bugfix import glm5_next_layer_types  # noqa: F401
    from vllm_sail.patch.bugfix import glm5_next_int8_mla_dequant  # noqa: F401
except ModuleNotFoundError as _exc:  # vLLM-less runner: nothing to patch
    if _exc.name != "vllm":
        raise
