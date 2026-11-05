# SPDX-License-Identifier: Apache-2.0
"""PPU INT8 query quantization for the GLM-5.3-Flash kpool indexer.

``vllm.models.glm5next.common.attention.Indexer.forward`` (the live model's
Indexer) rotates each query
with Hadamard-128 and calls ``fwht128_quant_fp8`` (defined in
``vllm.models.glm5next.nvidia.ops.kpool_compress``) before folding the query
scale into the logits ``weights``. The upstream helper returns
``float8_e4m3fn`` values; PPU 1.0 Triton cannot compile the fp8e4nv store, and
PPU DeepGEMM's indexer consumes INT8 Q/K.

This patch replaces the provider function on PPU without native fp8
(810E/SM80) with the INT8 variant (Hadamard unchanged,
``per_token_group_quant_int8`` for the values). fp8-capable PPUs (890P/SM89)
keep the upstream fp8 helper. The return contract changes from
``(float8_e4m3fn [rows,128], float32 [rows,1])``
to ``(int8 [rows,128], float32 [rows,1])``; the only consumer is
``Indexer.forward``'s ``q.view(-1, n_head, head_dim)`` ->
``_fused_indexer_weight_scale`` -> ``indexer_op`` chain, whose PPU op path
(``ppu_sparse_attn_indexer_kpool``) expects INT8 Q. Non-PPU platforms delegate
to the original function, so the CUDA contract is unchanged.

Aliases are patched on consumers that were imported before this plugin (module
scope ``from ... import fwht128_quant_fp8``); consumers imported later bind the
patched provider attribute directly.
"""

from __future__ import annotations

import sys

import torch
from vllm.platforms import current_platform

from vllm_sail.patch.utils import PATCH_MARKER, patch

_AFFECTED = ">=0.31.0,<0.32.0"
_PROVIDER = "vllm.models.glm5next.nvidia.ops.kpool_compress"
# ``common.attention`` is the live consumer. The legacy
# ``nvidia/attention.py`` copy imports ``model_executor.layers.
# sparse_attn_indexer_kpool`` and cannot even be imported after the glm5next
# package (Duplicate op name); it is listed defensively in case upstream
# restores that path, and the alias check only rewrites it if already loaded.
_CONSUMERS = (
    "vllm.models.glm5next.common.attention",
    "vllm.models.glm5next.nvidia.attention",
)
_REASON = (
    "PPU 1.0 Triton cannot compile upstream's fp8e4nv query cast, and PPU "
    "DeepGEMM's kpool indexer consumes INT8 Q/K. On PPUs without native fp8 "
    "(810E/SM80) this replacement rotates with the same Hadamard-128 matrix "
    "but returns INT8 values from per_token_group_quant_int8; the return "
    "contract changes from fp8 to int8 for the values (the fp32 scale shape is "
    "unchanged), and the only consumer is Indexer.forward's q.view -> "
    "_fused_indexer_weight_scale -> indexer_op chain, whose PPU path expects "
    "INT8. fp8-capable PPUs (890P/SM89) and non-PPU delegate to the original."
)
_REMOVE_WHEN = (
    "PPU 1.0 Triton supports fp8e4nv stores and PPU DeepGEMM accepts FP8 "
    "indexer queries, or upstream exposes a query-quantization format hook."
)

# Import the provider first so the original callable can be captured and any
# already-imported consumer alias can be recognized.
from vllm.models.glm5next.nvidia.ops import kpool_compress as _provider_module  # noqa: E402

_upstream_fwht = _provider_module.fwht128_quant_fp8
# Log through a vLLM model logger name: the CLI configures the "vllm" logger at
# INFO, while a plugin-owned logger inherits the WARNING root level. The provider
# module itself has no logger attribute.
from vllm.logger import init_logger  # noqa: E402

logger = init_logger("vllm.models.glm5next.common.attention")


@patch(
    _PROVIDER,
    "fwht128_quant_fp8",
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def fwht128_quant_fp8(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    original = getattr(fwht128_quant_fp8, PATCH_MARKER)[
        f"{_PROVIDER}.fwht128_quant_fp8"
    ]
    if current_platform.is_ppu() and not current_platform.supports_fp8():
        from vllm_sail.models.glm5_next.ops.kpool_quant import (
            fwht128_quant_int8,
        )

        logger.info_once(
            "Using PPU INT8 kpool indexer query quantization "
            "(fwht128_quant_int8 -> ppu_sparse_attn_indexer_kpool)"
        )
        return fwht128_quant_int8(q)
    return original(q)


for _consumer_name in _CONSUMERS:
    _consumer = sys.modules.get(_consumer_name)
    if (
        _consumer is not None
        and getattr(_consumer, "fwht128_quant_fp8", None) is _upstream_fwht
    ):
        patch(
            _consumer_name,
            "fwht128_quant_fp8",
            reason=_REASON,
            affected_versions=_AFFECTED,
            remove_when=_REMOVE_WHEN,
        )(fwht128_quant_fp8)
