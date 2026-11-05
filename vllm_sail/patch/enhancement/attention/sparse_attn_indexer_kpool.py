# SPDX-License-Identifier: Apache-2.0
"""PPU patches for the live glm5next kpool sparse-attention indexer.

The registry maps ``Glm5NextForConditionalGeneration`` to
``vllm.models.glm5next``; ``common.attention`` imports ``SparseAttnIndexerKpool``
from the package selector ``vllm.models.glm5next.sparse_indexer``, which picks
``nvidia/sparse_indexer.py`` on every non-ROCm platform -- PPU included. That
class's ``forward_cuda`` calls CUDA DeepGEMM's fp8/fp4 MQA logits and the
model's fp8 Triton kpool kernels unconditionally.

PPU runs the same layer through a dedicated INT8 module function
(``vllm_sail.attention.ops.mla_sparse_kpool``) registered as the
``ppu_sparse_attn_indexer_kpool`` torch op, mirroring how the shared
``SparseAttnIndexer`` is redirected to ``ppu_sparse_attn_indexer``:

* ``forward_ppu`` -- additive. Unpacks the optional ``(q, q_scale)`` tuple and
  calls the PPU op with the same arguments ``forward_cuda`` passes to the CUDA
  module function, plus the tail-cache coordinates.
* ``forward_native`` -- PPU redirects to ``forward_ppu`` so the disabled
  custom-op path also uses the PPU implementation; other platforms delegate to
  upstream. The enabled path is redirected by ``custom_op_dispatch``.

Do not import ``vllm.model_executor.layers.sparse_attn_indexer_kpool`` (or the
legacy ``vllm.models.glm5next.nvidia.attention`` that uses it): it registers the
same CustomOp name and raises ``Duplicate op name`` once the glm5next package
has been imported. The guard below fails loudly if the live class changes.
"""

from __future__ import annotations

import inspect
import sys

import torch
from vllm.models.glm5next.nvidia import sparse_indexer as _kpool_indexer
from vllm.platforms import current_platform
from vllm.utils.torch_utils import _encode_layer_name

import vllm_sail.ops  # noqa: F401  (registers torch.ops.vllm.ppu_sparse_attn_indexer_kpool)
from vllm_sail.patch.utils import patch

_AFFECTED = ">=0.31.0,<0.32.0"
_MODULE = "vllm.models.glm5next.nvidia.sparse_indexer"

_upstream_forward_native = _kpool_indexer.SparseAttnIndexerKpool.forward_native
# The pinned upstream (db9527a46) has no ``topk_backend``; newer builds do.
# Delegate with only the keywords the live upstream actually accepts.
_UPSTREAM_FORWARD_NATIVE_KEYS = frozenset(
    inspect.signature(_upstream_forward_native).parameters
)
_upstream_forward_cuda = getattr(
    _kpool_indexer.SparseAttnIndexerKpool, "forward_cuda", None
)
# Native fp8 PPUs (890P/SM89) keep the upstream fp8 kernels; only PPU hardware
# without native fp8 (810E/SM80) takes the INT8 port. Filter keywords the same
# way as for ``forward_native`` (the pin has no ``topk_backend``).
_UPSTREAM_FORWARD_CUDA_KEYS = (
    frozenset(inspect.signature(_upstream_forward_cuda).parameters)
    if _upstream_forward_cuda is not None
    else frozenset()
)

#: Modules that must expose the class patched here. ``common.attention`` builds
#: the live op; the selector re-exports the class for every other importer. A
#: future platform split (e.g. an ``amd/`` variant picked on PPU) would land in
#: one of these and must be adapted before PPU can run.
_EXPECTED_CONSUMERS = (
    "vllm.models.glm5next.sparse_indexer",
    "vllm.models.glm5next.common.attention",
)


@patch(
    _MODULE,
    "SparseAttnIndexerKpool.forward_ppu",
    allow_missing=True,
    reason=(
        "PPU implementation of the kpool sparse-attention indexer, calling the "
        "INT8 ppu_sparse_attn_indexer_kpool custom op registered by "
        "vllm_sail.ops on PPUs without native fp8 (810E/SM80). fp8-capable "
        "PPUs (890P/SM89) delegate to the upstream fp8 path. Upstream has no "
        "forward_ppu; installed additively and reached through the "
        "custom_op_dispatch patch (enabled path) or the forward_native "
        "redirect (disabled path)."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream's kpool indexer grows a platform hook, or PPU stops "
        "overriding this op."
    ),
)
def forward_ppu(
    self,
    hidden_states: torch.Tensor,
    q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    k: torch.Tensor,
    weights: torch.Tensor,
    *,
    gate_score: torch.Tensor | None = None,
    compress_ape: torch.Tensor | None = None,
    index_kpool: int = 1,
    positions: torch.Tensor | None = None,
    topk_backend: str = "auto",
):
    if not (
        current_platform.is_ppu() and not current_platform.supports_fp8()
    ):
        # fp8-capable PPU (890P/SM89): keep the upstream fp8/fp4 kernels.
        # Forcing the INT8 port here would lose performance and depends on
        # int8 kernels whose availability on SM89 is unverified.
        if _upstream_forward_cuda is None:  # pragma: no cover - pin has it
            raise RuntimeError(
                "SparseAttnIndexerKpool.forward_cuda is missing; cannot keep "
                "the native fp8 path on a fp8-capable PPU."
            )
        upstream_kwargs = {
            "gate_score": gate_score,
            "compress_ape": compress_ape,
            "index_kpool": index_kpool,
            "positions": positions,
        }
        if "topk_backend" in _UPSTREAM_FORWARD_CUDA_KEYS:
            upstream_kwargs["topk_backend"] = topk_backend
        return self.forward_cuda(
            hidden_states, q_quant, k, weights, **upstream_kwargs
        )

    # ``topk_backend`` selects among the CUDA top-k dispatches
    # (cooperative/persistent). The PPU op always uses the ``torch.ops._C``
    # top-k kernels exercised by the DSA INT8 path, so the flag is accepted for
    # signature parity with ``forward_cuda`` and ignored.
    del topk_backend
    if isinstance(q_quant, tuple):
        q_values, q_scale = q_quant
    else:
        q_values, q_scale = q_quant, None
    return torch.ops.vllm.ppu_sparse_attn_indexer_kpool(
        hidden_states,
        _encode_layer_name(self.k_cache.prefix),
        self.k_cache.kv_cache,
        q_values,
        q_scale,
        k,
        weights,
        self.quant_block_size,
        self.scale_fmt,
        self.topk_tokens,
        self.head_dim,
        self.max_pool_len,
        self.max_total_seq_len,
        self.topk_indices_buffer,
        self.skip_k_cache_insert,
        self.use_fp4_cache,
        gate_score,
        compress_ape,
        index_kpool,
        positions,
        self.tail_cache.kv_cache if self.tail_cache is not None else None,
        self.tail_cache.prefix if self.tail_cache is not None else None,
    )


@patch(
    _MODULE,
    "SparseAttnIndexerKpool.forward_native",
    reason=(
        "Fork-equivalent redirect: on PPU without native fp8 (810E/SM80) the "
        "kpool indexer must use the INT8 op even when the CustomOp is "
        "disabled, so forward_native dispatches to forward_ppu before the "
        "CUDA branch; other platforms and fp8-capable PPUs (890P/SM89, which "
        "keep the upstream fp8 path) delegate to the upstream implementation."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream's forward_native grows a platform hook, or PPU stops "
        "overriding this op."
    ),
)
def forward_native(
    self,
    hidden_states: torch.Tensor,
    q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    k: torch.Tensor,
    weights: torch.Tensor,
    *,
    gate_score: torch.Tensor | None = None,
    compress_ape: torch.Tensor | None = None,
    index_kpool: int = 1,
    positions: torch.Tensor | None = None,
    topk_backend: str = "auto",
):
    if current_platform.is_ppu() and not current_platform.supports_fp8():
        return self.forward_ppu(
            hidden_states,
            q_quant,
            k,
            weights,
            gate_score=gate_score,
            compress_ape=compress_ape,
            index_kpool=index_kpool,
            positions=positions,
            topk_backend=topk_backend,
        )
    upstream_kwargs = {
        "gate_score": gate_score,
        "compress_ape": compress_ape,
        "index_kpool": index_kpool,
        "positions": positions,
    }
    if "topk_backend" in _UPSTREAM_FORWARD_NATIVE_KEYS:
        upstream_kwargs["topk_backend"] = topk_backend
    return _upstream_forward_native(
        self, hidden_states, q_quant, k, weights, **upstream_kwargs
    )


def _check_live_consumer() -> None:
    """Fail loudly if a live consumer no longer uses the patched class."""
    for name in _EXPECTED_CONSUMERS:
        consumer = sys.modules.get(name)
        if consumer is None:
            continue
        live_cls = getattr(consumer, "SparseAttnIndexerKpool", None)
        if live_cls is not _kpool_indexer.SparseAttnIndexerKpool:
            raise RuntimeError(
                f"{name}.SparseAttnIndexerKpool is no longer "
                f"{_MODULE}.SparseAttnIndexerKpool; the PPU kpool indexer "
                "patch does not cover the model's live class. Port "
                "forward_ppu to the new class before running on PPU."
            )


_check_live_consumer()
