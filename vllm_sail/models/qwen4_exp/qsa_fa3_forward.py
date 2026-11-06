# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The FA3 QSA forward: scratch workspace plus one step's kernel sequence.

Held here rather than in the patch module so the device suite in
``tests/e2e/kernels/test_qsa_fa3_numeric.py`` exercises the same code the
patched ``Qwen4ExpQSAImpl.forward`` runs, instead of a copy that could drift.

Three things this sequence has to get right, all of them silent when wrong:

* **Absolute indices.** FA3 takes a ``(total_q, 1, topk)`` table of KV-pool token
  indices; the indexer emits request-relative logical positions.
  :func:`~vllm_sail.models.qwen4_exp.ops.qsa_fa3.qsa_fa3_prepare_page_table`
  translates once per step.
* **``seqused_k``, together with ``is_causal``.** FA3 bounds the columns it walks
  by ``seqused_k``, which is how the indexer's ``-1`` padding stays out of the
  softmax. It is honoured for QSA only under ``is_causal``; without it FA3 drops
  causality and reads the padding. Expressing a *per-row* count means handing
  every query row over as its own one-token batch, so ``cu_seqlens_q`` is an
  arange and ``max_seqlen_q`` is 1.
* **``num_splits=0``.** Let FA3 pick the split count. A forced ``1`` serialises
  each row's whole top-k walk onto one tile and measured slower than Triton.

The output gate is applied separately: upstream folds it into the Triton
kernel's epilogue, and FA3 has no such hook.
:func:`~vllm_sail.models.qwen4_exp.ops.qsa_gate.qsa_apply_output_gate`
reproduces that epilogue's rounding order exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from vllm.config import get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.utils.torch_utils import canonicalize_singleton_dim_strides

from vllm_sail.models.qwen4_exp.qsa_fa3 import qsa_fa3_selection_width

# Under ``vllm`` so the reservation line survives: vLLM raises only its own logger
# to INFO, and the root logger stays at WARNING. See qwen4_exp_qsa.py.
logger = init_logger("vllm.ppu.qsa_fa3_forward")


@dataclass(frozen=True)
class QSAFA3Workspace:
    """Scratch the FA3 QSA path needs on top of the indexer's own top-k output.

    ``page_table`` keeps the singleton dimension FA3 needs to recognise a QSA
    table; the prepare kernel takes a squeezed view of the same storage.
    ``cu_seqlens_q`` is a pre-filled arange that turns every query row into its
    own one-token batch, so FA3 reads each row's usable length from ``seqused_k``
    instead of one shared length for the whole step.

    One workspace is shared by every QSA layer in the process -- ~67 MB at 8192
    tokens, against ~800 MB if each of the model's dozen layers kept its own.
    """

    page_table: torch.Tensor
    seqused_k: torch.Tensor
    cu_seqlens_q: torch.Tensor
    max_tokens: int
    selection_width: int

    def views(self, num_tokens: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Leading slices sized for one step; all three stay contiguous."""
        return (
            self.page_table[:num_tokens],
            self.seqused_k[:num_tokens],
            self.cu_seqlens_q[: num_tokens + 1],
        )


_WORKSPACE: QSAFA3Workspace | None = None


def qsa_fa3_workspace() -> QSAFA3Workspace | None:
    """The reserved workspace, or ``None`` if it was never needed."""
    return _WORKSPACE


def reserve_qsa_fa3_workspace() -> tuple[QSAFA3Workspace | None, str | None]:
    """Reserve the process-global QSA scratch from the active vLLM config.

    Idempotent, and called from ``Qwen4ExpQSAImpl.__init__`` -- so always during
    model construction, never inside a CUDA graph capture, which the metadata
    builder's ``AttentionCGSupport.UNIFORM_BATCH`` would otherwise risk.

    Returns ``(workspace, None)``, or ``(None, reason)`` when there is no config
    to size from or the checkpoint does not report the indexer widths.
    """
    global _WORKSPACE
    if _WORKSPACE is not None:
        return _WORKSPACE, None

    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        return None, "no active vLLM config to size the FA3 QSA workspace from"
    selection_width = qsa_fa3_selection_width(vllm_config.model_config.hf_text_config)
    if selection_width is None:
        return None, (
            "the checkpoint reports no indexer_budget/indexer_compress_ratio, so "
            "the FA3 QSA workspace cannot be sized"
        )
    _WORKSPACE = make_qsa_fa3_workspace(
        vllm_config.scheduler_config.max_num_batched_tokens,
        selection_width,
        torch.device("cuda", torch.cuda.current_device()),
    )
    return _WORKSPACE, None


def make_qsa_fa3_workspace(
    max_tokens: int, selection_width: int, device: torch.device
) -> QSAFA3Workspace:
    """Allocate a workspace of an explicit size; ``reserve`` uses the config's."""
    workspace = QSAFA3Workspace(
        page_table=torch.empty(
            max_tokens, 1, selection_width, dtype=torch.int32, device=device
        ),
        seqused_k=torch.empty(max_tokens, dtype=torch.int32, device=device),
        cu_seqlens_q=torch.arange(max_tokens + 1, dtype=torch.int32, device=device),
        max_tokens=max_tokens,
        selection_width=selection_width,
    )
    logger.info_once(
        "Reserved an FA3 QSA workspace: %d tokens x %d selections on %s.",
        max_tokens,
        selection_width,
        device,
    )
    return workspace


def qsa_fa3_sparse_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    *,
    packed_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    output: torch.Tensor,
    output_gate: torch.Tensor,
    head_size: int,
    scale: float,
    workspace: QSAFA3Workspace,
) -> None:
    """Run one QSA step on FA3, writing the gated result into ``output``.

    Args:
        query: ``(num_tokens, num_heads, head_size)`` bf16.
        kv_cache: ``(num_blocks, num_kv_heads, block_size, 2 * head_size)``, the
            owner's packed main cache. Split, not copied.
        packed_indices: ``(num_tokens, selection_width + 1)`` int32, the owner's
            top-k buffer slice. Request-relative and ``-1`` padded; the trailing
            column is each row's valid-entry count and is dropped here.
        block_table: ``(num_requests, max_blocks_per_request)`` int32.
        token_to_req: ``(num_tokens,)`` int32.
        output: ``(num_tokens, num_heads, head_size)`` bf16, written in place.
            Rows past ``num_actual_tokens`` are left alone; the caller zeroes the
            full tensor, since the owner allocates it uninitialised.
        output_gate: ``(num_tokens, ...)`` bf16, pre-sigmoid.
        head_size: main attention head dim, 256.
        scale: softmax scale.
        workspace: shared scratch, at least ``num_tokens`` rows wide.
    """
    from vllm_sail.attention.flash_attn._qsa import flash_attn_qsa_varlen_func
    from vllm_sail.models.qwen4_exp.ops.qsa_fa3 import qsa_fa3_prepare_page_table
    from vllm_sail.models.qwen4_exp.ops.qsa_gate import qsa_apply_output_gate

    num_tokens = query.shape[0]
    if num_tokens == 0:
        return
    assert num_tokens <= workspace.max_tokens, (num_tokens, workspace.max_tokens)

    # Drop the trailing count column: it holds each row's valid-entry count,
    # never a token index. The prepare pass recounts anyway, because it also
    # rejects selections whose block-table slot has gone stale.
    logical_indices = packed_indices[:, : packed_indices.shape[1] - 1]
    assert logical_indices.shape[1] == workspace.selection_width, (
        f"QSA selection width {logical_indices.shape[1]} does not match the FA3 "
        f"workspace's {workspace.selection_width}"
    )

    page_table, seqused_k, cu_seqlens_q = workspace.views(num_tokens)
    key_cache, value_cache = kv_cache.transpose(1, 2).split(head_size, dim=-1)
    # Size-1 dims may carry a degenerate stride, which FA3's TMA load rejects.
    key_cache = canonicalize_singleton_dim_strides(key_cache)
    value_cache = canonicalize_singleton_dim_strides(value_cache)

    qsa_fa3_prepare_page_table(
        logical_indices,
        block_table,
        token_to_req,
        page_table.squeeze(1),
        seqused_k,
        key_cache.shape[1],
        key_cache.shape[0],
    )
    flash_attn_qsa_varlen_func(
        query,
        key_cache,
        value_cache,
        page_table,
        cu_seqlens_q,
        1,
        softmax_scale=scale,
        out=output,
        num_splits=0,
        seqused_k=seqused_k,
        qsa_allow_aiu=False,
    )
    qsa_apply_output_gate(output, output_gate)


__all__ = [
    "QSAFA3Workspace",
    "make_qsa_fa3_workspace",
    "qsa_fa3_sparse_attention",
    "qsa_fa3_workspace",
    "reserve_qsa_fa3_workspace",
]
