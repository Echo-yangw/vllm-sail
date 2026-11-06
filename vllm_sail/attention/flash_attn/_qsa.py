# SPDX-License-Identifier: Apache-2.0
"""FA3's query-sparse attention (QSA) path.

QSA has no dedicated op: ``flash_attn_3.fwd`` selects it from the *rank* of
``page_table``. A 3-D ``(total_q, 1, topk)`` int32 table of absolute KV-pool
token indices means QSA; a 2-D table means ordinary paged attention. Getting
that rank wrong silently produces dense attention, so the wrapper rejects a 2-D
table instead of passing it through.

Imported lazily: nothing in ``vllm_sail`` reaches this module at package import
time, so a missing ``flash_attn_3`` wheel costs an unavailable reason string and
never an exception.
"""

from __future__ import annotations

import torch

from vllm_sail.attention.flash_attn._kernels import (
    NVTX_PROFILE,
    fa_version_unsupported_reason,
    maybe_contiguous,
    th_nvtx_range_pop,
    th_nvtx_range_push,
)

#: Scattered, unsorted top-k list for the numerical self-test: no two entries are
#: adjacent when sorted, so a load path that fetches contiguous runs cannot match
#: the reference by accident. Entries are absolute indices into the probe's
#: 256-token pool.
_QSA_PROBE_TOPK = (3, 40, 77, 114, 151, 188, 225, 8, 45, 82, 119, 156, 193, 230, 31, 68)
#: Padding tail. Masked through ``seqused_k``, so a build whose QSA path
#: force-nulls ``seqused_k`` walks it, picks up stale scores, and fails.
_QSA_PROBE_PAD = (-1, -1, -1, -1)

_QSA_HEAD_DIM = 256
_QSA_MAX_QHEADS_PER_KVHEAD = 32

_FA3_QSA_SUPPORTED: bool | None = None
_FA3_QSA_UNSUPPORTED_REASON: str | None = None


def fa3_qsa_uses_allow_aiu_flag() -> bool:
    """Whether the installed ``flash_attn_3.fwd`` carries ``qsa_allow_aiu``.

    The AIU paged-KV bulk load reads one table entry per 16-column-aligned group
    and fetches 16 pool-contiguous tokens, ignoring the other 15 entries. That is
    wrong for the scattered lists the QSA indexer emits, so the caller must
    certify contiguity before FA3 may use it. Wheels without the argument cannot
    be told, and QSA is unavailable on them.
    """
    try:
        schema = torch.ops.flash_attn_3.fwd.default._schema
    except (AttributeError, RuntimeError):
        return False
    return any(arg.name == "qsa_allow_aiu" for arg in schema.arguments)


def flash_attn_qsa_varlen_func(
    q,
    k_cache,
    v_cache,
    topk_table,
    cu_seqlens_q,
    max_seqlen_q,
    softmax_scale=None,
    out=None,
    num_splits=0,
    scheduler_metadata=None,
    return_softmax_lse=False,
    qsa_allow_aiu: bool = False,
    seqused_k=None,
):
    """Query-sparse attention forward over a paged BF16 KV cache.

    Args:
        q: ``(total_q, nheads, headdim)``, headdim 256.
        k_cache: ``(num_pages, page_size, nheads_k, headdim)``, ``stride(-1) == 1``.
        v_cache: same shape and strides as ``k_cache``.
        topk_table: ``(total_q, 1, topk)`` int32, ``stride(-1) == 1``. Absolute
            KV-pool token indices; a negative entry makes the kernel skip the
            load. Columns past ``seqused_k`` are never walked, so a padded row
            needs its usable entries as a contiguous prefix.
        cu_seqlens_q: ``(batch_size + 1,)`` int32. Required by QSA.
        max_seqlen_q: int. Required together with ``cu_seqlens_q``.
        softmax_scale: defaults to ``1 / sqrt(headdim)``.
        out: ``(total_q, nheads, headdim)``, written in place when given.
        num_splits: ``0`` lets FA3 pick the split-KV count, ``1`` disables
            splitting. Defaults to ``0``: a forced ``1`` serialises every row's
            whole top-k walk onto one tile, which measured slower than the Triton
            kernel on Zhenwu M890.
        scheduler_metadata: pass ``None``. ``get_scheduler_metadata``'s dense
            precomputation assumes ``seqlen_k = pages * page_size``; under QSA
            ``seqlen_k`` is the table width.
        qsa_allow_aiu: certify that every 16-column-aligned group of
            ``topk_table`` lists 16 pool-contiguous tokens. The indexer emits
            scattered lists, so callers must leave this ``False``.
        seqused_k: ``(batch_size,)`` int32, per-batch count of usable top-k
            columns. Honoured for QSA only together with ``is_causal``; without
            it FA3 drops causality and walks the padded tail. To express a
            per-row count, give every query row its own one-token batch
            (``cu_seqlens_q = arange(total_q + 1)``, ``max_seqlen_q = 1``).

    Returns:
        ``out``, plus ``(nheads, total_q)`` float32 ``softmax_lse`` when
        ``return_softmax_lse``.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if q.shape[-1] != _QSA_HEAD_DIM:
        raise ValueError(f"QSA only supports head dim 256, got {q.shape[-1]}")
    if topk_table.dim() != 3 or topk_table.size(1) != 1:
        raise ValueError(
            "QSA topk_table must have shape (total_q, 1, topk); a 2-D table would be "
            "read as an ordinary page table and silently produce dense attention"
        )
    if topk_table.size(0) != q.size(0):
        raise ValueError("QSA topk_table must have exactly one row per query token")
    if topk_table.dtype != torch.int32:
        raise ValueError("QSA topk_table must be int32")
    if q.size(1) % k_cache.size(2):
        raise ValueError("QSA requires nheads to be a multiple of nheads_k")
    if q.size(1) // k_cache.size(2) > _QSA_MAX_QHEADS_PER_KVHEAD:
        raise ValueError("QSA supports at most 32 query heads per KV head")

    q = maybe_contiguous(q)
    topk_table = maybe_contiguous(topk_table)
    cu_seqlens_q = maybe_contiguous(cu_seqlens_q)
    seqused_k = maybe_contiguous(seqused_k)
    # k_cache/v_cache are transposed views of vLLM's packed cache: stride(-1) == 1
    # is required, the leading strides may be non-canonical, which FA3 reads from
    # the tensor itself.
    if k_cache.stride(-1) != 1 or v_cache.stride(-1) != 1:
        raise ValueError("QSA k_cache/v_cache must have a contiguous last dimension")

    if not fa3_qsa_uses_allow_aiu_flag():
        raise RuntimeError(
            "this FA3 build lacks the qsa_allow_aiu argument, so QSA cannot be "
            "routed off the AIU paged-KV bulk load; gate on is_fa3_qsa_supported()"
        )

    if NVTX_PROFILE:
        th_nvtx_range_push(
            f"[FW_FMHA] --format=flash_attn_3_qsa,Forward,type:QSA,"
            f"seqlen_q:{max_seqlen_q},head_dim:{q.shape[-1]},"
            f"num_heads:{q.shape[-2]},num_heads_k:{k_cache.shape[-2]},"
            f"batch_size:{len(cu_seqlens_q) - 1},topk:{topk_table.shape[-1]},"
            f"page_size:{k_cache.shape[1]},data_type:{q.dtype}"
        )

    out, softmax_lse, _, _ = torch.ops.flash_attn_3.fwd(
        q,
        k_cache,
        v_cache,
        None,  # k_new: QSA rejects it
        None,  # v_new: QSA rejects it
        None,  # q_v
        out,
        cu_seqlens_q,
        None,  # cu_seqlens_k: passing it would make seqlen_k = max_seqlen_k
        None,  # cu_seqlens_k_new
        None,  # seqused_q
        seqused_k,
        max_seqlen_q,
        None,  # max_seqlen_k: seqlen_k is already the table width
        topk_table,  # 3-D -> selects the QSA path
        None,  # kv_batch_idx: rejected
        None,  # leftpad_k: rejected, indices are already absolute
        None,
        None,
        None,  # rotary_cos, rotary_sin, seqlens_rotary
        None,
        None,
        None,  # q_descale, k_descale, v_descale
        softmax_scale,
        True,  # is_causal: seqused_k is only honoured for QSA under causal
        -1,  # window_size_left
        -1,  # window_size_right
        0,  # attention_chunk
        0.0,  # softcap
        True,  # is_rotary_interleaved
        scheduler_metadata,
        num_splits,
        True,  # pack_gqa: QSA carries qhead_per_khead rows in one kBlockM
        0,  # sm_margin
        None,  # s_aux
        qsa_allow_aiu=qsa_allow_aiu,
    )
    if NVTX_PROFILE:
        th_nvtx_range_pop()
    return (out, softmax_lse) if return_softmax_lse else out


def _probe_fa3_qsa(device=None) -> tuple[bool, str | None]:
    """Whether the installed FA3 wheel computes QSA correctly.

    Passing the 3-D ``page_table`` check is not enough. The probe issues one call
    in the exact convention the QSA path uses -- real paged cache,
    ``qsa_allow_aiu=False``, ``num_splits=0``, a padded table tail masked through
    ``seqused_k`` -- and compares against a PyTorch reference on a scattered,
    unsorted list. A build whose routing still lands on the AIU bulk load, or
    whose QSA path force-nulls ``seqused_k``, cannot match.

    Three cache layouts are probed, because FA3 dispatches on their strides: a
    contiguous K/V pool; the interleaved
    ``[num_blocks, num_kv_heads, block_size, 2 * head_dim]`` pool vLLM actually
    allocates with two KV heads, which reaches FA3's QSA-aware generic kernel and
    is what a tensor-parallel size of 1 runs; and the same pool with one KV head,
    the only shape that satisfies FA3's ``k_batch_stride == page_size *
    k_row_stride`` under that layout and so the only one that reaches its
    hand-tuned QSA kernels.

    Cached: the result is a property of the wheel, not of the arguments.
    """
    global _FA3_QSA_SUPPORTED, _FA3_QSA_UNSUPPORTED_REASON
    if _FA3_QSA_SUPPORTED is not None or _FA3_QSA_UNSUPPORTED_REASON is not None:
        return _FA3_QSA_SUPPORTED, _FA3_QSA_UNSUPPORTED_REASON

    def _fail(reason: str) -> tuple[bool, str]:
        global _FA3_QSA_SUPPORTED, _FA3_QSA_UNSUPPORTED_REASON
        _FA3_QSA_SUPPORTED = False
        _FA3_QSA_UNSUPPORTED_REASON = reason
        return False, reason

    reason = fa_version_unsupported_reason(3, device)
    if reason is not None:
        return _fail(reason)
    if not hasattr(torch.ops, "flash_attn_3") or not hasattr(
        torch.ops.flash_attn_3, "fwd"
    ):
        return _fail("torch.ops.flash_attn_3.fwd is not registered")
    if not fa3_qsa_uses_allow_aiu_flag():
        return _fail(
            "flash_attn_3.fwd has no qsa_allow_aiu argument, so QSA cannot be routed "
            "off the AIU paged-KV bulk load"
        )

    softmax_scale = _QSA_HEAD_DIM**-0.5
    width = len(_QSA_PROBE_TOPK)
    pages, page_size = 16, 16
    try:
        dev = device if device is not None else torch.cuda.current_device()
        # Generated off-device so the probe never disturbs the global RNG state.
        # device="cpu" is explicit: this process may run with a non-CPU default
        # device, where randn would target it and reject the CPU generator.
        gen = torch.Generator(device="cpu").manual_seed(0x5A1)

        def _bf16(*shape):
            return torch.randn(
                *shape, generator=gen, device="cpu", dtype=torch.float32
            ).to(device=dev, dtype=torch.bfloat16)

        # 2 query heads. Both pools hold pages * page_size == 256 token slots, so
        # one absolute-index table drives both.
        q = _bf16(1, 2, _QSA_HEAD_DIM)
        topk_table = torch.tensor(
            _QSA_PROBE_TOPK + _QSA_PROBE_PAD, dtype=torch.int32, device=dev
        ).view(1, 1, -1)
        cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=dev)
        seqused_k = torch.tensor([width], dtype=torch.int32, device=dev)

        # Separate contiguous K and V pools, one KV head shared by both query heads.
        layouts = {
            "paged": (
                _bf16(pages, page_size, 1, _QSA_HEAD_DIM),
                _bf16(pages, page_size, 1, _QSA_HEAD_DIM),
            )
        }

        def _interleaved(num_kv_heads: int):
            """Split the pool vLLM actually allocates, as Qwen4ExpQSAImpl does.

            One KV head leaves a singleton dimension whose stride PyTorch is free
            to pick; the call site canonicalizes it before handing FA3 the view,
            so the probe does the same.
            """
            from vllm.utils.torch_utils import canonicalize_singleton_dim_strides

            packed = torch.empty(
                pages,
                num_kv_heads,
                page_size,
                2 * _QSA_HEAD_DIM,
                device=dev,
                dtype=torch.bfloat16,
            )
            packed[:, :, :, :_QSA_HEAD_DIM] = _bf16(
                pages, num_kv_heads, page_size, _QSA_HEAD_DIM
            )
            packed[:, :, :, _QSA_HEAD_DIM:] = _bf16(
                pages, num_kv_heads, page_size, _QSA_HEAD_DIM
            )
            k, v = packed.transpose(1, 2).split(_QSA_HEAD_DIM, dim=-1)
            return (
                canonicalize_singleton_dim_strides(k),
                canonicalize_singleton_dim_strides(v),
            )

        # Two KV heads, so no singleton-dimension stride rewriting can hide the
        # layout: this is what a tensor-parallel size of 1 runs, and it reaches
        # FA3's QSA-aware generic kernel.
        layouts["interleaved-kv2"] = _interleaved(2)
        # One KV head satisfies FA3's k_batch_stride == page_size * k_row_stride
        # under this layout, so it is the only one that reaches the hand-tuned
        # QSA kernels. Tensor-parallel size 2 of the target model runs it.
        layouts["interleaved-kv1"] = _interleaved(1)

        outs = {}
        for name, (k_cache, v_cache) in layouts.items():
            result, _lse = flash_attn_qsa_varlen_func(
                q,
                k_cache,
                v_cache,
                topk_table,
                cu_seqlens_q,
                1,
                softmax_scale=softmax_scale,
                qsa_allow_aiu=False,
                seqused_k=seqused_k,
                return_softmax_lse=True,
            )
            torch.cuda.synchronize()
            outs[name] = (result, k_cache, v_cache)
    except Exception as e:  # noqa: BLE001 - any failure means "no QSA in this build"
        return _fail(f"{type(e).__name__}: {e}")

    # Reference: per-query-token gather attention over the usable prefix, plus an
    # all-bf16 PyTorch pass whose rounding error sizes the tolerance (the criterion
    # of hopper/test_qsa_cases.py, plus a hard cap: a load path that fetches the
    # wrong K/V misses by O(1), far past both bounds).
    idx = topk_table.view(-1)[:width].long()
    page = idx // page_size
    offset = idx % page_size
    for name, (out, k_cache, v_cache) in outs.items():
        group = q.size(1) // k_cache.size(2)
        ref = torch.empty(q.size(1), _QSA_HEAD_DIM, device=dev, dtype=torch.float32)
        pt = torch.empty_like(ref)
        for head in range(q.size(1)):
            k_rows = k_cache[page, offset, head // group]
            v_rows = v_cache[page, offset, head // group]
            scores = (q[0, head].float() @ k_rows.float().t()) * softmax_scale
            ref[head] = torch.softmax(scores, dim=-1) @ v_rows.float()
            pt[head] = (
                torch.softmax((q[0, head] @ k_rows.t()) * softmax_scale, dim=-1)
                @ v_rows
            ).float()

        diff = (out[0].float() - ref).abs().max().item()
        pt_diff = (pt - ref).abs().max().item()
        if diff > 2 * pt_diff + 1e-4 or diff > 0.1:
            return _fail(
                f"QSA self-test mismatch on the {name} KV layout, scattered top-k list "
                f"(max diff {diff:.3g} vs tolerance {2 * pt_diff + 1e-4:.3g}): the "
                "kernel returned wrong values for qsa_allow_aiu=False, the calling "
                "convention the QSA path uses"
            )

    _FA3_QSA_SUPPORTED = True
    _FA3_QSA_UNSUPPORTED_REASON = None
    return True, None


def is_fa3_qsa_supported(device=None) -> bool:
    return _probe_fa3_qsa(device)[0]


def fa3_qsa_unsupported_reason(device=None) -> str | None:
    return _probe_fa3_qsa(device)[1]


__all__ = [
    "fa3_qsa_uses_allow_aiu_flag",
    "fa3_qsa_unsupported_reason",
    "flash_attn_qsa_varlen_func",
    "is_fa3_qsa_supported",
]
