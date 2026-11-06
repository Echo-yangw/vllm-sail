# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numeric E2E oracles for PPU FA3's QSA (query-sparse attention) path.

Two levels, because they fail for different reasons:

* **Kernel level.** :func:`~vllm_sail.attention.flash_attn._qsa.flash_attn_qsa_varlen_func`
  against an fp32 gather-attention golden, over the interleaved KV layout vLLM
  actually allocates. This is what catches a wheel whose QSA routing lands on the
  AIU bulk load, or whose split-KV reduction is wrong.
* **Path level.** :func:`~vllm_sail.models.qwen4_exp.qsa_fa3_forward.qsa_fa3_sparse_attention`
  -- the exact function the patched ``Qwen4ExpQSAImpl.forward`` calls -- against
  upstream's Triton ``qsa_sparse_paged_attention`` on the same paged cache, block
  table and selection. This is what catches a wrong logical-to-absolute
  translation, a dropped padding mask or a gate applied in the wrong order.

Both PPU chips run FA3, so the cases carry ``ppu`` rather than a capability
marker. A wheel without a working QSA path is an environment property, not a
failure -- QSA is still landing upstream -- so the module skips with the probe's
reason instead of failing.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("requires a real PPU/CUDA device", allow_module_level=True)
pytest.importorskip("vllm")

import vllm_sail  # noqa: E402

vllm_sail.register_out_of_tree()

from vllm.platforms import current_platform  # noqa: E402

if not current_platform.is_ppu():
    pytest.skip("requires a PPU platform (is_ppu() is False)", allow_module_level=True)

from tests.e2e.kernels import _reference as ref  # noqa: E402
from vllm_sail.attention.flash_attn._qsa import (  # noqa: E402
    fa3_qsa_unsupported_reason,
    flash_attn_qsa_varlen_func,
    is_fa3_qsa_supported,
)

if not is_fa3_qsa_supported():
    pytest.skip(
        f"the installed FA3 wheel cannot run QSA: {fa3_qsa_unsupported_reason()}",
        allow_module_level=True,
    )

from vllm.models.qwen4_exp.nvidia.ops.qsa import (  # noqa: E402
    qsa_sparse_paged_attention,
)
from vllm.utils.torch_utils import canonicalize_singleton_dim_strides  # noqa: E402

from vllm_sail.models.qwen4_exp.qsa_fa3_forward import (  # noqa: E402
    make_qsa_fa3_workspace,
    qsa_fa3_sparse_attention,
)

pytestmark = pytest.mark.ppu

DEVICE = torch.device("cuda")
HEAD_DIM = 256
PAGE_SIZE = 16


def _split_packed(packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the owner's packed cache exactly as ``Qwen4ExpQSAImpl`` does."""
    key, value = packed.transpose(1, 2).split(HEAD_DIM, dim=-1)
    return (
        canonicalize_singleton_dim_strides(key),
        canonicalize_singleton_dim_strides(value),
    )


def _flat_pools(packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Slot-major ``(slots, num_kv_heads, head_dim)`` views, for the golden."""
    k = (
        packed[..., :HEAD_DIM]
        .permute(0, 2, 1, 3)
        .reshape(-1, packed.shape[1], HEAD_DIM)
    )
    v = (
        packed[..., HEAD_DIM:]
        .permute(0, 2, 1, 3)
        .reshape(-1, packed.shape[1], HEAD_DIM)
    )
    return k.contiguous(), v.contiguous()


def _golden(
    q: torch.Tensor,
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    indices: torch.Tensor,
    used: list[int],
    scale: float,
) -> torch.Tensor:
    """fp32 per-row gather attention over each row's usable prefix."""
    num_heads = q.shape[1]
    group = num_heads // key_pool.shape[1]
    out = torch.empty(
        q.shape[0], num_heads, HEAD_DIM, device=q.device, dtype=torch.float32
    )
    for row in range(q.shape[0]):
        slots = indices[row, : used[row]].long()
        k = key_pool[slots].float()
        v = value_pool[slots].float()
        for head in range(num_heads):
            kv = head // group
            scores = (q[row, head].float() @ k[:, kv].t()) * scale
            out[row, head] = torch.softmax(scores, dim=-1) @ v[:, kv]
    return out


def _packed_cache(pages: int, num_kv_heads: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(
        pages,
        num_kv_heads,
        PAGE_SIZE,
        2 * HEAD_DIM,
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    ).to(device=DEVICE, dtype=torch.bfloat16)


# The real model's width is 2048 + 4 - 1 = 2051, which is also the range FA3's
# qsa_small_decode_2048 tuned arm keys on. The narrow cases keep the fp32 golden
# cheap enough to sweep the other axes.
WIDTHS = [
    pytest.param(64, id="w64"),
    pytest.param(2051, id="w2051"),
]


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize(
    "num_kv_heads,num_heads",
    [
        pytest.param(1, 12, id="tp2-h12-kv1"),
        pytest.param(2, 24, id="tp1-h24-kv2"),
    ],
)
@pytest.mark.parametrize("num_splits", [0, 1], ids=["splits-auto", "splits-1"])
def test_kernel_matches_gather_golden(width, num_kv_heads, num_heads, num_splits):
    """FA3's QSA path against an fp32 golden, on the layout vLLM allocates.

    ``num_splits=1`` is checked alongside the shipped ``0`` because a wrong
    split-KV reduction is invisible at ``1`` and only appears once the kernel
    combines partials -- so the two together localise it.
    """
    rows = 3
    used = [width, max(1, width // 3), max(1, width - 1)]
    slots = min(4096, max(2 * width, 512))
    pages = -(-slots // PAGE_SIZE)
    slots = pages * PAGE_SIZE

    generator = torch.Generator(device="cpu").manual_seed(0xC0FFEE)
    # Scattered and unsorted, and a different permutation per row: a load path
    # that fetches contiguous runs cannot match the golden by accident.
    indices = torch.empty(rows, width, dtype=torch.int32)
    for row in range(rows):
        pick = torch.randperm(slots, generator=generator)[: used[row]]
        indices[row, : used[row]] = pick.to(torch.int32)
        indices[row, used[row] :] = -1
    indices = indices.to(DEVICE)

    packed = _packed_cache(pages, num_kv_heads, seed=7)
    key_cache, value_cache = _split_packed(packed)
    key_pool, value_pool = _flat_pools(packed)

    q = torch.randn(
        rows,
        num_heads,
        HEAD_DIM,
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    ).to(device=DEVICE, dtype=torch.bfloat16)
    out = torch.empty_like(q)
    scale = HEAD_DIM**-0.5

    flash_attn_qsa_varlen_func(
        q,
        key_cache,
        value_cache,
        indices.view(rows, 1, width),
        torch.arange(rows + 1, dtype=torch.int32, device=DEVICE),
        1,
        softmax_scale=scale,
        out=out,
        num_splits=num_splits,
        seqused_k=torch.tensor(used, dtype=torch.int32, device=DEVICE),
        qsa_allow_aiu=False,
    )
    torch.cuda.synchronize()

    ref.assert_close(
        out,
        _golden(q, key_pool, value_pool, indices, used, scale),
        kind="attention",
        msg=f"FA3 QSA mismatch at width={width} kv={num_kv_heads} splits={num_splits}",
    )


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("num_kv_heads,num_heads", [(1, 12), (2, 24)])
def test_padding_never_reaches_the_softmax(width, num_kv_heads, num_heads):
    """A padded row must equal the same row with its tail physically removed.

    This is the ``seqused_k`` guard. FA3 honours it for QSA only together with
    ``is_causal``; a build that drops either walks the ``-1`` tail and the two
    results diverge by O(1), far outside the band.
    """
    rows = 2
    used = [width, max(1, width // 2)]
    pages = -(-min(4096, 2 * width) // PAGE_SIZE)
    slots = pages * PAGE_SIZE

    generator = torch.Generator(device="cpu").manual_seed(0xBAD)
    indices = torch.empty(rows, width, dtype=torch.int32)
    for row in range(rows):
        pick = torch.randperm(slots, generator=generator)[: used[row]]
        indices[row, : used[row]] = pick.to(torch.int32)
        # Poison, not just padding: if the tail is walked it cannot be missed.
        indices[row, used[row] :] = 0
    indices = indices.to(DEVICE)

    packed = _packed_cache(pages, num_kv_heads, seed=11)
    key_cache, value_cache = _split_packed(packed)
    q = torch.randn(
        rows,
        num_heads,
        HEAD_DIM,
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    ).to(device=DEVICE, dtype=torch.bfloat16)
    scale = HEAD_DIM**-0.5
    cu = torch.arange(rows + 1, dtype=torch.int32, device=DEVICE)

    padded = torch.empty_like(q)
    flash_attn_qsa_varlen_func(
        q,
        key_cache,
        value_cache,
        indices.view(rows, 1, width),
        cu,
        1,
        softmax_scale=scale,
        out=padded,
        num_splits=0,
        seqused_k=torch.tensor(used, dtype=torch.int32, device=DEVICE),
        qsa_allow_aiu=False,
    )
    truncated = torch.empty_like(q)
    flash_attn_qsa_varlen_func(
        q,
        key_cache,
        value_cache,
        indices[:, : min(used)].contiguous().view(rows, 1, -1),
        cu,
        1,
        softmax_scale=scale,
        out=truncated,
        num_splits=0,
        seqused_k=torch.tensor([min(used)] * rows, dtype=torch.int32, device=DEVICE),
        qsa_allow_aiu=False,
    )
    torch.cuda.synchronize()

    # Only the row whose usable prefix is the shorter one is a real comparison;
    # the full-width row has no tail to mask.
    ref.assert_close(
        padded[1],
        truncated[1],
        kind="attention",
        msg="FA3 QSA walked the padded tail: seqused_k is not being honoured",
    )


# Path level: the FA3 sequence the patched forward runs, against upstream's
# Triton kernel on the same inputs. Two requests so the block table is exercised
# and token_to_req is not constant.
PATH_CASES = [
    pytest.param(1, 12, 64, False, id="tp2-w64-decode"),
    pytest.param(1, 12, 64, True, id="tp2-w64-prefill"),
    pytest.param(2, 24, 64, False, id="tp1-w64-decode"),
    pytest.param(2, 24, 2051, False, id="tp1-w2051-decode"),
]


@pytest.mark.parametrize("num_kv_heads,num_heads,width,use_prefill", PATH_CASES)
def test_path_matches_the_triton_kernel(num_kv_heads, num_heads, width, use_prefill):
    """FA3 and Triton must agree on a real paged cache and block table.

    Covers the logical-to-absolute translation, the packed buffer's trailing
    count column being dropped rather than read as an index, and the output gate's
    rounding order -- none of which the kernel-level cases above touch.
    """
    num_requests = 2
    rows_per_request = 3
    num_tokens = num_requests * rows_per_request
    seq_len = max(width + PAGE_SIZE, 2 * PAGE_SIZE)
    blocks_per_request = -(-seq_len // PAGE_SIZE)
    pages = num_requests * blocks_per_request

    generator = torch.Generator(device="cpu").manual_seed(0x5A1)
    packed = _packed_cache(pages, num_kv_heads, seed=3)
    key_pool, value_pool = _flat_pools(packed)

    # Request r owns a contiguous run of physical blocks, shuffled so a
    # translation that forgot the block table would land on the wrong tokens.
    order = torch.randperm(pages, generator=generator).tolist()
    block_table = torch.tensor(
        [
            order[r * blocks_per_request : (r + 1) * blocks_per_request]
            for r in range(num_requests)
        ],
        dtype=torch.int32,
        device=DEVICE,
    )
    token_to_req = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)
    token_to_req = torch.div(token_to_req, rows_per_request, rounding_mode="floor")

    # Scattered, unsorted selections with a per-row usable prefix, then -1 padding
    # and the count column the owner's expand kernel would have written. Each row
    # gets a different prefix length so a row-independent bound cannot pass.
    packed_indices = torch.full((num_tokens, width + 1), -1, dtype=torch.int32)
    for token in range(num_tokens):
        used = max(1, min(width, seq_len) - token)
        pick = torch.randperm(seq_len, generator=generator)[:used]
        packed_indices[token, :used] = pick.to(torch.int32)
        packed_indices[token, width] = used
    packed_indices = packed_indices.to(DEVICE)

    q = torch.randn(
        num_tokens,
        num_heads,
        HEAD_DIM,
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    ).to(device=DEVICE, dtype=torch.bfloat16)
    gate = torch.randn(
        num_tokens,
        num_heads * HEAD_DIM,
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    ).to(device=DEVICE, dtype=torch.bfloat16)

    key_cache, value_cache = _split_packed(packed)
    triton_out = torch.empty_like(q)
    qsa_sparse_paged_attention(
        q,
        key_cache,
        value_cache,
        packed_indices,
        block_table,
        token_to_req,
        use_prefill,
        triton_out,
        output_gate=gate,
    )

    workspace = make_qsa_fa3_workspace(num_tokens, width, DEVICE)
    fa3_out = torch.zeros_like(q)
    qsa_fa3_sparse_attention(
        q,
        packed,
        packed_indices=packed_indices,
        block_table=block_table,
        token_to_req=token_to_req,
        output=fa3_out,
        output_gate=gate,
        head_size=HEAD_DIM,
        scale=HEAD_DIM**-0.5,
        workspace=workspace,
    )
    torch.cuda.synchronize()

    # The golden settles which of the two is wrong when they disagree.
    absolute = torch.full((num_tokens, width), -1, dtype=torch.int32)
    used_list = []
    for token in range(num_tokens):
        request = int(token_to_req[token])
        logical = packed_indices[token, :width].cpu()
        used = int(packed_indices[token, width])
        used_list.append(used)
        page_rank = torch.div(logical[:used], PAGE_SIZE, rounding_mode="floor")
        absolute[token, :used] = (
            block_table.cpu()[request, page_rank.long()] * PAGE_SIZE
            + logical[:used] % PAGE_SIZE
        )
    golden = _golden(
        q, key_pool, value_pool, absolute.to(DEVICE), used_list, HEAD_DIM**-0.5
    )
    gated = golden.to(torch.bfloat16).float() * torch.sigmoid(
        gate.view(num_tokens, num_heads, HEAD_DIM).float()
    )

    ref.assert_close(
        fa3_out,
        gated,
        kind="attention",
        msg="the FA3 QSA path diverges from the fp32 golden",
    )
    ref.assert_close(
        triton_out,
        gated,
        kind="attention",
        msg="the Triton QSA baseline diverges from the fp32 golden",
    )
    ref.assert_close(
        fa3_out,
        triton_out,
        kind="attention",
        msg=f"FA3 and Triton QSA disagree (prefill={use_prefill}, kv={num_kv_heads})",
    )
