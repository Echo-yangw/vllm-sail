# SPDX-License-Identifier: Apache-2.0
"""Give the QSA sparse-attention kernel a PPU-measured launch-config table.

``vllm.models.qwen4_exp.nvidia.ops.qsa._select_config`` picks
``(block_n, num_warps, num_tiles, num_splits)`` for the split-K sparse paged
GQA kernel from ``base_programs = num_rows * num_kv_heads``. It dispatches on
exact device capability:

    _is_sm120() -> capability == (12, 0)
    _is_sm90()  -> capability == (9, 0)

Zhenwu M890 reports ``(8, 9)``, so it matches neither and falls through to the
branch whose own docstring says "This default table was tuned on GB300". There
is no PPU entry and no registration seam -- the table is an inline ``if/elif``
chain, not a lookup ``vllm_sail/registry/tuned_configs.py`` could feed.

0.31's default table also differs from the 0.28 one PPU was validated against,
in two places that both cost throughput:

* ``base_programs > 2048`` **prefill** moved from ``(64, 1, 2)`` to
  ``(32, 1, 1)`` -- half the tile width and a quarter of the warps.
* 0.28's single ``base_programs <= 256 -> (64, 8, 2)`` entry was split into
  three ``block_n=32, num_warps=1`` entries (``<= 64``, ``<= 128``, ``<= 256``).

Matching ``_is_sm90`` would *not* fix the first one: ``_select_sm90_config``
returns ``(32, 1, 1)`` for ``base_programs > 2048`` prefill too, identical to
the GB300 default.

Measured on Zhenwu M890 (capability (8,9), triton 3.7.0), Qwen3.8-Flash-Next
attention shapes (head_dim 256, topk 2048, compress_ratio 4 -> num_columns
2051), against the ``ops/qsa.py`` of the ``v0.31.0`` tag. ``gpu_us`` is
torch-profiler self-time; the ratio is ``0.31 table / 0.28 table``, so >1 is a
regression:

    topology   bp      tier    ctx=1000  ctx=2048  ctx=16384
    bf16 TP=1  16      <=24      1.25x     1.09x     1.04x
    bf16 TP=1  32      <=32      1.83x     2.01x     1.86x
    bf16 TP=1  64      <=64      1.97x     2.16x     2.09x
    bf16 TP=1  128     <=128     1.97x     2.17x     2.20x
    bf16 TP=1  256     <=256     1.70x     1.84x     1.84x
    bf16 TP=1  8192    >2048pf   2.66x     2.74x     2.68x
    fp8  TP=1  128     <=128     1.50x     1.55x     1.55x
    fp8  TP=1  8192    >2048pf   1.26x     1.26x     1.26x
    fp8  TP=4  128     <=128     1.64x     1.69x     1.67x
    fp8  TP=4  4096    >2048pf   1.59x     1.59x     1.60x

``base_programs >= 512`` decode is unaffected: both tables resolve to the same
``(64, 4, 2)`` / ``(64, 1, 2)`` there and measure within 0.5%.

Production decode runs DP8+EP8, i.e. attention TP=1, so ``num_kv_heads = 2``
and ``base_programs = 2 * num_rows``. With MTP=3 (4 rows per sequence) that
puts any concurrency up to 32 sequences in the regressed ``<= 256`` band --
the latency-sensitive end of the curve. ``MAX_NUM_SEQS=40`` and above land in
the safe ``<= 512`` band.

The shipped ``ppu_select_config`` was measured as-is -- loaded from this module
by file path, driving the ``ops/qsa.py`` of the ``v0.31.0`` tag (``db9527a4``,
which is what both ``csrc/upstream/manifest.toml`` and
``.github/vllm-main-verified.commit`` resolve to) -- and holds residual
0.99-1.01x against the 0.28 table at 99 of the 102 points of the bf16 TP=1,
fp8 TP=1 and fp8 TP=4 sweeps (overall range 0.87-1.01x). The three exceptions
are all fp8 TP=4 ``base_programs = 8``, where it is 0.87-0.97x, i.e. faster,
for the ``small_profile_limit`` reason below. At all 27 prefill points it picks
the byte-identical config to 0.28's and measures residual 1.00x, which is what
localises the loss to the table rather than the kernel.

The 0.31 *kernel* is not part of the problem and must not be reverted. It
replaced 0.28's contiguous split partition with a strided one clamped per row
by the packed selection buffer's trailing ``valid_count`` column, which skips
tiles past the valid entries. Measured against 0.28's own kernel with both
running the 0.28 table, 0.31's kernel is 1.6-1.8x faster at ctx=1000 (16 of 33
tiles are useful) and at parity for ctx >= 2048 (all tiles useful). It is never
slower.

Two deliberate deviations from 0.28's table, both because ``_select_config``'s
signature cannot see what 0.28 keyed on:

* 0.28 set ``small_profile_limit = 8 if block_m <= 8 else 4`` where
  ``block_m = next_power_of_2(num_query_heads // num_kv_heads)``. Neither
  ``num_query_heads`` nor ``block_m`` is passed here. This uses 4, which is
  exact for the production decode topology (TP=1 gives group size 12, so
  ``block_m = 16 > 8``) and differs only for ``base_programs`` in 5..8, where
  the two 0.28 branches both use ``block_n=16, num_warps=4`` and differ only in
  ``target_splits`` (64 vs 32). The lower value is the better one: on the TP=4
  topology (group size 6, so 0.28 would take ``limit = 8``) at
  ``base_programs = 8`` the patch's 32 splits measure 0.87-0.97x of 0.28's 64.
* 0.28's ``max_useful_splits = 1 << (num_tiles.bit_length() - 1)`` clamp is
  dropped in favour of upstream's ``min(target_splits, num_tiles)``, which is
  what the numbers above were measured with and preserves upstream's "never
  more splits than tiles, never empty" invariant. The two agree at every
  ``num_tiles`` this table can produce (33 -> 32, 129 -> 128).

``is_fp8`` is unused: 0.28 had no fp8 KV path at all, and its bf16-tuned values
still win on the fp8 cache wherever 0.31's table regresses, by 1.26-1.69x
(table above). An fp8-specific PPU sweep is a follow-up, not a blocker.

Leaf-style module (like ``enhancement/mxfp4``): importable on a bare CPU
runner, patches install when ``install()`` runs. ``ops/qsa.py`` imports only
``vllm.model_executor.warmup``, ``vllm.platforms`` and ``vllm.triton_utils``,
so -- unlike the model-side ``nvidia/qsa.py`` -- it can be imported directly
without first breaking the ``qsa`` <-> ``model`` import cycle.

Nothing is copied from upstream: ``install()`` wraps ``_select_config`` and
delegates to the original off PPU, so the sm_120/sm_90 branches and the GB300
default keep exactly upstream's behaviour and there is no body to drift. The
drift harness still hashes the original through ``patch()``'s source capture.
"""

from __future__ import annotations

import importlib
from functools import lru_cache

AFFECTED_VERSIONS = ">=0.31.0,<0.32.0"

_QSA_OPS_MODULE = "vllm.models.qwen4_exp.nvidia.ops.qsa"

REASON = (
    "QSA _select_config dispatches on exact device capability ((12,0) for "
    "sm_120, (9,0) for sm_90). Zhenwu M890 reports (8,9), matches neither, and "
    "falls through to a table whose docstring says it was tuned on GB300. "
    "Relative to the 0.28 table PPU was validated against, 0.31 halved "
    "block_n and quartered num_warps for base_programs>2048 prefill "
    "(64,1,2)->(32,1,1), and split 0.28's single base_programs<=256 entry "
    "(64,8,2) into three block_n=32/num_warps=1 entries. Measured on Zhenwu "
    "M890 that costs 2.7x on bf16 prefill, 1.26-1.60x on fp8 prefill, and "
    "1.7-2.2x on bf16 decode across the base_programs<=256 band (up to 1.7x "
    "on fp8) -- a band production decode (DP8+EP8, attention TP=1, "
    "num_kv_heads=2) reaches at up to 32 concurrent sequences with MTP=3. "
    "base_programs>=512 decode is unaffected. Matching _is_sm90 would not "
    "help: its base_programs>2048 prefill entry is (32,1,1) too. The table is "
    "an inline if/elif chain with no registration seam, so it is replaced by a "
    "PPU-measured one and the original is delegated to off PPU. The 0.31 "
    "kernel itself is kept -- with both running the 0.28 table it is 1.6-1.8x "
    "faster than 0.28's kernel at ctx=1000 and at parity above that."
)
REMOVE_WHEN = (
    "upstream gives _select_config a PPU branch (or any capability check that "
    "Zhenwu M890's (8,9) satisfies), or retunes the GB300 default table so "
    "base_programs>2048 prefill is (64,1,2) and base_programs<=256 is (64,8,2)."
)

#: 0.28 keyed this on ``block_m``, which ``_select_config`` is not passed. 4 is
#: exact for the production decode topology; see the module docstring.
_SMALL_PROFILE_LIMIT = 4

_installed = False


@lru_cache(maxsize=1)
def _on_ppu() -> bool:
    from vllm.platforms import current_platform

    return current_platform.is_ppu()


def ppu_select_config(
    num_rows: int,
    num_kv_heads: int,
    use_prefill_config: bool,
    num_columns: int,
    is_fp8: bool = False,
) -> tuple[int, int, int, int]:
    """PPU launch config for the QSA split-K sparse paged GQA kernel.

    Returns ``(block_n, num_warps, num_tiles, num_splits)`` -- note upstream's
    return order swaps ``num_warps`` and ``target_splits`` relative to the
    table below.

    ``use_prefill_config`` and ``is_fp8`` are accepted for call compatibility
    and deliberately unused: the measurements in the module docstring show one
    table serving both prefill and decode and both KV dtypes.
    """
    base_programs = num_rows * num_kv_heads
    if base_programs <= _SMALL_PROFILE_LIMIT:
        block_n, target_splits, num_warps = 16, 64, 4
    elif base_programs < 32:
        block_n, target_splits, num_warps = 16, 32, 4
    elif base_programs <= 256:
        block_n, target_splits, num_warps = 64, 8, 2
    elif base_programs <= 512:
        block_n, target_splits, num_warps = 64, 4, 2
    else:
        block_n, target_splits, num_warps = 64, 1, 2
    num_tiles = -(-num_columns // block_n)
    num_splits = min(target_splits, num_tiles)
    return block_n, num_warps, num_tiles, num_splits


def install() -> None:
    """Apply the patch. Requires vLLM to be importable; idempotent."""
    global _installed
    if _installed:
        return

    from vllm.logger import init_logger

    from vllm_sail.patch.utils import patch

    logger = init_logger(__name__)

    try:
        _ops = importlib.import_module(_QSA_OPS_MODULE)
    except ImportError as exc:
        logger.warning(
            "Skipping the QSA launch-config patch: the upstream qwen4_exp QSA "
            "ops module cannot be imported (%s).",
            exc,
        )
        return

    upstream_select_config = _ops._select_config

    def dispatch(
        num_rows: int,
        num_kv_heads: int,
        use_prefill_config: bool,
        num_columns: int,
        is_fp8: bool = False,
    ) -> tuple[int, int, int, int]:
        # Checked ahead of upstream's capability dispatch on purpose: on PPU the
        # sm_120/sm_90 branches are unreachable anyway (M890 reports (8,9)), and
        # the PPU table is the one measured here regardless of what a future PPU
        # might report.
        if _on_ppu():
            return ppu_select_config(
                num_rows, num_kv_heads, use_prefill_config, num_columns, is_fp8
            )
        return upstream_select_config(
            num_rows, num_kv_heads, use_prefill_config, num_columns, is_fp8
        )

    patch(
        _QSA_OPS_MODULE,
        "_select_config",
        reason=REASON,
        affected_versions=AFFECTED_VERSIONS,
        remove_when=REMOVE_WHEN,
    )(dispatch)

    _installed = True
