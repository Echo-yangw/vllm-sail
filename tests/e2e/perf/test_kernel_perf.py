# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Operator-level performance regression suite (CI plan PART 6, category 5).

Benchmarks the five PPU kernel families also covered numerically by
``tests/e2e/kernels`` -- DeepGEMM, ACEXT, PLA, FlashAttention and FlashMLA --
and compares latency / throughput against the per-chip baseline archive in
``tests/e2e/perf/baselines/<chip>/kernel.json`` (see ``_baseline_store.py``
for the storage, tolerance and promotion contract).

Measurement protocol (plan PART 6.1): warmup 5 + timed 20 iterations per case,
``torch.cuda.Event`` timing, median is the authoritative value and p90 the
tail.  Operand construction and quantisation happen outside the timed region;
only the kernel call itself is measured.

Chip routing follows the authoritative capability matrix (same split as the
numeric suites):

=========================  ==========  =====================
operator                   chips       marker
=========================  ==========  =====================
DeepGEMM bf16 dense        810E, 890P  ``ppu``
DeepGEMM w8a8-int8         810E, 890P  ``ppu``
DeepGEMM w8a8-fp8          890P only   ``cap89``
DeepGEMM mxfp4 grouped     890P only   ``cap89``
ACEXT fused MoE / GEMM     810E only   ``cap80``
PLA decode / prefill       810E, 890P  ``ppu``
FlashAttention FA2 / FA3   810E, 890P  ``ppu``
FlashMLA dense decode      810E, 890P  ``ppu``
=========================  ==========  =====================

Shapes reuse the ``tests/e2e/kernels`` conventions but keep only the
representative extremes (plan PART 6.2: decode / prefill / large GEMM), so the
nightly lane stays inside its timeout while every regime keeps an anchor.

Perf tests never compare numerics -- correctness is the numeric suites' job --
so a case whose operands would be too expensive to quantise per iteration is
still valid here.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# --- Guard chain: skip the whole module without a real PPU device. ------------
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("requires a real PPU/CUDA device", allow_module_level=True)
pytest.importorskip("vllm")

import vllm_sail  # noqa: E402

vllm_sail.register_out_of_tree()

from vllm.platforms import current_platform  # noqa: E402

if not current_platform.is_ppu():
    pytest.skip("requires a PPU platform (is_ppu() is False)", allow_module_level=True)

from vllm.third_party.flash_linear_attention.ops import (  # noqa: E402
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.v1.attention.ops import flashmla as mla_ops  # noqa: E402

from tests.e2e.kernels import _reference as ref  # noqa: E402
from tests.e2e.perf._perf_harness import (  # noqa: E402
    PerfHarness,
    attention_flops,
    gemm_flops,
)
from vllm_sail.attention import pla_decode, pla_prefill  # noqa: E402
from vllm_sail.attention.flash_attn import _kernels as fa  # noqa: E402
from vllm_sail.utils.deep_gemm import (  # noqa: E402
    bf16_gemm_nt,
    get_deep_gemm_config,
    get_mk_alignment_for_contiguous_layout,
    m_grouped_fp4_gemm_nt_nopad,
    m_grouped_int8_gemm_nt_nopad,
)

pytestmark = [pytest.mark.ppu, pytest.mark.perf]

DEVICE = torch.device("cuda")
SUITE = "kernel"

# ---------------------------------------------------------------------------
# Representative shapes (subset of the kernels/ numeric conventions; ids match
# the numeric suite so a baseline row can be traced back to its oracle case).
# ---------------------------------------------------------------------------

# DeepGEMM dense [M, K] x [N, K]^T: decode / prefill / large GEMM.
DENSE_SHAPES = [
    pytest.param(1, 4096, 4096, id="m1-n4096-k4096"),
    pytest.param(1024, 2048, 2048, id="m1024-n2048-k2048"),
    pytest.param(4096, 7168, 7168, id="m4096-n7168-k7168"),
]
# DeepGEMM row-contiguous grouped (MoE gate/up shape): tokens, experts, N, K.
GROUPED_SHAPES = [
    pytest.param(64, 16, 4096, 2048, id="t64-e16-n4096-k2048"),
]
# ACEXT fused MoE: tokens, experts, top_k, hidden, intermediate.
ACEXT_MOE_SHAPES = [
    pytest.param(1, 8, 2, 2048, 1024, id="t1-e8-k2-h2048"),
    pytest.param(64, 64, 8, 4096, 512, id="t64-e64-k8-h4096"),
    pytest.param(256, 128, 8, 2048, 512, id="t256-e128-k8-h2048"),
]
# ACEXT dense int8 GEMM (the w8a8_int8_matmul_acext op), decode + prefill.
ACEXT_DENSE_SHAPES = [
    pytest.param(1, 4096, 4096, id="m1-n4096-k4096"),
    pytest.param(1024, 2048, 2048, id="m1024-n2048-k2048"),
]
# PLA GDN decode (one step per call): batch, head_dim.
PLA_DECODE_SHAPES = [
    pytest.param(1, 128, id="b1-d128"),
    pytest.param(4, 128, id="b4-d128"),
]
# PLA GDN prefill (FlashQLA): total tokens, sequences.
PLA_PREFILL_SHAPES = [
    pytest.param(512, 1, id="t512-n1"),
    pytest.param(2048, 4, id="t2048-n4"),
]
# FlashAttention varlen prefill: causal, seq_len, heads, kv_heads, head_dim.
FA_PREFILL_SHAPES = [
    pytest.param(True, 1024, 32, 8, 128, id="causal-s1024-h32-kv8-d128"),
    pytest.param(False, 4096, 32, 8, 128, id="full-s4096-h32-kv8-d128"),
]
# FlashAttention paged decode: kv lengths of one ragged 2-request batch.
FA_DECODE_SEQ_LENS = (1024, 512)
FA_DECODE_HEADS = 16
FA_DECODE_KV_HEADS = 8
FA_DECODE_HEAD_DIM = 128
#: Page size per wheel.  The FA2 paged kernel is built for a 256-token page and
#: refuses anything else outright ("Paged KV cache block size must be divisible
#: by 256"), while FA3 walks the table page by page and accepts a finer page --
#: the same split ``test_paged_decode_with_block_table_matches_golden`` makes in
#: the numeric suite, so each wheel is benchmarked on a shape known to run.
FA_DECODE_BLOCK_SIZES = {2: 256, 3: 64}
# FlashMLA dense decode: kv_len of each of the two requests (MQA, 128 heads).
MLA_DECODE_KV_LENS = [
    pytest.param(512, id="kv512"),
    pytest.param(4096, id="kv4096"),
    pytest.param(16384, id="kv16384"),
]

# GDN head configuration (mirrors tests/e2e/kernels/test_pla_numeric.py).
PLA_H = 4
PLA_HV = 8

# DeepSeek MLA geometry (mirrors tests/e2e/kernels/test_flash_mla_numeric.py).
MLA_KV_LORA_RANK = 512
MLA_QK_ROPE_HEAD_DIM = 64
MLA_HEAD_DIM = MLA_KV_LORA_RANK + MLA_QK_ROPE_HEAD_DIM
MLA_BLOCK_SIZE = 64
MLA_NUM_Q_HEADS = 128
MLA_SOFTMAX_SCALE = MLA_HEAD_DIM**-0.5
MLA_DECODE_BATCH = 2


@pytest.fixture(scope="module", autouse=True)
def native_kernels():
    """Load the native extension and the PPU attention shims strictly.

    Same contract as the numeric suites: a missing operator must fail the run
    instead of silently dropping a baseline row.
    """
    from vllm_sail.attention.flash_attn_shim import install as install_fa_shim
    from vllm_sail.native import install
    from vllm_sail.native.extensions import import_kernels

    import_kernels(strict=True)
    install()
    install_fa_shim()
    # Registering the ScaledMM module exposes torch.ops.vllm.w8a8_* on PPU.
    import vllm_sail.model_executor.kernels.linear.scaled_mm.ppu  # noqa: F401


@pytest.fixture
def perf_harness(perf_baseline: dict) -> PerfHarness:
    """Kernel-suite harness bound to this chip's ``baselines/<chip>/kernel.json``.

    The chip key and baseline root come from the shared ``perf_baseline``
    fixture (see ``tests/conftest.py``), so the kernel and model suites
    resolve their store identically.
    """
    return PerfHarness(perf_baseline["store"](SUITE))


# ---------------------------------------------------------------------------
# Shared operand helpers (same conventions/seeds as the numeric suites)
# ---------------------------------------------------------------------------


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


def _randn(*shape: int, seed: int, dtype=torch.bfloat16) -> torch.Tensor:
    return torch.randn(*shape, generator=_generator(seed), device=DEVICE, dtype=dtype)


def _uniform_expert_ids(tokens: int, experts: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-contiguous expert assignment (DeepGEMM grouped layout contract)."""
    per_expert = tokens // experts
    block_m = get_mk_alignment_for_contiguous_layout()[0]
    if per_expert % block_m:
        raise ValueError(
            f"per-expert rows {per_expert} must align to block_m={block_m}"
        )
    expert_ids = torch.repeat_interleave(
        torch.arange(experts, device="cuda", dtype=torch.int32), per_expert
    )
    counts = torch.full((experts,), per_expert, device="cuda", dtype=torch.int32)
    return expert_ids, counts


def _cumsum(lengths, dtype=torch.int32) -> torch.Tensor:
    total = [0]
    for length in lengths:
        total.append(total[-1] + int(length))
    return torch.tensor(total, dtype=dtype, device=DEVICE)


# ---------------------------------------------------------------------------
# DeepGEMM (bf16 / w8a8-int8 on both chips; fp8 / mxfp4 on 890P only)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("m", "n", "k"), DENSE_SHAPES)
def test_deepgemm_bf16_dense_perf(perf_harness, m, n, k):
    """``bf16_gemm_nt`` latency + TFLOPS on both PPU chips."""
    x = _randn(m, k, seed=11)
    w = _randn(n, k, seed=12) * 0.05
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

    perf_harness.record_kernel(
        f"deepgemm.bf16_dense.m{m}-n{n}-k{k}",
        lambda: bf16_gemm_nt(x, w, out),
        flops=gemm_flops(m, n, k),
        config={"m": m, "n": n, "k": k, "dtype": "bf16"},
    )


@pytest.mark.parametrize(("m", "n", "k"), DENSE_SHAPES)
def test_deepgemm_w8a8_int8_dense_perf(perf_harness, m, n, k):
    """Registered ``w8a8_int8_matmul_deepgemm`` op, per-token x per-channel."""
    x = _randn(m, k, seed=21)
    w = (_randn(n, k, seed=22) * 0.05).to(torch.float32)
    x_q, x_s = ref.quant_per_token_int8(x)
    w_q, w_s = ref.quant_per_channel_int8(w, dim=0)

    def call():
        return torch.ops.vllm.w8a8_int8_matmul_deepgemm(
            x_q, w_q, scale_x=x_s, scale_w=w_s, out_dtype=torch.bfloat16
        )

    perf_harness.record_kernel(
        f"deepgemm.w8a8_int8_dense.m{m}-n{n}-k{k}",
        call,
        flops=gemm_flops(m, n, k),
        config={"m": m, "n": n, "k": k, "dtype": "w8a8-int8"},
    )


@pytest.mark.parametrize(("tokens", "experts", "n", "k"), GROUPED_SHAPES)
def test_deepgemm_w8a8_int8_grouped_perf(perf_harness, tokens, experts, n, k):
    """Row-contiguous grouped int8 GEMM (the DeepGEMM MoE gate/up shape)."""
    expert_ids, _ = _uniform_expert_ids(tokens, experts)
    x = _randn(tokens, k, seed=27)
    w = (_randn(experts, n, k, seed=28) * 0.05).to(torch.float32)
    x_q, x_s = ref.quant_per_token_int8(x)
    w_q = torch.round(w / 0.01).clamp(-127, 127).to(torch.int8).cuda()
    w_s = torch.full((experts, n, 1), 0.01, device="cuda", dtype=torch.float32)
    out = torch.empty((tokens, n), device="cuda", dtype=torch.bfloat16)
    config = get_deep_gemm_config(tokens, n, k, num_groups=experts)

    def call():
        m_grouped_int8_gemm_nt_nopad(
            (x_q.cuda(), x_s.cuda()), (w_q, w_s), out, expert_ids, None, configs=config
        )

    perf_harness.record_kernel(
        f"deepgemm.w8a8_int8_grouped.t{tokens}-e{experts}-n{n}-k{k}",
        call,
        flops=gemm_flops(tokens, n, k),
        config={
            "tokens": tokens,
            "experts": experts,
            "n": n,
            "k": k,
            "dtype": "w8a8-int8",
        },
    )


@pytest.mark.cap89
@pytest.mark.parametrize(("m", "n", "k"), DENSE_SHAPES[:2])
def test_deepgemm_w8a8_fp8_dense_perf(perf_harness, m, n, k):
    """``w8a8_fp8_matmul_deepgemm`` channelwise path (ZW-890P only)."""
    x = _randn(m, k, seed=31).to(torch.float32)
    w = (_randn(n, k, seed=32) * 0.05).to(torch.float32)
    x_q, x_s = ref.quant_per_token_fp8(x)
    w_q, w_s = ref.quant_per_token_fp8(w)  # per-row == per-output-channel
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    x_q, w_q = x_q.cuda(), w_q.cuda()
    x_s, w_s = x_s.cuda(), w_s.cuda()

    perf_harness.record_kernel(
        f"deepgemm.w8a8_fp8_dense.m{m}-n{n}-k{k}",
        lambda: torch.ops.vllm.w8a8_fp8_matmul_deepgemm(x_q, x_s, w_q, w_s, out),
        flops=gemm_flops(m, n, k),
        config={"m": m, "n": n, "k": k, "dtype": "w8a8-fp8"},
    )


@pytest.mark.cap89
@pytest.mark.parametrize(("tokens", "experts", "n", "k"), GROUPED_SHAPES)
def test_deepgemm_mxfp4_grouped_perf(perf_harness, tokens, experts, n, k):
    """Grouped MXFP4 GEMM (ZW-890P only), production downcast operands."""
    from deep_gemm import preprocess_mxfp4_scales

    from vllm_sail.model_executor.layers.quantization.utils.mxfp4_utils import (
        ScaleFormat,
        downcast_to_mxfp4,
    )

    expert_ids, counts = _uniform_expert_ids(tokens, experts)
    x = _randn(tokens, k, seed=41).to(torch.float32)
    w = (_randn(experts, n, k, seed=42) * 0.05).to(torch.float32)
    a_q, a_scale = downcast_to_mxfp4(x.cuda(), axis=1)
    w_q, w_scale_row = downcast_to_mxfp4(
        w.cuda(), axis=2, scale_format=ScaleFormat.UINT8_ROW_MAJOR
    )
    w_scale = preprocess_mxfp4_scales(w_scale_row)
    bias = torch.zeros((experts, n), device="cuda", dtype=torch.float32)
    out = torch.empty((tokens, n), device="cuda", dtype=torch.bfloat16)

    def call():
        m_grouped_fp4_gemm_nt_nopad(
            (a_q, a_scale), (w_q, w_scale), bias, out, expert_ids, counts
        )

    perf_harness.record_kernel(
        f"deepgemm.mxfp4_grouped.t{tokens}-e{experts}-n{n}-k{k}",
        call,
        # 4-bit MACs are conventionally credited once, not twice.
        flops=gemm_flops(tokens, n, k, mac_ops=1.0),
        config={
            "tokens": tokens,
            "experts": experts,
            "n": n,
            "k": k,
            "dtype": "mxfp4",
        },
    )


# ---------------------------------------------------------------------------
# ACEXT (ZW-810E only; 890P does not ship the acext backend)
# ---------------------------------------------------------------------------


def _routing(tokens: int, experts: int, top_k: int, seed: int):
    """Distinct experts per token plus normalized routing weights."""
    generator = _generator(seed)
    ids = torch.stack(
        [
            torch.randperm(experts, device="cuda", generator=generator)[:top_k]
            for _ in range(tokens)
        ]
    ).to(torch.int32)
    raw = torch.rand((tokens, top_k), generator=generator, device="cuda") + 0.1
    weights = raw / raw.sum(dim=-1, keepdim=True)
    return weights, ids


def _make_experts(quant_config):
    """Instantiate ``AcextExperts`` without a full ``FusedMoEConfig``."""
    from vllm_sail.model_executor.layers.fused_moe.experts.acext import AcextExperts

    expert = AcextExperts.__new__(AcextExperts)
    expert.quant_config = quant_config
    expert.ep_rank = 0
    expert.ep_size = 1
    return expert


def _acext_callables(
    scheme: str, tokens: int, experts: int, top_k: int, hidden: int, inter: int
):
    """Build ``(call, moe_flops)`` for one ACEXT fused-MoE precision scheme.

    Mirrors the operand construction of ``tests/e2e/kernels/test_acext_moe_numeric``
    so the perf rows and the numeric oracles describe the same kernels.
    """
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEQuantConfig,
        int8_w8a8_moe_quant_config,
    )

    generator = _generator(101)
    x = torch.randn(
        (tokens, hidden), generator=generator, device="cuda", dtype=torch.bfloat16
    )
    weights, ids = _routing(tokens, experts, top_k, seed=102)
    output = torch.empty_like(x)
    empty = x.new_empty((0,))

    if scheme == "bf16":
        w1 = (
            torch.randn(
                (experts, 2 * inter, hidden), generator=generator, device="cuda"
            ).to(torch.bfloat16)
            * 0.05
        )
        w2 = (
            torch.randn(
                (experts, hidden, inter), generator=generator, device="cuda"
            ).to(torch.bfloat16)
            * 0.05
        )
        expert = _make_experts(FusedMoEQuantConfig.make(torch.bfloat16))
    elif scheme == "w8a8_int8":
        # CPU generation + quantisation mirrors the numeric suite and keeps
        # the [E, 2I, H] fp32 intermediate off the device.
        cpu_generator = torch.Generator(device="cpu").manual_seed(101)
        w1 = (
            torch.randn((experts, 2 * inter, hidden), generator=cpu_generator).float()
            * 0.05
        )
        w2 = (
            torch.randn((experts, hidden, inter), generator=cpu_generator).float()
            * 0.05
        )
        # [E, N, K] expert stacks need the 3-D quantiser: reducing over every
        # axis except ``dim`` would collapse the expert axis too and give every
        # expert the same dequant scale, which changes the bytes the kernel is
        # timed on.
        w1, w1_s = ref.quant_per_channel_int8_3d(w1, dim=1)
        w2, w2_s = ref.quant_per_channel_int8_3d(w2, dim=1)
        w1, w2 = w1.cuda(), w2.cuda()
        w1_s, w2_s = w1_s.cuda(), w2_s.cuda()
        expert = _make_experts(int8_w8a8_moe_quant_config(w1_s, w2_s, None, None))
    else:
        raise ValueError(f"unknown ACEXT scheme {scheme!r}")

    def call():
        expert.apply(
            output,
            x,
            w1,
            w2,
            weights,
            ids,
            MoEActivation.SILU,
            w1.shape[0],
            None,  # expert_map
            None,  # a1q_scale
            None,  # a2_scale
            empty,  # workspace13 (ACEXT allocates internally)
            empty,  # workspace2
            None,  # expert_tokens_meta
            False,  # apply_router_weight_on_input
        )

    # gate/up + down projections over the selected experts; 4-bit-free MACs.
    flops = 2.0 * tokens * top_k * (2 * inter * hidden + hidden * inter)
    return call, flops


@pytest.mark.cap80
@pytest.mark.parametrize("scheme", ["bf16", "w8a8_int8"])
@pytest.mark.parametrize(
    ("tokens", "experts", "top_k", "hidden", "inter"), ACEXT_MOE_SHAPES
)
def test_acext_fused_moe_perf(
    perf_harness, scheme, tokens, experts, top_k, hidden, inter
):
    """ACEXT fused MoE latency + token throughput (ZW-810E only)."""
    call, flops = _acext_callables(scheme, tokens, experts, top_k, hidden, inter)
    prefix = f"acext.fused_moe.{scheme}.t{tokens}-e{experts}-k{top_k}-h{hidden}"
    measurement = perf_harness.record_kernel(
        prefix,
        call,
        flops=flops,
        config={
            "tokens": tokens,
            "experts": experts,
            "top_k": top_k,
            "hidden": hidden,
            "inter": inter,
            "scheme": scheme,
        },
    )
    perf_harness.record_rate(
        f"{prefix}.tokens_per_s", tokens, measurement, unit="tok/s"
    )


@pytest.mark.cap80
@pytest.mark.parametrize(
    ("tokens", "experts", "top_k", "hidden", "inter"), ACEXT_MOE_SHAPES
)
def test_acext_fused_moe_w4a8_int8_perf(
    perf_harness, tokens, experts, top_k, hidden, inter
):
    """ACEXT packed-int4 MoE (``W4AInt8MoEMethod.apply``), ZW-810E only."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    from vllm_sail.model_executor.layers.quantization.mixed_precision_w4 import (
        MixedPrecisionW4Config,
        W4AInt8MoEMethod,
    )

    quant_config = MixedPrecisionW4Config()
    method = W4AInt8MoEMethod.__new__(W4AInt8MoEMethod)
    method.quant_config = quant_config
    method.ep_rank = 0
    method.ep_size = 1

    generator = _generator(301)
    pack = quant_config.pack_factor
    layer = torch.nn.Module()
    layer.activation = MoEActivation.SILU
    layer.register_parameter(
        "w13_weight",
        torch.nn.Parameter(
            torch.randint(
                -128,
                128,
                (experts, 2 * inter, hidden // pack),
                generator=generator,
                device="cuda",
                dtype=torch.int8,
            ),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "w2_weight",
        torch.nn.Parameter(
            torch.randint(
                -128,
                128,
                (experts, hidden, inter // pack),
                generator=generator,
                device="cuda",
                dtype=torch.int8,
            ),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "w13_weight_scale",
        torch.nn.Parameter(
            torch.full((experts, 2 * inter, 1), 0.005, device="cuda"),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "w2_weight_scale",
        torch.nn.Parameter(
            torch.full((experts, hidden, 1), 0.005, device="cuda"), requires_grad=False
        ),
    )
    # Pure relabel into the kernel-facing views, exactly like the numeric test.
    method.process_weights_after_loading(layer)

    x = torch.randn(
        (tokens, hidden), generator=generator, device="cuda", dtype=torch.bfloat16
    )
    weights, ids = _routing(tokens, experts, top_k, seed=302)

    prefix = f"acext.fused_moe.w4a8_int8.t{tokens}-e{experts}-k{top_k}-h{hidden}"
    measurement = perf_harness.record_kernel(
        prefix,
        lambda: method.apply(layer, x, weights, ids, None, None),
        # int4 weights: one credited op per MAC (matches the mxfp4 convention).
        flops=1.0 * tokens * top_k * (2 * inter * hidden + hidden * inter),
        config={
            "tokens": tokens,
            "experts": experts,
            "top_k": top_k,
            "hidden": hidden,
            "inter": inter,
            "scheme": "w4a8-int8",
        },
    )
    perf_harness.record_rate(
        f"{prefix}.tokens_per_s", tokens, measurement, unit="tok/s"
    )


@pytest.mark.cap80
@pytest.mark.parametrize(("m", "n", "k"), ACEXT_DENSE_SHAPES)
def test_acext_w8a8_int8_dense_perf(perf_harness, m, n, k):
    """Registered ``w8a8_int8_matmul_acext`` dense op (ZW-810E only)."""
    x = _randn(m, k, seed=23)
    w = (_randn(n, k, seed=24) * 0.05).to(torch.float32)
    x_q, x_s = ref.quant_per_token_int8(x)
    w_q, w_s = ref.quant_per_channel_int8(w, dim=0)
    x_q, w_q = x_q.cuda(), w_q.cuda()
    x_s, w_s = x_s.cuda(), w_s.cuda()

    perf_harness.record_kernel(
        f"acext.w8a8_int8_dense.m{m}-n{n}-k{k}",
        lambda: torch.ops.vllm.w8a8_int8_matmul_acext(
            x_q, w_q, scale_a=x_s, scale_b=w_s, out_dtype=torch.bfloat16, bias=None
        ),
        flops=gemm_flops(m, n, k),
        config={"m": m, "n": n, "k": k, "dtype": "w8a8-int8"},
    )


# ---------------------------------------------------------------------------
# PLA (GDN linear attention; both chips)
# ---------------------------------------------------------------------------


def _pla_decode_inputs(batch: int, head_dim: int, seed: int) -> SimpleNamespace:
    """Single-step GDN decode operands (mirrors test_pla_numeric._decode_inputs)."""
    generator = _generator(seed)
    total = batch  # one decode step per sequence
    num_slots = batch + 1

    def randn(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=generator, device=DEVICE, dtype=dtype)

    q = randn(1, total, PLA_H, head_dim) * 0.1
    k = randn(1, total, PLA_H, head_dim) * 0.1
    v = randn(1, total, PLA_HV, head_dim) * 0.1
    b = randn(total, PLA_HV) * 0.1
    a = randn(total, PLA_HV) * 0.1
    a_log = randn(PLA_HV, dtype=torch.float32) * 0.1
    dt_bias = randn(PLA_HV) * 0.1
    state = randn(num_slots, PLA_HV, head_dim, head_dim, dtype=torch.float32) * 0.05
    indices = (
        torch.randperm(num_slots - 1, generator=generator, device=DEVICE)[:batch] + 1
    ).to(torch.int32)
    cu_seqlens = torch.arange(0, batch + 1, dtype=torch.int32, device=DEVICE)
    return SimpleNamespace(
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        a_log=a_log,
        dt_bias=dt_bias,
        state=state,
        indices=indices,
        cu_seqlens=cu_seqlens,
        scale=head_dim**-0.5,
    )


@pytest.mark.parametrize(("batch", "head_dim"), PLA_DECODE_SHAPES)
def test_pla_gdn_decode_perf(perf_harness, batch, head_dim):
    """Fused sigmoid-gating delta-rule decode step (PLA k_last on PPU)."""
    assert pla_decode.get_sail_cuda_pla_k_last() is not None, (
        "the PPU `pla` decode wheel is mandatory for this suite"
    )
    inp = _pla_decode_inputs(batch, head_dim, seed=501)
    state = inp.state.clone()

    def call():
        fused_sigmoid_gating_delta_rule_update(
            A_log=inp.a_log,
            a=inp.a,
            b=inp.b,
            dt_bias=inp.dt_bias,
            q=inp.q,
            k=inp.k,
            v=inp.v,
            scale=inp.scale,
            initial_state=state,
            inplace_final_state=True,
            cu_seqlens=inp.cu_seqlens,
            ssm_state_indices=inp.indices,
            use_qk_l2norm_in_kernel=True,
            is_kda=False,
        )

    perf_harness.record_kernel(
        f"pla.gdn_decode.b{batch}-d{head_dim}",
        call,
        config={"batch": batch, "head_dim": head_dim, "h": PLA_H, "hv": PLA_HV},
    )


@pytest.mark.parametrize(("total_tokens", "num_seqs"), PLA_PREFILL_SHAPES)
def test_pla_gdn_prefill_perf(perf_harness, total_tokens, num_seqs):
    """FlashQLA GDN prefill chunk kernel (opt-in ``pla.prefill`` build)."""
    fwd = pla_prefill.get_sail_cuda_pla_prefill_fwd()
    if fwd is None:
        pytest.skip(
            "the pla.prefill FlashQLA wheel is not installed; GDN prefill "
            "falls back to Triton/FLA, which this benchmark does not cover"
        )
    head_dim = 128  # FlashQLA only implements K == V == 128
    generator = _generator(601)
    q = torch.randn(
        (1, total_tokens, PLA_H, head_dim),
        generator=generator,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    k = torch.randn_like(q)
    v = torch.randn(
        (1, total_tokens, PLA_HV, head_dim),
        generator=generator,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    # The PLA prefill backend requires pre-normalized q/k (production runs
    # fused_post_conv_prep with apply_l2norm=True before forward_pla).
    q = q * torch.rsqrt((q.to(torch.float32) ** 2).sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt((k.to(torch.float32) ** 2).sum(-1, keepdim=True) + 1e-6)
    q, k = q.to(torch.bfloat16), k.to(torch.bfloat16)
    # g is the forget gate *in log space* -- always negative, never NaN -- and
    # beta is already activated, mirroring ``_prefill_inputs`` in the numeric
    # suite.  (``log()`` of a negative draw would be NaN, and benchmarking a
    # kernel fed NaN measures nothing worth recording.)
    g = -torch.nn.functional.softplus(
        torch.randn(
            (1, total_tokens, PLA_HV),
            generator=generator,
            device=DEVICE,
            dtype=torch.float32,
        )
    )
    beta = torch.sigmoid(
        torch.randn(
            (1, total_tokens, PLA_HV),
            generator=generator,
            device=DEVICE,
            dtype=torch.float32,
        )
    )
    seq_len = total_tokens // num_seqs
    cu_seqlens = torch.arange(
        0, (num_seqs + 1) * seq_len, seq_len, dtype=torch.int32, device=DEVICE
    )

    def call():
        # ``chunk_gated_delta_rule_fwd`` takes rank-4 ``[1, T, H, D]`` q/k/v and
        # rank-3 ``[1, T, H]`` g/beta, which is exactly how they are built above.
        # (``forward_pla`` squeezes vLLM's packed layout and re-adds the batch
        # dimension; here there is nothing to squeeze, so adding one would make
        # the operands rank-5/rank-4 and trip the kernel's shape checks.)
        fwd(
            q,
            k,
            v,
            g,
            beta,
            scale=None,
            initial_state=None,
            cu_seqlens=cu_seqlens,
            output_final_state=False,
        )

    perf_harness.record_kernel(
        f"pla.gdn_prefill.t{total_tokens}-n{num_seqs}",
        call,
        config={
            "total_tokens": total_tokens,
            "num_seqs": num_seqs,
            "head_dim": head_dim,
            "h": PLA_H,
            "hv": PLA_HV,
        },
    )


# ---------------------------------------------------------------------------
# FlashAttention (FA2 + FA3 wheels; both chips)
# ---------------------------------------------------------------------------


def _self_attention_inputs(seq_len, num_heads, kv_heads, head_dim, seed):
    """Single-sequence varlen operands plus matching ``cu_seqlens``."""
    q = _randn(seq_len, num_heads, head_dim, seed=seed)
    k = _randn(seq_len, kv_heads, head_dim, seed=seed + 1)
    v = _randn(seq_len, kv_heads, head_dim, seed=seed + 2)
    cu = _cumsum([seq_len])
    return q, k, v, cu


@pytest.mark.parametrize(
    "fa_version", [pytest.param(2, id="fa2"), pytest.param(3, id="fa3")]
)
@pytest.mark.parametrize(
    ("causal", "seq_len", "heads", "kv_heads", "head_dim"), FA_PREFILL_SHAPES
)
def test_flash_attn_prefill_perf(
    perf_harness, fa_version, causal, seq_len, heads, kv_heads, head_dim
):
    """Varlen prefill latency + TFLOPS through the PPU FA wrapper."""
    q, k, v, cu = _self_attention_inputs(
        seq_len, heads, kv_heads, head_dim, seed=1000 + seq_len + heads
    )

    def call():
        return fa.flash_attn_varlen_func(
            q,
            k,
            v,
            seq_len,
            cu,
            seq_len,
            cu_seqlens_k=cu,
            causal=causal,
            fa_version=fa_version,
        )

    flops = attention_flops(seq_len, seq_len, heads, head_dim, causal=causal)
    perf_harness.record_kernel(
        f"fa{fa_version}.prefill.{'causal' if causal else 'full'}"
        f".s{seq_len}-h{heads}-kv{kv_heads}-d{head_dim}",
        call,
        flops=flops,
        config={
            "fa_version": fa_version,
            "causal": causal,
            "seq_len": seq_len,
            "heads": heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
        },
    )


def _paged_decode_inputs(fa_version: int, seed: int):
    """Paged KV operands for a ragged 2-request decode batch.

    Mirrors ``test_paged_decode_with_block_table_matches_golden`` in the numeric
    suite, which pins the shape contract the wheels actually accept:

    * ``q`` is ``(batch, heads, head_dim)`` -- one decode token per request,
      packed varlen-style, *not* ``(batch, 1, heads, head_dim)``;
    * the page size is per-wheel (see ``FA_DECODE_BLOCK_SIZES``);
    * ``max_blocks`` is the longest request's page count, so a shorter request's
      trailing table entries are padding that ``seqused_k`` keeps unread, and
      ``num_blocks`` is exactly ``batch * max_blocks`` so the shuffled
      permutation reshapes onto the table without leftovers.
    """
    seq_lens = FA_DECODE_SEQ_LENS
    batch = len(seq_lens)
    num_heads = FA_DECODE_HEADS
    kv_heads = FA_DECODE_KV_HEADS
    head_dim = FA_DECODE_HEAD_DIM
    block_size = FA_DECODE_BLOCK_SIZES[fa_version]
    max_len = max(seq_lens)
    max_blocks = max((length + block_size - 1) // block_size for length in seq_lens)
    num_blocks = batch * max_blocks

    q = _randn(batch, num_heads, head_dim, seed=seed)
    k_cache = _randn(num_blocks, block_size, kv_heads, head_dim, seed=seed + 1)
    v_cache = _randn(num_blocks, block_size, kv_heads, head_dim, seed=seed + 2)
    permutation = torch.randperm(
        num_blocks, generator=_generator(seed + 3), device=DEVICE
    )
    block_table = permutation.reshape(batch, max_blocks).to(torch.int32)
    seqused_k = torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE)
    cu_seqlens_q = _cumsum([1] * batch)
    return q, k_cache, v_cache, block_table, seqused_k, cu_seqlens_q, max_len


@pytest.mark.parametrize(
    "fa_version", [pytest.param(2, id="fa2"), pytest.param(3, id="fa3")]
)
def test_flash_attn_paged_decode_perf(perf_harness, fa_version):
    """Paged single-token decode latency (the serving hot path)."""
    q, k_cache, v_cache, block_table, seqused_k, cu_seqlens_q, max_len = (
        _paged_decode_inputs(fa_version, seed=7004)
    )

    def call():
        return fa.flash_attn_varlen_func(
            q,
            k_cache,
            v_cache,
            1,
            cu_seqlens_q,
            max_len,
            seqused_k=seqused_k,
            block_table=block_table,
            causal=True,
            fa_version=fa_version,
        )

    flops = sum(
        attention_flops(1, length, FA_DECODE_HEADS, FA_DECODE_HEAD_DIM, causal=True)
        for length in FA_DECODE_SEQ_LENS
    )
    perf_harness.record_kernel(
        f"fa{fa_version}.paged_decode.kv{'-'.join(str(s) for s in FA_DECODE_SEQ_LENS)}"
        f".h{FA_DECODE_HEADS}-d{FA_DECODE_HEAD_DIM}",
        call,
        flops=flops,
        config={
            "fa_version": fa_version,
            "seq_lens": list(FA_DECODE_SEQ_LENS),
            "heads": FA_DECODE_HEADS,
            "kv_heads": FA_DECODE_KV_HEADS,
            "head_dim": FA_DECODE_HEAD_DIM,
            "block_size": FA_DECODE_BLOCK_SIZES[fa_version],
        },
    )


# ---------------------------------------------------------------------------
# FlashMLA dense decode (both chips)
# ---------------------------------------------------------------------------


def _mla_paged_layout(seq_lens, seed: int):
    """Shuffled block table + slot mapping (mirrors the numeric suite)."""
    lengths = [int(length) for length in seq_lens]
    per_request = [
        (length + MLA_BLOCK_SIZE - 1) // MLA_BLOCK_SIZE for length in lengths
    ]
    used = sum(per_request)
    num_blocks = used + 1
    order = torch.randperm(num_blocks, generator=_generator(seed), device=DEVICE)
    spare = int(order[used].item())

    rows = []
    slots = []
    cursor = 0
    width = max(per_request)
    for index, length in enumerate(lengths):
        physical = [int(b) for b in order[cursor : cursor + per_request[index]]]
        cursor += per_request[index]
        rows.append(physical + [spare] * (width - len(physical)))
        for position in range(length):
            block = physical[position // MLA_BLOCK_SIZE]
            slots.append(block * MLA_BLOCK_SIZE + position % MLA_BLOCK_SIZE)
    block_table = torch.tensor(rows, dtype=torch.int32, device=DEVICE)
    slot_mapping = torch.tensor(slots, dtype=torch.int64, device=DEVICE)
    return block_table, slot_mapping, num_blocks


@pytest.mark.parametrize("kv_len", MLA_DECODE_KV_LENS)
def test_flashmla_dense_decode_perf(perf_harness, kv_len):
    """``flash_mla_with_kvcache`` over a paged 576-wide cache (MQA decode)."""
    seq_lens = (kv_len, max(kv_len // 2, 64))
    batch = len(seq_lens)
    total = sum(int(length) for length in seq_lens)
    query_len = 1

    q = _randn(batch, query_len, MLA_NUM_Q_HEADS, MLA_HEAD_DIM, seed=20250101)
    tokens = _randn(total, MLA_HEAD_DIM, seed=20250102)
    block_table, slot_mapping, num_blocks = _mla_paged_layout(seq_lens, seed=7)
    cache = torch.zeros(
        (num_blocks, MLA_BLOCK_SIZE, MLA_HEAD_DIM), dtype=torch.bfloat16, device=DEVICE
    )
    cache.view(-1, MLA_HEAD_DIM).index_copy_(0, slot_mapping, tokens)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE)
    scheduler_metadata, _ = mla_ops.get_mla_metadata(
        seq_lens_t, query_len * MLA_NUM_Q_HEADS, 1, is_fp8_kvcache=False
    )
    kv_paged = cache.unsqueeze(-2)  # add the single KV head, as upstream does

    def call():
        return mla_ops.flash_mla_with_kvcache(
            q=q,
            k_cache=kv_paged,
            block_table=block_table,
            cache_seqlens=seq_lens_t,
            head_dim_v=MLA_KV_LORA_RANK,
            tile_scheduler_metadata=scheduler_metadata,
            softmax_scale=MLA_SOFTMAX_SCALE,
            causal=True,
            is_fp8_kvcache=False,
        )

    flops = sum(
        attention_flops(query_len, length, MLA_NUM_Q_HEADS, MLA_HEAD_DIM, causal=True)
        for length in seq_lens
    )
    perf_harness.record_kernel(
        f"flashmla.dense_decode.kv{kv_len}",
        call,
        flops=flops,
        config={
            "query_len": query_len,
            "seq_lens": [int(length) for length in seq_lens],
            "num_q_heads": MLA_NUM_Q_HEADS,
            "head_dim": MLA_HEAD_DIM,
            "block_size": MLA_BLOCK_SIZE,
        },
    )
