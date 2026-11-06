# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numeric E2E oracles for the PPU DeepGEMM family (CI plan PART 4, category 3).

Chip routing follows the authoritative capability matrix:

=========================  ==========  =======================
operator                   chips       marker
=========================  ==========  =======================
bf16 dense / grouped       810E, 890P  ``ppu``
w8a8-int8 DeepGEMM         810E, 890P  ``ppu``
w8a8-int8 ACEXT            810E only   ``cap80``
w8a8-fp8 (channel/block)   890P only   ``cap89``
mxfp4 grouped              890P only   ``cap89``
=========================  ==========  =======================

Every case compares the kernel output against the pure-PyTorch fp32 golden in
``tests/e2e/kernels/_reference.py``.  A missing DeepGEMM entry point raises
``RuntimeError`` from ``vllm_sail.utils.deep_gemm._missing``, and the
``native_kernels`` fixture loads the extension with ``strict=True``, so an
absent operator fails the run instead of skipping it.
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

from vllm.model_executor.layers.quantization.utils.int8_utils import (  # noqa: E402
    per_token_group_quant_int8,
)

from tests.e2e.kernels import _reference as ref  # noqa: E402
from vllm_sail.utils.deep_gemm import (  # noqa: E402
    bf16_gemm_nt,
    get_deep_gemm_config,
    get_mk_alignment_for_contiguous_layout,
    m_grouped_fp4_gemm_nt_nopad,
    m_grouped_int8_gemm_nt_nopad,
    per_block_cast_to_fp8,
    should_use_deepgemm_for_bf16_linear,
    should_use_deepgemm_for_fp8_linear,
)

pytestmark = pytest.mark.ppu

# Dense (M, N, K) problems from the CI plan.  N and K are multiples of 128 so
# the same shapes exercise the fp8 block-scaled path as well.
DENSE_SHAPES = [
    pytest.param(1, 4096, 4096, id="m1-n4096-k4096"),
    pytest.param(16, 4096, 7168, id="m16-n4096-k7168"),
    pytest.param(128, 7168, 2048, id="m128-n7168-k2048"),
    pytest.param(1024, 2048, 2048, id="m1024-n2048-k2048"),
    pytest.param(4096, 7168, 7168, id="m4096-n7168-k7168"),
]
# Reduced subset for the fp8/mxfp4 paths, whose fp32 golden is the dominant
# cost; the extremes (M=1 decode and M=128 prefill) are still covered.
SMALL_SHAPES = [
    pytest.param(1, 4096, 4096, id="m1-n4096-k4096"),
    pytest.param(16, 4096, 7168, id="m16-n4096-k7168"),
    pytest.param(128, 7168, 2048, id="m128-n7168-k2048"),
]
# Grouped (MoE-shaped) problems: (tokens, experts, N, K).
GROUPED_SHAPES = [
    pytest.param(8, 8, 2048, 2048, id="t8-e8-n2048-k2048"),
    pytest.param(64, 16, 4096, 2048, id="t64-e16-n4096-k2048"),
]


@pytest.fixture(scope="module", autouse=True)
def native_kernels():
    """Load the native extension strictly; a missing operator must fail."""
    from vllm_sail.native import install
    from vllm_sail.native.extensions import import_kernels

    import_kernels(strict=True)
    install()
    # Importing the ScaledMM module registers torch.ops.vllm.w8a8_* on PPU.
    import vllm_sail.model_executor.kernels.linear.scaled_mm.ppu  # noqa: F401


def _generator(seed: int = 1234) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


def _randn(*shape: int, seed: int = 1234, dtype=torch.bfloat16) -> torch.Tensor:
    return torch.randn(*shape, generator=_generator(seed), device="cuda", dtype=dtype)


def _uniform_expert_ids(tokens: int, experts: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-contiguous expert assignment plus the per-expert token counts.

    The DeepGEMM contiguous grouped kernels require rows to be sorted by
    expert, so the counts are made uniform and ``expert_ids`` is built with
    ``repeat_interleave`` exactly like ``_get_grouped_gemm_params`` does.
    """
    if tokens % experts:
        raise ValueError(f"tokens={tokens} must be divisible by experts={experts}")
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


# ---------------------------------------------------------------------------
# bf16
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("m", "n", "k"), DENSE_SHAPES)
def test_bf16_dense_gemm_nt(m, n, k):
    """``bf16_gemm_nt`` against an fp32 golden on both PPU chips."""
    x = _randn(m, k, seed=11)
    w = _randn(n, k, seed=12) * 0.05
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

    bf16_gemm_nt(x, w, out)
    torch.cuda.synchronize()

    expected = ref.ref_gemm(x, w)
    assert out.shape == (m, n) and out.dtype == torch.bfloat16
    ref.assert_close(out, expected, "bf16")


@pytest.mark.parametrize(("tokens", "experts", "n", "k"), GROUPED_SHAPES)
def test_m_grouped_bf16_gemm_nt_nopad(tokens, experts, n, k):
    """Grouped BF16 GEMM against the per-expert FP32 golden."""
    from vllm_sail.utils.deep_gemm import m_grouped_bf16_gemm_nt_nopad

    expert_ids, _ = _uniform_expert_ids(tokens, experts)
    x = _randn(tokens, k, seed=16)
    w = _randn(experts, n, k, seed=17) * 0.05
    out = torch.empty((tokens, n), device="cuda", dtype=torch.bfloat16)

    m_grouped_bf16_gemm_nt_nopad(x, w, out, expert_ids, None)
    torch.cuda.synchronize()

    expected = ref.ref_grouped_gemm(x, w, expert_ids)
    ref.assert_close(out, expected, "bf16")


@pytest.mark.parametrize(("tokens", "experts", "n", "k"), GROUPED_SHAPES)
def test_bf16_dense_linear_dispatch_gating(tokens, experts, n, k, monkeypatch):
    """BF16 dense DeepGEMM support is chip- *and* env-gated (capability != default).

    Both PPU chips can build bf16 DeepGEMM candidates, but auto-enablement
    additionally requires PPU 1.5 (890P) plus ``VLLM_SAIL_DENSE_BF16_DEEPGEMM``.
    """
    from vllm_sail import envs as ppu_envs

    x = _randn(tokens, k, seed=13)
    w = _randn(n, k, seed=14) * 0.05
    is_890p = current_platform.is_device_capability((8, 9))

    monkeypatch.setenv("VLLM_SAIL_DENSE_BF16_DEEPGEMM", "0")
    assert ppu_envs.VLLM_SAIL_DENSE_BF16_DEEPGEMM is False
    assert should_use_deepgemm_for_bf16_linear(x, w) is False

    monkeypatch.setenv("VLLM_SAIL_DENSE_BF16_DEEPGEMM", "1")
    assert should_use_deepgemm_for_bf16_linear(x, w) is is_890p
    # A bias forces the acblas fallback on every chip.
    bias = _randn(n, seed=15)
    assert should_use_deepgemm_for_bf16_linear(x, w, bias) is False


# ---------------------------------------------------------------------------
# w8a8 int8
# ---------------------------------------------------------------------------

_INT8_BACKENDS = [
    pytest.param("deepgemm", id="int8-deepgemm"),
    pytest.param("acext", marks=pytest.mark.cap80, id="int8-acext"),
]


@pytest.mark.parametrize("backend", _INT8_BACKENDS)
@pytest.mark.parametrize(("m", "n", "k"), DENSE_SHAPES[:4])
def test_w8a8_int8_dense_matmul(backend, m, n, k):
    """Registered ``w8a8_int8_matmul_*`` ops against a dequantized fp32 golden."""
    x = _randn(m, k, seed=21)
    w = (_randn(n, k, seed=22) * 0.05).to(torch.float32)
    x_q, x_s = ref.quant_per_token_int8(x)
    w_q, w_s = ref.quant_per_channel_int8(w, dim=0)
    x_q, w_q = x_q.cuda(), w_q.cuda()
    x_s, w_s = x_s.cuda(), w_s.cuda()

    if backend == "deepgemm":
        out = torch.ops.vllm.w8a8_int8_matmul_deepgemm(
            x_q, w_q, scale_x=x_s, scale_w=w_s, out_dtype=torch.bfloat16
        )
    else:
        out = torch.ops.vllm.w8a8_int8_matmul_acext(
            x_q, w_q, scale_a=x_s, scale_b=w_s, out_dtype=torch.bfloat16, bias=None
        )
    torch.cuda.synchronize()

    expected = ref.ref_gemm(ref.dequant_int8(x_q, x_s), ref.dequant_int8(w_q, w_s))
    assert out.shape == (m, n) and out.dtype == torch.bfloat16
    ref.assert_close(out, expected, "int8")


@pytest.mark.parametrize(("m", "n", "k"), DENSE_SHAPES[:3])
def test_w8a8_int8_backend_parity(m, n, k):
    """DeepGEMM and ACEXT must agree on 810E, where both int8 paths exist."""
    if not current_platform.is_device_capability((8, 0)):
        pytest.skip("ACEXT int8 GEMM only ships on ZW-810E (8,0)")
    x = _randn(m, k, seed=23)
    w = (_randn(n, k, seed=24) * 0.05).to(torch.float32)
    x_q, x_s = ref.quant_per_token_int8(x)
    w_q, w_s = ref.quant_per_channel_int8(w, dim=0)
    x_q, w_q, x_s, w_s = x_q.cuda(), w_q.cuda(), x_s.cuda(), w_s.cuda()

    deepgemm = torch.ops.vllm.w8a8_int8_matmul_deepgemm(
        x_q, w_q, scale_x=x_s, scale_w=w_s, out_dtype=torch.bfloat16
    )
    acext = torch.ops.vllm.w8a8_int8_matmul_acext(
        x_q, w_q, scale_a=x_s, scale_b=w_s, out_dtype=torch.bfloat16, bias=None
    )
    torch.cuda.synchronize()
    ref.assert_close(acext, deepgemm, "int8")


def _int8_layer(w_q: torch.Tensor, w_s: torch.Tensor) -> torch.nn.Module:
    """Minimal stand-in for a compressed-tensors int8 ``Linear`` layer."""
    layer = torch.nn.Module()
    layer.logical_widths = []
    layer.register_parameter(
        "weight", torch.nn.Parameter(w_q.clone(), requires_grad=False)
    )
    layer.register_parameter(
        "weight_scale", torch.nn.Parameter(w_s.clone(), requires_grad=False)
    )
    layer.input_scale = None
    layer.input_zero_point = None
    layer.azp_adj = None
    return layer


@pytest.mark.parametrize("backend", ["deepgemm", "acext", "triton"])
def test_w8a8_int8_scaled_mm_backend_forcing(backend, monkeypatch):
    """``VLLM_SAIL_DENSE_BACKEND`` selects the int8 ScaledMM implementation.

    Mirrors the fork_port monkeypatch pattern: the backend is forced through
    the documented environment switch and the resulting ``apply_weights``
    output is compared against the same fp32 golden.

    ``triton`` is asserted to be unsupported rather than compared numerically:
    it disables both PPU int8 GEMMs, so ``apply_weights`` falls through to
    ``ops.cutlass_scaled_mm``, which the PPU kernel manifest excludes (tier
    ``'x'``, CUTLASS/CuTe in the include closure).  There is no Triton int8
    *dense GEMM* on PPU -- ``VLLM_SAIL_USE_TRITON_INT8_QUANT`` only replaces
    the activation *quantiser* -- so the documented contract for that backend
    is a loud ``NotImplementedError``.
    """
    from vllm.model_executor.kernels.linear import init_int8_linear_kernel

    if backend == "acext" and not current_platform.is_device_capability((8, 0)):
        pytest.skip("ACEXT int8 GEMM only ships on ZW-810E (8,0)")
    monkeypatch.setenv("VLLM_SAIL_DENSE_BACKEND", backend)
    monkeypatch.setenv("VLLM_SAIL_USE_TRITON_INT8_QUANT", "1")

    m, n, k = 32, 4096, 2048
    x = _randn(m, k, seed=25)
    w = (_randn(n, k, seed=26) * 0.05).to(torch.float32)
    w_q, w_s = ref.quant_per_channel_int8(w, dim=0)
    layer = _int8_layer(w_q.cuda(), w_s.cuda()).to("cuda")

    kernel = init_int8_linear_kernel(
        is_channelwise=True,
        is_static_input_scheme=False,
        input_symmetric=True,
        module_name="kernels.test_deepgemm_numeric",
    )
    kernel.process_weights_after_loading(layer)
    assert kernel.use_deepgemm_int8_gemm is (backend == "deepgemm")
    assert kernel.use_acext_int8_gemm is (backend == "acext")
    assert kernel.weight_RowMajor is (backend != "triton")

    if backend == "triton":
        with pytest.raises(
            NotImplementedError, match="cutlass_scaled_mm has no PPU implementation"
        ):
            kernel.apply_weights(layer, x, None)
        return

    out = kernel.apply_weights(layer, x, None)
    torch.cuda.synchronize()

    # ``apply_weights`` re-quantises the activations itself, and with
    # ``VLLM_SAIL_USE_TRITON_INT8_QUANT=1`` (forced above) that is the Triton
    # ``per_token_group_quant_int8``, not ``ref.quant_per_token_int8``.  The two
    # break rounding ties the other way round: measured on ZW-890P for this
    # m=32/k=2048 shape they produce bit-identical scales and differ by at most
    # 1 LSB on 142 of 65536 activations (129 of them in row 31), every
    # disagreement sitting on a ``|x / scale|`` rounding boundary.  Feeding the
    # PyTorch-quantised golden a GEMM that consumed the Triton-quantised
    # activations therefore leaves 0.0637 of pure quantiser noise in the golden,
    # which is exactly the 18 row-31 violations this case used to report; the
    # DeepGEMM kernel itself is inside the int8 band (max abs 0.0305 against
    # either golden) and ``apply_weights`` is bitwise equal to calling
    # ``torch.ops.vllm.w8a8_int8_matmul_deepgemm`` on the same operands.
    # So: build the golden from the operand the kernel really used, and keep a
    # tight cross-check so a genuine quantiser regression still fails here.
    x_q, x_s = ref.quant_per_token_int8(x)
    x_q_kernel, x_s_kernel = per_token_group_quant_int8(
        x.contiguous(),
        x.shape[-1],
        dtype=torch.int8,
        use_triton=True,
        use_rounding=True,
    )
    torch.cuda.synchronize()
    assert torch.equal(x_s_kernel, x_s.cuda()), "activation scales diverged"
    lsb = (x_q_kernel.to(torch.int32) - x_q.cuda().to(torch.int32)).abs()
    assert int(lsb.max()) <= 1, (
        "the Triton and PyTorch per-token int8 quantisers disagree by more than "
        f"1 LSB (max {int(lsb.max())} over {int((lsb != 0).sum())} elements)"
    )

    expected = ref.ref_gemm(
        ref.dequant_int8(x_q_kernel, x_s_kernel),
        ref.dequant_int8(w_q.cuda(), w_s.cuda()),
    )
    assert out.shape == (m, n) and out.dtype == torch.bfloat16
    ref.assert_close(out, expected, "int8")


@pytest.mark.parametrize(("tokens", "experts", "n", "k"), GROUPED_SHAPES)
def test_m_grouped_int8_gemm_nt_nopad(tokens, experts, n, k):
    """Row-contiguous grouped int8 GEMM (the DeepGEMM MoE gate/up shape)."""
    expert_ids, _ = _uniform_expert_ids(tokens, experts)
    x = _randn(tokens, k, seed=27)
    w = (_randn(experts, n, k, seed=28) * 0.05).to(torch.float32)
    x_q, x_s = ref.quant_per_token_int8(x)
    w_q = torch.round(w / 0.01).clamp(-127, 127).to(torch.int8).cuda()
    w_s = torch.full((experts, n, 1), 0.01, device="cuda", dtype=torch.float32)
    out = torch.empty((tokens, n), device="cuda", dtype=torch.bfloat16)

    config = get_deep_gemm_config(tokens, n, k, num_groups=experts)
    m_grouped_int8_gemm_nt_nopad(
        (x_q.cuda(), x_s.cuda()), (w_q, w_s), out, expert_ids, None, configs=config
    )
    torch.cuda.synchronize()

    expected = ref.ref_grouped_gemm(
        ref.dequant_int8(x_q, x_s), ref.dequant_int8(w_q, w_s), expert_ids
    )
    ref.assert_close(out, expected, "int8")


# ---------------------------------------------------------------------------
# w8a8 fp8 (ZW-890P only)
# ---------------------------------------------------------------------------


@pytest.mark.cap89
@pytest.mark.parametrize(("m", "n", "k"), SMALL_SHAPES)
def test_w8a8_fp8_dense_channelwise(m, n, k):
    """``w8a8_fp8_matmul_deepgemm`` with per-token / per-channel E4M3FN scales."""
    x = _randn(m, k, seed=31).to(torch.float32)
    w = (_randn(n, k, seed=32) * 0.05).to(torch.float32)
    x_q, x_s = ref.quant_per_token_fp8(x)
    w_q, w_s = ref.quant_per_token_fp8(w)  # per-row == per-output-channel
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

    torch.ops.vllm.w8a8_fp8_matmul_deepgemm(
        x_q.cuda(), x_s.cuda(), w_q.cuda(), w_s.cuda(), out
    )
    torch.cuda.synchronize()

    expected = ref.ref_gemm(ref.dequant_fp8(x_q, x_s), ref.dequant_fp8(w_q, w_s))
    ref.assert_close(out, expected, "fp8")


@pytest.mark.cap89
@pytest.mark.parametrize(("m", "n", "k"), SMALL_SHAPES)
def test_w8a8_fp8_dense_blockwise(m, n, k):
    """``w8a8_fp8_matmul_deepgemm`` with (1, 128) x (128, 128) block scales."""
    x = _randn(m, k, seed=33).to(torch.float32)
    w = (_randn(n, k, seed=34) * 0.05).to(torch.float32)
    x_q, x_s = per_block_cast_to_fp8(x, block_size=[1, 128])
    w_q, w_s = per_block_cast_to_fp8(w, block_size=[128, 128])
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

    torch.ops.vllm.w8a8_fp8_matmul_deepgemm(x_q, x_s, w_q, w_s, out)
    torch.cuda.synchronize()

    # ``x_s`` is ``[m, k // 128]`` -- a ``[1, 128]`` block is one scale per
    # token, so the row axis is *not* repeated -- while ``w_s`` is the
    # ``[n // 128, k // 128]`` grid of ``[128, 128]`` blocks.
    expected = ref.ref_gemm(
        ref.dequant_fp8(x_q.float(), ref.expand_blocks(x_s, (1, 128), m, k)),
        ref.dequant_fp8(w_q.float(), ref.expand_blocks(w_s, (128, 128), n, k)),
    )
    ref.assert_close(out, expected, "fp8")


@pytest.mark.cap89
def test_fp8_linear_kernel_selection_and_numerics():
    """The ScaledMM layer picks the PPU DeepGEMM fp8 kernel on 890P.

    A static per-tensor activation scale and a static per-channel weight scale
    keep every operand exactly representable in E4M3FN, so the golden is the
    plain integer matmul rather than an approximation of it.
    """
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
    from vllm.model_executor.layers.quantization.utils import quant_utils

    from vllm_sail.model_executor.kernels.linear.scaled_mm.ppu import (
        PPUDeepGemmFP8ScaledMMLinearKernel,
    )

    m, n, k = 16, 4096, 2048
    generator = _generator(35)
    x = torch.randint(-2, 3, (m, k), generator=generator, device="cuda").float()
    w = torch.randint(-1, 2, (n, k), generator=generator, device="cuda").float()
    x_scale = torch.tensor([2.0 / 448.0], device="cuda")
    w_scale = (
        torch.arange(n, device="cuda").remainder(2).add(1).float().view(n, 1) / 448.0
    )
    fp8 = torch.float8_e4m3fn
    # |x| <= 2 with scale 2/448 and |w| <= 1 with scale <= 2/448 both land on
    # exact E4M3FN codes (448 is the largest finite E4M3FN value).
    x_q = (x / x_scale).to(fp8)
    w_q = (w / w_scale).to(fp8)
    reference = ref.ref_gemm(
        ref.dequant_fp8(x_q.float(), x_scale), ref.dequant_fp8(w_q.float(), w_scale)
    )

    with set_current_vllm_config(VllmConfig()):
        kernel = init_fp8_linear_kernel(
            quant_utils.kFp8StaticTensorSym,
            quant_utils.kFp8StaticChannelSym,
            torch.bfloat16,
            torch.bfloat16,
            (n, k),
        )
        assert isinstance(kernel, PPUDeepGemmFP8ScaledMMLinearKernel)
        layer = torch.nn.Module()
        # The ScaledMM base class stores the weight column-major, ``[K, N]``,
        # and ``PPUDeepGemmFP8ScaledMMLinearKernel.process_weights_after_loading``
        # transposes it into the ``[N, K]`` row-major operand the DeepGEMM
        # channelwise kernel wants.  Handing it the ``[N, K]`` quantised weight
        # directly would leave the layer ``[K, N]`` after that transpose, so
        # ``apply_scaled_mm`` reads ``N`` from the wrong axis and the
        # ``[N, 1]`` channel scale can no longer be expanded to it.
        layer.weight = torch.nn.Parameter(w_q.t(), requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(w_scale, requires_grad=False)
        layer.input_scale = torch.nn.Parameter(x_scale, requires_grad=False)
        layer.input_scale_ub = None
        kernel.process_weights_after_loading(layer)
        assert layer.weight.shape == (n, k), "weight must be row-major [N, K]"
        assert layer.weight_scale.shape == (n, 1)
        out = kernel.apply_weights(layer, x.to(torch.bfloat16), None)
        torch.cuda.synchronize()

    assert out.shape == (m, n) and out.dtype == torch.bfloat16
    ref.assert_close(out, reference, "fp8_relaxed")


@pytest.mark.cap80
def test_fp8_deepgemm_kernel_rejected_on_810e():
    """ZW-810E has no fp8 tensor cores: the kernel must advertise itself out."""
    from vllm_sail.model_executor.kernels.linear.scaled_mm.ppu import (
        PPUDeepGemmFp8BlockScaledMMKernel,
        PPUDeepGemmFP8ScaledMMLinearKernel,
    )

    supported, reason = PPUDeepGemmFP8ScaledMMLinearKernel.is_supported()
    assert supported is False
    assert reason is not None and "SM80" in reason
    assert PPUDeepGemmFp8BlockScaledMMKernel.is_supported()[0] is False
    assert (
        should_use_deepgemm_for_fp8_linear(
            torch.bfloat16, (4096, 2048), supports_deep_gemm=False
        )
        is False
    )


def test_fp8_linear_shape_gating_contract():
    """``should_use_deepgemm_for_fp8_linear`` enforces N % 64 and K % 128."""
    assert (
        should_use_deepgemm_for_fp8_linear(
            torch.bfloat16, (4096, 2048), supports_deep_gemm=True
        )
        is True
    )
    assert (
        should_use_deepgemm_for_fp8_linear(
            torch.bfloat16, (4090, 2048), supports_deep_gemm=True
        )
        is False
    )
    assert (
        should_use_deepgemm_for_fp8_linear(
            torch.bfloat16, (4096, 2040), supports_deep_gemm=True
        )
        is False
    )
    assert (
        should_use_deepgemm_for_fp8_linear(
            torch.float16, (4096, 2048), supports_deep_gemm=True
        )
        is False
    )


# ---------------------------------------------------------------------------
# mxfp4 (ZW-890P only)
# ---------------------------------------------------------------------------


@pytest.mark.cap89
@pytest.mark.parametrize(("tokens", "experts", "n", "k"), GROUPED_SHAPES)
def test_m_grouped_fp4_gemm_nt_nopad(tokens, experts, n, k):
    """Grouped MXFP4 GEMM against a dequantized fp32 golden.

    Inputs are produced by the same ``downcast_to_mxfp4`` the production MoE
    path uses, then decoded back to fp32 by ``_reference.dequantize_mxfp4`` so
    the comparison covers the E2M1/E8M0 wire format end to end.
    """
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

    assert a_q.dtype == torch.uint8 and a_q.shape == (tokens, k // 2)
    assert w_q.shape == (experts, n, k // 2)

    m_grouped_fp4_gemm_nt_nopad(
        (a_q, a_scale), (w_q, w_scale), bias, out, expert_ids, counts
    )
    torch.cuda.synchronize()

    a_deq = ref.dequantize_mxfp4(a_q, a_scale, uint16_scale=True)
    w_deq = ref.dequantize_mxfp4(w_q, w_scale_row, uint16_scale=False)
    expected = ref.ref_grouped_gemm(a_deq, w_deq, expert_ids, bias)
    ref.assert_close(out, expected, "mxfp4", chip="cap89")
