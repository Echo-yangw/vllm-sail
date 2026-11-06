# SPDX-License-Identifier: Apache-2.0
"""Compare the V4.1 delayed prenorm path with upstream's torch reference."""

import pytest

pytestmark = pytest.mark.ppu


@pytest.mark.parametrize("tokens", [1, 33])
@pytest.mark.parametrize("carried", [False, True])
def test_delayed_mhc_prenorm_matches_reference(tokens, carried):
    import torch

    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm.model_executor.kernels.mhc.tilelang import mhc_pre_delayed_tilelang
    from vllm.model_executor.kernels.mhc.torch import mhc_pre_delayed_torch

    generator = torch.Generator().manual_seed(41)
    hidden = 5120
    residual = torch.randn(tokens, 4, hidden, generator=generator).bfloat16()
    fn = torch.randn(24, 4 * hidden, generator=generator) * 0.02
    scale = torch.tensor([0.5, 0.25, 1.0])
    base = torch.randn(24, generator=generator)
    pre = torch.rand(tokens, 4, generator=generator) if carried else None
    norm = torch.ones(hidden, dtype=torch.bfloat16)
    args = (residual, fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 20)
    expected = list(mhc_pre_delayed_torch(*args, pre_mix=pre))
    collapsed = expected[2].float()
    expected[2] = (
        collapsed * torch.rsqrt(collapsed.square().mean(-1, keepdim=True) + 1e-6)
    ).bfloat16()
    actual = mhc_pre_delayed_tilelang(
        *(item.cuda() if isinstance(item, torch.Tensor) else item for item in args),
        pre_mix=pre.cuda() if pre is not None else None,
        norm_weight=norm.cuda(),
        norm_eps=1e-6,
    )
    for index in (0, 1, 3):
        torch.testing.assert_close(
            actual[index].cpu(), expected[index], atol=1e-3, rtol=1e-3
        )
    torch.testing.assert_close(actual[2].cpu(), expected[2], atol=0.016, rtol=0.01)


@pytest.mark.parametrize("tokens", [1, 128])
def test_delayed_mhc_fused_post_pre_and_aux(tokens):
    import torch

    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm.model_executor.kernels.mhc.tilelang import (
        mhc_fused_post_pre_delayed_tilelang,
    )
    from vllm.model_executor.kernels.mhc.torch import mhc_pre_delayed_torch

    generator = torch.Generator().manual_seed(42)
    hidden = 5120
    residual = torch.randn(tokens, 4, hidden, generator=generator).bfloat16()
    x = torch.randn(tokens, hidden, generator=generator).bfloat16()
    old_post = torch.rand(tokens, 4, 1, generator=generator)
    old_comb = torch.rand(tokens, 4, 4, generator=generator)
    fn = torch.randn(24, 4 * hidden, generator=generator) * 0.02
    scale = torch.tensor([0.5, 0.25, 1.0])
    base = torch.randn(24, generator=generator)
    pre = torch.rand(tokens, 4, generator=generator)
    norm = torch.ones(hidden, dtype=torch.bfloat16)
    args = (fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 20)
    post_fp32 = torch.bmm(old_comb.mT, residual.float())
    post_fp32 += x.float().unsqueeze(1) * old_post
    next_residual = post_fp32.bfloat16()
    # Small fused buckets project the FP32 intermediate; the fallback projects
    # stored BF16 residuals. Both must stay within the upstream 5e-3 tolerance.
    expected = list(
        mhc_pre_delayed_torch(next_residual, *args, pre_mix=pre, x=post_fp32.flatten(1))
    )
    collapsed = expected[2].float()
    expected[2] = (
        collapsed * torch.rsqrt(collapsed.square().mean(-1, keepdim=True) + 1e-6)
    ).bfloat16()
    actual = mhc_fused_post_pre_delayed_tilelang(
        *(item.cuda() for item in (x, residual, old_post, old_comb)),
        *(item.cuda() if isinstance(item, torch.Tensor) else item for item in args),
        pre_mix=pre.cuda(),
        norm_weight=norm.cuda(),
        capture_aux=True,
    )
    torch.testing.assert_close(actual[0].cpu(), next_residual, atol=0.016, rtol=0.01)
    for index in (0, 1, 3):
        torch.testing.assert_close(
            actual[index + 1].cpu(), expected[index], atol=5e-3, rtol=5e-3
        )
    torch.testing.assert_close(actual[3].cpu(), expected[2], atol=0.016, rtol=0.01)
    torch.testing.assert_close(
        actual[5].cpu(), next_residual.mean(1), atol=0.016, rtol=0.01
    )
