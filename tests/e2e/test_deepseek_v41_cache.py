# SPDX-License-Identifier: Apache-2.0
"""Check both V4.1 paged-cache records through the installed PPU dispatch."""

import pytest

pytestmark = pytest.mark.ppu


@pytest.mark.parametrize("record_bytes", [528, 584])
@pytest.mark.parametrize("tail_only", [False, True])
def test_cache_gather_after_preloaded_module(record_bytes, tail_only):
    import torch
    from vllm.models.deepseek_v41.common.ops import cache_utils

    import vllm_sail

    # Reproduce model-module import before the general plugin patches the probe.
    vllm_sail.register_out_of_tree()
    assert not cache_utils.has_cutedsl()

    blocks, block_size, offset = 4, 64, 3
    mxfp8 = record_bytes == 528
    fp8_dim, group_size = (512, 32) if mxfp8 else (448, 64)
    scale_dim, data_bytes = (16, 512) if mxfp8 else (8, 576)
    values = torch.arange(blocks * block_size * fp8_dim).reshape(
        blocks, block_size, fp8_dim
    )
    quantized = ((values % 31 - 15).float() / 4).to(torch.float8_e4m3fn)
    encoded = (
        torch.arange(blocks * block_size * scale_dim)
        .reshape(blocks, block_size, scale_dim)
        .remainder(7)
        .add(124)
        .to(torch.uint8)
    )
    scales = torch.exp2(encoded[..., : fp8_dim // group_size].float() - 127)
    reference = torch.zeros(blocks, block_size, 512, dtype=torch.bfloat16)
    reference[..., :fp8_dim] = quantized.float() * scales.repeat_interleave(
        group_size, dim=-1
    )
    data = torch.empty(blocks, block_size, data_bytes, dtype=torch.uint8)
    data[..., :fp8_dim] = quantized.view(torch.uint8)
    if not mxfp8:
        rope = (
            torch.arange(blocks * block_size * 64)
            .reshape(blocks, block_size, 64)
            .remainder(113)
            .float()
            .div(32)
            .to(torch.bfloat16)
        )
        data[..., fp8_dim:] = rope.view(torch.uint8)
        reference[..., fp8_dim:] = rope
    # Each page stores ALL token data, then ALL scales, not interleaved records.
    packed = torch.cat((data.flatten(1), encoded.flatten(1)), dim=1)
    cache = packed.reshape(blocks, block_size, record_bytes).cuda()
    table = torch.tensor([[2, 0], [3, 1]], dtype=torch.int32)
    seq_lens = torch.tensor([70, 93], dtype=torch.int32)
    lengths = torch.tensor([11, 68], dtype=torch.int32) if tail_only else seq_lens
    expected = torch.full((2, 100, 512), -77, dtype=torch.bfloat16)
    for request in range(2):
        count = int(lengths[request])
        positions = torch.arange(int(seq_lens[request]) - count, int(seq_lens[request]))
        expected[request, offset : offset + count] = reference[
            table[request, positions // block_size].long(), positions % block_size
        ]
    # Nontrivial request/token strides, with untouched output sentinels.
    storage = torch.full((2, 200, 512), -77, dtype=torch.bfloat16, device="cuda")
    actual = storage[:, ::2]
    cache_utils.dequantize_and_gather_k_cache(
        actual,
        cache,
        seq_lens.cuda(),
        lengths.cuda() if tail_only else None,
        table.cuda(),
        block_size,
        offset,
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    assert (storage[:, 1::2] == -77).all()
