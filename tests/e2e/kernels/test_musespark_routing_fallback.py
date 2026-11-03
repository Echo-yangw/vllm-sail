# SPDX-License-Identifier: Apache-2.0
"""Check the retained MuseSpark Triton path against independent Torch math."""

import pytest

pytestmark = [pytest.mark.ppu, pytest.mark.cap89]


@pytest.mark.parametrize("normalization", ["sigmoid", "softmax"])
def test_musespark_ppu_routing_and_changed_input_graph(normalization):
    import torch

    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm.models.muse_spark.common.ops.fused_routing import (
        fused_musespark1x_routing,
    )

    generator = torch.Generator().manual_seed(7321)
    logits = torch.randn(4, 256, generator=generator).cuda()
    bias = (torch.randn(256, generator=generator) * 0.1).cuda()
    mask = torch.tensor([False, True, False, True], device="cuda")
    weights = torch.empty(4, 8, device="cuda")
    ids = torch.empty(4, 8, dtype=torch.int32, device="cuda")

    def run():
        return fused_musespark1x_routing(
            logits,
            bias,
            8,
            0.5,
            1.25,
            normalization,
            1e-6,
            weights,
            ids,
            is_padding=mask,
        )

    def check():
        scaled = logits.cpu() * 1.25
        scores = scaled.sigmoid() if normalization == "sigmoid" else scaled.softmax(-1)
        expected_ids = (scores + bias.cpu()).topk(8, dim=-1).indices
        selected = scores.gather(-1, expected_ids)
        expected_weights = selected * (0.5 / (selected.sum(-1, keepdim=True) + 1e-6))
        padding = mask.cpu()
        expected_ids[padding] = -1
        expected_weights[padding] = 0
        torch.testing.assert_close(ids.cpu().long(), expected_ids, rtol=0, atol=0)
        torch.testing.assert_close(
            weights.cpu(), expected_weights, rtol=2e-5, atol=2e-6
        )

    actual = run()
    assert (
        actual[0].data_ptr() == weights.data_ptr()
        and actual[1].data_ptr() == ids.data_ptr()
    )
    torch.cuda.synchronize()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    logits.mul_(-1.15)
    bias.copy_(bias.flip(0))
    mask.logical_not_()
    graph.replay()
    torch.cuda.synchronize()
    check()
