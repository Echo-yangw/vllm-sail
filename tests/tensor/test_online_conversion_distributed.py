# SPDX-License-Identifier: Apache-2.0
"""Real two-rank collective: input shards must match full-row quantization."""

from datetime import timedelta

import pytest

torch = pytest.importorskip("torch")


def _rank(rank, rendezvous):
    from vllm_sail.model_executor.layers.quantization.online_conversion.quant_utils import (
        quantize_int8,
    )

    torch.distributed.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        torch.manual_seed(50)
        full = torch.randn((129, 256), dtype=torch.bfloat16)
        full[:, 128:] *= 3
        local = full.chunk(2, 1)[rank].contiguous()
        q, scale = quantize_int8(local, tp_group=torch.distributed.group.WORLD)
        expected_q, expected_scale = quantize_int8(full)
        assert torch.equal(scale, expected_scale)
        assert torch.equal(q, expected_q.chunk(2, 1)[rank])
    finally:
        torch.distributed.destroy_process_group()


def test_real_two_rank_full_row_scales(tmp_path):
    if not torch.distributed.is_gloo_available():
        pytest.skip("requires Gloo")
    torch.multiprocessing.spawn(
        _rank, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2
    )


def _axes_rank(rank, rendezvous):
    from vllm_sail.model_executor.layers.quantization.online_conversion.quant_utils import (
        quantize_int8,
    )

    torch.distributed.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=30),
    )
    try:
        # Orthogonal TP and DP/PCP axes, in the same creation order on all ranks.
        groups = []
        for ranks in ([0, 1], [2, 3], [0, 2], [1, 3]):
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                groups.append(group)
        torch.manual_seed(61)
        full = torch.randn((7, 512), dtype=torch.bfloat16)
        full[:, 384:] *= 9
        q, scale = quantize_int8(full.chunk(4, 1)[rank], tp_group=tuple(groups))
        expected_q, expected_scale = quantize_int8(full)
        assert torch.equal(scale, expected_scale)
        assert torch.equal(q, expected_q.chunk(4, 1)[rank])
    finally:
        torch.distributed.destroy_process_group()


def test_real_four_rank_flattened_axes(tmp_path):
    if not torch.distributed.is_gloo_available():
        pytest.skip("requires Gloo")
    torch.multiprocessing.spawn(
        _axes_rank, args=((tmp_path / "axes").as_uri(),), nprocs=4
    )
