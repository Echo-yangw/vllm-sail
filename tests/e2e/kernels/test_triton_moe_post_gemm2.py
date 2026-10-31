# SPDX-License-Identifier: Apache-2.0
"""Verify PPU expert output ownership against a dense CPU oracle."""

from __future__ import annotations

from math import prod
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.ppu


@pytest.fixture(scope="module")
def runtime():
    import torch
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FUSED_MOE_UNQUANTIZED_CONFIG,
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts

    import vllm_sail
    from vllm_sail.native.extensions import import_kernels

    import_kernels(strict=True)
    vllm_sail.register_out_of_tree()

    return SimpleNamespace(
        torch=torch,
        MoEActivation=MoEActivation,
        FUSED_MOE_UNQUANTIZED_CONFIG=FUSED_MOE_UNQUANTIZED_CONFIG,
        FusedMoEConfig=FusedMoEConfig,
        FusedMoEParallelConfig=FusedMoEParallelConfig,
        RoutingMethodType=RoutingMethodType,
        TritonExperts=TritonExperts,
    )


def make_case(runtime, dtype_name, remote):
    torch = runtime.torch
    dtype = getattr(torch, dtype_name)
    generator = torch.Generator().manual_seed(873)
    tokens, hidden, intermediate, topk = 3, 64, 64, 2
    local_experts = 2 if remote else 4
    x = (torch.randn(tokens, hidden, generator=generator) / 4).to("cuda", dtype)
    w1 = (
        torch.randn(local_experts, 2 * intermediate, hidden, generator=generator) / 8
    ).to("cuda", dtype)
    w2 = (torch.randn(local_experts, hidden, intermediate, generator=generator) / 8).to(
        "cuda", dtype
    )
    ids = torch.tensor([[0, 1], [2, 3], [3, 0]], device="cuda", dtype=torch.int32)
    weights = torch.tensor([[0.3, 0.7], [0.25, 0.75], [0.6, 0.4]], device="cuda")
    expert_map = (
        torch.tensor([0, -1, 1, -1], device="cuda", dtype=torch.int32)
        if remote
        else None
    )
    config = runtime.FusedMoEConfig(
        num_experts=4,
        num_local_experts=local_experts,
        num_logical_experts=4,
        experts_per_token=topk,
        hidden_dim=hidden,
        intermediate_size=intermediate,
        activation=runtime.MoEActivation.SILU,
        device="cuda",
        in_dtype=dtype,
        routing_method=runtime.RoutingMethodType.TopK,
        moe_parallel_config=runtime.FusedMoEParallelConfig.make_no_parallel(),
    )
    experts = runtime.TritonExperts(config, runtime.FUSED_MOE_UNQUANTIZED_CONFIG)
    shapes = experts.workspace_shapes(
        M=tokens,
        N=2 * intermediate,
        K=hidden,
        topk=topk,
        global_num_experts=4,
        local_num_experts=local_experts,
        expert_tokens_meta=None,
        activation=runtime.MoEActivation.SILU,
    )
    arguments = dict(
        output=torch.empty(tokens, hidden, device="cuda", dtype=dtype),
        hidden_states=x,
        w1=w1,
        w2=w2,
        topk_weights=weights,
        topk_ids=ids,
        activation=runtime.MoEActivation.SILU,
        global_num_experts=4,
        expert_map=expert_map,
        a1q_scale=None,
        a2_scale=None,
        workspace13=torch.empty(prod(shapes[0]), device="cuda", dtype=dtype),
        workspace2=torch.empty(prod(shapes[1]), device="cuda", dtype=dtype),
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )
    return experts, arguments


def dense_rows(runtime, arguments):
    torch = runtime.torch
    x, w1, w2 = [
        arguments[name].cpu().float() for name in ("hidden_states", "w1", "w2")
    ]
    ids = arguments["topk_ids"].cpu()
    mapping = arguments["expert_map"]
    mapping = mapping.cpu() if mapping is not None else torch.arange(w1.shape[0])
    rows = torch.zeros(ids.shape[0], ids.shape[1], x.shape[1])
    for token in range(ids.shape[0]):
        for route in range(ids.shape[1]):
            local = int(mapping[int(ids[token, route])])
            if local < 0:
                continue
            projected = (
                (x[token] @ w1[local].T).to(arguments["hidden_states"].dtype).float()
            )
            gate, up = projected.chunk(2)
            activated = (
                (torch.nn.functional.silu(gate) * up)
                .to(arguments["hidden_states"].dtype)
                .float()
            )
            rows[token, route] = (
                (activated @ w2[local].T).to(arguments["hidden_states"].dtype).float()
            )
    return rows


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("hooked", [False, True])
def test_expert_tail_weights_once_masks_remote_and_replays(
    runtime, dtype_name, remote, hooked
):
    torch = runtime.torch
    experts, args = make_case(runtime, dtype_name, remote)
    received = []

    def finalize(unfinalized, output):
        assert unfinalized.route_major
        assert output is args["output"]
        assert unfinalized.expert_weights is args["topk_weights"]
        received.append(unfinalized)
        rows = unfinalized.gemm2_permuted.view(3, 2, 64)
        valid = unfinalized.expanded_idx_to_permuted_idx >= 0
        output.copy_(
            (rows.float() * unfinalized.expert_weights[..., None])
            .masked_fill(~valid[..., None], 0)
            .sum(1)
        )

    if hooked:
        experts.post_gemm2_hook = finalize
    experts.apply(**args)
    torch.cuda.synchronize()
    expected_rows = dense_rows(runtime, args)
    if hooked:
        assert len(received) == 1
        record = received[0]
        expected_map = torch.arange(6, dtype=torch.int32).view(3, 2)
        if remote:
            expected_map[args["expert_map"].cpu()[args["topk_ids"].cpu()] < 0] = -1
        torch.testing.assert_close(
            record.expanded_idx_to_permuted_idx.cpu(), expected_map
        )
        valid = expected_map >= 0
        torch.testing.assert_close(
            record.gemm2_permuted.cpu().float().view(3, 2, 64)[valid],
            expected_rows[valid],
            rtol=0.04,
            atol=0.003,
        )
    expected = (expected_rows * args["topk_weights"].cpu()[..., None]).sum(1)
    torch.testing.assert_close(
        args["output"].cpu().float(), expected, rtol=0.04, atol=0.003
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        experts.apply(**args)
    args["hidden_states"].mul_(0.75)
    args["topk_weights"].copy_(args["topk_weights"].flip(1))
    graph.replay()
    torch.cuda.synchronize()
    expected = (dense_rows(runtime, args) * args["topk_weights"].cpu()[..., None]).sum(
        1
    )
    torch.testing.assert_close(
        args["output"].cpu().float(), expected, rtol=0.04, atol=0.003
    )


@pytest.mark.parametrize("unsupported", ["input-weight", "lora"])
def test_hook_rejects_input_weighting_and_lora_before_gemm(runtime, unsupported):
    experts, args = make_case(runtime, "bfloat16", False)
    experts.post_gemm2_hook = lambda *a: pytest.fail("unsupported hook must not run")
    if unsupported == "input-weight":
        args["apply_router_weight_on_input"] = True
    else:
        experts.set_lora_context(SimpleNamespace())
    with pytest.raises(NotImplementedError, match="neither.*LoRA"):
        experts.apply(**args)


def test_empty_batch_does_not_invoke_the_hook(runtime):
    experts, args = make_case(runtime, "bfloat16", True)
    for key in ("output", "hidden_states", "topk_weights", "topk_ids"):
        args[key] = args[key][:0]
    experts.post_gemm2_hook = lambda *a: pytest.fail("empty batch has no expert tail")
    experts.apply(**args)
    runtime.torch.cuda.synchronize()
    assert args["output"].shape == (0, 64)
