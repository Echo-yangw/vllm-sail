# SPDX-License-Identifier: Apache-2.0
"""Check batched MXFP4 workspace bounds without importing device libraries."""

import ast
from types import SimpleNamespace as NS

import pytest

from tests.support.source import ROOT, definition, function

SOURCE = "vllm_sail/model_executor/layers/fused_moe/experts/batched_deep_gemm_moe.py"


class Counts:
    def __init__(self, values):
        self.values = values
        self.cpu_copies = 0

    def cpu(self):
        self.cpu_copies += 1
        return self

    def numel(self):
        return len(self.values)

    def min(self):
        return NS(item=lambda: min(self.values))

    def max(self):
        return NS(item=lambda: max(self.values))


def workspace_shapes(M, counts, capturing=False, compact=True, hidden_dim=2048):
    namespace = {
        "torch": NS(cuda=NS(is_current_stream_capturing=lambda: capturing)),
        "cdiv": lambda value, divisor: (value + divisor - 1) // divisor,
        "round_up": lambda value, alignment: (
            (value + alignment - 1) // alignment * alignment
        ),
    }
    function(SOURCE, "_mxfp4_workspace_tokens", namespace)
    build = function(
        SOURCE, "PPUBatchedDeepGemmExpertsMXFP4.workspace_shapes", namespace
    )
    experts = NS(
        _compact_workspace=compact,
        num_dispatchers=4,
        max_num_tokens=8192,
        adjust_N_for_activation=lambda N, activation: N // 2,
    )
    metadata = NS(expert_num_tokens=counts, expert_num_tokens_cpu=None)
    return build(experts, M, 4096, hidden_dim, 6, 256, 64, metadata, "silu")


def test_workspace_uses_valid_expert_rows_but_keeps_combine_capacity():
    workspace13, workspace2, output = workspace_shapes(32768, Counts([0, 129]))
    assert workspace13 == (64, 256, 4096)
    assert workspace2 == (64, 256, 2048)
    assert output == (64, 32768, 2048)
    old_bytes = 64 * 32768 * (4096 + 2048) * 2
    new_bytes = (max(64 * 256 * 4096, 64 * 32768 * 2048) + 64 * 256 * 2048) * 2
    assert old_bytes == 24 * 1024**3
    assert new_bytes < 9 * 1024**3


def test_workspace_stays_bounded_when_all_tokens_hit_one_expert():
    workspace13, workspace2, output = workspace_shapes(32768, Counts([32768, 0]))
    assert workspace13 == (64, 2048, 4096)
    assert workspace2 == (64, 2048, 2048)
    assert output == (64, 32768, 2048)


def test_unpacked_output_and_disjoint_chunk_scratch_stay_below_old_peak():
    shapes = workspace_shapes(32768, Counts([32768, 0]), hidden_dim=4096)
    workspace13, workspace2, output = shapes
    assert workspace13 == (64, 2048, 4096)
    assert workspace2 == (64, 2048, 2048)
    assert output == (64, 32768, 4096)
    old_bytes = 64 * 32768 * (4096 + 2048) * 2
    fixed_output = 64 * 32768 * 4096 * 2
    chunk_scratch = 64 * 2048 * 4096 * 2
    activation_scratch = 64 * 2048 * 2048 * 2
    overlapping_chunk_outputs = 2 * 64 * 2048 * 4096 * 2
    visible_allocations = (
        fixed_output + chunk_scratch + activation_scratch + overlapping_chunk_outputs
    )
    assert visible_allocations == int(19.5 * 1024**3)
    assert visible_allocations < old_bytes


@pytest.mark.parametrize("counts", [[0, 0], []])
def test_workspace_handles_no_received_tokens(counts):
    workspace13, workspace2, output = workspace_shapes(32768, Counts(counts))
    assert workspace13[1] == workspace2[1] == 128
    assert output[1] == 32768


def test_graph_capture_keeps_full_capacity_without_reading_device_counts():
    counts = Counts([5, 129])
    shapes = workspace_shapes(32768, counts, capturing=True)
    assert shapes == (
        (64, 32768, 4096),
        (64, 32768, 2048),
        (64, 32768, 2048),
    )
    assert counts.cpu_copies == 0


def test_graph_profile_and_capture_reserve_the_same_full_capacity():
    counts = Counts([5, 129])
    profile = workspace_shapes(32768, counts, compact=False)
    capture = workspace_shapes(32768, counts, capturing=True, compact=False)
    assert profile == capture
    assert profile[0][1] == profile[1][1] == profile[2][1] == 32768
    assert counts.cpu_copies == 0


@pytest.mark.parametrize("counts", [[-1, 2], [32769, 0]])
def test_invalid_counts_do_not_silently_truncate_tokens(counts):
    with pytest.raises(AssertionError):
        workspace_shapes(32768, Counts(counts))


def test_workspace_never_expands_past_a_small_receive_capacity():
    workspace13, workspace2, output = workspace_shapes(19, Counts([19, 0]))
    assert workspace13[1] == workspace2[1] == output[1] == 19


def test_workspace_reads_device_counts_only_once_per_metadata():
    namespace = {
        "torch": NS(cuda=NS(is_current_stream_capturing=lambda: False)),
        "round_up": lambda value, alignment: (
            (value + alignment - 1) // alignment * alignment
        ),
    }
    bound = function(SOURCE, "_mxfp4_workspace_tokens", namespace)
    counts = Counts([5, 129])
    metadata = NS(expert_num_tokens=counts, expert_num_tokens_cpu=None)
    assert bound(32768, 2048, metadata) == 256
    assert bound(32768, 2048, metadata) == 256
    assert counts.cpu_copies == 1


@pytest.mark.parametrize("counts", [[0, 0, 0, 0], [1, 17, 129, 0], [513, 0, 0, 1]])
def test_tiled_experts_preserve_rows_when_output_aliases_workspace(counts, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    from vllm.config import (
        CompilationConfig,
        CUDAGraphMode,
        VllmConfig,
        get_current_vllm_config,
        set_current_vllm_config,
    )
    from vllm.logger import init_logger
    from vllm.model_executor.layers.fused_moe import modular_kernel as mk
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        FusedMoEQuantConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.utils import _resize_cache

    scale_inputs = []

    def gemm(lhs, rhs, bias, output, valid_counts, expected_m):
        values = lhs[0]
        if rhs[0].shape[-1] * 2 == values.shape[-1]:
            weight = rhs[0].repeat_interleave(2, dim=-1)
            scale_inputs.append(lhs[1].clone())
        else:
            weight = rhs[0]
        result = torch.bmm(values, weight.transpose(1, 2))
        for expert, count in enumerate(valid_counts.tolist()):
            output[expert, :count].copy_(result[expert, :count])

    def activate(values, valid_counts, **kwargs):
        gate, up = values.chunk(2, dim=-1)
        result = torch.nn.functional.silu(gate) * up
        return result, torch.zeros(
            (values.shape[0], 1, values.shape[1]), dtype=torch.uint16
        )

    namespace = {
        "__name__": "batched_workspace_oracle",
        "torch": torch,
        "mk": mk,
        "CUDAGraphMode": CUDAGraphMode,
        "get_current_vllm_config": get_current_vllm_config,
        "logger": init_logger(__name__),
        "is_forward_context_available": lambda: False,
        "round_up": lambda n, a: (n + a - 1) // a * a,
        "cdiv": lambda n, d: (n + d - 1) // d,
        "_resize_cache": _resize_cache,
        "MoEActivation": MoEActivation,
        "fp4_m_grouped_gemm_nt_masked": gemm,
        "silu_mul_mxfp4_quant_masked": activate,
    }
    function(SOURCE, "_mxfp4_workspace_tokens", namespace)
    cls = definition(ROOT / SOURCE, "PPUBatchedDeepGemmExpertsMXFP4")
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), SOURCE, "exec"), namespace)
    parallel = FusedMoEParallelConfig(
        tp_size=1,
        pcp_size=4,
        dp_size=1,
        ep_size=4,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=1,
        use_ep=True,
        all2all_backend="deepep_low_latency",
        enable_eplb=False,
    )
    config = FusedMoEConfig(
        num_experts=16,
        experts_per_token=2,
        hidden_dim=128,
        intermediate_size=128,
        num_local_experts=4,
        num_logical_experts=16,
        activation=MoEActivation.SILU,
        device="cpu",
        routing_method=RoutingMethodType.TopK,
        moe_parallel_config=parallel,
        in_dtype=torch.float32,
    )
    quant = FusedMoEQuantConfig.make(quant_dtype="mxfp4")
    runtime_config = VllmConfig(
        compilation_config=CompilationConfig(cudagraph_mode=CUDAGraphMode.NONE)
    )
    with set_current_vllm_config(runtime_config):
        expert = namespace[cls.name](config, quant, 512, 4)
    generator = torch.Generator().manual_seed(37)
    hidden = torch.randn((4, 1024, 128), generator=generator) * 0.01
    w1 = torch.randn((4, 128, 64), generator=generator) * 0.01
    w2 = torch.randn((4, 128, 64), generator=generator) * 0.01
    scales = torch.arange(4 * 2 * 1024, dtype=torch.int32).to(torch.uint16)
    scales = scales.view(4, 2, 1024).transpose(1, 2)
    metadata = mk.ExpertTokensMetadata(torch.tensor(counts, dtype=torch.int32), None)
    shapes = expert.workspace_shapes(
        1024, 128, 128, 2, 16, 4, metadata, MoEActivation.SILU
    )
    backing = torch.full((4 * 1024 * 128,), -123.0)
    output = backing.view(4, 1024, 128)
    workspace13 = _resize_cache(backing, shapes[0])
    workspace2 = torch.empty(shapes[1])
    assert output.data_ptr() == workspace13.data_ptr()
    expert.apply(
        output,
        hidden,
        w1,
        w2,
        torch.ones((2, 2)),
        torch.zeros((2, 2), dtype=torch.int64),
        MoEActivation.SILU,
        16,
        None,
        scales,
        None,
        workspace13,
        workspace2,
        metadata,
        False,
    )
    projected = torch.bmm(hidden, w1.repeat_interleave(2, dim=-1).transpose(1, 2))
    gate, up = projected.chunk(2, dim=-1)
    golden = torch.bmm(torch.nn.functional.silu(gate) * up, w2.transpose(1, 2))
    for index, count in enumerate(counts):
        torch.testing.assert_close(output[index, :count], golden[index, :count])
    for index, actual in enumerate(scale_inputs):
        start = index * shapes[0][1]
        end = start + actual.shape[1]
        torch.testing.assert_close(
            actual.to(torch.int32), scales[:, start:end].to(torch.int32), rtol=0, atol=0
        )
    assert output.shape == (4, 1024, 128)
