# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for the PPU DeepSeek V4 PCP integration."""

from __future__ import annotations

import ast
import sys
import types
from types import SimpleNamespace as NS

import numpy as np
import pytest

from tests.support.source import ROOT, definition, function


class FakeTensor:
    def __init__(self, values, dtype="int32"):
        self.values = np.asarray(values)
        self.dtype = dtype
        self.device = "ppu"

    @property
    def shape(self):
        return self.values.shape

    def numel(self):
        return self.values.size

    def __getitem__(self, key):
        return FakeTensor(self.values[key], self.dtype)

    def zero_(self):
        self.values.fill(0)
        return self

    def copy_(self, other):
        self.values[...] = other.values
        return self

    def clamp_(self, min=None):
        if min is not None:
            np.maximum(self.values, min, out=self.values)
        return self

    def max(self):
        return FakeTensor(np.asarray(self.values.max()), self.dtype)

    def item(self):
        return self.values.item()

    def contiguous(self):
        return self

    def view(self, *shape):
        return FakeTensor(self.values.reshape(shape), self.dtype)

    def unsqueeze(self, dim):
        return FakeTensor(np.expand_dims(self.values, dim), self.dtype)

    def __iadd__(self, value):
        self.values += value
        return self


class FakeTorch:
    int32 = "int32"
    int64 = "int64"
    uint8 = "uint8"

    @staticmethod
    def empty(shape, dtype, device):
        return FakeTensor(np.empty(shape, dtype=np.int64), dtype)

    @staticmethod
    def tensor(values, dtype, device):
        return FakeTensor(values, dtype)


def _pcp_config(**overrides):
    config = NS(
        parallel_config=NS(
            prefill_context_parallel_size=2,
            decode_context_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            nnodes=1,
            use_ubatching=False,
        ),
        speculative_config=None,
        kv_transfer_config=None,
        cache_config=NS(enable_prefix_caching=False, cache_dtype="fp8"),
        kernel_config=NS(moe_backend="ppu_deep_gemm"),
        compilation_config=NS(cudagraph_mode="NONE"),
    )
    for path, value in overrides.items():
        owner, attr = path.rsplit(".", 1) if "." in path else ("", path)
        target = config
        for part in owner.split(".") if owner else ():
            target = getattr(target, part)
        setattr(target, attr, value)
    return config


def _validate_pcp_config():
    return function(
        "vllm_sail/models/deepseek_v4/model.py",
        "_validate_pcp_config",
        {
            "CUDAGraphMode": NS(NONE="NONE"),
            "MEGA_MOE_BACKENDS": {"mega_moe"},
            "VllmConfig": object,
        },
    )


def test_pcp_config_accepts_only_the_initial_supported_scope():
    validate = _validate_pcp_config()
    validate(_pcp_config())

    for override, message in (
        ({"cache_config.enable_prefix_caching": True}, "no prefix caching"),
        ({"parallel_config.decode_context_parallel_size": 2}, "DCP=PP=DP=1"),
        ({"parallel_config.nnodes": 2}, "one node"),
        ({"speculative_config": object()}, "no prefix caching"),
        ({"kernel_config.moe_backend": "mega_moe"}, "no prefix caching"),
    ):
        with pytest.raises(NotImplementedError, match=message):
            validate(_pcp_config(**override))

    with pytest.raises(NotImplementedError, match="cudagraph_mode=NONE"):
        validate(_pcp_config(**{"compilation_config.cudagraph_mode": "FULL"}))
    with pytest.raises(NotImplementedError, match="FP8"):
        validate(_pcp_config(**{"cache_config.cache_dtype": "bfloat16"}))


def test_pcp_config_leaves_single_rank_behavior_unchanged():
    validate = _validate_pcp_config()
    validate(
        _pcp_config(
            **{
                "parallel_config.prefill_context_parallel_size": 1,
                "cache_config.enable_prefix_caching": True,
                "compilation_config.cudagraph_mode": "FULL",
            }
        )
    )


def test_compressor_metadata_gathers_uniform_rank_major_rows(monkeypatch):
    class Group:
        rank_in_group = 1

        def __init__(self):
            self.calls = 0

        def all_gather(self, value, dim):
            self.calls += 1
            if self.calls == 1:
                return FakeTensor([2, 1])
            if self.calls == 2:
                rank0 = np.array([[9, 9], [0, 0]])
                return FakeTensor(np.concatenate((rank0, value.values), axis=0))
            if self.calls == 3:
                rank0 = np.zeros(4, dtype=np.int32)
                return FakeTensor(np.concatenate((rank0, value.values)))
            rank0 = np.array([1, 2, -1, -1])
            return FakeTensor(np.concatenate((rank0, value.values)), "int64")

    group = Group()
    distributed = types.ModuleType("vllm.distributed")
    distributed.get_pcp_group = lambda: group
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed)

    gather = function(
        "vllm_sail/patch/enhancement/models/deepseek_v4_compressor.py",
        "_gather_compressor_metadata",
        {"_compressor": NS(torch=FakeTorch)},
    )
    builder = NS(
        _ppu_pcp_size=2,
        _ppu_pcp_max_reqs=4,
        _ppu_pcp_local_req_indices=FakeTensor(np.empty(4, dtype=np.int32)),
        _ppu_pcp_token_to_req_indices=FakeTensor(np.empty(8, dtype=np.int32)),
        _ppu_pcp_positions=FakeTensor(np.empty(8, dtype=np.int64), "int64"),
        _ppu_pcp_local_block_table=None,
        _ppu_pcp_block_table=None,
        device="ppu",
    )
    metadata = NS(
        slot_mapping=FakeTensor(np.arange(8), "int64"),
        block_table=FakeTensor([[5, 6]]),
        token_to_req_indices=FakeTensor([0, 0]),
        num_decode_tokens=1,
        c128_boundary=True,
    )
    common = NS(
        num_reqs=1,
        num_actual_tokens=2,
        positions=FakeTensor([7, 8, -1, -1], "int64"),
    )

    gather(builder, metadata, common)

    assert metadata.block_table.shape == (4, 2)
    assert metadata.token_to_req_indices.values.tolist() == [0, 0, 0, 0, 2, 2, 2, 2]
    assert metadata.cache_positions.values.tolist() == [1, 2, -1, -1, 7, 8, -1, -1]
    assert metadata.num_decode_tokens is None
    assert metadata.c128_boundary is None


def test_hash_routing_ids_follow_pcp_dispatch_rows():
    calls = []
    group = NS(all_gather=lambda value, dim: "rank-major-ids")

    def upstream(*args):
        calls.append(args)
        return "result"

    apply_quant = function(
        "vllm_sail/patch/enhancement/models/deepseek_v4.py",
        "_apply_quant_method",
        {
            "current_platform": NS(is_ppu=lambda: True),
            "get_pcp_group": lambda: group,
            "_upstream_apply_quant_method": upstream,
        },
    )
    runner = NS(
        moe_config=NS(
            pcp_size=2,
            moe_parallel_config=NS(use_all2all_kernels=False),
        )
    )
    input_ids = FakeTensor([11, 12])

    assert apply_quant(runner, "hidden", "logits", None, input_ids) == "result"
    assert calls[-1][4] == "rank-major-ids"


def test_compressed_slots_gather_the_padded_rank_width():
    gathered = []

    def all_gather(value, dim):
        gathered.append(value.values.copy())
        return FakeTensor(np.concatenate((value.values, value.values)), "int64")

    gather_slots = function(
        "vllm_sail/models/deepseek_v4/flashmla.py",
        "_gather_padded_compressed_slots",
        {
            "torch": NS(Tensor=FakeTensor),
            "get_pcp_group": lambda: NS(all_gather=all_gather),
        },
    )
    buffer = FakeTensor([10, 11, -1, -1, 99, 99], "int64")
    local_slot_mapping = FakeTensor([3, 4, -1, -1], "int64")

    result = gather_slots(buffer, local_slot_mapping)

    assert gathered[0].tolist() == [10, 11, -1, -1]
    assert result.values.tolist() == [10, 11, -1, -1, 10, 11, -1, -1]


def test_swa_pcp_separates_global_cache_write_from_local_query():
    calls = []

    def fused(*args):
        calls.append(args)
        return "query-output"

    method = function(
        "vllm_sail/models/deepseek_v4/flashmla.py",
        "DeepseekV4FlashMLAAttention._fused_qnorm_rope_kv_insert",
        {
            "torch": NS(
                uint8="uint8",
                ops=NS(_C=NS(fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert=fused)),
            ),
            "get_pcp_group": lambda: NS(
                all_gather=lambda value, dim: FakeTensor(
                    np.arange(8).reshape(2, 4), "bf16"
                )
            ),
        },
    )
    cache = FakeTensor(np.empty((3, 32)), "uint8")
    local_slots = FakeTensor([3, -1], "int64")
    metadata = NS(
        slot_mapping=local_slots,
        cache_slot_mapping=FakeTensor([3, -1, 4, -1], "int64"),
        cache_positions=FakeTensor([5, -1, 6, -1], "int64"),
        block_size=256,
    )
    layer = NS(
        use_pcp=True,
        swa_cache_layer=NS(prefix="swa", kv_cache=cache),
        rotary_emb=NS(cos_sin_cache="rope"),
        eps=1e-6,
        padded_heads=32,
    )
    q = FakeTensor(np.empty((2, 32, 4)), "bf16")
    kv = FakeTensor(np.empty((2, 4)), "bf16")
    positions = FakeTensor([5, -1], "int64")

    assert method(layer, q, kv, positions, {"swa": metadata}) == "query-output"
    assert len(calls) == 2
    assert calls[0][6] == 0
    assert calls[0][3] is metadata.cache_slot_mapping
    assert calls[1][3].numel() == 0
    assert calls[1][6] == layer.padded_heads


PCP_SOURCE = "vllm_sail/models/deepseek_v4/pcp.py"


def _shared_metadata_class(group):
    namespace = {
        "torch": FakeTorch,
        "get_pcp_group": lambda: group,
        "ModelSpecificAttnMetadata": object,
        "CUDAGraphMode": NS(NONE="NONE"),
    }
    node = definition(ROOT / PCP_SOURCE, "PCPSharedMetadata")
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), PCP_SOURCE, "exec"), namespace)
    return namespace[node.name], namespace


@pytest.mark.parametrize("num_tokens", [0, 2, 4])
def test_shared_layout_reuses_collectives_but_keeps_group_tables(
    num_tokens, monkeypatch
):
    class Group:
        rank_in_group = 1
        world_size = 2

        def __init__(self):
            self.calls = []

        def all_gather(self, value, dim):
            self.calls.append((value.shape, value.dtype))
            return FakeTensor(np.concatenate((value.values, value.values)), value.dtype)

    group = Group()
    shared_cls, _ = _shared_metadata_class(group)
    distributed = types.ModuleType("vllm.distributed")
    distributed.get_pcp_group = lambda: group
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed)
    gather = function(
        "vllm_sail/patch/enhancement/models/deepseek_v4_compressor.py",
        "_gather_compressor_metadata",
        {"_compressor": NS(torch=FakeTorch)},
    )
    shared = shared_cls()
    common = NS(
        num_reqs=1,
        num_actual_tokens=num_tokens,
        positions=FakeTensor(
            [11 + i if i < num_tokens else -1 for i in range(4)], "int64"
        ),
    )
    positions = shared.get_cache_positions(common)
    builders, outputs = [], []
    for table_values in ([5, 6], [21, 22]):
        builder = NS(
            _ppu_pcp_size=2,
            _ppu_pcp_max_reqs=4,
            _ppu_pcp_local_req_indices=FakeTensor(np.empty(4, dtype=np.int32)),
            _ppu_pcp_token_to_req_indices=FakeTensor(np.empty(8, dtype=np.int32)),
            _ppu_pcp_positions=FakeTensor(np.empty(8, dtype=np.int64), "int64"),
            _ppu_pcp_local_block_table=None,
            _ppu_pcp_block_table=None,
            device="ppu",
        )
        metadata = NS(
            slot_mapping=FakeTensor(np.arange(8), "int64"),
            block_table=FakeTensor([table_values]),
            token_to_req_indices=FakeTensor(np.zeros(num_tokens, dtype=np.int32)),
        )
        gather(builder, metadata, common, shared)
        builders.append(builder)
        outputs.append(metadata)
    assert len(group.calls) == 5
    assert outputs[0].cache_positions is outputs[1].cache_positions is positions
    assert outputs[0].token_to_req_indices is outputs[1].token_to_req_indices
    assert outputs[0].block_table.values.tolist() == [[5, 6], [5, 6]]
    assert outputs[1].block_table.values.tolist() == [[21, 22], [21, 22]]
    assert outputs[0].block_table is not outputs[1].block_table
    snapshots = [
        (
            item.cache_positions.values.copy(),
            item.token_to_req_indices.values.copy(),
            item.block_table.values.copy(),
        )
        for item in outputs
    ]
    for builder, output, snapshot in zip(builders, outputs, snapshots, strict=True):
        reference = NS(
            slot_mapping=FakeTensor(np.arange(8), "int64"),
            block_table=output.block_table[:1],
            token_to_req_indices=FakeTensor(np.zeros(num_tokens, dtype=np.int32)),
        )
        gather(builder, reference, common)
        np.testing.assert_array_equal(reference.cache_positions.values, snapshot[0])
        np.testing.assert_array_equal(
            reference.token_to_req_indices.values, snapshot[1]
        )
        np.testing.assert_array_equal(reference.block_table.values, snapshot[2])
    assert len(group.calls) == 13
    common.positions.values[:] = [101, 102, -1, -1]
    following = shared_cls().get_cache_positions(common)
    assert len(group.calls) == 14
    assert following.values.tolist() == [101, 102, -1, -1, 101, 102, -1, -1]
    assert following is not positions


def test_model_state_creates_a_new_shared_layout_for_every_forward():
    shared_cls, namespace = _shared_metadata_class(NS())

    class ParentState:
        def prepare_attn(self, *args, **kwargs):
            return kwargs["model_specific_attn_metadata"]

    namespace["DefaultModelState"] = ParentState
    node = definition(ROOT / PCP_SOURCE, "DeepseekV4PCPModelState")
    tree = ast.Module(body=[node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), PCP_SOURCE, "exec"), namespace)
    state = namespace[node.name]()
    batch = NS(pcp=object())
    first = state.prepare_attn(batch, "NONE", (), None, [], None)
    second = state.prepare_attn(batch, "NONE", (), None, [], None)
    assert isinstance(first, shared_cls)
    assert isinstance(second, shared_cls)
    assert first is not second
    assert (
        state.prepare_attn(batch, "NONE", (), None, [], None, for_capture=True) is None
    )
    assert state.prepare_attn(NS(pcp=None), "NONE", (), None, [], None) is None
    assert state.prepare_attn(batch, "FULL", (), None, [], None) is None


def test_shared_metadata_with_real_common_metadata_and_builder(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    from vllm.config import ParallelConfig, SchedulerConfig, VllmConfig
    from vllm.models.deepseek_v4.compressor import CompressorMetadataBuilder
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.kv_cache_interface import SlidingWindowMLASpec

    from vllm_sail.models.deepseek_v4 import pcp

    calls = []

    def all_gather(value, dim):
        calls.append(value.clone())
        return torch.cat((value, value), dim=dim)

    monkeypatch.setattr(
        pcp,
        "get_pcp_group",
        lambda: NS(
            world_size=2,
            rank_in_group=1,
            all_gather=all_gather,
        ),
    )
    query_start = torch.tensor([0, 2], dtype=torch.int32)
    cm = CommonAttentionMetadata(
        query_start_loc=query_start,
        query_start_loc_cpu=query_start,
        seq_lens=torch.tensor([12], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=2,
        max_query_len=2,
        max_seq_len=12,
        block_table_tensor=torch.tensor([[5, 6]], dtype=torch.int32),
        slot_mapping=torch.tensor([0, 1, -1, -1] * 2, dtype=torch.int64),
        positions=torch.tensor([10, 11, 0, 0], dtype=torch.int64),
    )
    config = VllmConfig(
        parallel_config=ParallelConfig(
            prefill_context_parallel_size=2, distributed_executor_backend="uni"
        ),
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=16,
            max_num_seqs=4,
            max_model_len=64,
            is_encoder_decoder=False,
            enable_chunked_prefill=True,
        ),
    )
    spec = SlidingWindowMLASpec(
        block_size=4,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.float32,
        sliding_window=8,
    )
    builder = CompressorMetadataBuilder(spec, ["state"], config, torch.device("cpu"))
    shared = pcp.PCPSharedMetadata()
    assert shared.get_extra_attn_kwargs(builder, 1) == {"pcp_shared_metadata": shared}
    assert shared.get_extra_attn_kwargs(object(), 1) == {}
    first = shared.get_cache_positions(cm)
    assert shared.get_cache_positions(cm) is first
    assert shared.get_gather_num_reqs(cm, "cpu") == 1
    tokens = shared.get_token_to_req_indices(
        cm,
        4,
        torch.zeros(2, dtype=torch.int32),
        torch.empty(4, dtype=torch.int32),
        torch.empty(8, dtype=torch.int32),
    )
    assert len(calls) == 3
    assert tokens.tolist() == [1] * 8
    torch.testing.assert_close(first, torch.tensor([10, 11, 0, 0] * 2), rtol=0, atol=0)
    cm.positions.copy_(torch.tensor([30, 40, 41, 0]))
    query_start_next = torch.tensor([0, 1, 3], dtype=torch.int32)
    following_cm = cm.replace(
        num_reqs=2,
        num_actual_tokens=3,
        query_start_loc=query_start_next,
        query_start_loc_cpu=query_start_next,
        seq_lens=torch.tensor([31, 42], dtype=torch.int32),
        block_table_tensor=torch.tensor([[15, 16], [17, 18]], dtype=torch.int32),
    )
    following_shared = pcp.PCPSharedMetadata()
    second = following_shared.get_cache_positions(following_cm)
    assert following_shared.get_gather_num_reqs(following_cm, "cpu") == 2
    following_tokens = following_shared.get_token_to_req_indices(
        following_cm,
        4,
        torch.tensor([0, 1, 1], dtype=torch.int32),
        torch.empty(4, dtype=torch.int32),
        torch.empty(8, dtype=torch.int32),
    )
    assert len(calls) == 6
    assert first.tolist() == [10, 11, 0, 0] * 2
    assert second.tolist() == [30, 40, 41, 0] * 2
    assert following_tokens.tolist() == [2, 3, 3, 2] * 2


def test_shared_metadata_routes_through_real_model_state_and_attention_groups(
    tmp_path, monkeypatch
):
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from transformers import GPT2Config
    from vllm.config import (
        CompilationConfig,
        CUDAGraphMode,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
    )
    from vllm.models.deepseek_v4.compressor import (
        CompressorBackend,
        CompressorMetadataBuilder,
    )
    from vllm.v1.attention.backend import AttentionMetadataBuilder
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        SlidingWindowMLASpec,
    )
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
    from vllm.v1.worker.gpu.model_states import resolve_model_state_cls
    from vllm.v1.worker.gpu.pcp_manager import PCPBatchMetadata
    from vllm.v1.worker.utils import AttentionGroup

    from vllm_sail.models.deepseek_v4 import pcp
    from vllm_sail.models.deepseek_v4.flashmla import (
        PPUDeepseekSparseSWAFlashMLABackend,
        PPUDeepseekSparseSWAMetadataBuilder,
    )

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setattr(
        pcp,
        "get_pcp_group",
        lambda: NS(
            world_size=2,
            rank_in_group=1,
            all_gather=lambda value, dim: torch.cat((value, value), dim=dim),
        ),
    )
    model_path = tmp_path / "routing-model"
    GPT2Config(
        n_layer=1,
        n_head=1,
        n_embd=16,
        n_positions=64,
        architectures=["GPT2LMHeadModel"],
    ).save_pretrained(model_path)
    config = VllmConfig(
        model_config=ModelConfig(
            model=str(model_path), max_model_len=64, dtype="float32"
        ),
        parallel_config=ParallelConfig(
            prefill_context_parallel_size=2, distributed_executor_backend="uni"
        ),
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=16,
            max_num_seqs=4,
            max_model_len=64,
            is_encoder_decoder=False,
            enable_chunked_prefill=True,
        ),
        compilation_config=CompilationConfig(cudagraph_mode=CUDAGraphMode.NONE),
    )

    class HookModel(torch.nn.Module):
        get_model_state_cls = function(
            "vllm_sail/models/deepseek_v4/model.py",
            "DeepseekV4ForCausalLM.get_model_state_cls",
            {},
        )

        def __init__(self):
            super().__init__()
            self.model = torch.nn.Identity()
            self.model.parallel_config = config.parallel_config

    def route(builder, cm, shared):
        local_tokens = cm.positions.numel()
        local_indices = torch.repeat_interleave(
            torch.arange(cm.num_reqs, dtype=torch.int32), cm.naive_query_lens()
        )
        mapping = shared.get_token_to_req_indices(
            cm,
            local_tokens,
            local_indices,
            torch.empty(local_tokens, dtype=torch.int32),
            torch.empty(local_tokens * 2, dtype=torch.int32),
        )
        return shared, shared.get_cache_positions(cm), mapping

    class RoutingSWA(PPUDeepseekSparseSWAMetadataBuilder):
        def __init__(self, *args):
            AttentionMetadataBuilder.__init__(self, *args)

        def build(self, common_prefix_len, cm, pcp_shared_metadata=None):
            return route(self, cm, pcp_shared_metadata)

    class RoutingCompressor(CompressorMetadataBuilder):
        def __init__(self, *args):
            AttentionMetadataBuilder.__init__(self, *args)

        def build(self, common_prefix_len, cm, pcp_shared_metadata=None):
            return route(self, cm, pcp_shared_metadata)

    model = HookModel()
    state_type = resolve_model_state_cls(config, model)
    assert state_type is pcp.DeepseekV4PCPModelState
    state = state_type(config, model, None, torch.device("cpu"))
    spec = SlidingWindowMLASpec(
        block_size=4,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.float32,
        sliding_window=8,
    )
    groups = [
        [
            AttentionGroup(
                PPUDeepseekSparseSWAFlashMLABackend,
                ["swa"],
                spec,
                0,
                metadata_builders=[
                    RoutingSWA(spec, ["swa"], config, torch.device("cpu"))
                ],
            )
        ],
        [
            AttentionGroup(
                CompressorBackend,
                ["compressor"],
                spec,
                1,
                metadata_builders=[
                    RoutingCompressor(spec, ["compressor"], config, torch.device("cpu"))
                ],
            )
        ],
    ]
    kv_config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["swa"], spec),
            KVCacheGroupSpec(["compressor"], spec),
        ],
    )
    buffers = InputBuffers(4, 16, torch.device("cpu"))
    outputs = []
    for num_reqs, num_tokens in [(1, 2), (2, 3)]:
        batch = InputBatch.make_dummy(num_reqs, num_tokens, buffers, is_padding=False)
        batch.pcp = PCPBatchMetadata(
            global_num_tokens=num_tokens * 2,
            global_query_lens_np=batch.num_scheduled_tokens * 2,
            global_seq_lens_cpu_upper_bound_np=batch.seq_lens_cpu_upper_bound.numpy()
            * 2,
        )
        batch.positions.copy_(torch.arange(num_tokens, dtype=torch.int64))
        tables = (
            torch.full((num_reqs, 2), 5, dtype=torch.int32),
            torch.full((num_reqs, 2), 15, dtype=torch.int32),
        )
        slots = torch.arange(num_tokens * 2, dtype=torch.int64).repeat(2, 1)
        result = state.prepare_attn(
            batch, CUDAGraphMode.NONE, tables, slots, groups, kv_config
        )
        assert result["swa"][0] is result["compressor"][0]
        assert result["swa"][1] is result["compressor"][1]
        outputs.append(result)
    assert outputs[0]["swa"][0] is not outputs[1]["swa"][0]
    assert outputs[0]["swa"][2].tolist() == [1, 1, 1, 1]
    assert outputs[1]["swa"][2].tolist() == [2, 3, 3, 2, 3, 3]
