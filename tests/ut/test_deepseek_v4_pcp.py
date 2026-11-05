# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for the PPU DeepSeek V4 PCP integration."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace as NS

import numpy as np
import pytest

from tests.support.source import function


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
