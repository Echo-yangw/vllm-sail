# SPDX-License-Identifier: Apache-2.0
"""Preserve new prefill row-shard metadata in the selected upstream build."""

from types import SimpleNamespace as NS

import pytest


class Values(list):
    def __getitem__(self, index):
        value = super().__getitem__(index)
        return Values(value) if isinstance(index, slice) else value

    def __floordiv__(self, divisor):
        return Values(value // divisor for value in self)

    def max(self):
        return NS(item=lambda: max(self))


@pytest.fixture
def row_shard_case():
    def make(compress_ratio, shard):
        compressed_calls, shard_calls = [], []
        table, slot_mapping, compressed_slots = object(), object(), object()
        shard_sizes = [1] * 8
        config = NS(attention_config=NS(indexer_tp_shard_min_rows=37))
        common = NS(
            num_reqs=1,
            num_actual_tokens=8,
            query_start_loc=object(),
            query_start_loc_cpu=Values([0, 8]),
            seq_lens=Values([17]),
            seq_lens_cpu_upper_bound=Values([17]),
            slot_mapping=slot_mapping,
            block_table_tensor=table,
            dcp_local_seq_lens=None,
            max_seq_len=17,
        )
        builder = NS(
            compress_ratio=compress_ratio,
            use_pcp=False,
            pcp_world_size=1,
            dcp_world_size=1,
            dcp_rank=0,
            cp_kv_cache_interleave_size=1,
            compressed_cp_interleave_size=1,
            kv_cache_spec=NS(num_states=256),
            compressed_slot_mapping_buffer=compressed_slots,
            decode_threshold=1,
            use_flattening=False,
            supports_varlen=False,
            max_prefill_buffer_size=128,
            indexer_uses_fp4=True,
            vllm_config=config,
            tp_size=8,
            _prefill_split_seq_lens=lambda value: value,
            _split_indexer_prefill_chunks=lambda *args, **kwargs: [
                (slice(0, 1), slice(0, 8))
            ],
        )

        def compressed(*args, **kwargs):
            compressed_calls.append((args, kwargs))
            return compressed_slots

        def eligible(*args):
            assert args == (config, 1, False, 8)
            return shard

        def balanced(*args):
            shard_calls.append(args)
            return shard_sizes

        namespace = {
            "get_compressed_slot_mapping": compressed,
            "torch": NS(
                diff=lambda value: Values([value[1] - value[0]]),
                bfloat16="bf16",
                float32="f32",
            ),
            "envs": NS(VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=1),
            "split_decodes_and_prefills": lambda *args, **kwargs: (0, 1, 0, 8),
            "build_prefill_chunk_metadata": lambda *args, **kwargs: NS(
                block_table=args[7]
            ),
            "tp_prefill_row_sharding_supported": eligible,
            "balanced_prefill_row_shard": balanced,
            "DeepseekV32IndexerMetadata": lambda **kwargs: NS(**kwargs),
            "DeepseekV32IndexerPrefillMetadata": lambda chunks,
            row_shard_sizes=None,
            **kwargs: NS(chunks=chunks, row_shard_sizes=row_shard_sizes, **kwargs),
        }
        return builder, common, namespace, compressed_calls, shard_calls, shard_sizes

    return make


def check_result(body, row_shard_case, compress_ratio, shard):
    builder, common, namespace, compressed_calls, shard_calls, shard_sizes = (
        row_shard_case(compress_ratio, shard)
    )
    assert not hasattr(builder, "kernel_block_size")
    result = body(namespace)(builder, 0, common)
    assert result.block_table is common.block_table_tensor
    assert result.prefill.chunks[0].block_table is common.block_table_tensor
    assert result.prefill.row_shard_sizes is (shard_sizes if shard else None)
    assert len(shard_calls) == int(shard)
    if shard:
        assert shard_calls[0] == (Values([17]), Values([8]), compress_ratio, 8, 37)
    assert len(compressed_calls) == int(compress_ratio > 1)
    if compressed_calls:
        args, kwargs = compressed_calls[0]
        assert args[4] is common.block_table_tensor
        assert (
            result.slot_mapping
            is kwargs["out"]
            is builder.compressed_slot_mapping_buffer
        )
    else:
        assert result.slot_mapping is common.slot_mapping


@pytest.mark.upstream_source
@pytest.mark.parametrize("compress_ratio", [1, 8])
@pytest.mark.parametrize("shard", [False, True])
def test_actual_upstream_producer_has_same_geometry(
    upstream_source_root, row_shard_case, compress_ratio, shard
):
    import ast

    from tests.support.source import definition

    path = upstream_source_root / "vllm/v1/attention/backends/mla/indexer.py"
    node = definition(path, "DeepseekV32IndexerMetadataBuilder.build")
    node.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )

    def load(namespace):
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
        return namespace["build"]

    check_result(load, row_shard_case, compress_ratio, shard)
