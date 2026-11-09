# SPDX-License-Identifier: Apache-2.0
"""Execute the delegating indexer build wrapper and its builder-aware helpers."""

from types import SimpleNamespace as NS
from unittest.mock import ANY

import pytest

from tests.support.source import definition, function

PATCH = "vllm_sail/patch/enhancement/attention/mla_indexer.py"


class Row:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class Span:
    shape = (3, 1)

    @staticmethod
    def dim():
        return 2


def _paged_namespace(modules, *, ppu, builder=None):
    calls = []
    fallback = []

    def paged(context_lens, block_size, num_sms, metadata_extra=None, **kwargs):
        calls.append((context_lens, block_size, num_sms, metadata_extra))
        return "ppu-metadata"

    def upstream(context_lens, block_size, num_sms, indices=None):
        fallback.append((context_lens, block_size, num_sms, indices))
        return "upstream-metadata"

    namespace = {
        "_current_builder": [builder] if builder is not None else [],
        "_upstream_paged_metadata": upstream,
        "_ppu_deep_gemm": NS(get_paged_mqa_logits_metadata=paged),
        "_indexer_module": NS(PPU_FP4_INDEXER_ELEMENT_SIZE=1),
    }
    modules("vllm.platforms", current_platform=NS(is_ppu=lambda: ppu))
    return namespace, calls, fallback


def test_wrapper_delegates_with_the_builder_visible(modules):
    seen = []
    metadata = NS(decode=NS(schedule_metadata="upstream"))
    builder = NS()
    namespace = {"_current_builder": []}

    def upstream(*args):
        seen.append(list(namespace["_current_builder"]))
        return metadata

    namespace["_upstream_build"] = upstream
    result = function(PATCH, "_build_wrapper", namespace)(builder, 0, None)

    assert result is metadata
    assert seen == [[builder]]
    assert namespace["_current_builder"] == []


def test_wrapper_pops_the_builder_when_upstream_raises(modules):
    namespace = {"_current_builder": []}

    def upstream(*args):
        raise RuntimeError("upstream build failed")

    namespace["_upstream_build"] = upstream

    with pytest.raises(RuntimeError, match="upstream build failed"):
        function(PATCH, "_build_wrapper", namespace)(NS(), 0, None)

    assert namespace["_current_builder"] == []


def test_draft_refresh_wrapper_keeps_the_builder_visible():
    seen = []
    builder = NS()
    metadata = NS(decode=None)
    namespace = {"_current_builder": []}

    def upstream(self, value):
        seen.append((self, value, list(namespace["_current_builder"])))
        return "refreshed"

    namespace["_upstream_update_draft_decode"] = upstream
    result = function(PATCH, "_update_draft_decode_wrapper", namespace)(
        builder, metadata
    )

    assert result == "refreshed"
    assert seen == [(builder, metadata, [builder])]
    assert namespace["_current_builder"] == []


def test_paged_metadata_derives_the_fp4_extra_from_the_builder(modules):
    builder = NS(indexer_n_head=64, indexer_q_head_dim=128)
    namespace, calls, fallback = _paged_namespace(modules, ppu=True, builder=builder)

    result = function(PATCH, "_ppu_paged_metadata", namespace)(Span(), 64, 7)

    assert result == "ppu-metadata"
    assert calls == [(ANY, 64, 7, (1, 64, 128, 1))]
    assert not fallback


def test_paged_metadata_keeps_the_standard_layout_without_fp4_heads(modules):
    namespace, calls, fallback = _paged_namespace(modules, ppu=True)
    namespace["_current_builder"] = [NS(indexer_n_head=0, indexer_q_head_dim=128)]

    function(PATCH, "_ppu_paged_metadata", namespace)(Span(), 64, 7)

    assert calls[0][3] is None
    assert not fallback


def test_paged_metadata_rejects_varlen_indices_on_ppu(modules):
    builder = NS(indexer_n_head=64, indexer_q_head_dim=128)
    namespace, _, _ = _paged_namespace(modules, ppu=True, builder=builder)

    with pytest.raises(NotImplementedError, match="fixed-length"):
        function(PATCH, "_ppu_paged_metadata", namespace)(
            Span(), 64, 7, indices=Span()
        )


def test_paged_metadata_delegates_on_non_ppu(modules):
    builder = NS(indexer_n_head=64, indexer_q_head_dim=128)
    namespace, calls, fallback = _paged_namespace(modules, ppu=False, builder=builder)

    result = function(PATCH, "_ppu_paged_metadata", namespace)(
        Span(), 64, 7, indices=None
    )

    assert result == "upstream-metadata"
    assert not calls
    assert fallback == [(ANY, 64, 7, None)]


@pytest.mark.parametrize("fp4,chunks", [(True, 2), (False, 4)])
def test_chunk_budget_follows_the_builder_logits_dtype(fp4, chunks):
    namespace = {
        "_current_builder": [NS(indexer_uses_fp4=fp4)],
        "torch": NS(
            bfloat16="bfloat16",
            float32="float32",
            empty=lambda shape, dtype=None, **kwargs: NS(
                element_size=lambda: 2 if dtype == "bfloat16" else 4
            ),
        ),
    }
    split = function(PATCH, "_split_indexer_prefill_chunks_body", namespace)

    assert len(split([Row(100)], [Row(8)], 10_000, 1024)) == chunks


def test_chunk_budget_falls_back_to_fp32_outside_build():
    namespace = {
        "_current_builder": [],
        "torch": NS(
            bfloat16="bfloat16",
            float32="float32",
            empty=lambda shape, dtype=None, **kwargs: NS(
                element_size=lambda: 2 if dtype == "bfloat16" else 4
            ),
        ),
    }
    split = function(PATCH, "_split_indexer_prefill_chunks_body", namespace)

    assert len(split([Row(100)], [Row(8)], 10_000, 1024)) == 4


@pytest.mark.upstream_source
def test_selected_upstream_producer_preserves_the_same_table(
    upstream_source_root,
):
    import ast

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

    class CpuValues(list):
        def __getitem__(self, index):
            value = super().__getitem__(index)
            return CpuValues(value) if isinstance(index, slice) else value

        def max(self):
            return NS(item=lambda: max(self))

    block_table = object()
    common = NS(
        num_reqs=1,
        num_actual_tokens=1,
        query_start_loc=object(),
        query_start_loc_cpu=CpuValues([0, 1]),
        seq_lens=object(),
        slot_mapping=object(),
        block_table_tensor=block_table,
        dcp_local_seq_lens=None,
        max_seq_len=17,
        seq_lens_cpu_upper_bound=CpuValues([17]),
    )
    builder = NS(
        compress_ratio=1,
        decode_threshold=1,
        use_flattening=False,
        supports_varlen=False,
        use_pcp=False,
        max_prefill_buffer_size=128,
        indexer_uses_fp4=False,
        vllm_config=object(),
        dcp_world_size=1,
        tp_size=1,
        _prefill_split_seq_lens=lambda values: values,
        _split_indexer_prefill_chunks=lambda *args, **kwargs: [],
    )
    namespace = {
        "torch": NS(
            diff=lambda values: [values[1] - values[0]],
            bfloat16="bfloat16",
            float32="float32",
        ),
        "envs": NS(VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=1),
        "tp_prefill_row_sharding_supported": lambda config, dcp, pcp, tp: False,
        "split_decodes_and_prefills": lambda *args, **kwargs: (0, 1, 0, 1),
        "DeepseekV32IndexerMetadata": lambda **kwargs: NS(**kwargs),
        "DeepseekV32IndexerPrefillMetadata": lambda chunks, **kwargs: NS(
            chunks=chunks, **kwargs
        ),
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    result = namespace["build"](builder, 0, common)
    assert result.block_table is block_table
