# SPDX-License-Identifier: Apache-2.0
"""Bare-Python regression for a copied constructor using super()."""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest

from tests.support.source import assert_accepts_upstream_keywords
from tests.ut.test_ppu_kernel_capabilities import load_patch
from tests.ut.test_ppu_kernel_capabilities import modules as _modules

modules = _modules

PATCH_PATH = (
    Path(__file__).parents[2] / "vllm_sail/patch/enhancement/attention/mla_indexer.py"
)


@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("max_tokens", [32, 1024])
def test_indexer_reserves_only_decode_capacity(
    modules, patch_utils_module, adaptive, max_tokens
):
    """Run the real constructor through all allocations using strict CPU doubles."""
    from types import SimpleNamespace as NS

    class Parent:
        def __init__(self, config):
            self.vllm_config = config
            self.kv_cache_spec = object()
            self.device = NS(index=0)

    class Builder(Parent):
        def __init__(self, *args, **kwargs):
            pytest.fail("the registered constructor must replace this entry")

        def _split_indexer_prefill_chunks(self, *args, **kwargs):
            pytest.fail("this constructor must not split a prefill")

        def build(self, *args, **kwargs):
            pytest.fail("this constructor must not build step metadata")

    class Tensor:
        def __init__(self, shape):
            self.shape = shape

    config = NS(
        scheduler_config=NS(max_num_seqs=16, max_num_batched_tokens=max_tokens),
        parallel_config=NS(
            decode_context_parallel_size=1,
            tensor_parallel_size=8,
            prefill_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
        ),
        speculative_config=NS(
            num_speculative_tokens=3, enable_adaptive_verification=adaptive
        ),
    )
    torch = modules(
        "torch",
        arange=lambda size, **kwargs: Tensor((size,)),
        zeros=lambda shape, **kwargs: Tensor(shape),
        empty=lambda shape, **kwargs: Tensor(shape),
        tensor=lambda value, **kwargs: Tensor(()),
        int32="int32",
        int64="int64",
        float32="float32",
    )
    platform = NS(is_ppu=lambda: False)
    modules("vllm.platforms", current_platform=platform)
    modules("vllm_sail.patch.utils", patch_value=patch_utils_module.patch_value)
    modules("vllm_sail.utils.deep_gemm")
    provider = modules(
        "vllm.v1.attention.backends.mla.indexer",
        DeepseekV32IndexerMetadataBuilder=Builder,
        MLAAttentionSpec=type("MLAAttentionSpec", (), {}),
        get_max_prefill_buffer_size=lambda config: 128,
        dsa_indexer_uses_fp4=lambda config: False,
        _use_flattening=lambda config: False,
        _supports_varlen_paged_mqa_logits=lambda: True,
        current_platform=platform,
        get_paged_mqa_logits_metadata=lambda *args, **kwargs: None,
        logger=NS(info_once=lambda *args: None),
        num_compute_units=lambda index: 80,
        torch=torch,
    )
    load_patch("vllm_sail/patch/enhancement/attention/mla_indexer.py")
    builder = provider.DeepseekV32IndexerMetadataBuilder(config, block_table_width=29)
    assert builder.expanded_block_table_buffer.shape == (min(16 * 4, max_tokens), 29)
    assert builder.use_adaptive_verification is adaptive
    assert builder.num_speculative_tokens == 3
    assert builder.supports_draft_decode_metadata_update


@pytest.mark.parametrize("subclass", [False, True])
def test_rebased_indexer_init_calls_parent_once(
    monkeypatch, patch_utils_module, subclass
):
    """Execute the real copied body, globals rebasing and patch installation.

    Stop in the parent constructor: reproducing the missing class cell needs
    neither vLLM configuration nor device allocations later in the body.
    The subclass case also catches the recursive super(type(self), self) fix.
    """
    calls = []

    class ParentReached(Exception):
        pass

    class Parent:
        def __init__(self, *args, **kwargs):
            calls.append((self, args, kwargs))
            raise ParentReached

    class DeepseekV32IndexerMetadataBuilder(Parent):
        pass

    class DerivedBuilder(DeepseekV32IndexerMetadataBuilder):
        pass

    target = types.ModuleType("_test_mla_indexer_init_target")
    target.DeepseekV32IndexerMetadataBuilder = DeepseekV32IndexerMetadataBuilder
    monkeypatch.setitem(sys.modules, target.__name__, target)
    tree = ast.parse(PATCH_PATH.read_text())
    selected = ast.Module(
        body=[
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in ("_with_target_globals", "_builder_init_body")
        ],
        type_ignores=[],
    )
    from vllm_sail.patch.bodies import bind_body

    namespace = {"bind_body": bind_body, "_indexer_module": target}
    exec(compile(selected, str(PATCH_PATH), "exec"), namespace)
    replacement = namespace["_with_target_globals"](namespace["_builder_init_body"])
    patch_utils_module.patch(
        target.__name__,
        "DeepseekV32IndexerMetadataBuilder.__init__",
        reason="exercise the copied indexer constructor's parent dispatch",
        affected_versions=">=0.27.0,<0.28.0",
        remove_when="the indexer constructor no longer needs a copied body",
    )(replacement)

    builder_class = DerivedBuilder if subclass else DeepseekV32IndexerMetadataBuilder
    config = object()
    with pytest.raises(ParentReached):
        builder_class(config, block_table_width=29, device="test-device")
    assert len(calls) == 1
    assert type(calls[0][0]) is builder_class
    assert calls[0][1:] == ((config,), {"device": "test-device"})
    assert replacement.__globals__ is target.__dict__


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "local,local_name,upstream,upstream_name",
    [
        (
            "attention/mla_indexer",
            "_builder_init_body",
            "v1/attention/backends/mla/indexer",
            "DeepseekV32IndexerMetadataBuilder.__init__",
        ),
        (
            "attention/mla_indexer",
            "_split_indexer_prefill_chunks_body",
            "v1/attention/backends/mla/indexer",
            "DeepseekV32IndexerMetadataBuilder._split_indexer_prefill_chunks",
        ),
    ],
)
def test_replacements_accept_upstream_keywords(
    upstream_source_root, local, local_name, upstream, upstream_name
):
    assert_accepts_upstream_keywords(
        local, local_name, upstream_source_root, upstream, upstream_name
    )
