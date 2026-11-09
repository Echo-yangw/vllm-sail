# SPDX-License-Identifier: Apache-2.0
"""Exercise the one-argument batch reinitialization call and PPU invalidation."""

from contextlib import contextmanager
from types import SimpleNamespace as NS

import pytest

from tests.support.source import function

PATCH = "vllm_sail/patch/_upstream/runtime.py"


@pytest.fixture
def batch_geometry():
    def make(case, signature=None):
        calls, warmup = [], []
        token_mode = object()
        config = NS(reasoning_config=object())
        old_batch = NS(
            max_model_len=1024,
            logitsprocs=object(),
            logitsprocs_need_output_token_ids=True,
        )

        @contextmanager
        def activate():
            warmup.append("enter")
            yield
            warmup.append("exit")

        def allocate(**kwargs):
            if signature is not None:
                signature.bind(None, **kwargs)
            assert "kernel_block_sizes" not in kwargs
            calls.append(kwargs)
            return NS(**kwargs)

        spec = NS(
            block_size=512 if case == "block-geometry" else 256,
            max_num_blocks_per_req=lambda config, max_len: 8,
        )
        cache = NS(kv_cache_groups=[NS(kv_cache_spec=spec)])
        owner = NS(
            max_model_len=2048 if case == "context-only" else 1024,
            max_encoder_len=0,
            _init_block_sizes=[256],
            _init_max_num_blocks=[8],
            _init_slot_mapping_modes=[token_mode],
            cp_kv_cache_interleave_size=1,
            parallel_config=NS(cp_kv_cache_interleave_size=1),
            input_batch=old_batch,
            vllm_config=config,
            jit_warmup_registry=NS(activate=activate),
            max_num_reqs=128,
            max_num_tokens=1024,
            device=object(),
            model_config=NS(get_vocab_size=lambda: 128),
            num_spec_tokens=7,
            is_pooling_model=False,
            cache_config=NS(use_replayssm=False),
        )
        namespace = {
            "KVCacheSpecKind": NS(ENCODER_ONLY_ATTENTION="encoder", MAMBA="mamba"),
            "SlotMappingMode": NS(NONE=object(), TOKEN_TO_KV_SLOT=token_mode),
            "get_kv_cache_spec_kind": lambda spec: "attention",
            "InputBatch": allocate,
        }
        return owner, cache, namespace, calls, warmup, old_batch

    return make


def check_body(load, batch_geometry, case, expected, signature=None):
    owner, cache, namespace, calls, warmup, old = batch_geometry(case, signature)
    load(namespace)(owner, cache)
    assert len(calls) == expected
    assert warmup == (["enter", "exit"] if expected else [])
    if expected:
        assert owner.input_batch.max_model_len == owner.max_model_len
        assert calls[0]["logitsprocs"] is old.logitsprocs
        assert calls[0]["num_spec_tokens"] == 7
        assert calls[0]["block_sizes"] == [512 if case == "block-geometry" else 256]
    else:
        assert owner.input_batch is old


@pytest.mark.parametrize(
    "case,expected", [("stable", 0), ("context-only", 1), ("block-geometry", 1)]
)
def test_one_argument_call_keeps_ppu_context_invalidation(
    batch_geometry, case, expected
):
    check_body(
        lambda ns: function(PATCH, "may_reinitialize_input_batch", ns),
        batch_geometry,
        case,
        expected,
    )


def test_context_only_invalidation_check_detects_removed_condition(
    batch_geometry, tmp_path
):
    import ast

    from tests.support.source import ROOT

    tree = ast.parse((ROOT / PATCH).read_text())
    body = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "may_reinitialize_input_batch"
    )
    branch = next(node for node in body.body if isinstance(node, ast.If))
    # The fault removes the context comparison while retaining block geometry.
    branch.test.values.pop()
    source = tmp_path / "runtime.py"
    source.write_text(ast.unparse(ast.fix_missing_locations(tree)))
    with pytest.raises(AssertionError):
        check_body(
            lambda ns: function(source, "may_reinitialize_input_batch", ns),
            batch_geometry,
            "context-only",
            1,
        )


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "case,expected", [("stable", 0), ("context-only", 0), ("block-geometry", 1)]
)
def test_current_upstream_call_and_input_batch_abi(
    upstream_source_root, batch_geometry, case, expected
):
    import ast
    import inspect

    from tests.support.source import definition

    source = upstream_source_root / "vllm/v1/worker/gpu_model_runner.py"
    node = definition(source, "GPUModelRunner.may_reinitialize_input_batch")
    node.decorator_list = []
    batch_source = upstream_source_root / "vllm/v1/worker/gpu_input_batch.py"
    constructor = definition(batch_source, "InputBatch.__init__")
    constructor.body = [ast.Pass()]
    constructor.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            constructor,
            node,
        ],
        type_ignores=[],
    )
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    signature = inspect.signature(namespace["__init__"])

    def load(values):
        exec(compile(ast.fix_missing_locations(module), str(source), "exec"), values)
        return values["may_reinitialize_input_batch"]

    check_body(load, batch_geometry, case, expected, signature)
    plugin_expected = 1 if case != "stable" else 0
    check_body(
        lambda ns: function(PATCH, "may_reinitialize_input_batch", ns),
        batch_geometry,
        case,
        plugin_expected,
        signature,
    )
