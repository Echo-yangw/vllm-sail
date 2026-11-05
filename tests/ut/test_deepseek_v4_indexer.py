# SPDX-License-Identifier: Apache-2.0
"""Exercise the V4 indexer patch boundary without torch, vLLM or a device."""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

from tests.support.source import assert_accepts_upstream_keywords

ROOT = Path(__file__).parents[2]
BASE = "vllm.models.deepseek_v4"
Q_PROVIDER = f"{BASE}.common.ops.fused_indexer_q"
K_PROVIDER = f"{BASE}.common.ops.fused_compress_quant_cache"
OPS = "vllm_sail.models.deepseek_v4.ops.indexer"


@pytest.fixture
def installed(monkeypatch, patch_utils_module):
    modules = {}

    def module(name, **attrs):
        if name not in modules:
            result = types.ModuleType(name)
            result.__path__ = []
            modules[name] = result
            monkeypatch.setitem(sys.modules, name, result)
            parent, _, leaf = name.rpartition(".")
            if parent:
                setattr(module(parent), leaf, result)
        modules[name].__dict__.update(attrs)
        return modules[name]

    platform = types.SimpleNamespace(
        ppu=True,
        capability=(8, 0),
        is_ppu=lambda: platform.ppu,
        is_device_capability=lambda cap: platform.capability == cap,
    )
    module("torch", float32="float32", bfloat16="bfloat16")
    module("vllm.platforms", current_platform=platform)
    module(
        "vllm.distributed",
        get_pcp_group=lambda: types.SimpleNamespace(
            all_gather=lambda value, dim: value
        ),
    )

    class MoERunner:
        def _apply_quant_method(self, *args, **kwargs):
            return args, kwargs

    module(
        "vllm.model_executor.layers.fused_moe.runner.moe_runner",
        MoERunner=MoERunner,
    )
    module(
        "vllm.v1.attention.backends.registry",
        AttentionBackendEnum=types.SimpleNamespace(
            FLASHMLA_SPARSE="flashmla",
            FLASHMLA_SPARSE_DSV4="flashmla_dsv4",
        ),
    )
    attention_cls = type("Attention", (), {})
    module(f"{BASE}.nvidia.flashmla", DeepseekV4FlashMLAAttention=attention_cls)
    module(f"{BASE}.nvidia.model", _select_dsv4_attn_cls=lambda _: attention_cls)
    module("vllm_sail.models.deepseek_v4.flashmla", DeepseekV4FlashMLAAttention=object)
    module(
        "vllm_sail.patch.utils",
        **{key: getattr(patch_utils_module, key) for key in ("patch", "PATCH_MARKER")},
    )

    calls = []

    def upstream_q(
        positions,
        index_q,
        index_q_cos_sin_cache,
        index_weights,
        index_weights_softmax_scale,
        index_weights_head_scale,
        use_fp4=False,
        weights_out_dtype="float32",
    ):
        args = (
            positions,
            index_q,
            index_q_cos_sin_cache,
            index_weights,
            index_weights_softmax_scale,
            index_weights_head_scale,
        )
        kwargs = {"use_fp4": use_fp4, "weights_out_dtype": weights_out_dtype}
        calls.append(("upstream_q", args, kwargs))
        if platform.ppu and platform.capability == (8, 0) and not use_fp4:
            raise ValueError("type fp8e4nv not supported in this architecture")
        return "upstream_q"

    def upstream_k(*args, **kwargs):
        calls.append(("upstream_k", args, kwargs))
        return "upstream_k"

    def int8_q(*args, **kwargs):
        calls.append(("int8_q", args, kwargs))
        return "int8_q"

    def int8_k(*args, **kwargs):
        calls.append(("int8_k", args, kwargs))
        return "int8_k"

    module(Q_PROVIDER, fused_indexer_q_rope_quant=upstream_q)
    module(f"{BASE}.common.ops", fused_indexer_q_rope_quant=upstream_q)
    module(f"{BASE}.attention", fused_indexer_q_rope_quant=upstream_q)
    module(K_PROVIDER, compress_norm_rope_store_triton=upstream_k)
    module(f"{BASE}.compressor", compress_norm_rope_store_triton=upstream_k)
    module(
        OPS,
        fused_indexer_q_rope_quant_int8=int8_q,
        compress_indexer_rope_store_int8=int8_k,
    )
    module(
        "vllm_sail.models.deepseek_v4.ops.cache",
        compress_mla_rope_store_fp8=lambda *a: (
            calls.append(("software_fp8", a, {})) or "software_fp8"
        ),
    )
    path = ROOT / "vllm_sail/patch/enhancement/models/deepseek_v4.py"
    spec = importlib.util.spec_from_file_location("_test_v4_indexer_patch", path)
    patched = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patched)
    return platform, modules, calls, patched


@pytest.mark.parametrize(
    "consumer", [Q_PROVIDER, f"{BASE}.common.ops", f"{BASE}.attention"]
)
def test_ppu10_query_bypasses_unsupported_fp8_kernel(installed, consumer):
    _, modules, calls, _ = installed
    assert modules[consumer].fused_indexer_q_rope_quant(*range(6)) == "int8_q"
    assert calls[-1][2] == {"output_buffers": None}


@pytest.mark.parametrize("buffers", [None, (object(), object())])
def test_attention_alias_forwards_int8_output_buffers(installed, buffers):
    _, modules, calls, _ = installed
    modules[f"{BASE}.attention"].fused_indexer_q_rope_quant(
        *range(6), output_buffers=buffers
    )
    assert calls[-1][0] == "int8_q"
    assert calls[-1][2]["output_buffers"] is buffers


@pytest.mark.parametrize(
    "ppu,capability,use_fp4",
    [
        (False, (8, 0), False),
        (True, (8, 9), False),
        (True, (8, 0), True),
    ],
)
def test_other_query_paths_delegate(installed, ppu, capability, use_fp4):
    platform, modules, calls, _ = installed
    platform.ppu, platform.capability = ppu, capability
    assert (
        modules[f"{BASE}.attention"].fused_indexer_q_rope_quant(
            *range(6), use_fp4=use_fp4
        )
        == "upstream_q"
    )
    assert calls[-1][2] == {
        "use_fp4": use_fp4,
        "weights_out_dtype": "float32",
    }


@pytest.mark.parametrize(
    "consumer", [Q_PROVIDER, f"{BASE}.common.ops", f"{BASE}.attention"]
)
@pytest.mark.parametrize(
    "ppu,capability,use_fp4",
    [(False, (8, 0), False), (True, (8, 9), False), (True, (8, 0), True)],
)
def test_delegated_query_rejects_output_buffers(
    installed, consumer, ppu, capability, use_fp4
):
    platform, modules, calls, _ = installed
    platform.ppu, platform.capability = ppu, capability
    query = modules[consumer].fused_indexer_q_rope_quant
    with pytest.raises(ValueError, match="output_buffers.*PPU 1.0 INT8"):
        query(*range(6), use_fp4=use_fp4, output_buffers=(object(), object()))
    assert not calls
    assert query(*range(6), use_fp4=use_fp4, output_buffers=None) == "upstream_q"
    assert calls[-1][1] == tuple(range(6))
    assert calls[-1][2] == {
        "use_fp4": use_fp4,
        "weights_out_dtype": "float32",
    }


def test_fp4_query_forwards_scoring_weights_dtype(installed):
    _, modules, calls, _ = installed
    modules[Q_PROVIDER].fused_indexer_q_rope_quant(
        *range(6), use_fp4=True, weights_out_dtype="bfloat16"
    )
    assert calls[-1][2]["weights_out_dtype"] == "bfloat16"


def test_int8_query_requires_float32_scoring_weights(installed):
    _, modules, calls, _ = installed
    with pytest.raises(AssertionError):
        modules[Q_PROVIDER].fused_indexer_q_rope_quant(
            *range(6), weights_out_dtype="bfloat16"
        )
    assert not calls


@pytest.mark.parametrize("consumer", [K_PROVIDER, f"{BASE}.compressor"])
@pytest.mark.parametrize(
    "ppu,capability,head_dim,use_fp4,expected",
    [
        (True, (8, 0), 128, False, "int8_k"),
        (True, (8, 9), 128, False, "upstream_k"),
        (False, (8, 0), 128, False, "upstream_k"),
        (True, (8, 0), 512, False, "software_fp8"),
        (True, (8, 0), 128, True, "upstream_k"),
    ],
)
def test_compressed_keys_match_the_int8_query_path(
    installed, consumer, ppu, capability, head_dim, use_fp4, expected
):
    platform, modules, calls, _ = installed
    platform.ppu, platform.capability = ppu, capability
    args = list(range(22))
    args[12], args[16] = head_dim, use_fp4
    assert modules[consumer].compress_norm_rope_store_triton(*args) == expected
    assert calls[-1][1] == tuple(args)


def test_rebinding_is_idempotent_and_preserves_other_overrides(installed):
    _, modules, _, patched = installed

    def replacement():
        return None

    modules[f"{BASE}.attention"].fused_indexer_q_rope_quant = replacement
    patched._rebind_indexer_aliases()
    patched._rebind_indexer_aliases()
    assert modules[f"{BASE}.attention"].fused_indexer_q_rope_quant is replacement
    assert modules[f"{BASE}.common.ops"].fused_indexer_q_rope_quant is (
        modules[Q_PROVIDER].fused_indexer_q_rope_quant
    )


def test_consumer_inventory_and_signatures_match_vllm_source(installed):
    """Catch new by-value imports and upstream signature changes independently."""
    _, _, _, patched = installed
    source = os.environ.get("VLLM_SOURCE_ROOT")
    if source:
        vllm_root = Path(source) / "vllm"
    else:
        # The fixture intentionally stubs vLLM; source validation is opt-in here.
        pytest.skip("set VLLM_SOURCE_ROOT to a vLLM checkout for source validation")
    discovered = {name: set() for name in patched._INDEXER_ALIAS_CONSUMERS}
    for path in (vllm_root / "models/deepseek_v4").rglob("*.py"):
        parts = path.relative_to(vllm_root).with_suffix("").parts
        is_package = parts[-1] == "__init__"
        module = ".".join(("vllm", *(parts[:-1] if is_package else parts)))
        package = module if is_package else module.rpartition(".")[0]
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ImportFrom):
                continue
            provider = importlib.util.resolve_name(
                "." * node.level + (node.module or ""), package
            )
            if provider not in (Q_PROVIDER, K_PROVIDER, f"{BASE}.common.ops"):
                continue
            for alias in node.names:
                if alias.name in discovered:
                    assert alias.asname is None, "new renamed consumer needs rebinding"
                    discovered[alias.name].add(module)
    assert discovered == {
        name: set(consumers)
        for name, consumers in patched._INDEXER_ALIAS_CONSUMERS.items()
    }
    patch_tree = ast.parse(
        (ROOT / "vllm_sail/patch/enhancement/models/deepseek_v4.py").read_text()
    )
    for provider, name in (
        (Q_PROVIDER, "fused_indexer_q_rope_quant"),
        (K_PROVIDER, "compress_norm_rope_store_triton"),
    ):
        original = ast.parse(
            (vllm_root.parent / (provider.replace(".", "/") + ".py")).read_text()
        )
        upstream_fn = next(
            n
            for n in original.body
            if isinstance(n, ast.FunctionDef) and n.name == name
        )
        patched_fn = next(
            n
            for n in patch_tree.body
            if isinstance(n, ast.FunctionDef) and n.name == name
        )
        assert [arg.arg for arg in upstream_fn.args.args] == [
            arg.arg for arg in patched_fn.args.args
        ]
        assert [ast.dump(n) for n in upstream_fn.args.defaults] == [
            ast.dump(n) for n in patched_fn.args.defaults
        ]


@pytest.mark.parametrize("buffered", [False, True])
@pytest.mark.parametrize("tokens", [0, 3])
def test_query_wrapper_returns_int8_and_preserves_output_buffers(
    monkeypatch, buffered, tokens
):
    """Run actual allocation/casting code on CPU; the device launch is recorded."""
    torch = pytest.importorskip("torch")
    launch_args = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launch_args.append((grid, args, kwargs))
                args[7].fill_(-3.75)
                args[15].fill_(0.125)

            return launch

    triton_utils = types.ModuleType("vllm.triton_utils")
    triton_utils.triton = types.SimpleNamespace(jit=lambda fn: Kernel())
    triton_utils.tl = types.SimpleNamespace(constexpr=object())
    monkeypatch.setitem(sys.modules, "vllm.triton_utils", triton_utils)
    spec = importlib.util.spec_from_file_location(
        "_test_ppu_indexer_ops", ROOT / "vllm_sail/models/deepseek_v4/ops/indexer.py"
    )
    ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ops)
    q = torch.zeros(tokens, 2, 128, dtype=torch.bfloat16)
    weights = torch.ones(tokens, 2)
    # Non-contiguous token strides reflect slices of reusable scratch buffers.
    buffers = (
        (
            torch.empty(tokens * 2, 2, 128, dtype=torch.int8)[::2],
            torch.empty(tokens * 2, 2, dtype=torch.float32)[::2],
        )
        if buffered
        else None
    )
    result_q, result_w = ops.fused_indexer_q_rope_quant_int8(
        torch.arange(tokens),
        q,
        torch.zeros(8, 64),
        weights,
        0.5,
        0.25,
        output_buffers=buffers,
    )
    assert result_q.dtype == torch.int8 and result_q.shape == q.shape
    assert result_w.dtype == torch.float32 and result_w.shape == weights.shape
    if buffered:
        assert result_q is buffers[0] and result_w is buffers[1]
    if tokens:
        assert launch_args[0][0] == (tokens, 2)
        assert launch_args[0][1][13:15] == (0.5, 0.25)
        assert (result_q == -3).all()  # INT8 casts truncate, not round.
        assert (result_w == 0.125).all()
    else:
        assert not launch_args


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "local,local_name,upstream,upstream_name",
    [
        (
            "attention/sparse_attn_indexer",
            "sparse_attn_indexer_init",
            "model_executor/layers/sparse_attn_indexer",
            "SparseAttnIndexer.__init__",
        )
    ],
)
def test_replacements_accept_upstream_keywords(
    upstream_source_root, local, local_name, upstream, upstream_name
):
    assert_accepts_upstream_keywords(
        local, local_name, upstream_source_root, upstream, upstream_name
    )
