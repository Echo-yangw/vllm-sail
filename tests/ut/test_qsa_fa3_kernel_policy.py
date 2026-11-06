# SPDX-License-Identifier: Apache-2.0
"""Qwen4Exp QSA kernel choices stay usable without torch, vLLM or a device.

Covers the pure half of the FA3-vs-Triton decision
(:mod:`vllm_sail.models.qwen4_exp.qsa_fa3`) plus a source-level guard on the FA3
call site, whose three load-bearing arguments -- ``num_splits``, ``seqused_k``
and ``qsa_allow_aiu`` -- each fail silently rather than loudly:

* ``num_splits=1`` serialises a row's whole top-k walk onto one tile, which
  measured slower than the Triton kernel on Zhenwu M890.
* A missing ``seqused_k`` makes FA3 drop ``is_causal`` and walk the indexer's
  ``-1`` padding, so the softmax picks up stale scores.
* ``qsa_allow_aiu=True`` lets the AIU bulk load fetch 16 pool-contiguous tokens
  per 16-column table group, ignoring 15 of every 16 entries the indexer picked.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from vllm_sail import envs
from vllm_sail.models.qwen4_exp.qsa_fa3 import (
    QSA_FA3_KV_CACHE_DTYPES,
    qsa_fa3_kv_dtype_reason,
    qsa_fa3_selection_width,
    qsa_fa3_shape_reason,
    resolve_qsa_fa3_kernel,
)

_SOURCE_ROOT = pathlib.Path(__file__).parents[2]
PATCH_PATH = (
    _SOURCE_ROOT / "vllm_sail" / "patch" / "enhancement" / "models" / "qwen4_exp_qsa.py"
)
FORWARD_PATH = (
    _SOURCE_ROOT / "vllm_sail" / "models" / "qwen4_exp" / "qsa_fa3_forward.py"
)


class _Config:
    """Stand-in for ``Qwen4ExpTextConfig``: only the two widths are read."""

    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


# ---------------------------------------------------------------------------
# resolve_qsa_fa3_kernel
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "requested,cuda_reason,active,use_fa3",
    [
        ("auto", None, "cuda", True),
        ("auto", "no wheel", "triton", False),
        ("cuda", None, "cuda", True),
        ("triton", None, "triton", False),
        # A pinned triton never even asks the probes, so a would-be reason is
        # irrelevant and must not be reported.
        ("triton", "no wheel", "triton", False),
    ],
)
def test_kernel_selection(requested, cuda_reason, active, use_fa3):
    choice = resolve_qsa_fa3_kernel(requested, cuda_reason=cuda_reason)
    assert choice.requested == requested
    assert choice.active == active
    assert choice.use_fa3 is use_fa3


def test_auto_fallback_keeps_the_probe_verdict_as_its_reason():
    choice = resolve_qsa_fa3_kernel("auto", cuda_reason="no wheel")
    assert choice.reason == "no wheel"


def test_pinned_triton_names_the_pin_as_its_reason():
    choice = resolve_qsa_fa3_kernel("triton", cuda_reason=None)
    assert choice.reason == "VLLM_SAIL_QSA_KERNEL=triton"


def test_explicit_cuda_fails_early_rather_than_falling_back():
    """A pinned choice is a measurement; a quiet fallback would make it a lie."""
    with pytest.raises(
        ValueError, match=r"VLLM_SAIL_QSA_KERNEL=cuda is not supported: no wheel"
    ):
        resolve_qsa_fa3_kernel("cuda", cuda_reason="no wheel")


def test_unknown_kernel_is_rejected():
    with pytest.raises(ValueError, match="Invalid QSA kernel"):
        resolve_qsa_fa3_kernel("fa3", cuda_reason=None)


def test_env_default_is_auto(monkeypatch):
    monkeypatch.delenv("VLLM_SAIL_QSA_KERNEL", raising=False)
    monkeypatch.delenv("VLLM_PPU_QSA_KERNEL", raising=False)
    assert envs.VLLM_SAIL_QSA_KERNEL == "auto"
    assert envs.QSA_KERNELS == ("auto", "cuda", "triton")


# ---------------------------------------------------------------------------
# shape gate: the head shapes the target model reaches at each TP size
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "num_heads,num_kv_heads",
    [
        (24, 2),  # TP=1
        (12, 1),  # TP=2
        (6, 1),  # TP=4
        (3, 1),  # TP=8
        (32, 1),  # the GQA ceiling itself
    ],
)
def test_qwen4_exp_head_shapes_are_accepted(num_heads, num_kv_heads):
    assert qsa_fa3_shape_reason(256, num_heads, num_kv_heads) is None


@pytest.mark.parametrize("head_size", [64, 128, 192, 512])
def test_other_head_dims_are_rejected(head_size):
    reason = qsa_fa3_shape_reason(head_size, 12, 1)
    assert reason is not None and "head dim 256" in reason


def test_gqa_group_above_32_is_rejected():
    reason = qsa_fa3_shape_reason(256, 66, 2)
    assert reason is not None and "at most 32 query heads" in reason


@pytest.mark.parametrize(
    "num_heads,num_kv_heads",
    [(7, 2), (12, 5), (12, 0), (12, -1)],
)
def test_non_integral_or_empty_gqa_is_rejected(num_heads, num_kv_heads):
    assert qsa_fa3_shape_reason(256, num_heads, num_kv_heads) is not None


# ---------------------------------------------------------------------------
# KV cache dtype gate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", QSA_FA3_KV_CACHE_DTYPES)
def test_bf16_caches_are_accepted(dtype):
    assert qsa_fa3_kv_dtype_reason(dtype) is None


@pytest.mark.parametrize("dtype", ["fp8", "fp8_e4m3", "fp8_e5m2", "uint8"])
def test_quantized_caches_are_rejected_before_the_cpp_abort(dtype):
    """FA3 requires K and V to share Q's BF16 dtype; upstream TORCH_CHECKs it."""
    reason = qsa_fa3_kv_dtype_reason(dtype)
    assert reason is not None and dtype in reason


# ---------------------------------------------------------------------------
# selection width: mirrors QSAIndexer.output_width
# ---------------------------------------------------------------------------
def test_selection_width_matches_the_indexer_formula():
    config = _Config(indexer_budget=2048, indexer_compress_ratio=4)
    assert qsa_fa3_selection_width(config) == 2051


def test_selection_width_of_an_uncompressed_indexer():
    config = _Config(indexer_budget=512, indexer_compress_ratio=1)
    assert qsa_fa3_selection_width(config) == 512


@pytest.mark.parametrize(
    "kwargs",
    [
        {},  # neither field
        {"indexer_budget": 2048},  # no ratio
        {"indexer_compress_ratio": 4},  # no budget
        {"indexer_budget": "2048", "indexer_compress_ratio": 4},  # not an int
        {"indexer_budget": 0, "indexer_compress_ratio": 4},
        {"indexer_budget": 2048, "indexer_compress_ratio": 0},
    ],
)
def test_unusable_configs_size_no_workspace(kwargs):
    assert qsa_fa3_selection_width(_Config(**kwargs)) is None


# ---------------------------------------------------------------------------
# FA3 call-site guard (source level: the patch module needs torch and vLLM)
# ---------------------------------------------------------------------------
def _qsa_call_kwargs() -> dict[str, ast.AST]:
    tree = ast.parse(FORWARD_PATH.read_text(), filename=str(FORWARD_PATH))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "flash_attn_qsa_varlen_func"
    ]
    assert len(calls) == 1, f"expected one FA3 QSA call, got {len(calls)}"
    return {kw.arg: kw.value for kw in calls[0].keywords if kw.arg}


def test_fa3_lets_the_kernel_pick_the_split_count():
    kwargs = _qsa_call_kwargs()
    assert isinstance(kwargs["num_splits"], ast.Constant)
    assert kwargs["num_splits"].value == 0


def test_fa3_is_handed_the_padding_bound():
    kwargs = _qsa_call_kwargs()
    assert isinstance(kwargs["seqused_k"], ast.Name)
    assert kwargs["seqused_k"].id == "seqused_k"


def test_fa3_is_kept_off_the_aiu_bulk_load():
    kwargs = _qsa_call_kwargs()
    assert isinstance(kwargs["qsa_allow_aiu"], ast.Constant)
    assert kwargs["qsa_allow_aiu"].value is False


def test_fa3_selection_is_resolved_once_in_the_constructor():
    """Per-step re-resolution would re-probe the wheel on every forward."""
    tree = ast.parse(PATCH_PATH.read_text(), filename=str(PATCH_PATH))
    functions = {
        node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    for name, expect in (("impl_init", True), ("impl_forward", False)):
        calls = {
            node.func.id
            for node in ast.walk(functions[name])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert ("select_qsa_fa3_kernel" in calls) is expect, name


def test_patch_imports_the_cycle_owner_before_qsa():
    """``qsa`` and ``model`` import each other, so ``model`` must come first.

    ``qsa`` does ``from . import model`` and ``model`` does ``from .qsa import
    Qwen4ExpQSAAttention``. Importing ``qsa`` first raises on the partially
    initialised module, and this patch module is imported at plugin
    registration, before vLLM has loaded any model and therefore before anything
    else has established the working order. A CPU-only test cannot import either
    module, so the ordering is asserted on the source instead.
    """
    tree = ast.parse(PATCH_PATH.read_text(), filename=str(PATCH_PATH))
    imported = [
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "vllm.models.qwen4_exp.nvidia"
        for alias in node.names
    ]
    assert "model" in imported and "qsa" in imported, imported
    assert imported.index("model") < imported.index("qsa"), imported


@pytest.mark.parametrize("path", [PATCH_PATH, FORWARD_PATH], ids=["patch", "forward"])
def test_operator_facing_loggers_live_under_the_vllm_namespace(path):
    """``info_once`` on a ``vllm_sail.*`` logger is dropped in a real server.

    vLLM configures only its own ``vllm`` logger to INFO and leaves the root
    logger at WARNING, so a logger named after its module propagates to root and
    loses every INFO record -- silently, with nothing to distinguish it from a
    code path that never ran. Both of these messages are the operator's only
    evidence of what a server is doing: which QSA kernel was selected, and that
    the FA3 workspace was sized from the live config rather than never reserved.

    Asserted on the source because the drop happens in vLLM's logging setup,
    which a CPU-only test does not run.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    names = [
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "init_logger"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ]
    assert names, f'no init_logger("...") call found in {path.name}'
    assert all(name.startswith("vllm.") for name in names), names
