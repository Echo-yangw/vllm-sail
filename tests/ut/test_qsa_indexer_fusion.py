# SPDX-License-Identifier: Apache-2.0
"""Mixed-precision checkpoints must not enter the fused QKV+indexer GEMM.

Regression tests for ``ValueError: Detected some but not all shards of
language_model.model.layers.3.self_attn.qkv_proj are quantized``, raised while
building ``qkv_proj`` for ``Qwen3.8-Flash-Next-MoE-Quant-W-MXFP4-A-MXFP4-Dense-
Quant-W-FP8-PerChannel-A-FP8-PerToken`` on vLLM 0.32.

The helpers are pure Python, so they are extracted from source and exercised on
a bare runner without torch, vLLM or a device (see tests/conftest.py).
"""

from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import Any

import pytest

from tests.support.source import ROOT, assert_accepts_upstream_keywords, function

MODULE = "vllm_sail/patch/enhancement/qsa_indexer_fusion.py"

FOUR_SHARDS = ["q_proj", "k_proj", "v_proj", "indexer.index_qk_proj"]
THREE_SHARDS = ["q_proj", "k_proj", "v_proj"]


def _module_namespace() -> dict[str, Any]:
    """Exec the patch module without its ``vllm_sail.patch.utils`` import."""
    path = ROOT / MODULE
    tree = ast.parse(path.read_text())
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.ImportFrom) and node.module == "vllm_sail.patch.utils"
        )
    ]
    namespace: dict[str, Any] = {"__name__": "qsa_indexer_fusion_under_test"}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


_NS = _module_namespace()
block_reason = _NS["indexer_fusion_block_reason"]
trim = _NS["trim_indexer_shard"]

# Layer-3 shapes of the four published Qwen3.8-Flash-Next checkpoints, reduced
# to the entries the detector reads. ``LM`` is the checkpoint's
# ``model.language_model.`` prefix.
LM = "model.language_model.layers.3.self_attn"
MTP = "mtp.layers.0.self_attn"

MIXED_MXFP4 = [
    f"{LM}.q_norm",
    f"{LM}.indexer.index_qk_proj",
    f"{LM}.indexer.q_layernorm",
    f"{MTP}.q_proj",
    f"{MTP}.indexer.index_qk_proj",
]
HOMOGENEOUS_OFFICIAL_FP8 = [
    f"{LM}.q_proj",
    f"{LM}.k_proj",
    f"{LM}.v_proj",
    f"{LM}.o_proj",
    f"{LM}.indexer.index_qk_proj",
    f"{MTP}.q_proj",
    f"{MTP}.indexer.index_qk_proj",
]
HOMOGENEOUS_CHANNELWISE_FP8 = [
    f"{LM}.indexer.q_layernorm",
    f"{LM}.indexer.k_layernorm",
    f"{MTP}.q_proj",
    f"{MTP}.indexer.index_qk_proj",
]


@pytest.mark.parametrize(
    "attr,skipped",
    [
        ("ignored_layers", MIXED_MXFP4),
        ("ignore", MIXED_MXFP4),
        ("modules_to_not_convert", MIXED_MXFP4),
    ],
)
def test_mixed_precision_checkpoint_blocks_the_fusion(attr, skipped):
    config = SimpleNamespace(**{attr: list(skipped)})
    reason = block_reason(config)
    assert reason is not None
    assert f"{LM}.q_proj" in reason
    assert f"{LM}.indexer.index_qk_proj" in reason


@pytest.mark.parametrize(
    "skipped",
    [
        HOMOGENEOUS_OFFICIAL_FP8,
        HOMOGENEOUS_CHANNELWISE_FP8,
        [],
    ],
)
def test_homogeneous_checkpoint_keeps_the_fusion(skipped):
    assert block_reason(SimpleNamespace(ignored_layers=list(skipped))) is None


def test_unquantized_checkpoint_keeps_the_fusion():
    assert block_reason(None) is None
    assert block_reason(SimpleNamespace()) is None


def test_block_reason_is_deterministic():
    config = SimpleNamespace(ignored_layers=list(reversed(MIXED_MXFP4)))
    expected = block_reason(SimpleNamespace(ignored_layers=MIXED_MXFP4))
    assert block_reason(config) == expected


def test_trim_drops_only_the_indexer_shard():
    original = list(FOUR_SHARDS)
    config = SimpleNamespace(
        packed_modules_mapping={
            "qkv_proj": original,
            "gate_up_proj": ["gate_proj", "up_proj"],
        }
    )
    assert trim(config) is True
    assert config.packed_modules_mapping["qkv_proj"] == THREE_SHARDS
    assert config.packed_modules_mapping["gate_up_proj"] == ["gate_proj", "up_proj"]
    # The model class's own list object must not be mutated in place.
    assert original == FOUR_SHARDS
    # Idempotent: every QSA attention layer trims the shared quant config.
    assert trim(config) is False


def test_trim_is_a_noop_without_the_indexer_shard():
    config = SimpleNamespace(packed_modules_mapping={"qkv_proj": list(THREE_SHARDS)})
    assert trim(config) is False
    assert config.packed_modules_mapping["qkv_proj"] == THREE_SHARDS
    assert trim(SimpleNamespace(packed_modules_mapping={})) is False
    assert trim(SimpleNamespace()) is False


def test_fusion_gate_ands_the_upstream_flag_with_the_block():
    namespace: dict[str, Any] = {}
    getter = function(MODULE, "install._fuse_indexer_projection_get", namespace)
    setter = function(MODULE, "install._fuse_indexer_projection_set", namespace)

    class Layer:
        """Stands in for the nn.Module, whose plain attrs land in ``__dict__``."""

        def __init__(self, **attrs):
            self.__dict__.update(attrs)

    layer = Layer()
    setter(layer, True)
    assert getter(layer) is True

    blocked = Layer(_ppu_indexer_fusion_blocked=True)
    setter(blocked, True)
    assert getter(blocked) is False

    # Upstream's own LoRA gate still wins: never fuse when it says no.
    setter(blocked, False)
    assert getter(blocked) is False
    assert getter(Layer()) is False


@pytest.mark.upstream_source
def test_init_replacement_accepts_upstream_keywords(upstream_source_root):
    assert_accepts_upstream_keywords(
        "qsa_indexer_fusion",
        "install._init",
        upstream_source_root,
        "models/qwen4_exp/nvidia/qsa",
        "Qwen4ExpQSAAttention.__init__",
    )


def _upstream_qkv_shards(model_source: str) -> list[str] | None:
    """Read ``Qwen4ExpForCausalLM.packed_modules_mapping["qkv_proj"]`` from AST."""
    for node in ast.parse(model_source).body:
        if not isinstance(node, ast.ClassDef) or node.name != "Qwen4ExpForCausalLM":
            continue
        for child in node.body:
            if not isinstance(child, ast.Assign):
                continue
            names = [t.id for t in child.targets if isinstance(t, ast.Name)]
            if "packed_modules_mapping" not in names:
                continue
            return ast.literal_eval(child.value)["qkv_proj"]
    return None


@pytest.mark.upstream_source
def test_upstream_still_fuses_the_indexer_into_qkv(upstream_source_root):
    """The patch is dead code if upstream drops either half of the seam."""
    qsa = (upstream_source_root / "vllm/models/qwen4_exp/nvidia/qsa.py").read_text()
    assert "self.fuse_indexer_projection = " in qsa
    # With the gate off, the indexer weight must still reach a real module: the
    # shard-3 mapper is what would otherwise swallow it into qkv_proj.
    lines = [line.strip() for line in qsa.splitlines()]
    guarded = [
        i for i, line in enumerate(lines) if line == "if self.fuse_indexer_projection:"
    ]
    assert any(lines[i + 1] == "mapper = WeightsMapper(" for i in guarded), (
        "the indexer shard-3 weight mapper must stay behind the fusion gate"
    )
    model = (upstream_source_root / "vllm/models/qwen4_exp/nvidia/model.py").read_text()
    assert _upstream_qkv_shards(model) == FOUR_SHARDS
