# SPDX-License-Identifier: Apache-2.0
"""Qlean V4.1 metadata contracts, without torch, vLLM or PPU dependencies."""

from types import SimpleNamespace

import pytest

from tests.support.modules import load_patch
from vllm_sail.models.deepseek_v41.config import (
    checkpoint_layer_name,
    normalize_expert_dtype,
)


@pytest.mark.parametrize("kind", ["deepseek_v41", "deepseek_v41_text"])
def test_exported_expert_alias_is_normalized_for_target_and_draft(kind):
    quant = {
        "quant_method": "mxfp4",
        "expert_dtype": "mxfp4",
        "fp8_channelwise_layers": ["layers.0.attn.wq_a"],
    }
    target = SimpleNamespace(
        model_type=kind, expert_dtype="mxfp4", quantization_config=quant
    )
    draft = SimpleNamespace(
        model_type=kind, expert_dtype="mxfp4", quantization_config=quant
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=target),
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=draft)
        ),
    )
    normalize_expert_dtype(config)
    normalize_expert_dtype(config)
    assert target.expert_dtype == draft.expert_dtype == "fp4"
    assert quant["expert_dtype"] == "mxfp4"


@pytest.mark.parametrize(
    "prefix,expected",
    [
        ("language_model.model.layers.14.engram.wkv", "layers.14.engram.wkv"),
        ("model.layers.40.attn.wo_a", "mtp.0.attn.wo_a"),
        ("model.layers.42.ffn.shared_experts.down_proj", "mtp.2.ffn.shared_experts.w2"),
        ("model.main_proj", "mtp.0.main_proj"),
    ],
)
def test_checkpoint_names_preserve_target_and_draft_indices(prefix, expected):
    assert checkpoint_layer_name(prefix, 40) == expected
