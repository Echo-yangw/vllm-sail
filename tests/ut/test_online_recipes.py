# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy
from types import SimpleNamespace

import pytest

from vllm_sail.model_executor.layers.quantization.online_conversion.recipe import (
    select_recipe,
    target_exclusions,
    validate_recipe,
)


@pytest.mark.parametrize("source", ["fp8", "bf16"])
def test_model_specific_recipe_precedes_fallback(source):
    recipe = select_recipe(
        "/models/Qwen3-Next-80B-A3B-Instruct",
        SimpleNamespace(model_type="qwen3_next"),
        source,
    )
    assert recipe["id"] == "qwen3_next_80b_a3b_instruct"
    assert "re:.*linear_attn.*" in recipe["ignored_layers"]


def test_distinct_source_fallbacks():
    assert select_recipe("unknown", None, "fp8") is None
    assert select_recipe("unknown", None, "bf16")["id"] == "_default"


def test_explicit_recipe_and_unknown_id():
    hf = SimpleNamespace(
        sail_quantization_recipe="qwen3_next_80b_a3b_instruct", model_type="qwen3_next"
    )
    assert select_recipe("/renamed", hf, "fp8")["id"] == hf.sail_quantization_recipe
    hf.sail_quantization_recipe = "../../bad"
    with pytest.raises(ValueError, match="Unknown"):
        select_recipe("unknown", hf, "bf16")


def test_ambiguous_model_match():
    with pytest.raises(ValueError, match="Ambiguous"):
        select_recipe("Qwen3-Next-80B-A3B-Qwen3-235B-A22B", None, "fp8")


def test_mixed_layers_are_target_policy_and_validate_depth():
    hf = SimpleNamespace(num_hidden_layers=94, model_type="qwen3_moe")
    recipe = select_recipe("Qwen3-235B-A22B", hf, "fp8")
    assert r"re:(^|\.)layers\.88\." in target_exclusions(recipe)
    hf.num_hidden_layers = 80
    with pytest.raises(ValueError, match="depth"):
        select_recipe("Qwen3-235B-A22B", hf, "fp8")


def test_specific_recipe_checks_structure_for_both_sources():
    for source in ("fp8", "bf16"):
        with pytest.raises(ValueError, match="architecture"):
            select_recipe("Qwen3.8-27B", SimpleNamespace(model_type="other"), source)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("quantization", "int4"),
        ("ignored_layers", "all"),
        ("ignored_layers", ["re:["]),
        ("mix_layers", [-1]),
        ("mix_layers", [True]),
    ],
)
def test_invalid_recipe(field, value):
    recipe = deepcopy(select_recipe("unknown", None, "bf16"))
    recipe[field] = value
    with pytest.raises(ValueError):
        validate_recipe(recipe)
