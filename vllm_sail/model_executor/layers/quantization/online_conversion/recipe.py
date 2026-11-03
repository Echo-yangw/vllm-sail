# SPDX-License-Identifier: Apache-2.0
"""Versioned bundled policies; source storage is never inferred from a recipe."""

import fnmatch
import json
import re
from pathlib import Path

_RECIPE_DIR = Path(__file__).with_name("recipes")


def validate_recipe(recipe):
    if not isinstance(recipe, dict):
        raise ValueError("Recipe must be a JSON object")
    if not isinstance(recipe.get("id"), str) or not recipe["id"]:
        raise ValueError("Recipe requires a nonempty ID")
    if type(recipe.get("schema_version")) is not int or recipe["schema_version"] != 1:
        raise ValueError("Unsupported online quantization recipe schema")
    if recipe.get("quantization") != "w8a8_dynamic_token_symmetric_channel":
        raise ValueError("Unsupported online quantization recipe algorithm")
    for key in ("model_patterns", "ignored_layers"):
        entries = recipe.get(key)
        if not isinstance(entries, list) or any(
            not isinstance(x, str) for x in entries
        ):
            raise ValueError(f"Recipe {key} must be a list of strings")
    for entry in recipe["ignored_layers"]:
        if entry.startswith("re:"):
            try:
                re.compile(entry[3:])
            except re.error as exc:
                raise ValueError(f"Invalid recipe regex: {entry}") from exc
    mixed = recipe.get("mix_layers", [])
    if not isinstance(mixed, list) or any(type(x) is not int or x < 0 for x in mixed):
        raise ValueError("Recipe mix_layers must be nonnegative layer indices")
    constraints = recipe.get("model_constraints")
    if constraints is not None and (
        not isinstance(constraints, dict)
        or not isinstance(constraints.get("model_types"), list)
        or any(not isinstance(v, str) for v in constraints["model_types"])
        or not isinstance(constraints.get("text_config"), dict)
    ):
        raise ValueError("Invalid recipe model_constraints")
    return recipe


def select_recipe(model_name, hf_config, source_dtype):
    """Explicit bundled ID, then unique model-name match, then source fallback.

    Explicit IDs support renamed local checkpoints. Structural constraints, when
    supplied by a recipe, apply equally to automatic and explicit selection.
    """
    requested = getattr(hf_config, "sail_quantization_recipe", None)
    if requested is None:
        requested = getattr(hf_config, "ppu_quantization_recipe", None)
    recipes = [
        validate_recipe(json.loads(p.read_text()))
        for p in sorted(_RECIPE_DIR.glob("*.json"))
    ]
    if requested is not None:
        matches = [r for r in recipes if r["id"] == requested]
        if not matches:
            raise ValueError(f"Unknown SAIL quantization recipe: {requested!r}")
    else:
        candidates = (model_name, Path(model_name.rstrip("/")).name)
        matches = [
            r
            for r in recipes
            if r["id"] != "_default"
            and any(
                fnmatch.fnmatchcase(candidate.lower(), pattern.lower())
                for candidate in candidates
                for pattern in r["model_patterns"]
            )
        ]
    if len(matches) > 1:
        raise ValueError(
            "Ambiguous model recipe; select sail_quantization_recipe explicitly"
        )
    if not matches:
        if source_dtype == "fp8":
            return None
        matches = [r for r in recipes if r["id"] == "_default"]
    if len(matches) != 1:
        raise ValueError("Missing or duplicate default recipe")
    recipe = matches[0]
    spec = recipe.get("model_constraints")
    if spec:
        text = getattr(hf_config, "text_config", hf_config)
        if getattr(hf_config, "model_type", None) not in spec["model_types"]:
            raise ValueError(
                f"Model architecture is incompatible with recipe {recipe['id']}"
            )
        for key, expected in spec["text_config"].items():
            if getattr(text, key, None) != expected:
                raise ValueError(f"Recipe {recipe['id']} requires {key}={expected!r}")
    text = getattr(hf_config, "text_config", hf_config)
    depth = getattr(text, "num_hidden_layers", None)
    if recipe.get("mix_layers") and (
        depth is None or max(recipe["mix_layers"]) >= depth
    ):
        raise ValueError(f"Recipe {recipe['id']} mix_layers exceed model depth")
    return recipe


def target_exclusions(recipe):
    entries = list(recipe["ignored_layers"])
    entries.extend(
        rf"re:(^|\.)layers\.{index}\." for index in recipe.get("mix_layers", [])
    )
    return entries
