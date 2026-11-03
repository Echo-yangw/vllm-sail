#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build smoke and accuracy matrices from the checked-in model catalog.

Catalog entries are partitioned by ``category``. Missing categories retain the
legacy ``smoke`` meaning. A requested subset is validated against the complete
catalog, then each output contains only entries belonging to its own category.

Usage:
    ppu_e2e_select.py [MODELS]

``MODELS`` is a comma-separated list of catalog keys. An empty value selects all
models in each category. Unknown keys are rejected so a typo fails fast. The
script emits ``matrix=<json>`` for smoke and ``accuracy_matrix=<json>`` for the
GSM8K accuracy workflow.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CATALOG = Path(__file__).resolve().parent / "ppu_e2e_models.json"

# Catalog keys are CLI-facing and use hyphens; the evaluator historically uses
# underscores for the two DeepSeek variants. Keep the translation explicit so
# workflow filtering cannot silently dispatch the wrong accuracy model.
ACCURACY_MODEL_KEYS = {
    "deepseek-v4-int8": "deepseek_v4_int8",
    "deepseek-v4-mxfp4": "deepseek_v4_mxfp4",
    "minimax-m3-fp8": "minimax-m3-fp8",
    "qwen3.8-flash-next": "qwen3.8-flash-next",
}


def _category(entry: dict) -> str:
    return entry.get("category", "smoke")


def _select(catalog: list[dict], models: str, category: str) -> list[dict]:
    by_key = {entry["key"]: entry for entry in catalog}
    requested = [item.strip() for item in models.split(",") if item.strip()]
    unknown = [key for key in requested if key not in by_key]
    if unknown:
        available = ", ".join(by_key)
        raise ValueError(
            f"unknown model key(s): {', '.join(unknown)}; available: {available}"
        )
    entries = [by_key[key] for key in requested] if requested else list(catalog)
    return [entry for entry in entries if _category(entry) == category]


def build_matrix(catalog: list[dict], models: str) -> dict:
    """Return smoke entries plus the aliases default-branch callers read."""
    include = []
    for entry in _select(catalog, models, "smoke"):
        selected = dict(entry)
        # Default-branch workflow_dispatch callers may still read `checkpoint`.
        # Its value is a model ID, never a runner-specific filesystem path.
        if "model_id" in selected:
            selected.setdefault("checkpoint", selected["model_id"])
        # A PR run takes its workflow files from the default branch and its
        # catalog from the PR head, so the two disagree on the card count's
        # name until the rename reaches the default branch. Answering both
        # keeps a PR from failing on its own catalog; the alias can go once
        # every caller reads `ppu_count`.
        if "ppu_count" in selected:
            selected.setdefault("tp", selected["ppu_count"])
        include.append(selected)
    return {"include": include}


def build_accuracy_matrix(catalog: list[dict], models: str) -> dict:
    """Return the evaluator matrix for accuracy catalog entries only."""
    include = []
    for entry in _select(catalog, models, "accuracy"):
        catalog_key = entry["key"]
        try:
            model_key = ACCURACY_MODEL_KEYS[catalog_key]
        except KeyError as error:
            raise ValueError(
                f"missing accuracy model-key mapping for {catalog_key!r}"
            ) from error
        include.append(
            {
                "model_key": model_key,
                "board": entry["board"],
                "ppu_count": int(entry["ppu_count"]),
            }
        )
    return {"include": include}


def main(argv: list[str]) -> int:
    models = argv[0] if argv else ""
    catalog = json.loads(CATALOG.read_text())
    matrix = build_matrix(catalog, models)
    accuracy_matrix = build_accuracy_matrix(catalog, models)
    print(f"matrix={json.dumps(matrix, separators=(',', ':'))}")
    print(f"accuracy_matrix={json.dumps(accuracy_matrix, separators=(',', ':'))}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
