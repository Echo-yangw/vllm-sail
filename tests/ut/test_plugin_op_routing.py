# SPDX-License-Identifier: Apache-2.0
"""PPU-private ops must be reached through ``vllm_sail._custom_ops``.

``torch.ops._ppu_C`` / ``_ppu_moe_C`` are registered as a static-initializer
side effect of importing ``vllm_sail._C`` / ``vllm_sail._moe_C``, and nothing
imports those extensions eagerly. A direct ``torch.ops._ppu_C.<op>`` call in
model code therefore raises a bare ``AttributeError`` unless some earlier call
happened to touch a wrapper first -- which is exactly how the MXFP4 indexer
top-k path failed on device. These tests are stdlib-only.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).parents[2]
PACKAGE = ROOT / "vllm_sail"
WRAPPER = Path("vllm_sail/_custom_ops.py")

# Namespaces registered by vllm_sail._C / vllm_sail._moe_C.
PLUGIN_NAMESPACES = frozenset({"_ppu_C", "_ppu_moe_C"})


def _plugin_namespace_uses(path: Path) -> list[tuple[int, str]]:
    """(line, namespace) for every ``torch.ops.<plugin namespace>`` reference."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    uses = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        owner = node.value
        if not (isinstance(owner, ast.Attribute) and owner.attr == "ops"):
            continue
        if node.attr in PLUGIN_NAMESPACES:
            uses.append((node.lineno, node.attr))
    return uses


def test_plugin_namespaces_are_reached_only_through_the_wrapper() -> None:
    offenders = {
        str(path.relative_to(ROOT)): uses
        for path in sorted(PACKAGE.rglob("*.py"))
        if (uses := _plugin_namespace_uses(path)) and path.relative_to(ROOT) != WRAPPER
    }
    assert not offenders, (
        "Call torch.ops._ppu_* only via vllm_sail._custom_ops: the extension "
        f"import that registers the namespace happens there. Offenders: {offenders}"
    )


def test_the_wrapper_still_owns_both_plugin_namespaces() -> None:
    # Guards the previous test against passing vacuously.
    assert {name for _, name in _plugin_namespace_uses(ROOT / WRAPPER)} == set(
        PLUGIN_NAMESPACES
    )
