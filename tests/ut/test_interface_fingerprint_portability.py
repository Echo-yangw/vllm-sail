# SPDX-License-Identifier: Apache-2.0
"""Preserve interface fingerprints across the supported Python minors."""

import ast
import hashlib

import pytest

from tests.ut.test_upstream_version_precheck import fingerprint, signature_text


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("def probe() -> bool: return True", "lambda: None -> bool"),
        ("async def probe(): pass", "lambda: None"),
        (
            "env = {'factory': lambda: (lambda: 1), 'literal': 'lambda : None'}",
            "env = {'factory': lambda: lambda: 1, 'literal': 'lambda : None'}",
        ),
        (
            "def probe(value='lambda : None', *, factory=lambda: 1): pass",
            "lambda value='lambda : None', *, factory=lambda: 1: None",
        ),
    ],
)
def test_signature_rendering_keeps_literals_and_normalizes_empty_lambdas(
    source, expected
):
    node = ast.parse(source).body[0]
    assert signature_text(node) == expected
    assert (
        fingerprint(node) == "sha256:" + hashlib.sha256(expected.encode()).hexdigest()
    )


def test_fingerprint_distinguishes_changed_defaults_and_ignores_function_body():
    nodes = [
        ast.parse(source).body[0]
        for source in (
            "def probe(value=1): return True",
            "def probe(value=1): raise RuntimeError",
            "def probe(value=2): return True",
        )
    ]
    assert fingerprint(nodes[0]) == fingerprint(nodes[1])
    assert fingerprint(nodes[0]) != fingerprint(nodes[2])
