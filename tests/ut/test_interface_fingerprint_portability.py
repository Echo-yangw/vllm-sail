# SPDX-License-Identifier: Apache-2.0
"""Keep portable callable binding coverage after retiring rendered hashes."""

import ast
import inspect

import pytest

from tests.support.upstream import callable_signature


@pytest.mark.parametrize(
    "source",
    [
        "def probe() -> bool: return True",
        "async def probe(): pass",
        "def probe(value='lambda : None', *, factory=lambda: 1): pass",
        "def probe(value='lambda : None', *, factory=lambda: (lambda: 1)): pass",
    ],
)
def test_literal_and_lambda_defaults_do_not_add_required_inputs(source):
    signature = callable_signature(ast.parse(source).body[0])
    assert signature.bind().arguments == {}


def test_default_expressions_are_not_evaluated_and_keyword_only_stays_keyword_only():
    signature = callable_signature(
        ast.parse("def probe(value=unavailable(), *, factory=lambda: 1): pass").body[0]
    )
    value, factory = object(), object()
    assert signature.bind(value, factory=factory).arguments == {
        "value": value,
        "factory": factory,
    }
    assert signature.parameters["factory"].kind is inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(TypeError):
        signature.bind(value, factory)


@pytest.mark.parametrize(
    "source",
    [
        "def probe(value=1): return True",
        "def probe(value=1): raise RuntimeError",
        "def probe(value=2): return True",
    ],
)
def test_body_and_default_value_changes_preserve_the_call_surface(source):
    signature = callable_signature(ast.parse(source).body[0])
    value = object()
    assert signature.bind().arguments == {}
    assert signature.bind(value).arguments == {"value": value}
    with pytest.raises(TypeError):
        signature.bind(value, value)
