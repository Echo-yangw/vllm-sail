# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the PPU MXFP4 candidates and upstream selection.

The plugin does not copy ``select_mxfp4_moe_backend``: ``map_mxfp4_backend``
returns each standard PPU candidate followed by its batched family and
upstream's shared activation-format check picks one. These tests therefore run
upstream's selector body against the patched PPU helpers.
"""

import ast
import sys
from enum import Enum
from types import SimpleNamespace as NS

import pytest

from tests.support.source import ROOT, definition, function
from vllm_sail.registry.moe_backends._extend import extend_enum

PATCH = "vllm_sail/registry/moe_backends/mxfp4.py"
Backend = Enum(
    "Backend",
    {
        name: name
        for name in (
            "MARLIN",
            "BATCHED_MARLIN",
            "EMULATION",
            "XPU",
            "PPU_DEEPGEMM_MXFP4",
            "BATCHED_PPU_DEEPGEMM_MXFP4",
            "PPU_DEEPGEMM_MXFP4_BF16",
            "BATCHED_PPU_DEEPGEMM_MXFP4_BF16",
            "PPU_DEEPGEMM_MXFP4_BF16_MMA",
            "BATCHED_PPU_DEEPGEMM_MXFP4_BF16_MMA",
        )
    },
)

# vllm_sail's map_mxfp4_backend returns each standard PPU candidate followed by
# its batched family.
BATCHED_SIBLING = {
    "PPU_DEEPGEMM_MXFP4": "BATCHED_PPU_DEEPGEMM_MXFP4",
    "PPU_DEEPGEMM_MXFP4_BF16": "BATCHED_PPU_DEEPGEMM_MXFP4_BF16",
    "PPU_DEEPGEMM_MXFP4_BF16_MMA": "BATCHED_PPU_DEEPGEMM_MXFP4_BF16_MMA",
}


def selection(
    candidates,
    *,
    requested="fp8",
    explicit=False,
    batched=False,
    supports=lambda *_: True,
    upstream,
    lora=False,
):
    """Run upstream's selector body with the PPU helpers installed as globals."""
    calls, warnings = [], []
    standard, batch = object(), object()

    def own_format(backend):
        return batch if backend.name.startswith("BATCHED_") else standard

    def kernel(backend):
        class Experts:
            @staticmethod
            def is_supported_config(cls, config, weight, activation, activation_format):
                calls.append((backend, weight, activation, activation_format))
                # Upstream's FusedMoEExperts.is_supported_config() rejects a
                # kernel whose activation format differs from the MoE layout,
                # which is what drops the sibling PPU candidate.
                if activation_format is not own_format(backend):
                    return False, "activation format fixture"
                return supports(backend, activation), "unsupported candidate fixture"

        return Experts

    def map_backend(_runner_backend):
        expanded = []
        for candidate in candidates:
            expanded.append(candidate)
            sibling = BATCHED_SIBLING.get(candidate.name)
            if sibling is not None:
                expanded.append(Backend[sibling])
        return expanded

    namespace = {
        "Mxfp4MoeBackend": Backend,
        "mk": NS(FusedMoEActivationFormat=NS(Standard=standard, BatchedExperts=batch)),
        "kMxfp4Static": "weights",
        "kFp8Dynamic128Sym": "fp8-block",
        "kMxfp8Dynamic": "mxfp8",
        "kFp8StaticTensorSym": "fp8",
        "kMxfp4Dynamic": "mxfp4",
        "envs": NS(VLLM_B12X_MOE_FP4_FORCE_A16=False),
        "oracle": NS(kMxfp4Dynamic="mxfp4"),
        "map_mxfp4_backend": map_backend,
        "backend_to_kernel_cls": lambda backend: [kernel(backend)],
        # The auto path reads the patched priority list; expanding the
        # requested candidates keeps one candidate source for both paths.
        "_get_priority_backends_for_gpt_oss": lambda: map_backend(None),
        "_resolve_activation_key": lambda key: key,
        "_make_log_backend": lambda backend: f"Using '{backend.value}' Mxfp4 MoE.",
        "_make_log_unsupported": lambda backend, reason: reason or backend.value,
        "logger": NS(
            info_once=lambda *a, **kw: None,
            warning_once=lambda *a, **kw: warnings.append(Backend(a[-1])),
            debug_once=lambda *a, **kw: None,
        ),
        "current_platform": NS(is_xpu=lambda: False),
    }
    namespace["_upstream_backend_activation_key"] = function(
        upstream, "_backend_activation_key", namespace
    )
    function(PATCH, "_backend_activation_key", namespace)
    constants = [
        node
        for node in ast.parse(upstream.read_text()).body
        if isinstance(node, ast.Assign)
        and any(
            getattr(target, "id", None) == "TRTLLM_BACKENDS" for target in node.targets
        )
    ]
    assert len(constants) == 1
    exec(
        compile(ast.Module(body=constants, type_ignores=[]), str(upstream), "exec"),
        namespace,
    )
    for helper in (
        "_filter_by_activation",
        "_get_requested_backends",
        "_warn_on_bf16_fallback",
        "_requested_kernel_cls",
        "_return_or_raise",
    ):
        if helper != "_requested_kernel_cls" or any(
            isinstance(node, ast.FunctionDef) and node.name == helper
            for node in ast.parse(upstream.read_text()).body
        ):
            function(upstream, helper, namespace)
    config = NS(
        moe_backend="explicit" if explicit else "auto",
        moe_parallel_config=NS(use_batched_activation_format=batched),
        is_lora_enabled=lora,
    )
    result = function(upstream, "select_mxfp4_moe_backend", namespace)(
        config, activation_key=requested
    )
    return result, calls, warnings, batch if batched else standard


def test_plugin_does_not_copy_the_selector_body() -> None:
    tree = ast.parse((ROOT / PATCH).read_text(encoding="utf-8"))
    defined = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    assert "select_mxfp4_moe_backend" not in defined


def test_automatic_bf16_candidate_uses_own_activation_key(real_oracle_source):
    (backend, _), calls, warnings, format = selection(
        [Backend.MARLIN],
        supports=lambda _, key: key is None,
        upstream=real_oracle_source,
    )
    assert backend is Backend.MARLIN
    assert calls == [(Backend.MARLIN, "weights", None, format)]
    assert warnings == [Backend.MARLIN]


@pytest.mark.parametrize("explicit", [False, True])
def test_successful_bf16_fallback_warns(real_oracle_source, explicit):
    (backend, _), _, warnings, _ = selection(
        [Backend.MARLIN], explicit=explicit, upstream=real_oracle_source
    )
    assert backend is Backend.MARLIN
    assert warnings == [Backend.MARLIN]


@pytest.mark.parametrize("explicit", [False, True])
def test_emulation_uses_checkpoint_activation_without_bf16_warning(
    real_oracle_source, explicit
):
    (backend, _), calls, warnings, format = selection(
        [Backend.EMULATION], explicit=explicit, upstream=real_oracle_source
    )
    assert backend is Backend.EMULATION
    assert calls == [(backend, "weights", "fp8", format)]
    assert not warnings


@pytest.mark.parametrize("lora", [False, True])
def test_requested_kernel_helper_retains_the_ppu_candidate(real_oracle_source, lora):
    (backend, _), calls, warnings, format = selection(
        [Backend.PPU_DEEPGEMM_MXFP4_BF16],
        explicit=True,
        requested=None,
        upstream=real_oracle_source,
        lora=lora,
    )
    assert backend is Backend.PPU_DEEPGEMM_MXFP4_BF16
    assert calls == [(backend, "weights", None, format)]
    assert not warnings


@pytest.mark.parametrize(
    "name",
    [
        "PPU_DEEPGEMM_MXFP4",
        "PPU_DEEPGEMM_MXFP4_BF16",
        "PPU_DEEPGEMM_MXFP4_BF16_MMA",
    ],
)
@pytest.mark.parametrize("batched", [False, True])
def test_explicit_ppu_variant_resolves_its_activation_format(
    real_oracle_source, name, batched
):
    (backend, _), calls, warnings, format = selection(
        [Backend[name]],
        explicit=True,
        batched=batched,
        requested=None,
        upstream=real_oracle_source,
    )
    expected = Backend[("BATCHED_" if batched else "") + name]
    activation = "mxfp4" if name.endswith("MXFP4") else None
    assert backend is expected
    assert calls[-1] == (expected, "weights", activation, format)
    if batched:
        # The standard candidate is rejected by the activation-format check.
        assert len(calls) == 2
        assert calls[0] == (Backend[name], "weights", activation, format)
    else:
        assert calls == [(expected, "weights", activation, format)]
    assert not warnings


@pytest.mark.parametrize("explicit", [False, True])
def test_failed_candidate_does_not_warn_before_successful_fallback(
    real_oracle_source, explicit
):
    (backend, _), calls, warnings, _ = selection(
        [Backend.MARLIN, Backend.TRITON],
        explicit=explicit,
        upstream=real_oracle_source,
        supports=lambda candidate, _: candidate is Backend.TRITON,
        requested="fp8",
    )
    assert backend is Backend.TRITON
    assert [c[0] for c in calls] == [Backend.MARLIN, Backend.TRITON]
    assert calls[-1][2] is None
    # Only the selected BF16 backend warns about the activation fallback.
    assert warnings == [Backend.TRITON]


def test_w4a4_candidate_rejection_warns_for_bf16_fallback(real_oracle_source):
    (backend, _), calls, warnings, format = selection(
        [Backend.PPU_DEEPGEMM_MXFP4, Backend.MARLIN],
        requested="mxfp4",
        explicit=True,
        upstream=real_oracle_source,
        supports=lambda candidate, key: candidate is Backend.MARLIN and key is None,
    )
    assert backend is Backend.MARLIN
    assert calls == [
        (Backend.PPU_DEEPGEMM_MXFP4, "weights", "mxfp4", format),
        (Backend.BATCHED_PPU_DEEPGEMM_MXFP4, "weights", "mxfp4", format),
        (Backend.MARLIN, "weights", None, format),
    ]
    assert warnings == [Backend.MARLIN]


@pytest.fixture
def real_oracle_source(monkeypatch, upstream_source_root):
    path = upstream_source_root / "vllm/model_executor/layers/fused_moe/oracle/mxfp4.py"
    node = definition(path, "Mxfp4MoeBackend")
    namespace = {"Enum": Enum}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    enum = namespace["Mxfp4MoeBackend"]
    for member in Backend:
        if not hasattr(enum, member.name):
            extend_enum(enum, member.name, member.value)
    monkeypatch.setattr(sys.modules[__name__], "Backend", enum)
    return path
