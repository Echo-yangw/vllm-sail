# SPDX-License-Identifier: Apache-2.0
"""Load-reordering patch for the Qwen3.5 fused QK-RMSNorm + RoPE + gate kernel.

``patch/performance/fused_qk_norm_rope.py`` replaces the upstream Triton kernel
with the load-reordered one, so the load-bearing properties are: the target is
*replaced* (not shadowed by a same-name launcher patch), the launcher keeps
working because the parameter list is unchanged, and the loads that matter are
issued before the RMSNorm reduction.

Runs without torch, vLLM or Triton: the patch module is loaded by path against a
fake ``vllm.triton_utils`` and a fake upstream module, mirroring
``test_fla_autotune_cache.py``.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import types
from pathlib import Path

import pytest

_MODULE = "vllm.model_executor.layers.fused_qk_norm_rope"
_KERNEL = "_fused_qk_rmsnorm_rope_gate_kernel"
_TARGET = f"{_MODULE}.{_KERNEL}"
_AFFECTED = ">=0.32.0,<0.33.0"
_PATCH_FILE = (
    Path(__file__).parents[2] / "vllm_sail/patch/performance/fused_qk_norm_rope.py"
)

#: Upstream's parameter list (v0.32.0), positional order included. The unpatched
#: launcher in the same module passes these by position and name, so a renamed,
#: dropped or reordered parameter here is a launch-time break, not a style nit.
_UPSTREAM_PARAMETERS = (
    "q_gate_ptr",
    "k_ptr",
    "q_out_ptr",
    "k_out_ptr",
    "gate_out_ptr",
    "q_weight_ptr",
    "k_weight_ptr",
    "cos_sin_cache_ptr",
    "positions_ptr",
    "q_gate_stride_t",
    "k_stride_t",
    "q_out_stride_t",
    "k_out_stride_t",
    "gate_out_stride_t",
    "cache_stride_p",
    "positions_stride_m",
    "positions_stride_t",
    "num_q_heads",
    "num_kv_heads",
    "head_dim",
    "rotary_dim",
    "half_rotary",
    "eps",
    "norm_beta",
    "INPUT_DTYPE",
    "HEAD_BLOCK",
    "ROT_HALF_BLOCK",
    "HAS_PASS",
    "HAS_MROPE",
    "MROPE_SECTION_H",
    "MROPE_SECTION_W",
)


def _upstream_kernel_stub():
    """Stand-in for the upstream kernel, with upstream's parameter names."""

    def kernel(*args, **kwargs):
        return args, kwargs

    kernel.__signature__ = inspect.Signature(
        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for name in _UPSTREAM_PARAMETERS
    )
    return kernel


@pytest.fixture
def installed(monkeypatch, patch_utils_module):
    """Load the patch module against fakes; returns (module, upstream, original)."""
    triton_utils = types.ModuleType("vllm.triton_utils")
    triton_utils.tl = types.SimpleNamespace(constexpr=object())
    triton_utils.triton = types.SimpleNamespace(jit=lambda fn: fn)
    monkeypatch.setitem(sys.modules, "vllm.triton_utils", triton_utils)
    monkeypatch.setitem(sys.modules, "vllm_sail.patch.utils", patch_utils_module)

    upstream = types.ModuleType(_MODULE)
    original = _upstream_kernel_stub()
    setattr(upstream, _KERNEL, original)
    # The launcher lives in the same upstream module and is deliberately not
    # patched; it must survive the kernel rebind.
    upstream.fused_qk_rmsnorm_rope_gate = lambda *args, **kwargs: (_KERNEL, args)
    monkeypatch.setitem(sys.modules, _MODULE, upstream)

    spec = importlib.util.spec_from_file_location(
        "_test_fused_qk_norm_rope_patch", _PATCH_FILE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, upstream, original


def test_kernel_target_is_replaced(installed, patch_utils_module):
    module, upstream, original = installed

    replacement = getattr(upstream, _KERNEL)
    assert replacement is module._fused_qk_rmsnorm_rope_gate_kernel
    assert replacement is not original
    # Installed under upstream's name, not the module's.
    assert replacement.__name__ == _KERNEL
    assert patch_utils_module.original_of(replacement, _TARGET) is original


def test_launcher_is_left_untouched(installed):
    module, upstream, _ = installed

    assert upstream.fused_qk_rmsnorm_rope_gate() == (_KERNEL, ())
    # The patch module must not export a launcher copy that a same-module
    # caller could pick up by import.
    assert not hasattr(module, "fused_qk_rmsnorm_rope_gate")


def test_registry_record_documents_the_patch(installed, patch_utils_module):
    module, upstream, _ = installed
    assert getattr(upstream, _KERNEL) is module._fused_qk_rmsnorm_rope_gate_kernel

    records = patch_utils_module.PATCH_REGISTRY
    assert [record.target for record in records] == [_TARGET]
    record = records[0]
    assert record.kind == "module"
    assert record.was_missing is False
    assert record.affected_versions == _AFFECTED
    assert "above the reduction" in record.reason
    assert "before the RMSNorm reduction" in record.remove_when


def test_reapplying_the_patch_raises(installed, patch_utils_module):
    _, upstream, _ = installed

    with pytest.raises(RuntimeError, match="already patched"):
        patch_utils_module.patch(
            _MODULE,
            _KERNEL,
            reason="second application must be rejected",
            affected_versions=_AFFECTED,
            remove_when="never; this asserts the guard",
        )(upstream._fused_qk_rmsnorm_rope_gate_kernel)


def test_parameter_list_matches_upstream(installed):
    """The unpatched launcher calls this positionally; the shape must not move."""
    module, _, original = installed

    signature = inspect.signature(module._fused_qk_rmsnorm_rope_gate_kernel)
    patched = list(signature.parameters)
    assert patched == list(inspect.signature(original).parameters)
    assert patched == list(_UPSTREAM_PARAMETERS)


def test_cache_and_gate_loads_are_hoisted_above_the_reduction(installed):
    """The whole point of the patch: those loads overlap the tl.sum dependency."""
    module, _, _ = installed

    source = inspect.getsource(module._fused_qk_rmsnorm_rope_gate_kernel)
    reduction = source.index("var = tl.sum(")

    assert source.index("tl.load(positions_ptr") < reduction
    assert source.index("cos = tl.load(cos_sin_cache_ptr") < reduction
    assert source.index("sin = tl.load(") < reduction
    assert source.index("g = tl.load(gate_in_base") < reduction
    # Upstream reloads the gate after the RoPE stores; the patch loads it up
    # front and stores it from registers.
    assert source.index("tl.store(gate_out_base") > reduction
