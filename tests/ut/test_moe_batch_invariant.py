# SPDX-License-Identifier: Apache-2.0
"""The upstream batch-invariant contract overrides optional native MoE paths."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from uuid import uuid4

import pytest

from vllm_sail.native import stubs
from vllm_sail.patch.utils import PATCH_REGISTRY


@pytest.fixture
def installed_moe(monkeypatch):
    before = len(PATCH_REGISTRY)

    def install(invariant, native):
        modules = {}

        def module(name):
            parts = name.split(".")
            for depth in range(1, len(parts) + 1):
                full = ".".join(parts[:depth])
                if full not in modules:
                    value = ModuleType(full)
                    value.__path__ = []
                    modules[full] = value
                    monkeypatch.setitem(sys.modules, full, value)
                    if depth > 1:
                        setattr(
                            modules[".".join(parts[: depth - 1])],
                            parts[depth - 1],
                            value,
                        )
            return modules[name]

        module("vllm.envs").VLLM_BATCH_INVARIANT = invariant
        module("vllm.platforms").current_platform = SimpleNamespace(is_ppu=lambda: True)
        provider = module("vllm.model_executor.layers.fused_moe.fused_moe")

        def original(*args, **kwargs):
            pytest.fail("the registered PPU implementation must handle the call")

        for name in (
            "invoke_fused_moe_triton_kernel",
            "invoke_fused_moe_wna16_triton_kernel",
            "dispatch_fused_moe_kernel",
            "get_config_file_name",
            "get_moe_configs",
            "get_moe_wna16_block_config",
            "try_get_optimal_moe_config",
            "fused_experts_impl",
            "should_moe_wna16_use_cuda",
        ):
            setattr(provider, name, original)
        module(
            "vllm.model_executor.layers.fused_moe.experts.triton_moe"
        ).TritonExperts = type("TritonExperts", (), {"apply": original})

        def unsupported():
            return [] if native else ["_moe_C::moe_wna16_gemm"]

        monkeypatch.setattr(stubs, "unsupported_ops", unsupported)
        name = "_sail_moe_batch_invariant_" + uuid4().hex
        path = Path(__file__).parents[2] / "vllm_sail/patch/enhancement/triton_moe.py"
        spec = importlib.util.spec_from_file_location(name, path)
        target = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, target)
        spec.loader.exec_module(target)
        return provider.should_moe_wna16_use_cuda

    yield install
    del PATCH_REGISTRY[before:]


@pytest.mark.parametrize("invariant", [False, True])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("disabled", [False, True])
def test_invariance_precedes_native_and_force(
    monkeypatch, installed_moe, invariant, native, force, disabled
):
    monkeypatch.setenv("VLLM_SAIL_FORCE_MOE_WNA16_CUDA", str(int(force)))
    monkeypatch.setenv("VLLM_SAIL_DISABLE_MOE_WNA16_CUDA", str(int(disabled)))
    choose = installed_moe(invariant, native)
    if invariant or disabled:
        assert choose(32, 64, 4, 4) is False
    elif not native and force:
        with pytest.raises(NotImplementedError, match="moe_wna16_gemm is unavailable"):
            choose(32, 64, 4, 4)
    else:
        assert choose(32, 64, 4, 4) is native
