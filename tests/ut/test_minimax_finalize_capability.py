# SPDX-License-Identifier: Apache-2.0
"""Check the registered platform guard before NVIDIA-only initialization."""

from types import SimpleNamespace as NS

import pytest

from tests.ut.test_ppu_kernel_capabilities import load_patch
from tests.ut.test_ppu_kernel_capabilities import modules as _modules

modules = _modules


@pytest.mark.parametrize("ppu", [False, True])
def test_registered_finalize_retains_ordinary_ppu_path(modules, ppu):
    calls = []
    result = object()

    class Model:
        def _init_fused_moe_finalize(self, config):
            if ppu:
                raise ModuleNotFoundError("NVIDIA-only dependency", name="quack")
            calls.append((self, config))
            return result

    modules(
        "vllm.models.minimax_m3.nvidia.model",
        MiniMaxM3Model=Model,
        MiniMaxM3MLP=type("MLP", (), {}),
    )
    modules(
        "vllm.platforms", current_platform=NS(is_ppu=lambda: ppu, is_cuda=lambda: True)
    )
    load_patch("vllm_sail/patch/enhancement/models/minimax_m3.py")
    instance, config = Model(), object()
    instance.fuse_final_norm_allreduce = True
    actual = instance._init_fused_moe_finalize(config)
    assert actual is (None if ppu else result)
    assert calls == ([] if ppu else [(instance, config)])
    assert instance.fuse_final_norm_allreduce is True
