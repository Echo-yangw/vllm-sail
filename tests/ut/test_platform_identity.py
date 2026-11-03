# SPDX-License-Identifier: Apache-2.0
"""PPU identity must be available before the general-plugin patch lifecycle."""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path


def test_ppu_identity_exists_before_general_patches(monkeypatch):
    cuda_enum = object()

    class NvmlCudaPlatform:
        @classmethod
        def get_hisparse_backend_cls(cls):
            return "vllm.v1.hisparse.backend.CUDAHiSparseBackend"

        def __getattr__(self, name):
            # vLLM's fallback returns None for unknown platform capabilities.
            return None

        def is_cuda(self):
            return self._enum is cuda_enum

    modules = {
        "torch": types.SimpleNamespace(
            backends=types.SimpleNamespace(
                cuda=types.SimpleNamespace(enable_cudnn_sdp=lambda value: None)
            )
        ),
        "vllm.logger": types.SimpleNamespace(init_logger=logging.getLogger),
        "vllm.platforms.cuda": types.SimpleNamespace(NvmlCudaPlatform=NvmlCudaPlatform),
        "vllm.platforms.interface": types.SimpleNamespace(
            DeviceCapability=tuple, PlatformEnum=types.SimpleNamespace(CUDA=cuda_enum)
        ),
        "vllm.v1.attention.backends.registry": types.SimpleNamespace(
            AttentionBackendEnum=object
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).parents[2] / "vllm_sail/platform.py"
    spec = importlib.util.spec_from_file_location("_ppu_identity_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    platform = module.PPUPlatform()
    assert platform.is_ppu() is True
    assert module.PPUPlatform.is_ppu() is True
    assert platform.is_cuda() is True
    assert NvmlCudaPlatform.get_hisparse_backend_cls() is not None
    assert platform.get_hisparse_backend_cls() is None
    assert module.PPUPlatform.get_hisparse_backend_cls() is None
    # Installing a predicate on the upstream base later must not change PPU
    # identity or become a prerequisite for it.
    NvmlCudaPlatform.is_ppu = lambda self: False
    assert platform.is_ppu() is True
