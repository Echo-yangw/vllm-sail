# SPDX-License-Identifier: Apache-2.0
"""A compute stub must not select the fused MiniMax DSpark context writer."""

import importlib.util
import math
import sys
from types import SimpleNamespace as NS

import pytest

from tests.support.modules import load_patch


@pytest.mark.upstream_source
@pytest.mark.parametrize("ppu", [False, True])
@pytest.mark.parametrize("stub_present", [False, True])
def test_registered_capability_rejects_ppu_compute_stub(
    modules, monkeypatch, upstream_source_root, ppu, stub_present
):
    class Tensor:
        is_cuda = True

        def __init__(self, shape, dtype):
            self.shape, self.dtype = shape, dtype

        def numel(self):
            return math.prod(self.shape)

        def dim(self):
            return len(self.shape)

        def is_contiguous(self):
            return True

        def stride(self):
            return tuple(math.prod(self.shape[i + 1 :]) for i in range(self.dim()))

        def data_ptr(self):
            return 4096

        def element_size(self):
            return 2

    calls = []

    def stub(*args):
        calls.append(args)
        raise NotImplementedError("SAIL compute stub")

    ops = NS(**({"fused_minimax_m3_dspark_context_kv": stub} if stub_present else {}))
    modules(
        "torch",
        Tensor=Tensor,
        uint8="uint8",
        bfloat16="bf16",
        float32="fp32",
        ops=NS(_C=ops),
    )
    modules("vllm._custom_ops")
    modules(
        "vllm.platforms", current_platform=NS(is_cuda=lambda: True, is_ppu=lambda: ppu)
    )
    name = "vllm.models.minimax_m3.common.ops.dspark_context_kv"
    modules(name)
    source = (
        upstream_source_root / "vllm/models/minimax_m3/common/ops/dspark_context_kv.py"
    )
    spec = importlib.util.spec_from_file_location(name, source)
    provider = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, provider)
    spec.loader.exec_module(provider)
    layer = NS(
        kv_cache_dtype="bfloat16",
        kv_cache=Tensor((1, 1, 64, 256), "bf16"),
        _k_scale=Tensor((), "fp32"),
        _v_scale=Tensor((), "fp32"),
    )
    assert provider.DSparkContextKVWriter.supported([layer]) is stub_present
    load_patch("vllm_sail/patch/enhancement/attention/minimax_dspark.py")
    assert provider.DSparkContextKVWriter.supported([layer]) is (
        stub_present and not ppu
    )
    assert not calls
