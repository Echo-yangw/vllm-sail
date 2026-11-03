# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual registered INT8 guard and unchanged fp8 delegation."""

from types import SimpleNamespace as NS

import pytest

from tests.ut.test_ppu_kernel_capabilities import load_patch
from tests.ut.test_ppu_kernel_capabilities import modules as _modules

modules = _modules


@pytest.fixture
def kpool(modules):
    def load(*, fp8=False):
        calls = []
        modules(
            "torch",
            ops=NS(
                vllm=NS(
                    ppu_sparse_attn_indexer_kpool=lambda *args: calls.append(args)
                    or "int8"
                )
            ),
        )
        modules(
            "vllm.platforms",
            current_platform=NS(is_ppu=lambda: True, supports_fp8=lambda: fp8),
        )
        modules("vllm.utils.torch_utils", _encode_layer_name=lambda name: name)
        modules("vllm_sail.ops")

        class Indexer:
            def forward_native(self, *args, **kwargs):
                raise AssertionError("INT8 redirect must reach the registered PPU path")

            def forward_cuda(
                self,
                hidden_states,
                q_quant,
                k,
                weights,
                *,
                gate_score=None,
                compress_ape=None,
                index_kpool=1,
                positions=None,
                topk_backend="auto",
            ):
                return "upstream-fp8", self.dcp_world_size, topk_backend

        modules(
            "vllm.models.glm5next.nvidia.sparse_indexer", SparseAttnIndexerKpool=Indexer
        )
        for name in (
            "vllm.models.glm5next.sparse_indexer",
            "vllm.models.glm5next.common.attention",
        ):
            modules(name, SparseAttnIndexerKpool=Indexer)
        patch = load_patch(
            "vllm_sail/patch/enhancement/attention/sparse_attn_indexer_kpool.py"
        )
        assert Indexer.forward_ppu is patch.forward_ppu
        return Indexer, calls

    return load


@pytest.mark.parametrize("method", ["forward_ppu", "forward_native"])
def test_int8_dcp_fails_before_accessing_or_mutating_cache(kpool, method):
    cls, calls = kpool()
    layer = cls()
    layer.dcp_world_size = 2
    with pytest.raises(NotImplementedError, match="decode context parallel size 1"):
        getattr(layer, method)(*[object() for _ in range(4)])
    assert calls == []


def test_fp8_dcp_keeps_upstream_dispatch(kpool):
    cls, calls = kpool(fp8=True)
    layer = cls()
    layer.dcp_world_size = 2
    assert layer.forward_ppu(*[object() for _ in range(4)], topk_backend="triton") == (
        "upstream-fp8",
        2,
        "triton",
    )
    assert calls == []
