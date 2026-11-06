# SPDX-License-Identifier: Apache-2.0
"""PPU channelwise Engram storage and factory contracts."""

from types import SimpleNamespace

import pytest

from tests.ut import test_ppu_kernel_capabilities as harness

modules = harness.modules
load_patch = harness.load_patch


@pytest.mark.parametrize("offload", [False, True])
def test_engram_channelwise_storage_keeps_real_fp32_scales(modules, offload):
    allocations = []

    def allocate(*shape, **kwargs):
        allocations.append((shape, kwargs))
        return SimpleNamespace()

    def set_weight_attrs(weight, attrs):
        for key, value in attrs.items():
            assert not hasattr(weight, key), key
            setattr(weight, key, value)

    modules(
        "torch",
        empty=allocate,
        float8_e4m3fn="fp8",
        float32="fp32",
    )
    modules("vllm.model_executor.utils", set_weight_attrs=set_weight_attrs)

    class Upstream:
        lookup = object()
        _storage = object()

        def __init__(self, n, dim, heads, **kwargs):
            self.dim, self.part_num_embeddings = dim, n
            self.cpu_offload = kwargs["cpu_offload"]
            assert kwargs["block_size"] == dim
            self.weight, self.weight_scale_inv = self._allocate_weights()
            set_weight_attrs(self.weight_scale_inv, {"dummy_weight_value": 127})

    modules("vllm.models.deepseek_v41.common.engram", ParallelEngramEmbedding=Upstream)
    cls = load_patch(
        "vllm_sail/models/deepseek_v41/engram.py"
    ).ChannelwiseEngramEmbedding
    embedding = cls(10, 256, (10,), cpu_offload=offload)
    assert [item[0] for item in allocations] == [(10, 256), (10, 1)]
    assert allocations[1][1]["dtype"] == "fp32"
    assert (allocations[1][1].get("device") == "cpu") is offload
    assert embedding.weight_scale_inv.dummy_weight_value == 1.0
    assert (
        embedding.lookup is Upstream.lookup and embedding._storage is Upstream._storage
    )
    with pytest.raises(ValueError, match="dp_shared_memory"):
        cls(10, 256, (10,), dp_shared_memory=True)
    peer_embedding = cls.__new__(cls)
    peer_embedding.peer_rows = 8
    with pytest.raises(NotImplementedError, match="peer-table"):
        peer_embedding._allocate_weights()


def test_engram_factory_keeps_offload_and_original_layouts(modules):
    platform = SimpleNamespace(ppu=True)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                quantization_config={
                    "fp8_channelwise_layers": ["layers.14.engram.embed"],
                }
            )
        ),
        engram_config=SimpleNamespace(cpu_offload=True, dp_shared_memory=False),
    )
    modules("vllm.config", get_current_vllm_config=lambda: config)
    modules(
        "vllm.platforms", current_platform=SimpleNamespace(is_ppu=lambda: platform.ppu)
    )
    modules(
        "vllm.triton_utils",
        tl=SimpleNamespace(constexpr=object()),
        triton=SimpleNamespace(jit=lambda **kw: lambda fn: fn),
    )
    calls = []
    modules(
        "vllm_sail.models.deepseek_v41.engram",
        ChannelwiseEngramEmbedding=lambda *args, **kw: calls.append((args, kw)),
    )
    original = object()

    class Engram:
        def _create_embedding(self, *args):
            return original

    original_factory = Engram._create_embedding
    factory = modules("vllm.models.deepseek_v41.common.engram", Engram=Engram)
    kernel = modules(
        "vllm.models.deepseek_v41.common.engram", _engram_lookup_kernel=lambda: None
    )
    path = "vllm_sail/patch/enhancement/models/deepseek_v41_engram.py"
    patch = load_patch(path)
    assert factory.Engram._create_embedding is patch._create_embedding
    assert kernel._engram_lookup_kernel is patch._engram_lookup_kernel
    marker = getattr(patch._create_embedding, patch.PATCH_MARKER)
    assert marker[f"{patch._FACTORY}.Engram._create_embedding"] is original_factory
    layout = SimpleNamespace(
        layer_ids=[1, 14],
        num_embeddings=[10, 20],
        head_dim=256,
        primes=[[(4, 6)], [(8, 12)]],
    )
    embedding = Engram()
    assert embedding._create_embedding(layout, 0) is original
    embedding._create_embedding(layout, 1)
    assert calls == [
        ((20, 256, (8, 12)), {"cpu_offload": True, "dp_shared_memory": False})
    ]
    platform.ppu = False
    assert embedding._create_embedding(layout, 1) is original
    with pytest.raises(RuntimeError, match="already patched"):
        load_patch(path)
