# SPDX-License-Identifier: Apache-2.0
"""Exercise the installed V4.1 indexer patch with strict upstream doubles."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from tests.ut import test_ppu_kernel_capabilities as harness

modules = harness.modules


@pytest.mark.parametrize("ppu", [True, False])
@pytest.mark.parametrize("ratio", [1, 2])
def test_indexer_publishes_geometry_and_physical_pages(modules, ppu, ratio):
    @dataclass(frozen=True)
    class Spec:
        block_size: int = 256
        tokens_per_state: int = ratio
        page_size_padded: int | None = 33792
        indexer_n_head: int | None = None
        indexer_q_head_dim: int | None = None

    class Cache:
        def get_kv_cache_spec(self, config):
            return Spec()

    class Indexer:
        def __init__(self, cache):
            self.k_cache = cache
            self.n_head = 32
            self.head_dim = 128

    class Backend:
        @staticmethod
        def get_supported_kernel_block_sizes(kv_cache_spec=None):
            return [128]

    modules("vllm.platforms", current_platform=SimpleNamespace(is_ppu=lambda: ppu))
    modules(
        "vllm.models.deepseek_v41.attention",
        DeepseekV4IndexerCache=Cache,
        DeepseekV4Indexer=Indexer,
    )
    modules("vllm.v1.attention.backends.mla.indexer", DeepseekV41IndexerBackend=Backend)
    harness.load_patch("vllm_sail/patch/enhancement/models/deepseek_v41_indexer.py")
    cache = Cache()
    cache.compress_ratio = ratio
    Indexer(cache)
    spec = cache.get_kv_cache_spec(object())
    if ppu:
        assert spec.block_size == 64 * ratio
        assert spec.page_size_padded is None
        assert (spec.indexer_n_head, spec.indexer_q_head_dim) == (32, 128)
        assert Backend.get_supported_kernel_block_sizes(spec) == [64, 128]
    else:
        assert spec == Spec()
        assert Backend.get_supported_kernel_block_sizes(spec) == [128]
