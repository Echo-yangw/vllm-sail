# SPDX-License-Identifier: Apache-2.0
"""Give each V4.1 indexer cache the physical page size required by SAIL."""

from dataclasses import replace

from vllm.models.deepseek_v41 import attention
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.indexer import DeepseekV41IndexerBackend

from vllm_sail.patch.utils import patch

_cache_spec = attention.DeepseekV4IndexerCache.get_kv_cache_spec
_kernel_sizes = DeepseekV41IndexerBackend.get_supported_kernel_block_sizes
_AFFECTED = ">=0.32.0,<0.33.0"

_indexer_init = attention.DeepseekV4Indexer.__init__


@patch(
    "vllm.models.deepseek_v41.attention",
    "DeepseekV4Indexer.__init__",
    reason="SAIL paged logits metadata needs the V4.1 query head count and head dimension.",
    affected_versions=_AFFECTED,
    remove_when="The upstream V4.1 indexer publishes query dimensions in its cache spec.",
)
def indexer_init(self, *args, **kwargs):
    _indexer_init(self, *args, **kwargs)
    if current_platform.is_ppu():
        self.k_cache.n_head = self.n_head
        self.k_cache.q_head_dim = self.head_dim


@patch(
    "vllm.models.deepseek_v41.attention",
    "DeepseekV4IndexerCache.get_kv_cache_spec",
    reason=(
        "SAIL paged FP8 logits requires 64 stored rows per page. V4.1 C1 and "
        "C2 indexers therefore need independent 64- and 128-token cache blocks; "
        "splitting the main cache's shared padded pages cannot express this."
    ),
    affected_versions=_AFFECTED,
    remove_when="V4.1 indexer cache sizing delegates to a platform-specific kernel contract.",
)
def get_kv_cache_spec(self, vllm_config):
    spec = _cache_spec(self, vllm_config)
    if current_platform.is_ppu():
        # Size the indexer's own manager blocks before cache grouping. Virtual
        # splitting cannot represent the shared, padded BLHNC allocation.
        # Reset the old padding so MLAAttentionSpec recomputes its alignment.
        spec = replace(
            spec,
            block_size=64 * self.compress_ratio,
            page_size_padded=None,
            indexer_n_head=getattr(self, "n_head", None),
            indexer_q_head_dim=getattr(self, "q_head_dim", None),
        )
    return spec


@patch(
    "vllm.v1.attention.backends.mla.indexer",
    "DeepseekV41IndexerBackend.get_supported_kernel_block_sizes",
    reason=(
        "PPU V4.1 indexer cache specs select 64/128 uncompressed tokens for "
        "C1/C2, respectively, so both retain 64 physical rows without virtual "
        "splitting of interleaved, padded cache pages."
    ),
    affected_versions=_AFFECTED,
    remove_when="The indexer backend obtains supported page sizes from the PPU kernel contract.",
)
def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int]:
    if current_platform.is_ppu():
        return [64, 128]
    return _kernel_sizes(kv_cache_spec=kv_cache_spec)
