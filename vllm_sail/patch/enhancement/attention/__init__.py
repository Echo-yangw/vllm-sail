# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: I001  (FlashMLA provider patches must precede their consumers)
"""PPU patches for upstream's attention code (Phase 4, Group 1).

Import order notes:

* ``flashmla_ops`` must come before ``flashmla_backend`` /
  ``flashmla_sparse_backend``: their builders call the availability probes and
  symbol getters at import/construction time.
* ``mla_indexer`` rebinds ``has_deep_gemm`` / ``get_paged_mqa_logits_metadata``
  on the indexer module, so it must come after ``enhancement.deep_gemm_redirect``
  (already imported by the parent package) but is otherwise independent.
* The ``forward_ppu`` additions (``mhc``, ``sparse_attn_indexer``) rely on the
  parent package's ``custom_op_dispatch`` having installed the dispatch reroute
  first; the parent ``__init__`` guarantees that.
"""

from vllm_sail.patch.enhancement.attention import (
    fa_utils,  # noqa: F401
    gdn,  # noqa: F401
    gdn_decode,  # noqa: F401
    kda,  # noqa: F401
    flash_attn,  # noqa: F401
    flashmla_ops,  # noqa: F401
    flashmla_backend,  # noqa: F401
    flashmla_sparse_backend,  # noqa: F401
    kv_cache_interface,  # noqa: F401
    mhc,  # noqa: F401
    mhc_tilelang,  # noqa: F401
    mla_attention,  # noqa: F401
    mla_indexer,  # noqa: F401
    minimax_dspark,  # noqa: F401
    mla_prefill_flash_attn,  # noqa: F401
    sparse_attn_indexer,  # noqa: F401
    sparse_attn_indexer_kpool,  # noqa: F401
    triton_decode_attention,  # noqa: F401
)
