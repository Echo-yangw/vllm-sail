# SPDX-License-Identifier: Apache-2.0
"""PPU patches for ``vllm.v1.attention.backends.mla.flashmla_sparse``.

Three PPU-gated fork changes, applied by delegation:

* ``FlashMLASparseBackend.supports_compute_capability`` accepts capability
  majors 8, 9 and 10 on PPU (upstream: 9 and 10).
* ``FlashMLASparseBackend.supports_combination`` accepts GLM5Next's NoPE-512
  head on PPU capability major 8 (upstream: only SM90, major 9).
* ``FlashMLASparseMetadataBuilder.__init__`` uses the full SM count for
  ``max_num_sm_parts`` on PPU ("PPU uses full SM count"). That value only feeds
  the ``tile_scheduler_metadata_buffer`` allocation, so the upstream ``__init__``
  runs first and the buffer is re-allocated with the PPU size afterwards.
"""

from __future__ import annotations

import torch
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.attention.backends.mla import flashmla_sparse as _flashmla_sparse_backend

from vllm_sail.patch.utils import patch

_AFFECTED = ">=0.31.0,<0.32.0"
_MODULE = "vllm.v1.attention.backends.mla.flashmla_sparse"

_upstream_supports_compute_capability = (
    _flashmla_sparse_backend.FlashMLASparseBackend.supports_compute_capability
)
_upstream_supports_combination = (
    _flashmla_sparse_backend.FlashMLASparseBackend.supports_combination
)
_upstream_builder_init = _flashmla_sparse_backend.FlashMLASparseMetadataBuilder.__init__


@patch(
    _MODULE,
    "FlashMLASparseBackend.supports_compute_capability",
    reason=(
        "PPU runs sparse FlashMLA on capability major 8. Upstream accepts only "
        "majors 9 and 10, so without this PPU could never select the sparse "
        "FlashMLA backend."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream accepts PPU's capability for sparse FlashMLA natively.",
)
def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
    if current_platform.is_ppu():
        return capability.major in [8, 9, 10]
    return _upstream_supports_compute_capability(capability)


@patch(
    _MODULE,
    "FlashMLASparseBackend.supports_combination",
    reason=(
        "GLM5Next serves its rope-free (NoPE) head_size 512 through the direct "
        "bf16 cache, which upstream allows only on SM90 (capability major 9). "
        "PPU reports major 8, so without this the sparse FlashMLA backend is "
        "rejected for GLM-5.3-Flash before the bf16 NoPE-512 cache can be used. "
        "The kv-cache dtype and qk_rope_head_dim == 0 checks stay in force."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream's head_size == 512 NoPE gate accepts PPU capability major 8, "
        "or the backend exposes a per-platform capability hook for it."
    ),
)
def supports_combination(
    cls,
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype,
    block_size: int | None,
    use_mla: bool,
    has_sink: bool,
    use_sparse: bool,
    use_mm_prefix: bool,
    device_capability: DeviceCapability,
) -> str | None:
    if not current_platform.is_ppu() or head_size != 512:
        return _upstream_supports_combination(
            head_size,
            dtype,
            kv_cache_dtype,
            block_size,
            use_mla,
            has_sink,
            use_sparse,
            use_mm_prefix,
            device_capability,
        )
    # PPU reports capability major 8. Other majors keep the upstream gate.
    if device_capability.major != 8:
        return _upstream_supports_combination(
            head_size,
            dtype,
            kv_cache_dtype,
            block_size,
            use_mla,
            has_sink,
            use_sparse,
            use_mm_prefix,
            device_capability,
        )
    # The direct bf16 NoPE-512 cache is only correct without a quantized
    # envelope; plain fp8 caches need the 576/656B DS-MLA layout.
    if kv_cache_dtype not in (None, "auto", "bfloat16", "float16"):
        return (
            "FLASHMLA_SPARSE supports head_size 512 only with a bf16 kv-cache "
            f"on PPU (NoPE), got kv_cache_dtype={kv_cache_dtype}"
        )
    # Direct bf16 NoPE-512 is only correct for rope-free models.
    from vllm.config import get_current_vllm_config

    vllm_config = get_current_vllm_config()
    if vllm_config.model_config is not None:
        hf_text_config = vllm_config.model_config.hf_text_config
        if getattr(hf_text_config, "qk_rope_head_dim", 64) != 0:
            return (
                "FLASHMLA_SPARSE supports head_size 512 only for "
                "rope-free (NoPE) models"
            )
    return None


@patch(
    _MODULE,
    "FlashMLASparseMetadataBuilder.__init__",
    reason=(
        "PPU uses the full SM count for max_num_sm_parts (the fork: 'PPU uses "
        "full SM count') instead of the SM90/SM100 formulas. The value only "
        "sizes tile_scheduler_metadata_buffer, so this delegates to upstream and "
        "re-allocates that buffer on PPU."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream grows a PPU branch in the max_num_sm_parts formula, or the "
        "PPU scheduler stops needing the full SM count."
    ),
)
def builder_init(
    self,
    kv_cache_spec,
    layer_names,
    vllm_config,
    device: torch.device,
) -> None:
    _upstream_builder_init(self, kv_cache_spec, layer_names, vllm_config, device)
    if current_platform.is_ppu():
        sm_count = num_compute_units(device.index)
        self.tile_scheduler_metadata_buffer = torch.empty(
            # TileSchedulerMetaDataSize = 8
            # see: FlashMLA/csrc/params.h
            (sm_count, 8),
            dtype=torch.int32,
            device=device,
        )
