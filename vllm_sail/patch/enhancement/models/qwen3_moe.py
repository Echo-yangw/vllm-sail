# SPDX-License-Identifier: Apache-2.0
"""PPU changes to ``vllm.model_executor.models.qwen3_moe``.

Ports the fork's +36 hunk set: per-layer quant-config plumbing so an HF
``quantization_config.mix_layer`` list can mark layers that must stay
unquantized in a mixed-precision checkpoint. Three delegating patches —
upstream reads ``vllm_config.quant_config`` internally, so each replacement
substitutes a shallow-copied ``VllmConfig`` carrying the requested quant
config and then calls upstream untouched:

* ``Qwen3MoeSparseMoeBlock.__init__`` and ``Qwen3MoeDecoderLayer.__init__``
  gain the fork's ``quant_config`` parameter.
* ``Qwen3MoeModel.__init__`` swaps in a layer factory that applies the
  per-layer override when ``mix_layer`` is non-empty.

One deliberate, documented divergence from the fork: a missing (not ``None``)
``quant_config`` argument keeps upstream behaviour (read
``vllm_config.quant_config``), where the fork would silently build
unquantized layers. ``None`` means "force unquantized", exactly as in the
fork; callers added by the fork always pass the argument explicitly.
"""

from __future__ import annotations

from dataclasses import replace

from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3_moe import (
    Qwen3MoeDecoderLayer,
    Qwen3MoeModel,
    Qwen3MoeSparseMoeBlock,
)
from vllm.model_executor.models.utils import extract_layer_index

from vllm_sail.patch.utils import patch

_AFFECTED = ">=0.32.0,<0.33.0"

#: Sentinel distinguishing "argument not passed" (keep upstream behaviour)
#: from ``None`` ("force this layer unquantized", the fork's mix_layer case).
_UNSET = object()

# Captured before patching so the replacements can delegate to upstream.
_upstream_moe_block_init = Qwen3MoeSparseMoeBlock.__init__
_upstream_decoder_layer_init = Qwen3MoeDecoderLayer.__init__
_upstream_model_init = Qwen3MoeModel.__init__


def _with_quant_config(vllm_config: VllmConfig, quant_config) -> VllmConfig:
    """Shallow VllmConfig copy with a substituted quant_config.

    ``dataclasses.replace`` shares every other field by reference — the fork
    explicitly avoids deep copies here because they OOM at this model size.
    """
    return replace(vllm_config, quant_config=quant_config)


def _mix_layers(hf_config) -> list[int]:
    """Read ``quantization_config.mix_layer`` the way the fork does."""
    quant_cfg = getattr(hf_config, "quantization_config", None)
    if quant_cfg is None:
        return []
    if hasattr(quant_cfg, "get"):
        try:
            return quant_cfg.get("mix_layer") or []
        except Exception:
            return []
    return getattr(quant_cfg, "mix_layer", []) or []


@patch(
    "vllm.model_executor.models.qwen3_moe",
    "Qwen3MoeSparseMoeBlock.__init__",
    reason=(
        "Mixed-precision PPU checkpoints mark some layers mix_layer to keep "
        "them unquantized; that requires overriding vllm_config.quant_config "
        "per layer. Delegates to upstream with a shallow VllmConfig copy "
        "carrying the requested quant config."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream threads a per-layer quant_config through "
        "Qwen3MoeSparseMoeBlock itself."
    ),
)
def sparse_moe_block_init(
    self,
    vllm_config: VllmConfig,
    quant_config=_UNSET,
    prefix: str = "",
    is_fused_checkpoint_transposed: bool = False,
):
    if quant_config is not _UNSET:
        vllm_config = _with_quant_config(vllm_config, quant_config)
    _upstream_moe_block_init(
        self,
        vllm_config=vllm_config,
        prefix=prefix,
        is_fused_checkpoint_transposed=is_fused_checkpoint_transposed,
    )


@patch(
    "vllm.model_executor.models.qwen3_moe",
    "Qwen3MoeDecoderLayer.__init__",
    reason=(
        "Threads the per-layer quant config into attention, MoE block and "
        "MLP for mix_layer checkpoints. Delegates to upstream with a "
        "shallow VllmConfig copy carrying the requested quant config."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream threads a per-layer quant_config through Qwen3MoeDecoderLayer itself."
    ),
)
def decoder_layer_init(
    self,
    vllm_config: VllmConfig,
    quant_config=_UNSET,
    prefix: str = "",
    is_fused_checkpoint_transposed: bool = False,
    mrope_positions_factor: int = 4,
) -> None:
    if quant_config is not _UNSET:
        vllm_config = _with_quant_config(vllm_config, quant_config)
    _upstream_decoder_layer_init(
        self,
        vllm_config=vllm_config,
        prefix=prefix,
        is_fused_checkpoint_transposed=is_fused_checkpoint_transposed,
        mrope_positions_factor=mrope_positions_factor,
    )


@patch(
    "vllm.model_executor.models.qwen3_moe",
    "Qwen3MoeModel.__init__",
    reason=(
        "Reads quantization_config.mix_layer from the HF config and builds "
        "those layers unquantized. Delegates to upstream, substituting a "
        "layer factory that applies the per-layer override only when "
        "mix_layer is non-empty."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream Qwen3MoeModel honours quantization_config.mix_layer itself."
    ),
)
def model_init(
    self,
    *,
    vllm_config: VllmConfig,
    prefix: str = "",
    decoder_layer_type: type = Qwen3MoeDecoderLayer,
):
    mix_layer = _mix_layers(vllm_config.model_config.hf_text_config)
    # PPU MODIFICATION: begin — conversion owns source/target layer selection.
    conversion = getattr(vllm_config.quant_config, "handles_source_mixed_layers", False)
    if not mix_layer or conversion:
        _upstream_model_init(
            self,
            vllm_config=vllm_config,
            prefix=prefix,
            decoder_layer_type=decoder_layer_type,
        )
        return
    # PPU MODIFICATION: end

    # PPU MODIFICATION: begin — per-layer quant override (fork's create_layer).
    # Upstream calls decoder_layer_type(vllm_config=..., prefix=...), so the
    # substituted factory keeps that calling convention.
    def create_layer(*, vllm_config: VllmConfig, prefix: str, **kwargs):
        layer_idx = extract_layer_index(prefix)
        per_layer_qcfg = None if layer_idx in mix_layer else vllm_config.quant_config
        return decoder_layer_type(
            vllm_config=vllm_config,
            quant_config=per_layer_qcfg,
            prefix=prefix,
            **kwargs,
        )

    _upstream_model_init(
        self,
        vllm_config=vllm_config,
        prefix=prefix,
        decoder_layer_type=create_layer,
    )
    # PPU MODIFICATION: end
