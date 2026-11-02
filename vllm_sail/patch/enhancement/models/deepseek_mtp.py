# SPDX-License-Identifier: Apache-2.0
"""PPU changes to ``vllm.model_executor.models.deepseek_mtp``.

Ports the fork's +19 hunk set, the MTP-side twin of the ``deepseek_v2``
indexer changes:

* ``DeepSeekMTP.__init__`` — delegating; only adds ``is_fp4_ckpt``.
* ``DeepSeekMTP.load_weights`` — verbatim upstream body with two marked
  guards. Unavoidable: both guards control method-local loader tables (the
  fused indexer-wk stacked mapping and the FP8 wk upcast loader only apply
  to modelopt_fp4 checkpoints).
"""

from __future__ import annotations

import typing
from collections.abc import Callable, Iterable

import torch
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.model_loader.mtp_validation import (
    is_mtp_completeness_check_enabled,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.deepseek_mtp import DeepSeekMTP
from vllm.model_executor.models.deepseek_v2 import _try_load_fp8_indexer_wk
from vllm.model_executor.models.utils import (
    get_pp_missing_layer_names,
    get_spec_layer_idx_from_weight_name,
)

from vllm_sail.patch.utils import patch

_AFFECTED = ">=0.32.0,<0.33.0"

# Captured before patching so the delegating replacement can call upstream.
_upstream_init = DeepSeekMTP.__init__


@patch(
    "vllm.model_executor.models.deepseek_mtp",
    "DeepSeekMTP.__init__",
    reason=(
        "Adds the is_fp4_ckpt flag the load_weights patch keys the fused "
        "indexer-wk weight mapping on. Delegates to upstream unchanged."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream load_weights stops special-casing modelopt_fp4 indexer "
        "weights, making the flag unused."
    ),
)
def init(self, *, vllm_config, prefix: str = ""):
    _upstream_init(self, vllm_config=vllm_config, prefix=prefix)
    # PPU MODIFICATION: begin
    self.is_fp4_ckpt = (
        self.quant_config is not None
        and self.quant_config.get_name() == "modelopt_fp4"
    )
    # PPU MODIFICATION: end


@patch(
    "vllm.model_executor.models.deepseek_mtp",
    "DeepSeekMTP.load_weights",
    reason=(
        "Restricts the fused indexer wk/weights_proj stacked-parameter "
        "mapping and the FP8 wk upcast loader to modelopt_fp4 checkpoints; "
        "non-fp4 PPU quant schemes load separate wk/weights_proj weights. "
        "Verbatim upstream body with two marked guards; both control "
        "method-local loader tables, so delegation is impossible."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream load_weights stops special-casing modelopt_fp4 indexer "
        "weights, making both guards no-ops."
    ),
)
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    stacked_params_mapping = [
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
        ("fused_qkv_a_proj", "q_a_proj", 0),
        ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
    ]

    # PPU MODIFICATION: begin
    if self.is_fp4_ckpt:
        # Fused indexer wk + weights_proj (shard 0 = wk, shard 1 = weights_proj)
        indexer_fused_mapping = [
            ("wk_weights_proj", "wk", 0),
            ("wk_weights_proj", "weights_proj", 1),
        ]
        stacked_params_mapping.extend(indexer_fused_mapping)
    # PPU MODIFICATION: end

    expert_params_mapping = fused_moe_make_expert_params_mapping(
        self,
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=self.config.n_routed_experts
        + (
            self.config.n_shared_experts
            if self.is_fused_shared_expert_enabled
            else 0
        ),
    )

    pp_missing_layer_names = get_pp_missing_layer_names(self)
    params_dict = dict(self.named_parameters())
    loaded_params: set[str] = set()
    # Weight and scale can arrive in separate load_weights calls.
    _pending_wk_fp8 = getattr(self, "_pending_indexer_wk_fp8", None)
    if _pending_wk_fp8 is None:
        self._pending_indexer_wk_fp8 = _pending_wk_fp8 = {}
    for name, loaded_weight in weights:
        if "rotary_emb.inv_freq" in name:
            continue
        spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
        if spec_layer is None:
            # A tied top-level embed_tokens has no spec layer to rewrite
            # from; the draft needs its own copy under PP.
            param = params_dict.get(name) if "embed_tokens" in name else None
            if param is not None:
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
            continue
        is_fusion_moe_shared_experts_layer = (
            self.is_fused_shared_expert_enabled and ("mlp.shared_experts" in name)
        )
        name = self._rewrite_spec_layer_name(spec_layer, name)

    # PPU MODIFICATION: begin
        if self.is_fp4_ckpt and _try_load_fp8_indexer_wk(
    # PPU MODIFICATION: end
            name,
            loaded_weight,
            _pending_wk_fp8,
            params_dict,
            loaded_params,
            pp_missing_layer_names,
        ):
            continue

        for param_name, weight_name, shard_id in stacked_params_mapping:
            # Skip non-stacked layers and experts (experts handled below).
            if weight_name not in name:
                continue
            # We have mlp.experts[0].gate_proj in the checkpoint.
            # Since we handle the experts below in expert_params_mapping,
            # we need to skip here BEFORE we update the name, otherwise
            # name will be updated to mlp.experts[0].gate_up_proj, which
            # will then be updated below in expert_params_mapping
            # for mlp.experts[0].gate_gate_up_proj, which breaks load.
            if ("mlp.experts." in name) and name not in params_dict:
                continue
            if is_fusion_moe_shared_experts_layer:
                continue
            name_mapped = name.replace(weight_name, param_name)

            # QKV fusion is optional, fall back to normal
            # weight loading if it's not enabled
            if (
                param_name == "fused_qkv_a_proj"
            ) and name_mapped not in params_dict:
                continue
            else:
                name = name_mapped

            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue

            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, loaded_weight, shard_id)
            break
        else:
            # Special handling: when AITER fusion_shared_experts is enabled,
            # checkpoints may provide a single widened shared_experts tensor
            # without explicit expert indices
            # (e.g. ...mlp.shared_experts.gate_proj.weight).
            # For models with multiple shared experts, split that tensor
            # evenly into per-shared-expert slices and load them into
            # appended expert slots mlp.experts.{n_routed_experts + j}.*
            # accordingly.
            num_chunks = 1
            if is_fusion_moe_shared_experts_layer:
                num_chunks = getattr(self.config, "n_shared_experts", 1) or 1
                # Determine split axis based on op type
                # gate/up: ColumnParallel → split along dim 0
                # down: RowParallel → split along dim 1
                split_dim = (
                    1
                    if ("down_proj.weight" in name and loaded_weight.ndim > 1)
                    else 0
                )
                total = loaded_weight.shape[split_dim]
                assert total % num_chunks == 0, (
                    f"Shared expert weight dim {total} "
                    f"not divisible by num_chunks {num_chunks}"
                )
                chunk_size = total // num_chunks

            for j in range(num_chunks):
                chunk_name = name
                weight_to_load = loaded_weight

                if is_fusion_moe_shared_experts_layer:
                    chunk_slice = slice(j * chunk_size, (j + 1) * chunk_size)
                    if loaded_weight.ndim == 1:
                        weight_to_load = loaded_weight[chunk_slice]
                    elif split_dim == 0:
                        weight_to_load = loaded_weight[chunk_slice, :]
                    else:
                        weight_to_load = loaded_weight[:, chunk_slice]
                    # Synthesize an expert-style name so expert mapping
                    # can route it
                    chunk_name = name.replace(
                        "mlp.shared_experts",
                        f"mlp.experts.{self.config.n_routed_experts + j}",
                    )

                # Use expert_params_mapping to locate the destination
                # param and delegate to its expert-aware weight_loader
                # with expert_id.
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, expert_shard_id = mapping
                    if weight_name not in chunk_name:
                        continue

                    # Anyway, this is an expert weight and should not be
                    # attempted to load as other weights later
                    is_expert_weight = True

                    # Do not modify `name` since the loop may continue here
                    # Instead, create a new variable
                    name_mapped = chunk_name.replace(weight_name, param_name)

                    param = params_dict[name_mapped]
                    # We should ask the weight loader to return success or
                    # not here since otherwise we may skip experts with
                    # other available replicas.
                    weight_loader = typing.cast(
                        Callable[..., bool], param.weight_loader
                    )
                    success = weight_loader(
                        param,
                        weight_to_load,
                        name_mapped,
                        shard_id=expert_shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        if not is_fusion_moe_shared_experts_layer:
                            name = name_mapped
                        else:
                            loaded_params.add(name_mapped)
                        break
                else:
                    if is_expert_weight:
                        # We've checked that this is an expert weight
                        # However it's not mapped locally to this rank
                        # So we simply skip it
                        continue

                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    remapped_name = maybe_remap_kv_scale_name(name, params_dict)
                    if remapped_name is None:
                        continue
                    name = remapped_name

                    # According to DeepSeek-V3 Technical Report, MTP modules
                    # shares embedding layer. We only load the first weights.
                    if (
                        spec_layer != self.model.mtp_start_layer_idx
                        and ".layers" not in name
                    ):
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
        if not is_fusion_moe_shared_experts_layer:
            loaded_params.add(name)

    # Validate that weights were loaded for each expected MTP layer.
    loaded_layers: set[int] = set()
    for param_name in loaded_params:
        spec_layer = get_spec_layer_idx_from_weight_name(self.config, param_name)
        if spec_layer is not None:
            loaded_layers.add(spec_layer)
    for layer_idx in range(
        self.model.mtp_start_layer_idx,
        self.model.mtp_start_layer_idx + self.model.num_mtp_layers,
    ):
        if layer_idx not in loaded_layers and is_mtp_completeness_check_enabled():
            raise ValueError(
                f"MTP speculative decoding layer {layer_idx} weights "
                f"missing from checkpoint. The checkpoint may have "
                f"been quantized without including the MTP layers. "
                f"Use a checkpoint that includes MTP layer weights, "
                f"or disable speculative decoding."
            )

    return loaded_params
