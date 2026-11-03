# SPDX-License-Identifier: Apache-2.0
"""PPU changes to ``vllm.model_executor.models.deepseek_v2``.

Ports the fork's +108 hunk set. Four patches:

* ``Indexer.__init__`` — delegating. Upstream builds the fused
  ``wk_weights_proj`` projection; the fork splits it into separate ``wk`` /
  ``weights_proj`` linears for non-modelopt_fp4 checkpoints (PPU int8/int4
  channelwise quant wants ``wk`` quantized) and adds an int8 indexer path for
  PPU without FP8 support. Expressed as "call upstream, then adjust" because
  every change is post-construction state.
* ``Indexer.forward`` — verbatim upstream body with two marked branches
  (fp4-fused vs separate wk projection; int8 vs fp8 q quant). Unavoidable:
  the branches sit mid-body with no interceptable seam.
* ``DeepseekV2Model.__init__`` — delegating; only adds ``is_fp4_ckpt``.
* ``DeepseekV2Model.load_weights`` — verbatim upstream body with two marked
  guards. Unavoidable: both guards control method-local loader tables.
"""

from __future__ import annotations

import typing
from collections.abc import Callable, Iterable

import torch
from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.quantization.utils.int8_utils import (
    per_token_group_quant_int8,
)
from vllm.model_executor.layers.sparse_attn_indexer import (
    fused_indexer_q_rope_quant,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2Model,
    Indexer,
    _try_load_fp8_indexer_wk,
)
from vllm.model_executor.models.utils import (
    extract_layer_index,
    get_pp_missing_layer_names,
    get_spec_layer_idx_from_weight_name,
    is_pp_missing_parameter,
)
from vllm.platforms import current_platform

from vllm_sail.patch.utils import patch

_AFFECTED = ">=0.32.0,<0.33.0"

# Captured before patching so delegating replacements can call upstream.
_upstream_indexer_init = Indexer.__init__
_upstream_model_init = DeepseekV2Model.__init__


def _is_fp4_ckpt(quant_config) -> bool:
    return quant_config is not None and quant_config.get_name() == "modelopt_fp4"


@patch(
    "vllm.model_executor.models.deepseek_v2",
    "Indexer.__init__",
    reason=(
        "For non-modelopt_fp4 checkpoints PPU quant schemes need the indexer "
        "wk projection quantized, which the upstream fused wk_weights_proj "
        "cannot be; also adds the int8 indexer q-quant path for PPU without "
        "FP8 support. Delegates to upstream, then rebuilds the projection and "
        "sets the PPU flags."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream parameterises the Indexer wk projection on quant_config "
        "and exposes a platform hook for the q-quant format."
    ),
)
def indexer_init(
    self,
    vllm_config,
    config,
    hidden_size: int,
    q_lora_rank: int,
    quant_config,
    cache_config,
    topk_indices_buffer,
    prefix: str = "",
    is_inplace_rope: bool = False,
):
    _upstream_indexer_init(
        self,
        vllm_config,
        config,
        hidden_size,
        q_lora_rank,
        quant_config,
        cache_config,
        topk_indices_buffer,
        prefix=prefix,
        is_inplace_rope=is_inplace_rope,
    )

    # PPU MODIFICATION: begin
    self.is_fp4_ckpt = _is_fp4_ckpt(self.quant_config)
    self.use_ppu_int8_indexer = (
        current_platform.is_ppu() and not current_platform.supports_fp8()
    )

    if not self.is_fp4_ckpt:
        # Upstream fused wk + weights_proj into one unquantized GEMM. The
        # fork keeps that only for modelopt_fp4 checkpoints; otherwise wk
        # must accept quant_config, so rebuild the two projections
        # separately (weights_proj stays unquantized, as in the fork).
        del self.wk_weights_proj
        self.wk = ReplicatedLinear(
            hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{self.prefix}.wk",
        )
        self.weights_proj = ReplicatedLinear(
            hidden_size,
            self.n_head,
            bias=False,
            quant_config=None,
            prefix=f"{self.prefix}.weights_proj",
        )
        # Upstream's fused-indexer-q gate assumes the fused projection.
        self.use_fused_indexer_q = False
    # PPU MODIFICATION: end


@patch(
    "vllm.model_executor.models.deepseek_v2",
    "Indexer.forward",
    reason=(
        "Selects between the fused wk_weights_proj path (modelopt_fp4 only) "
        "and the separate wk/weights_proj projections installed by the "
        "Indexer.__init__ patch, and uses int8 q quantization on PPU without "
        "FP8 support. Verbatim upstream body with two marked branches; the "
        "changes sit mid-body with no delegation seam."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream parameterises the Indexer wk projection on quant_config "
        "and exposes a platform hook for the q-quant format."
    ),
)
def forward(
    self, hidden_states: torch.Tensor, qr: torch.Tensor, positions, rotary_emb
) -> torch.Tensor:
    q, _ = self.wq_b(qr)
    q = q.view(-1, self.n_head, self.head_dim)

    if current_platform.is_rocm() and self.is_inplace_rope:
        # This path should works on all platform, will remove extra
        # branches in the future
        # This fast path relies on rotary_emb mutating q and k inplace.
        # On ROCm, this is only valid for kernels used as custom ops.
        # In pytorch-native rope for inductor fusion, rotated q/k tensors
        # are not mutated inplace but returned as new tensors.
        # Fused wk + weights_proj: one GEMM, then split
        kw, _ = self.wk_weights_proj(hidden_states)
        k = kw[:, : self.head_dim]
        weights = kw[:, self.head_dim :]

        k = self.k_norm(k)

        rotary_emb(
            positions, q[..., : self.rope_dim], k[..., : self.rope_dim].unsqueeze(1)
        )
    elif self.use_fused_indexer_q and q.dtype == torch.bfloat16:
        # fused wk + weights_proj: one GEMM, then split
        kw, _ = self.wk_weights_proj(hidden_states)
        k = kw[:, : self.head_dim]
        weights = kw[:, self.head_dim :]

        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )

        q_fp8, weights = fused_indexer_q_rope_quant(
            positions,
            q,
            rotary_emb.cos_sin_cache,
            weights,
            self.softmax_scale,
            self.n_head_scale,
            rotary_emb.is_neox_style,
        )

        # rotate only the MQA K
        k_pe = k_pe.unsqueeze(1)
        q_dummy = torch.empty_like(k_pe)
        _, k_pe = rotary_emb(positions, q_dummy, k_pe)
        k_pe = k_pe.reshape(-1, self.rope_dim)
        k = torch.cat([k_pe, k_nope], dim=-1)

        return self.indexer_op(hidden_states, q_fp8, k, weights)
    else:
        q_pe, q_nope = torch.split(
            q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )
    # PPU MODIFICATION: begin
        if self.is_fp4_ckpt:
            # Fused wk + weights_proj: one GEMM, then split
            kw, _ = self.wk_weights_proj(hidden_states)
            k = kw[:, : self.head_dim]
            weights = kw[:, self.head_dim :]
        else:
            k, _ = self.wk(hidden_states)
            weights, _ = self.weights_proj(hidden_states)
    # PPU MODIFICATION: end

        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )

        q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        # Note: RoPE (NeoX) can introduce extra leading dimensions during
        # compilation so we need to reshape back to token-flattened shapes
        q_pe = q_pe.reshape(-1, self.n_head, self.rope_dim)
        k_pe = k_pe.reshape(-1, self.rope_dim)

        # `rotary_emb` is shape-preserving; `q_pe` is already
        # [num_tokens, n_head, rope_dim].
        q = torch.cat([q_pe, q_nope], dim=-1)
        # `k_pe` is [num_tokens, rope_dim] (MQA).
        k = torch.cat([k_pe, k_nope], dim=-1)

    # we only quant q here since k quant is fused with cache insertion
    q = q.view(-1, self.head_dim)
    # PPU MODIFICATION: begin

    if self.use_ppu_int8_indexer:
        q_fp8, q_scale = per_token_group_quant_int8(q, self.quant_block_size)
    else:
        q_fp8, q_scale = per_token_group_quant_fp8(
            q,
            self.quant_block_size,
            column_major_scales=False,
            use_ue8m0=self.scale_fmt is not None,
        )
    # PPU MODIFICATION: end
    q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
    q_scale = q_scale.view(-1, self.n_head)

    weights = weights * q_scale * self.softmax_scale * self.n_head_scale

    return self.indexer_op(hidden_states, q_fp8, k, weights)


@patch(
    "vllm.model_executor.models.deepseek_v2",
    "DeepseekV2Model.__init__",
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
def model_init(self, *, vllm_config, prefix: str = ""):
    _upstream_model_init(self, vllm_config=vllm_config, prefix=prefix)
    # PPU MODIFICATION: begin
    self.is_fp4_ckpt = _is_fp4_ckpt(vllm_config.quant_config)
    # PPU MODIFICATION: end


@patch(
    "vllm.model_executor.models.deepseek_v2",
    "DeepseekV2Model.load_weights",
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
        # (param_name, shard_name, shard_id)
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]
    mla_params_mapping = [
        ("fused_qkv_a_proj", "q_a_proj", 0),
        ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
    ]
    mha_params_mapping = [
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
    ]
    # PPU MODIFICATION: begin
    if self.is_fp4_ckpt:
        # Fused indexer wk + weights_proj (shard 0 = wk, shard 1 = weights_proj)
        _pending_wk_fp8 = getattr(self, "_pending_indexer_wk_fp8", None)
        if _pending_wk_fp8 is None:
            self._pending_indexer_wk_fp8 = _pending_wk_fp8 = {}
    # PPU MODIFICATION: end

    # PPU MODIFICATION: begin
        indexer_fused_mapping = [
            ("wk_weights_proj", "wk", 0),
            ("wk_weights_proj", "weights_proj", 1),
        ]
        stacked_params_mapping.extend(indexer_fused_mapping)
    # PPU MODIFICATION: end

    if self.use_mha:
        stacked_params_mapping.extend(mha_params_mapping)
    else:
        stacked_params_mapping.extend(mla_params_mapping)

    # Params for weights, fp8 weight scales, fp8 activation scales
    # (param_name, weight_name, expert_id, shard_id)
    expert_params_mapping = fused_moe_make_expert_params_mapping(
        self,
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=self.config.n_routed_experts
        + (self.config.n_shared_experts if self.fused_shared_expert_layers else 0),
        num_redundant_experts=self.num_redundant_experts,
    )

    pp_missing_layer_names = get_pp_missing_layer_names(self)
    params_dict = dict(self.named_parameters())
    loaded_params: set[str] = set()
    # With index_topk_freq>1 only some layers build an indexer, yet the
    # checkpoint ships indexer weights for all of them; track the built ones.
    indexer_present_prefixes = {
        n.rsplit(".indexer.", 1)[0] for n in params_dict if ".indexer." in n
    }
    for name, loaded_weight in weights:
        if "rotary_emb.inv_freq" in name:
            continue

        spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
        if spec_layer is not None:
            continue  # skip spec decode layers for main model

        if ".indexer." in name and (
            name.rsplit(".indexer.", 1)[0] not in indexer_present_prefixes
        ):
            continue  # this layer has no indexer; drop its checkpoint weights

        is_fusion_moe_shared_experts_layer = (
            "mlp.shared_experts" in name
            and bool(rocm_aiter_ops.is_fusion_moe_shared_experts_enabled())
            and extract_layer_index(name) in self.fused_shared_expert_layers
        )

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
            # if go with fusion option, then update name
            if (
                param_name == "fused_qkv_a_proj"
            ) and name_mapped not in params_dict:
                continue
            else:
                name = name_mapped
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue

            if is_pp_missing_parameter(name, self):
                continue

            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, loaded_weight, shard_id)
            break
        else:
            is_expert_weight = False

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
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in chunk_name:
                        continue

                    # Anyway, this is an expert weight and should not be
                    # attempted to load as other weights later
                    is_expert_weight = True

                    # Do not modify `name` since the loop may continue here
                    # Instead, create a new variable
                    name_mapped = chunk_name.replace(weight_name, param_name)

                    if is_pp_missing_parameter(name_mapped, self):
                        continue

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
                        shard_id=shard_id,
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

                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue

                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
        if name is not None and not is_fusion_moe_shared_experts_layer:
            loaded_params.add(name)

    return loaded_params
