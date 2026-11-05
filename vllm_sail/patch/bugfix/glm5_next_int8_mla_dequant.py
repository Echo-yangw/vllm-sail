# SPDX-License-Identifier: Apache-2.0
"""Dequantise INT8 MLA projections of GLM-5.3-Flash while loading.

``Glm5NextDecoderLayer`` builds ``Glm5NextMLAAttention`` with
``quant_config=None`` ("MLA projections are BF16 in checkpoint").  That holds
for the FP8 checkpoints the model was written against, whose q_a_proj /
kv_a_proj_with_mqa / q_b_proj / o_proj are stored block-FP8 and dequantised to
BF16 by ``_try_load_fp8_attn_proj``.  It does *not* hold for INT8 dense
checkpoints (``mixed_precision_w4``): there the same four projections are
``int8`` with a per-output-channel ``float32`` ``weight_scale``.

With the upstream loader those weights fall through to the regular path and
``default_weight_loader`` copies the raw int8 values into the BF16 parameters
(``param.data.copy_`` casts silently), while the ``.weight_scale`` tensors are
captured by the FP8 branch (``is_scale``), find no ``weight_scale_inv`` target
and are dropped without a missing-parameter error.  Every MLA layer then runs
with weights inflated by ``1 / weight_scale`` (measured on the GLM-5.3-Flash
INT4-MoE/INT8-dense checkpoint: q_a 2171x, kv_a 1519x, q_b 1155x, o_proj
1207x), the attention scores saturate and the residual stream grows from 0.008
to ~2500 rms over the 11 DSA layers -> garbled first token.

The replacement below is the upstream function with three additions:

* ``int8`` weights are accepted by ``is_weight`` and dequantised with their
  per-channel scale (``weight.float() * scale``) before being handed to the
  parameter's ``weight_loader``;
* FP8 per-output-channel weights (Quark ``weight_scale`` with shape [out] or
  [out, 1], used by FP8-PerChannel checkpoints) are dequantised row-wise
  instead of being forced through the block-FP8 map;
* when the target parameter is itself quantised (``weight_scale_inv`` or
  ``weight_scale`` present), both the weight and its scale are handed back to
  the regular quantised loader instead of being dequantised.

The FP8 block-dequant path, the NoPE kv_a padding and the loading order are
unchanged.
"""

from __future__ import annotations

import torch
from vllm.models.glm5next.common import model as _glm5next_model

from vllm_sail.patch.utils import patch

# Log through the target module's logger: the vLLM logger tree is configured at
# the requested VLLM_LOGGING_LEVEL, while a plugin-owned logger inherits the
# root level (WARNING) and would silently drop these load-time lines.
logger = _glm5next_model.logger

_AFFECTED = ">=0.31.0,<0.32.0"
_MODULE = "vllm.models.glm5next.common.model"

_upstream_try_load_fp8_attn_proj = _glm5next_model._try_load_fp8_attn_proj
_upstream_dequant_fp8_block = _glm5next_model._dequant_fp8_block
_FP8_ATTN_PROJS = _glm5next_model._FP8_ATTN_PROJS


@patch(
    _MODULE,
    "_try_load_fp8_attn_proj",
    reason=(
        "GLM5Next keeps the four MLA projections in BF16 (the decoder passes "
        "quant_config=None), which is only true for FP8 checkpoints: the FP8 "
        "loader dequantises them on load. For INT8 dense checkpoints "
        "(mixed_precision_w4) the same projections are int8 with a "
        "per-output-channel weight_scale, so upstream copies the raw int8 "
        "values into the bf16 parameters (1155-2171x inflation) and silently "
        "drops the scales. That breaks every GLM-5.3-Flash layer of type "
        "deepseek_sparse_attention (first-token garbage) while KDA layers are "
        "unaffected. It also dequantises FP8 per-output-channel weights "
        "(FP8-PerChannel checkpoints) row-wise; upstream's block-only map "
        "materialises an out*in tensor and fails on those checkpoints."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream glm5next passes the quant config to Glm5NextMLAAttention for "
        "INT8 checkpoints, handles per-channel FP8 MLA projections, or "
        "vllm-for-sail carries an equivalent load-time dequantisation."
    ),
)
def _try_load_fp8_attn_proj(
    name,
    tensor,
    buf,
    params_dict,
    loaded_params,
    kv_a_pad_size: int,
) -> bool:
    matched = None
    for suffix, info in _FP8_ATTN_PROJS.items():
        if suffix in name:
            matched = (suffix, info)
            break
    if matched is None:
        return False
    suffix, (key, target_base, shard_id, is_kva) = matched
    is_fp8_weight = name.endswith(".weight") and tensor.dtype == torch.float8_e4m3fn
    is_int8_weight = name.endswith(".weight") and tensor.dtype == torch.int8
    is_weight = is_fp8_weight or is_int8_weight
    # Need to accept both the DeepSeek-native ``weight_scale_inv`` and the Quark
    # ``weight_scale`` names before feeding the shared block dequant below.
    is_scale = "weight_scale_inv" in name or name.endswith(".weight_scale")
    if not is_weight and not is_scale:
        return False

    layer_prefix = name.rsplit(suffix, 1)[0]
    target_w = f"{layer_prefix}.{target_base}.weight"
    target_s = f"{layer_prefix}.{target_base}.weight_scale_inv"
    target_s_pc = f"{layer_prefix}.{target_base}.weight_scale"
    # If the model actually kept this projection quantised, let the normal path
    # handle it: block-FP8 targets carry ``weight_scale_inv``, per-channel FP8
    # targets carry ``weight_scale``.
    if target_s in params_dict or target_s_pc in params_dict:
        return False
    # If the target module is natively int8-quantised, the regular quantised
    # loader must handle both the weight and its weight_scale.
    target_param = params_dict.get(target_w)
    if target_param is not None and target_param.dtype == torch.int8:
        return False

    entry = buf.setdefault(layer_prefix, {}).setdefault(key, {})
    entry["weight" if is_weight else "scale"] = tensor
    if "weight" not in entry or "scale" not in entry:
        return True

    ckpt_weight, ckpt_scale = entry["weight"], entry["scale"]
    buf[layer_prefix].pop(key, None)
    if ckpt_weight.dtype == torch.int8:
        # Per-output-channel int8 dequantisation: scale is [out] or [out, 1].
        scale = ckpt_scale.to(torch.float32)
        if scale.dim() == 1:
            scale = scale.reshape(-1, 1)
        elif scale.dim() != 2 or scale.shape[1] != 1:
            raise ValueError(
                f"unexpected int8 weight_scale shape {tuple(scale.shape)} for {name}"
            )
        if scale.shape[0] != ckpt_weight.shape[0]:
            raise ValueError(
                f"int8 weight_scale has {scale.shape[0]} rows but weight has "
                f"{ckpt_weight.shape[0]} for {name}"
            )
        weight_bf16 = (ckpt_weight.to(torch.float32) * scale).to(torch.bfloat16)
        logger.info(
            "dequantised int8 MLA projection %s%s -> bf16 %s",
            layer_prefix,
            suffix,
            tuple(weight_bf16.shape),
        )
    else:
        scale = ckpt_scale.to(torch.float32)
        out_dim = ckpt_weight.shape[0]
        # FP8 per-output-channel (Quark ``weight_scale``, shape [out] or
        # [out, 1]) is not block-FP8; multiplying by a 128x128 block map would
        # materialise an out*in tensor and fail. Detect it by row count.
        per_channel = (
            (scale.dim() == 1 and scale.numel() == out_dim)
            or (
                scale.dim() == 2
                and scale.shape[1] == 1
                and scale.shape[0] == out_dim
            )
        )
        if per_channel:
            weight_bf16 = (
                ckpt_weight.to(torch.float32) * scale.reshape(-1, 1)
            ).to(torch.bfloat16)
            logger.info(
                "dequantised fp8 per-channel MLA projection %s%s -> bf16 %s",
                layer_prefix,
                suffix,
                tuple(weight_bf16.shape),
            )
        elif scale.numel() == 1:
            # FP8 per-tensor scale.
            weight_bf16 = (
                ckpt_weight.to(torch.float32) * scale.reshape(())
            ).to(torch.bfloat16)
        else:
            block_size = ckpt_weight.shape[1] // ckpt_scale.shape[1]
            weight_bf16 = _upstream_dequant_fp8_block(
                ckpt_weight, ckpt_scale, block_size
            )
    # NoPE: pad kv_a rope portion (kv_lora_rank -> kv_lora_rank + qk_rope_head_dim).
    if is_kva and kv_a_pad_size > 0:
        pad = torch.zeros(
            kv_a_pad_size,
            weight_bf16.shape[1],
            dtype=weight_bf16.dtype,
            device=weight_bf16.device,
        )
        weight_bf16 = torch.cat([weight_bf16, pad], dim=0)

    param = params_dict[target_w]
    if shard_id is None:
        param.weight_loader(param, weight_bf16)
    else:
        param.weight_loader(param, weight_bf16, shard_id)
    loaded_params.add(target_w)
    return True
