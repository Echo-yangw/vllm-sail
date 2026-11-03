# SPDX-License-Identifier: Apache-2.0
"""FP8/BF16 source loading for the vLLM 0.31 RoutedExperts interface."""

import torch
from torch.nn import Parameter
from vllm.model_executor.layers.fused_moe import FusedMoeWeightScaleSupported
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kInt8DynamicTokenSym,
    kInt8StaticChannelSym,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs

from .loading import guard_weights
from .quant_utils import dequantize_fp8, quantize_int8


class Fp8SourceMoE:
    """Shared source allocation; target kernels see only converted parameters."""

    supports_pre_processed_weights = False
    source_dtype = "fp8"
    converted = False

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        if params_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("Conversion requires BF16 or FP16 activations")
        if (
            self.source_dtype == "fp8"
            and self.moe.tp_size > 1
            and self.moe.intermediate_size_per_partition_unpadded % 128
        ):
            raise ValueError("FP8 MoE TP shards must align to 128 elements")
        self.source_hidden_size = self.moe.hidden_dim_unpadded
        self.source_intermediate_size = (
            self.moe.intermediate_size_per_partition_unpadded
        )
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = [128, 128] if self.source_dtype == "fp8" else None
        shards = 2 if self.moe.is_act_and_mul else 1
        shapes = {
            "w13": (shards * intermediate_size_per_partition, hidden_size),
            "w2": (hidden_size, intermediate_size_per_partition),
        }
        for name, (rows, cols) in shapes.items():
            weight = Parameter(
                torch.empty(
                    num_experts,
                    rows,
                    cols,
                    dtype=torch.float8_e4m3fn
                    if self.source_dtype == "fp8"
                    else torch.bfloat16,
                ),
                requires_grad=False,
            )
            layer.register_parameter(f"{name}_weight", weight)
            set_weight_attrs(weight, extra_weight_attrs)
            if self.source_dtype == "fp8":
                scale_rows = (
                    shards * ((intermediate_size_per_partition + 127) // 128)
                    if name == "w13"
                    else (rows + 127) // 128
                )
                scale = Parameter(
                    torch.full(
                        (num_experts, scale_rows, (cols + 127) // 128),
                        float("nan"),
                        dtype=torch.float32,
                    ),
                    requires_grad=False,
                )
                layer.register_parameter(f"{name}_weight_scale_inv", scale)
                set_weight_attrs(
                    scale,
                    {
                        **extra_weight_attrs,
                        "quant_method": FusedMoeWeightScaleSupported.BLOCK.value,
                    },
                )
            else:
                layer.register_parameter(f"{name}_weight_scale_inv", None)
            if self.moe.has_bias:
                bias = Parameter(
                    torch.zeros(
                        num_experts,
                        rows,
                        dtype=params_dtype,
                    ),
                    requires_grad=False,
                )
                layer.register_parameter(f"{name}_bias", bias)
                set_weight_attrs(bias, extra_weight_attrs)
        layer.w13_input_scale = None
        layer.w2_input_scale = None
        guard_weights(layer, ("w13_weight", "w2_weight"), layer.w13_weight.dtype)

    def _convert(self, layer, *, to_int8):
        group = None
        if to_int8 and self.moe.tp_size > 1:
            from vllm.distributed import get_dp_group, get_pcp_group, get_tp_group

            tp = get_tp_group()
            groups = [tp]
            if self.moe.dp_size > 1:
                groups.append(get_dp_group())
            if self.moe.pcp_size > 1:
                groups.append(get_pcp_group())
            size = 1
            for axis in groups:
                size *= axis.world_size
            if size != self.moe.tp_size:
                raise ValueError("MoE input shards do not match TP/DP/PCP groups")
            group = tuple(axis.device_group for axis in groups if axis.world_size > 1)
        for name in ("w13", "w2"):
            source = getattr(layer, f"{name}_weight")
            scales = getattr(layer, f"{name}_weight_scale_inv")
            target = torch.zeros_like(
                source, dtype=torch.int8 if to_int8 else layer.orig_dtype
            )
            target_scales = (
                torch.full(
                    (*source.shape[:-1], 1),
                    1.0,  # Match offline target scale initialization for padded rows.
                    device=source.device,
                    dtype=torch.float32,
                )
                if to_int8
                else None
            )
            shards = 2 if name == "w13" and self.moe.is_act_and_mul else 1
            for expert in range(source.shape[0]):
                for shard, (weight, scale) in enumerate(
                    zip(
                        source[expert].chunk(shards, 0),
                        scales[expert].chunk(shards, 0)
                        if scales is not None
                        else [None] * shards,
                        strict=True,
                    )
                ):
                    # Decode only checkpoint cells, never unpopulated padding.
                    hidden = getattr(
                        self,
                        "source_hidden_size",
                        weight.shape[1] if name == "w13" else weight.shape[0],
                    )
                    intermediate = getattr(
                        self,
                        "source_intermediate_size",
                        weight.shape[0] if name == "w13" else weight.shape[1],
                    )
                    rows, cols = (
                        (intermediate, hidden)
                        if name == "w13"
                        else (hidden, intermediate)
                    )
                    weight = weight[:rows, :cols]
                    if scale is not None:
                        scale = scale[: (rows + 127) // 128, : (cols + 127) // 128]
                    decoded = (
                        dequantize_fp8(weight, scale) if scale is not None else weight
                    )
                    dest = target[expert].chunk(shards, 0)[shard][:rows, :cols]
                    if to_int8:
                        q, s = quantize_int8(
                            decoded, tp_group=group if name == "w2" else None
                        )
                        dest.copy_(q)
                        target_scales[expert].chunk(shards, 0)[shard][:rows].copy_(s)
                    else:
                        dest.copy_(decoded)
            replace_parameter(layer, f"{name}_weight", target)
            if target_scales is not None:
                layer.register_parameter(
                    f"{name}_weight_scale",
                    Parameter(
                        target_scales,
                        requires_grad=False,
                    ),
                )
        layer.weight_block_size = None

    def _release_scales(self, layer):
        replace_parameter(layer, "w13_weight_scale_inv", None)
        replace_parameter(layer, "w2_weight_scale_inv", None)
        self.converted = True


class Fp8ToBf16MoEMethod(Fp8SourceMoE, UnquantizedFusedMoEMethod):
    def __init__(self, layer, source_dtype="fp8"):
        super().__init__(layer.moe_config)
        self.source_dtype = source_dtype
        self.converted = False
        self.supports_pre_processed_weights = False

    def process_weights_after_loading(self, layer):
        if self.converted:
            return
        self._convert(layer, to_int8=False)
        super().process_weights_after_loading(layer)
        self._release_scales(layer)


class Fp8ToInt8MoEMethod(Fp8SourceMoE, FusedMoEMethodBase):
    def __init__(self, layer, source_dtype="fp8"):
        super().__init__(layer.moe_config)
        self.source_dtype = source_dtype
        self.converted = False
        # Resolve after plugin installation, never capture an unpatched oracle
        # callable while quantization configs are being registered.
        from vllm.model_executor.layers.fused_moe.oracle import int8 as oracle

        self.int8_backend, self.experts_cls = oracle.select_int8_moe_backend(
            config=self.moe,
            weight_key=kInt8StaticChannelSym,
            activation_key=kInt8DynamicTokenSym,
        )

    def process_weights_after_loading(self, layer):
        if self.converted:
            return
        from vllm.model_executor.layers.fused_moe.oracle import int8 as oracle

        self._convert(layer, to_int8=True)
        w13, w2 = oracle.convert_to_int8_moe_kernel_format(
            self.int8_backend,
            layer.w13_weight,
            layer.w2_weight,
            layer=layer,
            w13_scale=layer.w13_weight_scale,
        )
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        self.moe_kernel = oracle.make_int8_moe_kernel(
            int8_backend=self.int8_backend,
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            experts_cls=self.experts_cls,
            routing_tables=layer._expert_routing_tables(),
        )
        self.moe_kernel.fused_experts.process_weights_after_loading(layer)
        self._release_scales(layer)

    def get_fused_moe_quant_config(self, layer):
        from vllm.model_executor.layers.fused_moe.oracle import int8 as oracle

        config = oracle.make_int8_moe_quant_config(
            int8_backend=self.int8_backend,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=None,
            a2_scale=None,
            per_act_token_quant=True,
            w1_bias=getattr(layer, "w13_bias", None),
            w2_bias=getattr(layer, "w2_bias", None),
            layer=layer,
        )
        config.gemm1_clamp_limit = getattr(layer, "swiglu_limit", None)
        config.gemm1_alpha = getattr(layer, "swiglu_alpha", None)
        config.gemm1_beta = getattr(layer, "swiglu_beta", None)
        return config

    def apply(
        self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    ):
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )
