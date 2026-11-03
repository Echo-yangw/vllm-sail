# SPDX-License-Identifier: Apache-2.0
"""Load FP8 or BF16 source parameters and delegate target execution."""

import torch
from torch.nn import Parameter
from vllm.model_executor.layers.linear import LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.parameter import BlockQuantScaleParameter
from vllm.model_executor.utils import replace_parameter, set_weight_attrs

from .loading import guard_weights
from .quant_utils import dequantize_fp8, quantize_int8


class Fp8ConversionLinearMethod(LinearMethodBase):
    def __init__(self, target_dtype, source_dtype="fp8"):
        self.target_dtype = target_dtype
        self.source_dtype = source_dtype
        self.converted = False
        self.kernel = None
        self.unquantized = UnquantizedLinearMethod()

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        if params_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("Conversion requires BF16 or FP16 activations")
        if self.source_dtype == "fp8" and any(
            n % 128 for n in output_partition_sizes[:-1]
        ):
            raise ValueError("Fused FP8 output boundaries must align to 128 rows")
        if (
            self.source_dtype == "fp8"
            and input_size != input_size_per_partition
            and input_size_per_partition % 128
        ):
            raise ValueError("FP8 input TP shards must align to 128 columns")
        if input_size % input_size_per_partition:
            raise ValueError("FP8 input TP shards must evenly divide the input")
        if (
            self.source_dtype == "fp8"
            and output_size != sum(output_partition_sizes)
            and any(n % 128 for n in output_partition_sizes)
        ):
            raise ValueError("FP8 output TP shards must align to 128 rows")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = sum(output_partition_sizes)
        layer.orig_dtype = params_dtype
        layer.weight_block_size = [128, 128] if self.source_dtype == "fp8" else None
        self.input_sharded = input_size != input_size_per_partition
        self.input_tp_size = input_size // input_size_per_partition
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=torch.float8_e4m3fn
                if self.source_dtype == "fp8"
                else torch.bfloat16,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            weight, {"input_dim": 1, "output_dim": 0, **extra_weight_attrs}
        )
        layer.register_parameter("weight", weight)
        guard_weights(layer, ("weight",), weight.dtype)
        if self.source_dtype == "bf16":
            layer.register_parameter("weight_scale_inv", None)
            return
        scale = BlockQuantScaleParameter(
            data=torch.full(
                (
                    (sum(output_partition_sizes) + 127) // 128,
                    (input_size_per_partition + 127) // 128,
                ),
                float("nan"),
                dtype=torch.float32,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=extra_weight_attrs.get("weight_loader"),
        )
        layer.register_parameter("weight_scale_inv", scale)

    def process_weights_after_loading(self, layer):
        if self.converted:
            return
        weight = (
            dequantize_fp8(layer.weight, layer.weight_scale_inv)
            if self.source_dtype == "fp8"
            else layer.weight.data
        )
        if self.target_dtype == "int8":
            from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
                Int8ScaledMMLinearLayerConfig,
            )

            from vllm_sail.model_executor.kernels.linear.scaled_mm.ppu import (
                PPUInt8ScaledMMLinearKernel,
            )

            group = None
            if self.input_sharded:
                from vllm.distributed import get_tp_group

                tp = get_tp_group()
                if tp.world_size != self.input_tp_size:
                    raise ValueError("Input shards do not match the active TP group")
                group = tp.device_group
            weight, scales = quantize_int8(weight, tp_group=group)
            layer.register_parameter(
                "weight_scale", Parameter(scales, requires_grad=False)
            )
            self.kernel = PPUInt8ScaledMMLinearKernel(
                Int8ScaledMMLinearLayerConfig(
                    is_channelwise=True,
                    is_static_input_scheme=False,
                    input_symmetric=True,
                ),
                [
                    "weight",
                    "weight_scale",
                    "input_scale",
                    "input_zero_point",
                    "azp_adj",
                ],
            )
        if self.target_dtype == "bf16":
            weight = weight.to(layer.orig_dtype)
        replace_parameter(layer, "weight", weight)
        layer.weight_block_size = None
        if self.kernel is not None:
            self.kernel.process_weights_after_loading(layer)
        else:
            self.unquantized.process_weights_after_loading(layer)
        replace_parameter(layer, "weight_scale_inv", None)
        self.converted = True

    def apply(self, layer, x, bias=None):
        if self.target_dtype == "int8":
            assert self.kernel is not None
            return self.kernel.apply_weights(layer, x, bias)
        return self.unquantized.apply(layer, x, bias)
