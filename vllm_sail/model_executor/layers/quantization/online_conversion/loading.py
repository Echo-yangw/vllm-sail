# SPDX-License-Identifier: Apache-2.0
"""Check source tensors before a loader can cast them to the parameter dtype."""

from functools import wraps

import torch
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import UnquantizedLinearMethod


def checked_loader(loader, expected_dtype):
    @wraps(loader)
    def load(param, loaded_weight, *args, **kwargs):
        if loaded_weight.dtype != expected_dtype:
            raise ValueError(
                f"Checkpoint storage/selection mismatch: expected {expected_dtype}, "
                f"received {loaded_weight.dtype}; check modules_to_not_convert"
            )
        return loader(param, loaded_weight, *args, **kwargs)

    return load


def guard_weights(layer, names, dtype):
    for name in names:
        param = getattr(layer, name)
        loader = getattr(param, "weight_loader", None)
        if loader is not None:
            param.weight_loader = checked_loader(loader, dtype)


class CheckedUnquantizedLinearMethod(UnquantizedLinearMethod):
    def create_weights(self, layer, *args, **kwargs):
        super().create_weights(layer, *args, **kwargs)
        guard_weights(layer, ("weight",), torch.bfloat16)


class CheckedUnquantizedMoEMethod(UnquantizedFusedMoEMethod):
    def create_weights(self, layer, *args, **kwargs):
        super().create_weights(layer, *args, **kwargs)
        guard_weights(layer, ("w13_weight", "w2_weight"), torch.bfloat16)
