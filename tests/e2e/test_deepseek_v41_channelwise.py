# SPDX-License-Identifier: Apache-2.0
"""Exercise channelwise checkpoint loading through the actual PPU FP8 kernel."""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.ppu


@pytest.mark.parametrize("input_size", [256, 576])
def test_channelwise_dense_loading_and_per_token_activation(monkeypatch, input_size):
    import torch
    from compressed_tensors.quantization import QuantizationArgs
    from vllm.config import VllmConfig, set_current_vllm_config

    import vllm_sail

    monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "1")
    vllm_sail.register_out_of_tree()
    # This is a single-device linear oracle; no distributed engine is started.
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (
        CompressedTensorsW8A8Fp8,
    )

    from vllm_sail.model_executor.kernels.linear.scaled_mm.ppu import (
        PPUDeepGemmFP8ScaledMMLinearKernel,
    )

    config = VllmConfig()
    config.model_config = SimpleNamespace(dtype=torch.bfloat16)
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with set_current_vllm_config(config), torch.device("cuda"):
            scheme = CompressedTensorsW8A8Fp8(
                QuantizationArgs(num_bits=8, type="float", strategy="channel"),
                is_static_input_scheme=False,
            )
            layer = torch.nn.Module()
            scheme.create_weights(
                layer,
                input_size,
                [128],
                input_size,
                128,
                torch.bfloat16,
                weight_loader=lambda param, value: param.copy_(value),
            )
            scales = torch.linspace(0.125, 0.5, 128, dtype=torch.float32)[:, None]
            layer.weight.data.fill_(2.0)
            layer.weight_scale.data.copy_(scales)
            scheme.process_weights_after_loading(layer)
            assert isinstance(scheme.fp8_linear, PPUDeepGemmFP8ScaledMMLinearKernel)
            assert layer.weight.dtype == torch.float8_e4m3fn
            assert layer.weight.shape == (128, input_size)
            x = torch.tensor([0.5, -1.0, 2.0], dtype=torch.bfloat16)
            x = x[:, None].expand(3, input_size).contiguous()
            actual = scheme.apply_weights(layer, x)
            expected = (x[:, :1].float() * (2 * input_size) * scales.T).to(
                torch.bfloat16
            )
            torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)
    finally:
        torch.set_default_dtype(previous_dtype)
