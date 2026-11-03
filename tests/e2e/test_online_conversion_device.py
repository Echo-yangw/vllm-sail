# SPDX-License-Identifier: Apache-2.0
"""Device gates for loading-time FP8 conversion and existing Dense backends."""

import pytest

pytestmark = pytest.mark.ppu


@pytest.fixture
def runtime(monkeypatch):
    torch = pytest.importorskip("torch")
    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm.platforms import current_platform

    if not current_platform.is_ppu():
        pytest.skip("requires PPU")
    # Standalone, unsharded layer test; no distributed process group is needed.
    import vllm.model_executor.parameter as parameter

    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.v1.worker.workspace import init_workspace_manager

    with set_current_vllm_config(VllmConfig()):
        init_workspace_manager(torch.device("cuda:0"))
        yield torch


@pytest.mark.parametrize(
    "source,target", [("fp8", "bf16"), ("fp8", "int8"), ("bf16", "int8")]
)
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("activation_dtype", ["bfloat16", "float16"])
def test_dense_conversion_forward_and_idempotency(
    runtime, source, target, bias, activation_dtype
):
    torch = runtime
    dtype = getattr(torch, activation_dtype)
    from vllm_sail.model_executor.layers.quantization.online_conversion.linear import (
        Fp8ConversionLinearMethod,
    )
    from vllm_sail.model_executor.layers.quantization.online_conversion.quant_utils import (
        dequantize_fp8,
        quantize_int8,
    )

    torch.manual_seed(23)
    with torch.device("cuda"):
        layer = torch.nn.Module()
        method = Fp8ConversionLinearMethod(target, source_dtype=source)
        method.create_weights(layer, 256, [128, 128], 256, 256, dtype)
        layer.weight.data.copy_(torch.randn(256, 256).to(layer.weight.dtype))
        if source == "fp8":
            layer.weight_scale_inv.data.fill_(0.03125)
            decoded = dequantize_fp8(layer.weight, layer.weight_scale_inv)
        else:
            decoded = layer.weight.detach().clone()
        expected_q, expected_scale = quantize_int8(decoded)
        method.process_weights_after_loading(layer)
        assert layer.weight_scale_inv is None
        pointer = layer.weight.data_ptr()
        method.process_weights_after_loading(layer)
        assert layer.weight.data_ptr() == pointer
        if target == "int8":
            expected_q = expected_q if method.kernel.weight_RowMajor else expected_q.t()
            assert torch.equal(layer.weight, expected_q)
            assert torch.equal(layer.weight_scale, expected_scale)
            reference_weight = (
                expected_q if method.kernel.weight_RowMajor else expected_q.t()
            ).float() * expected_scale
        else:
            assert torch.equal(layer.weight, decoded.to(dtype))
            reference_weight = decoded.float()
        x = torch.randn(2, 7, 256, dtype=dtype)
        b = torch.randn(256, dtype=dtype) if bias else None
        actual = method.apply(layer, x, b)
        # W8A8 adds dynamic activation quantization; compare against decoded
        # weight matmul with a bounded relative norm error, not exact outputs.
        expected = x.float() @ reference_weight.t()
        if b is not None:
            expected += b.float()
        assert actual.shape == expected.shape
        assert torch.isfinite(actual).all()
        relative = (actual.float() - expected).norm() / expected.norm()
        assert relative.item() < (0.025 if target == "int8" else 0.01)


@pytest.mark.parametrize(
    "source,target", [("fp8", "bf16"), ("fp8", "int8"), ("bf16", "int8")]
)
def test_moe_conversion_forward(runtime, source, target):
    torch = runtime
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    from vllm_sail.model_executor.layers.quantization.bf16_to_int8 import (
        Bf16ToInt8Config,
    )
    from vllm_sail.model_executor.layers.quantization.fp8_to_bf16 import Fp8ToBf16Config
    from vllm_sail.model_executor.layers.quantization.fp8_to_int8 import Fp8ToInt8Config
    from vllm_sail.model_executor.layers.quantization.online_conversion.quant_utils import (
        dequantize_fp8,
    )

    parallel = FusedMoEParallelConfig(
        tp_size=1,
        pcp_size=1,
        dp_size=1,
        ep_size=1,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=1,
        use_ep=False,
        all2all_backend="allgather_reducescatter",
        enable_eplb=False,
    )
    moe = FusedMoEConfig(
        num_experts=4,
        experts_per_token=2,
        hidden_dim=128,
        intermediate_size=256,
        num_local_experts=4,
        num_logical_experts=4,
        activation=MoEActivation.SILU,
        device="cuda",
        routing_method=RoutingMethodType.Default,
        moe_parallel_config=parallel,
        in_dtype=torch.bfloat16,
    )
    manager = ExpertMapManager(
        max_num_batched_tokens=8,
        top_k=2,
        global_num_experts=4,
        num_redundant_experts=0,
        num_expert_group=None,
        moe_parallel_config=parallel,
        placement_strategy="linear",
        enable_eplb=False,
    )
    if source == "bf16":
        config = Bf16ToInt8Config()
    else:
        cls = Fp8ToInt8Config if target == "int8" else Fp8ToBf16Config
        config = cls({"quant_method": "fp8", "weight_block_size": [128, 128]})
    config.maybe_update_config("unknown")
    torch.manual_seed(56)
    with torch.device("cuda"):
        layer = RoutedExperts(
            "model.layers.0.mlp.experts", torch.bfloat16, moe, config, manager
        )
        reference = {}
        for name in ("w13", "w2"):
            weight = getattr(layer, name + "_weight")
            weight.data.copy_(
                (torch.randn(weight.shape) * (1 if source == "fp8" else 0.03)).to(
                    weight.dtype
                )
            )
            scales = getattr(layer, name + "_weight_scale_inv")
            if scales is not None:
                scales.data.fill_(0.03125)
                reference[name] = torch.stack(
                    [dequantize_fp8(w, s) for w, s in zip(weight, scales, strict=True)]
                ).float()
            else:
                reference[name] = weight.detach().float().clone()
        x = torch.randn(4, 128, dtype=torch.bfloat16)
        ids = torch.tensor([[0, 1], [1, 2], [2, 3], [0, 3]], dtype=torch.int32)
        routing = torch.full((4, 2), 0.5, dtype=torch.float32)
        expected = torch.zeros_like(x, dtype=torch.float32)
        for token in range(4):
            for expert in ids[token].tolist():
                gate, up = (reference["w13"][expert] @ x[token].float()).chunk(2)
                expected[token] += 0.5 * (
                    reference["w2"][expert] @ (torch.nn.functional.silu(gate) * up)
                )
        layer.quant_method.process_weights_after_loading(layer)
        pointer = layer.w13_weight.data_ptr()
        layer.quant_method.process_weights_after_loading(layer)
        assert layer.w13_weight.data_ptr() == pointer
        actual = layer.quant_method.apply(layer, x, routing, ids, None, None)
        assert actual.shape == expected.shape
        assert torch.isfinite(actual).all()
        assert ((actual.float() - expected).norm() / expected.norm()).item() < (
            0.05 if target == "int8" else 0.02
        )
