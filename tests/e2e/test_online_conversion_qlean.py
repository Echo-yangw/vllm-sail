# SPDX-License-Identifier: Apache-2.0
"""Optional same-device reference against the installed qlean implementation."""

import importlib.metadata

import pytest

pytestmark = pytest.mark.ppu


@pytest.mark.parametrize("source", ["fp8", "bf16"])
@pytest.mark.parametrize("shape", [(128, 256), (129, 257)])
def test_qlean_weight_and_scale_equivalence(source, shape, record_property):
    torch = pytest.importorskip("torch")
    pytest.importorskip("qlean")
    from qlean.modifiers.dequant.base import weight_dequant
    from qlean.modifiers.utils import int8_channel_weight_quant

    from vllm_sail.model_executor.layers.quantization.online_conversion.quant_utils import (
        dequantize_fp8,
        quantize_int8,
    )

    record_property("qlean_version", importlib.metadata.version("qlean"))
    record_property("torch_version", torch.__version__)
    record_property("device", torch.cuda.get_device_name())
    torch.manual_seed(56)
    weight = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    if source == "fp8":
        weight = weight.to(torch.float8_e4m3fn)
        scales = (
            torch.rand(
                ((shape[0] + 127) // 128, (shape[1] + 127) // 128),
                device="cuda",
                dtype=torch.float32,
            )
            + 0.01
        )
        old_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            reference = weight_dequant(weight.float(), scales)
        finally:
            torch.set_default_dtype(old_dtype)
        decoded = dequantize_fp8(weight, scales)
        assert torch.equal(decoded, reference)
    else:
        decoded = reference = weight
    q, scale = quantize_int8(decoded)
    reference_q, reference_scale = int8_channel_weight_quant(reference)
    assert torch.equal(q, reference_q)
    assert torch.equal(scale, reference_scale)
