# SPDX-License-Identifier: Apache-2.0
"""Regression for INT8 MoE warmup after online FP8 scale release."""

from types import SimpleNamespace

import pytest


def test_moe_warmup_reads_converted_scales(monkeypatch):
    torch = pytest.importorskip("torch")
    import vllm.utils.import_utils as vllm_import_utils

    # The optional Humming package in the PPU test image predates vLLM's
    # dtype schema; this W8A8 test does not use that backend.
    monkeypatch.setattr(vllm_import_utils, "has_humming", lambda: False)
    from vllm_sail.model_executor.warmup import deep_gemm_warmup as warmup

    class FakeMoERunner:
        def __init__(self):
            self.routed_experts = SimpleNamespace(
                w13_weight=torch.zeros((2, 4, 4), dtype=torch.int8),
                w2_weight=torch.zeros((2, 4, 4), dtype=torch.int8),
                w13_weight_scale_inv=None,
                w2_weight_scale_inv=None,
                w13_weight_scale=torch.ones((2, 4, 1)),
                w2_weight_scale=torch.ones((2, 4, 1)),
                top_k=2,
            )

    monkeypatch.setattr(warmup, "MoERunner", FakeMoERunner)
    runner = FakeMoERunner()
    w13, w13_scale, w2, w2_scale, top_k = warmup._extract_data_from_fused_moe_module(
        runner
    )
    assert w13 is runner.routed_experts.w13_weight
    assert w2 is runner.routed_experts.w2_weight
    assert w13_scale is runner.routed_experts.w13_weight_scale
    assert w2_scale is runner.routed_experts.w2_weight_scale
    assert top_k == 2
