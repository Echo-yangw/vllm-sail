# SPDX-License-Identifier: Apache-2.0
"""Execute the installed upstream KDA constructor with a real CPU workspace."""

import builtins
import importlib
from types import SimpleNamespace as NS

import pytest


@pytest.mark.parametrize("backend", ["flashkda", "triton"])
def test_kda_constructor_reserves_workspace_before_norm(monkeypatch, backend):
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    import vllm_sail.patch

    vllm_sail.patch.install()
    from vllm.platforms import current_platform
    from vllm.v1.worker.workspace import WorkspaceManager

    if not current_platform.is_ppu():
        pytest.skip("requires the installed SAIL platform")
    provider = importlib.import_module("vllm.models.kimi_k3.nvidia.kda")
    attention = provider.KimiK3DeltaAttention
    manager = WorkspaceManager(torch.device("cpu"))
    actual_get = manager.get_simultaneous
    calls = []

    def reserve(*specs):
        tensors = actual_get(*specs)
        calls.append((specs, tensors))
        return tensors

    def base_init(self, config, vllm_config, prefix):
        torch.nn.Module.__init__(self)
        self.cache_config = NS(use_kda_recoverssm=False, use_replayssm=False)
        self.num_spec = 0
        self.tp_size = 1
        self.tp_rank = 0
        self.hidden_size = 256
        self.quant_config = None
        self.model_config = vllm_config.model_config

    class Linear(torch.nn.Module):
        def __init__(
            self,
            input_size,
            output_size,
            *,
            bias=False,
            quant_config=None,
            prefix="",
            params_dtype=None,
            replicated_shard_id=None,
            tp_size=None,
            padding_size=0,
        ):
            super().__init__()
            size = sum(output_size) if isinstance(output_size, list) else output_size
            self.weight = torch.nn.Parameter(
                torch.empty(
                    size + padding_size,
                    input_size,
                    dtype=params_dtype or torch.bfloat16,
                    device="cpu",
                )
            )
            self.weight.weight_loader = lambda *args, **kwargs: None

    class Norm(torch.nn.Module):
        def __init__(self, dim, activation):
            super().__init__()
            assert activation == "sigmoid"
            assert len(calls) == (1 if backend == "flashkda" else 0)
            self.weight = torch.nn.Parameter(torch.empty(dim, dtype=torch.bfloat16))
            self.eps = 1e-6

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name != "vllm._flashkda_C", "NVIDIA workspace imported on PPU"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(attention.__mro__[1], "__init__", base_init)
    monkeypatch.setattr(
        attention, "get_state_dtype", lambda self: (torch.bfloat16, torch.float32)
    )
    for name in (
        "_KimiGDNMergedColumnParallelLinear",
        "ColumnParallelLinear",
        "RowParallelLinear",
    ):
        monkeypatch.setattr(provider, name, Linear)
    monkeypatch.setattr(provider, "FusedRMSNormGated", Norm)
    monkeypatch.setattr(provider, "is_flashkda_supported", lambda *args: True)
    monkeypatch.setattr(provider, "is_workspace_manager_initialized", lambda: True)
    monkeypatch.setattr(provider, "current_workspace_manager", lambda: manager)
    monkeypatch.setattr(manager, "get_simultaneous", reserve)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    config = NS(
        linear_attn_config={
            "head_dim": 128,
            "num_heads": 2,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5,
            "use_full_rank_gate": True,
        }
    )
    vllm_config = NS(
        additional_config={
            "kda_prefill_backend": backend,
            "kda_decode_backend": "triton",
            "kda_spec_decode_backend": "auto",
        },
        model_config=NS(dtype=torch.bfloat16),
        scheduler_config=NS(max_num_batched_tokens=32, max_num_seqs=4),
        compilation_config=NS(static_forward_context={}),
    )
    layer = attention(config, vllm_config, prefix="test.layer")
    assert layer.kda_prefill_backend == backend
    assert vllm_config.compilation_config.static_forward_context["test.layer"] is layer
    if backend == "triton":
        assert calls == []
        return
    expected = (
        ((1, 32, 2, 128), torch.bfloat16),
        ((4, 2, 128, 128), torch.float32),
        ((4, 2, 128, 128), torch.float32),
        ((0,), torch.uint8),
    )
    assert len(calls) == 1
    assert calls[0][0] == layer._flashkda_buffer_specs == expected
    storage = [tensor.untyped_storage().data_ptr() for tensor in calls[0][1]]
    manager.lock()
    tensors = manager.get_simultaneous(*expected)
    assert [tensor.untyped_storage().data_ptr() for tensor in tensors] == storage
