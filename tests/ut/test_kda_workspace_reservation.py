# SPDX-License-Identifier: Apache-2.0
"""The registered constructor reserves shared buffers before KV sizing."""

from types import SimpleNamespace as NS

import pytest

from tests.ut.test_ppu_kernel_capabilities import load_patch
from tests.ut.test_ppu_kernel_capabilities import modules as _modules

modules = _modules


@pytest.mark.parametrize("initialized", [False, True])
@pytest.mark.parametrize("backend", ["flashkda", "flashinfer", "triton"])
def test_registered_constructor_reserves_selected_workspace(
    modules, initialized, backend
):
    events = []

    class Weight:
        def __init__(self, dtype="bf16", device="host-double"):
            self.dtype, self.device, self.data = dtype, device, self
            self.weight_loader = object()

        def unsqueeze(self, dim):
            assert dim == 1
            return self

    class Parent:
        def __init__(self, config, vllm_config, prefix):
            self.cache_config = NS(use_kda_recoverssm=False, use_replayssm=False)
            self.num_spec, self.tp_size, self.tp_rank, self.hidden_size = 0, 8, 0, 512
            self.quant_config, self.model_config = None, vllm_config.model_config

        def get_state_dtype(self):
            return "bf16", "fp32"

        def register_buffer(self, name, value, *, persistent):
            assert persistent is False
            setattr(self, name, value)

    def original(*args, **kwargs):
        pytest.fail("the registered constructor must handle this call")

    class Attention(Parent):
        __init__ = original
        _forward = original
        get_kv_cache_spec = original

    def linear(*args, **kwargs):
        assert set(kwargs) <= {
            "input_size",
            "output_size",
            "bias",
            "quant_config",
            "prefix",
            "params_dtype",
            "replicated_shard_id",
            "tp_size",
            "padding_size",
        }
        return NS(weight=Weight(kwargs.get("params_dtype", "bf16")), bias=None)

    def empty(*shape, dtype, device="host-double"):
        return Weight(dtype, device)

    def set_attrs(weight, attrs):
        for name, value in attrs.items():
            setattr(weight, name, value)

    def reserve(*specs):
        events.append(("reserve", specs))

    def manager():
        assert initialized
        return NS(get_simultaneous=reserve)

    def norm(head_dim, *, activation):
        assert head_dim == 128 and activation == "sigmoid"
        events.append(("norm",))
        return NS(weight=Weight())

    provider = modules(
        "vllm.models.kimi_k3.nvidia.kda",
        KimiK3DeltaAttention=Attention,
        is_flashkda_supported=original,
        is_fused_kda_decode_supported=original,
        _flashkda_prefill=original,
        divide=lambda n, d: n // d,
        ModelOptMixedPrecisionConfig=type("Mixed", (), {}),
        _KimiGDNMergedColumnParallelLinear=linear,
        ColumnParallelLinear=linear,
        RowParallelLinear=linear,
        torch=NS(float32="fp32", uint8="uint8", empty=empty),
        nn=NS(Parameter=lambda value: value),
        set_weight_attrs=set_attrs,
        sharded_weight_loader=lambda dim: ("sharded", dim),
        a_log_weight_loader=lambda dim: ("a-log", dim),
        _make_decode_conv1d_weight_loader=lambda *args: ("conv", args),
        resolve_kda_decode_backend=lambda *args: "triton",
        resolve_kda_spec_decode_backend=lambda *args: "triton",
        resolve_kda_prefill_backend=lambda *args: backend,
        FlashKDAPrefillCheckpointExporter=object,
        FusedRMSNormGated=norm,
        current_platform=NS(is_ppu=lambda: True),
        is_workspace_manager_initialized=lambda: initialized,
        current_workspace_manager=manager,
        eager_break_during_capture=lambda fn: fn,
    )
    load_patch("vllm_sail/patch/enhancement/attention/kda.py")
    config = NS(
        linear_attn_config={
            "head_dim": 128,
            "num_heads": 16,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
        }
    )
    vllm_config = NS(
        additional_config={},
        model_config=NS(dtype="bf16"),
        scheduler_config=NS(max_num_seqs=4, max_num_batched_tokens=32),
        compilation_config=NS(static_forward_context={}),
    )
    layer = provider.KimiK3DeltaAttention(config, vllm_config, prefix="layer0")
    assert vllm_config.compilation_config.static_forward_context == {"layer0": layer}
    if initialized and backend != "triton":
        expected = (
            (
                ((1, 32, 2, 128), "bf16"),
                ((4, 2, 128, 128), "fp32"),
                ((4, 2, 128, 128), "fp32"),
                ((0,), "uint8"),
            )
            if backend == "flashkda"
            else (((1, 32, 2, 128), "bf16"),)
        )
        assert events == [("reserve", expected), ("norm",)]
    else:
        assert events == [("norm",)]
