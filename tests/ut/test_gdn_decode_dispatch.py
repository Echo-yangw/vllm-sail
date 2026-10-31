# SPDX-License-Identifier: Apache-2.0
"""Exercise installed GDN selection patches and their upstream call sites."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from tests.support.source import function
from tests.ut.test_ppu_kernel_capabilities import load_patch
from tests.ut.test_ppu_kernel_capabilities import modules as _modules
from vllm_sail import envs
from vllm_sail.attention import gdn_decode

modules = _modules


@pytest.fixture
def gdn_provider(modules, monkeypatch, upstream_source_root):
    monkeypatch.delenv("VLLM_GDN_DECODE_KERNEL", raising=False)
    monkeypatch.delenv("VLLM_PPU_USE_PLA", raising=False)
    monkeypatch.setenv("VLLM_SAIL_USE_PLA", "1")
    modules("vllm_sail.envs", **envs.__dict__)
    modules("vllm_sail.attention.gdn_decode", **gdn_decode.__dict__)
    calls = []
    platform = NS(
        ppu=True,
        is_ppu=lambda: platform.ppu,
        is_rocm=lambda: False,
        has_device_capability=lambda _: True,
    )
    modules("vllm.platforms", current_platform=platform)
    logger = NS(info_once=lambda *a: None, warning_once=lambda *a: None)
    modules("vllm.logger", init_logger=lambda _: logger)
    torch = NS(
        float32="fp32",
        bfloat16="bf16",
        int32="int32",
        _C=NS(_dispatch_has_kernel_for_dispatch_key=lambda *a: False),
        ops=NS(_C=NS(fused_gdn_decode_post_conv_mtp=lambda *a: None)),
    )
    modules("torch", **torch.__dict__)
    modules(
        "vllm_sail.attention.pla_decode",
        get_sail_cuda_pla_k_last=lambda: calls.append("pla probe") or object(),
        get_sail_cuda_pla_k_last_packed=lambda: object(),
    )

    def original_getter():
        import os

        value = os.getenv("VLLM_GDN_DECODE_KERNEL", "cuda")
        if value not in ("cuda", "triton"):
            raise ValueError("upstream choice rejected")
        return value

    env_module = modules(
        "vllm.envs", environment_variables={"VLLM_GDN_DECODE_KERNEL": original_getter}
    )
    env_module.__getattr__ = lambda name: env_module.environment_variables[name]()
    path = (
        upstream_source_root
        / "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
    )
    namespace = dict(
        torch=torch,
        current_platform=platform,
        FUSED_GDN_STATE_DTYPES=("fp32", "bf16"),
        MAX_FUSED_GDN_MTP_TOKENS=8,
    )

    supports = function(path, "supports_fused_mtp_decode", namespace)

    class Layer:
        def __init__(self, config, vllm_config, prefix=""):
            self.head_k_dim = self.head_v_dim = 128
            self.num_k_heads, self.num_v_heads = 2, 4
            self.gqa_interleaved_layout = False
            self._ba_stream = None
            self._fused_gdn_gated_rms_norm = None
            self.norm = NS(activation="silu")
            self.state_dtype = vllm_config.state_dtype
            self.speculative_config = vllm_config.speculative_config
            self.gdn_decode_kernel = env_module.VLLM_GDN_DECODE_KERNEL
            if self.gdn_decode_kernel == "cuda":
                reason = self._fused_gdn_decode_unsupported_reason(vllm_config)
                if reason:
                    raise ValueError(reason)
            self.enable_fused_gdn_spec_decode = (
                self.gdn_decode_kernel == "cuda" and self.speculative_config is not None
            )

        def get_state_dtype(self):
            return "bf16", self.state_dtype

    for name in (
        "_fused_gdn_decode_unsupported_reason",
        "_can_use_fused_gdn_mtp_decode",
    ):
        setattr(
            Layer, name, function(path, f"QwenGatedDeltaNetAttention.{name}", namespace)
        )
    provider = modules(
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
        QwenGatedDeltaNetAttention=Layer,
        supports_fused_mtp_decode=supports,
    )
    originals = {
        name: getattr(Layer, name)
        for name in (
            "__init__",
            "_fused_gdn_decode_unsupported_reason",
            "_can_use_fused_gdn_mtp_decode",
        )
    }
    original_getters = env_module.environment_variables
    load_patch("vllm_sail/patch/enhancement/attention/gdn_decode.py")
    return NS(
        layer=Layer,
        env=env_module,
        torch=torch,
        platform=platform,
        calls=calls,
        path=path,
        config=NS(
            state_dtype="fp32",
            model_config=NS(dtype="bf16"),
            speculative_config=object(),
        ),
        provider=provider,
        originals=originals,
        original_getters=original_getters,
    )


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "attribute",
    [
        "environment_variables",
        "__init__",
        "_fused_gdn_decode_unsupported_reason",
        "_can_use_fused_gdn_mtp_decode",
    ],
)
def test_decode_patches_record_originals_and_reject_reapplication(
    gdn_provider, attribute
):
    from vllm_sail.patch.utils import PATCH_MARKER, patch

    f = gdn_provider
    if attribute == "environment_variables":
        module, target = "vllm.envs", attribute
        replacement, original = f.env.environment_variables, f.original_getters
    else:
        module = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
        target = f"QwenGatedDeltaNetAttention.{attribute}"
        replacement, original = getattr(f.layer, attribute), f.originals[attribute]
    assert replacement is not original
    assert getattr(replacement, PATCH_MARKER)[f"{module}.{target}"] is original
    with pytest.raises(RuntimeError, match="already patched"):
        patch(
            module,
            target,
            reason="GDN lifecycle regression",
            affected_versions=">=0.31.0,<0.32.0",
            remove_when="upstream registers platform GDN backends",
        )(lambda *a, **kw: None)


@pytest.mark.upstream_source
def test_stub_cannot_select_native_mtp_even_with_registered_schema(
    gdn_provider, monkeypatch
):
    f = gdn_provider
    monkeypatch.setenv("VLLM_SAIL_USE_PLA", "0")
    assert f.provider.supports_fused_mtp_decode() is True
    layer = f.layer(None, f.config)
    assert layer.gdn_decode_kernel == "triton"
    assert not layer.enable_fused_gdn_spec_decode
    layer.gdn_decode_kernel = "cuda"  # Also guard stale/prebuilt layer state.
    layer.kv_cache = (None, NS(dtype="fp32"))
    metadata = NS(
        spec_state_indices_tensor=NS(size=lambda _: 3),
        spec_sequence_masks=object(),
        num_decodes=0,
        num_spec_decodes=1,
    )
    assert not layer._can_use_fused_gdn_mtp_decode(metadata)


@pytest.mark.upstream_source
@pytest.mark.parametrize("requested", ["auto", "pla", "triton"])
def test_extended_choices_reach_layer_initialization(
    gdn_provider, monkeypatch, requested
):
    f = gdn_provider
    monkeypatch.setenv("VLLM_GDN_DECODE_KERNEL", requested)
    if requested == "pla":
        monkeypatch.setenv("VLLM_SAIL_USE_PLA", "0")
    layer = f.layer(None, f.config)
    assert f.env.VLLM_GDN_DECODE_KERNEL == requested
    assert f.env.environment_variables["VLLM_GDN_DECODE_KERNEL"]() == requested
    assert layer.gdn_decode_kernel == ("pla" if requested == "auto" else requested)
    assert not layer.enable_fused_gdn_spec_decode
    assert bool(f.calls) is (requested != "triton")


@pytest.mark.upstream_source
def test_explicit_cuda_requires_real_dispatch_kernel(gdn_provider, monkeypatch):
    f = gdn_provider
    monkeypatch.setenv("VLLM_GDN_DECODE_KERNEL", "cuda")
    with pytest.raises(ValueError, match="CUDA implementation"):
        f.layer(None, f.config)
    f.torch._C._dispatch_has_kernel_for_dispatch_key = lambda *a: True
    layer = f.layer(None, f.config)
    assert layer.gdn_decode_kernel == "cuda"
    assert layer.enable_fused_gdn_spec_decode
    assert not f.calls


@pytest.mark.upstream_source
def test_explicit_pla_rejects_bf16_state_without_changing_cache_dtype(
    gdn_provider, monkeypatch
):
    f = gdn_provider
    f.config.state_dtype = "bf16"
    monkeypatch.setenv("VLLM_GDN_DECODE_KERNEL", "pla")
    with pytest.raises(ValueError, match="float32.*mamba-ssm-cache-dtype"):
        f.layer(None, f.config)
    assert f.config.state_dtype == "bf16"
    assert not f.calls


@pytest.mark.upstream_source
def test_non_ppu_retains_upstream_parser_and_selection(gdn_provider, monkeypatch):
    f = gdn_provider
    f.platform.ppu = False
    f.torch._C._dispatch_has_kernel_for_dispatch_key = lambda *a: True
    assert f.layer(None, f.config).gdn_decode_kernel == "cuda"
    monkeypatch.setenv("VLLM_GDN_DECODE_KERNEL", "pla")
    with pytest.raises(ValueError, match="upstream choice rejected"):
        f.layer(None, f.config)


@pytest.mark.upstream_source
def test_stub_falls_through_real_upstream_forward_to_normal_core(
    gdn_provider, monkeypatch
):
    f = gdn_provider
    monkeypatch.setenv("VLLM_SAIL_USE_PLA", "0")
    layer = f.layer(None, f.config)
    calls = []

    class Tensor:
        dtype = "bf16"
        device = "cuda"

        def size(self, dim):
            return 3

        def split(self, sizes, **kwargs):
            return tuple(Tensor() for _ in sizes)

        def reshape(self, *shape):
            return self

        def contiguous(self):
            return self

    def fused(*args, **kwargs):
        raise AssertionError("The unsupported fused GDN op must never be reached")

    f.torch.ops.vllm = NS(
        qwen_gdn_attention_core=lambda *a, **kw: calls.append("normal core"),
        qwen_gdn_attention_core_fused_norm_packed=fused,
    )
    f.torch.zeros = lambda *a, **kw: Tensor()
    f.torch.empty = lambda *a, **kw: Tensor()
    layer.in_proj_qkvz = layer.in_proj_ba = lambda _: (Tensor(), None)
    layer.split_ba = lambda _: (Tensor(), Tensor())
    layer._output_projection = lambda *a: "norm + output"
    layer.key_dim, layer.value_dim, layer.tp_size, layer.prefix = 256, 512, 1, "gdn"
    forward = function(
        f.path,
        "QwenGatedDeltaNetAttention.forward_cuda",
        dict(torch=f.torch, _encode_layer_name=lambda name: name),
    )
    assert forward(layer, Tensor()) == "norm + output"
    assert calls == ["normal core"]


@pytest.mark.upstream_source
@pytest.mark.parametrize("speculative", [False, True])
def test_auto_native_selection_updates_the_spec_decode_flag(
    gdn_provider, monkeypatch, speculative
):
    f = gdn_provider
    monkeypatch.setenv("VLLM_SAIL_USE_PLA", "0")
    monkeypatch.setenv("VLLM_GDN_DECODE_KERNEL", "auto")
    f.torch._C._dispatch_has_kernel_for_dispatch_key = lambda *a: True
    f.config.speculative_config = object() if speculative else None
    layer = f.layer(None, f.config)
    assert layer.gdn_decode_kernel == "cuda"
    assert layer.enable_fused_gdn_spec_decode is speculative
