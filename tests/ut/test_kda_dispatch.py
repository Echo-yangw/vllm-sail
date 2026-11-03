# SPDX-License-Identifier: Apache-2.0
"""KDA dispatch, caller buffers and upstream API compatibility."""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tests.support.source import assert_accepts_upstream_keywords, function
from tests.ut.test_kda_workspace_reservation import configs as constructor_configs
from tests.ut.test_kda_workspace_reservation import kda_constructor as _kda_constructor
from tests.ut.test_ppu_kernel_capabilities import modules as _modules

kda_constructor = _kda_constructor
modules = _modules


@pytest.mark.parametrize("num_heads", [12, 24])
def test_pla_prefill_preserves_caller_buffers(monkeypatch, num_heads):
    calls = []
    pla = ModuleType("pla.prefill.flashkdapro")
    pla.flashkda_fwd = lambda **kw: calls.append(kw)
    monkeypatch.setitem(sys.modules, pla.__name__, pla)
    monkeypatch.setenv("VLLM_SAIL_USE_PLA", "1")

    class Tensor:
        shape = (1, 5, num_heads, 128)

        def view(self, *shape):
            return self

    class Bias:
        shape = (num_heads * 128,)

        def view(self, *shape):
            return SimpleNamespace(shape=shape)

    namespace = {
        "current_platform": SimpleNamespace(is_ppu=lambda: True),
        "ppu_envs": SimpleNamespace(VLLM_SAIL_USE_PLA=True),
        "logger": SimpleNamespace(info_once=lambda *args: None),
    }
    fn = function(
        "vllm_sail/patch/enhancement/attention/kda.py", "_flashkda_prefill", namespace
    )
    tensors = [Tensor() for _ in range(12)]
    q, k, v, g, beta, a, initial, seq, out, final, workspace, checkpoint = tensors
    bias = Bias()
    result = fn(q, k, v, g, beta, a, bias, -5.0, initial, seq, out, final, workspace)
    assert result == (out, final)
    assert calls[0]["out"] is out and calls[0]["final_state"] is final
    assert calls[0]["dt_bias"].shape == (num_heads, 128)
    assert bias.shape == (num_heads * 128,)
    with pytest.raises(NotImplementedError, match="checkpoint"):
        fn(
            q,
            k,
            v,
            g,
            beta,
            a,
            bias,
            -5.0,
            initial,
            seq,
            out,
            final,
            workspace,
            checkpoint_state=checkpoint,
        )
    assert len(calls) == 1


@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_ppu_kda_cannot_select_nvidia_kernel_when_pla_disabled(monkeypatch, mode):
    sdk = ModuleType("vllm_sail.attention.pla_kda")
    sdk.get_pla_kda_kernel = lambda mode: None
    monkeypatch.setitem(sys.modules, sdk.__name__, sdk)
    namespace = {
        "current_platform": SimpleNamespace(
            is_ppu=lambda: True,
            is_cuda=lambda: True,
            get_device_capability=lambda: SimpleNamespace(major=10),
            is_device_capability=lambda cap: True,
            is_device_capability_family=lambda cap: True,
        ),
        "torch": SimpleNamespace(bfloat16="bf16", float32="fp32"),
        "is_conv_state_dim_first": lambda: False,
    }
    fn = function(
        "vllm_sail/patch/enhancement/attention/kda.py",
        "is_flashkda_supported"
        if mode == "prefill"
        else "is_fused_kda_decode_supported",
        namespace,
    )
    args = (
        (128, "bf16", "fp32", -5.0)
        if mode == "prefill"
        else (12, 128, 4, 0, "bf16", "bf16", "fp32")
    )
    assert fn(*args) is False


@pytest.mark.parametrize(
    "name", ["is_flashkda_supported", "is_fused_kda_decode_supported"]
)
def test_device_predicate_calls_bind_all_required_arguments(name):
    fn = function("vllm_sail/patch/enhancement/attention/kda.py", name, {})
    path = Path(__file__).parents[1] / "e2e/gdn/test_kda.py"
    calls = [
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]
    assert calls
    for call in calls:
        inspect.signature(fn).bind(
            *call.args, **{kw.arg: kw.value for kw in call.keywords}
        )


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "local,local_name,upstream,upstream_name",
    [
        (
            "attention/kda",
            "_init",
            "models/kimi_k3/nvidia/kda",
            "KimiK3DeltaAttention.__init__",
        ),
        (
            "attention/kda",
            "_flashkda_prefill",
            "models/kimi_k3/nvidia/kda",
            "_flashkda_prefill",
        ),
        (
            "attention/kda",
            "is_flashkda_supported",
            "models/kimi_k3/nvidia/kda",
            "is_flashkda_supported",
        ),
        (
            "attention/kda",
            "is_fused_kda_decode_supported",
            "models/kimi_k3/nvidia/kda",
            "is_fused_kda_decode_supported",
        ),
    ],
)
def test_replacements_accept_upstream_keywords(
    upstream_source_root, local, local_name, upstream, upstream_name
):
    assert_accepts_upstream_keywords(
        local, local_name, upstream_source_root, upstream, upstream_name
    )


def wrapper_forward(monkeypatch, namespace, pla_calls):
    """Build the ``_forward`` replacement with a host PLA decode double."""
    pla = ModuleType("pla.decode.kda")
    pla.fused_kda_decode_mega_forward = lambda **kw: pla_calls.append(kw)
    monkeypatch.setitem(sys.modules, pla.__name__, pla)
    return function(
        "vllm_sail/patch/enhancement/attention/kda.py", "_forward", namespace
    )


class SlicedView:
    """A recorded view of a host double tensor."""

    def __init__(self, source, index):
        self.source, self.index = source, index


class HostTensor:
    def __init__(self, name):
        self.name = name

    def __getitem__(self, index):
        return SlicedView(self.name, index if isinstance(index, tuple) else (index,))

    def transpose(self, left, right):
        return SlicedView(self.name, "transposed")


def decode_metadata(**overrides):
    metadata = SimpleNamespace(
        num_actual_tokens=3,
        num_spec_decodes=0,
        num_prefills=0,
        num_decodes=1,
        non_spec_state_indices_tensor=HostTensor("state_indices"),
    )
    for key, value in overrides.items():
        setattr(metadata, key, value)
    return metadata


def decode_layer():
    return SimpleNamespace(
        prefix="kda",
        kda_decode_backend="cuda",
        decode_conv1d_weight=object(),
        decode_norm_weight=object(),
        kv_cache=(HostTensor("conv_state"), object()),
        conv1d=SimpleNamespace(bias=None),
        A_log=object(),
        dt_bias=object(),
        gate_lower_bound=-5.0,
        o_norm=SimpleNamespace(eps=1e-6),
    )


def test_forward_runs_the_pla_decode_kernel_on_ppu(monkeypatch):
    pla_calls, upstream_calls = [], []
    fn = wrapper_forward(
        monkeypatch,
        {
            "current_platform": SimpleNamespace(is_ppu=lambda: True),
            "ppu_envs": SimpleNamespace(VLLM_SAIL_USE_PLA=True),
            "get_forward_context": lambda: SimpleNamespace(
                attn_metadata={"kda": decode_metadata()}
            ),
            "logger": SimpleNamespace(info_once=lambda *args: None),
            "is_conv_state_dim_first": lambda: False,
            "_upstream_forward": lambda *args: upstream_calls.append(args),
        },
        pla_calls,
    )
    layer = decode_layer()
    fn(
        layer,
        HostTensor("mixed"),
        HostTensor("g1"),
        HostTensor("g2"),
        HostTensor("beta"),
        HostTensor("out"),
    )

    assert upstream_calls == []
    assert len(pla_calls) == 1
    call = pla_calls[0]
    assert call["weight"] is layer.decode_conv1d_weight
    assert call["norm_weight"] is layer.decode_norm_weight
    assert call["out"].source == "out"
    assert call["conv_state"].index == "transposed"
    assert call["state_indices"].source == "state_indices"


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_prefills": 1, "num_decodes": 0},
        {"num_spec_decodes": 1},
        {"num_decodes": 0},
    ],
)
def test_forward_leaves_other_batches_to_upstream(monkeypatch, overrides):
    pla_calls, upstream_calls = [], []
    fn = wrapper_forward(
        monkeypatch,
        {
            "current_platform": SimpleNamespace(is_ppu=lambda: True),
            "ppu_envs": SimpleNamespace(VLLM_SAIL_USE_PLA=True),
            "get_forward_context": lambda: SimpleNamespace(
                attn_metadata={"kda": decode_metadata(**overrides)}
            ),
            "logger": SimpleNamespace(info_once=lambda *args: None),
            "is_conv_state_dim_first": lambda: False,
            "_upstream_forward": lambda *args: upstream_calls.append(args),
        },
        pla_calls,
    )
    tensors = [decode_layer(), *(object() for _ in range(5))]
    fn(*tensors)

    assert pla_calls == []
    assert upstream_calls == [tuple(tensors)]


@pytest.mark.parametrize("is_ppu,pla", [(False, True), (True, False)])
def test_forward_delegates_to_upstream_off_the_pla_path(is_ppu, pla):
    upstream_calls = []
    fn = function(
        "vllm_sail/patch/enhancement/attention/kda.py",
        "_forward",
        {
            "current_platform": SimpleNamespace(is_ppu=lambda: is_ppu),
            "ppu_envs": SimpleNamespace(VLLM_SAIL_USE_PLA=pla),
            "_upstream_forward": lambda *args: upstream_calls.append(args),
        },
    )
    tensors = [object() for _ in range(6)]
    fn(*tensors)

    assert upstream_calls == [tuple(tensors)]


def test_init_delegates_to_upstream_off_ppu(kda_constructor):
    c = kda_constructor(ppu=False)
    config, cfg = constructor_configs("triton")
    layer = c.provider.KimiK3DeltaAttention(config, cfg, "model.kda", None, True)
    assert c.calls == [(layer, config, cfg, "model.kda", None, True)]


def test_init_restores_the_flashkda_prefill_contract_on_ppu(kda_constructor):
    c = kda_constructor()
    config, cfg = constructor_configs(tokens=64, seqs=4)
    layer = c.provider.KimiK3DeltaAttention(config, cfg, "model.kda", None, False)
    assert c.provider.resolve_kda_prefill_backend is c.resolver
    assert len(c.calls) == 1
    assert layer.kda_prefill_backend == "flashkda"
    assert layer._flashkda_buffer_specs == (
        ((1, 64, 2, 128), "bf16"),
        ((4, 2, 128, 128), "fp32"),
        ((4, 2, 128, 128), "fp32"),
        ((0,), "uint8"),
    )
    assert layer._checkpoint_exporter == "checkpoint-exporter"
