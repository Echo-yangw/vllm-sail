# SPDX-License-Identifier: Apache-2.0
"""Registered KDA delegation keeps the shared reservation before KV sizing."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Barrier
from types import SimpleNamespace as NS

import pytest

from tests.support.modules import load_patch
from vllm_sail.patch.bodies import bind_body


@dataclass
class CacheSpec:
    num_prefill_checkpoint_blocks: int = 3
    prefill_checkpoint_alignment: int | None = 128
    retained: str = "unchanged"


def upstream_constructor(
    self, config, vllm_config, prefix="", aux_stream=None, run_gemm_rs_ar=False
):
    """Strict external constructor contract; the registered adapter runs intact."""
    record_ctor(self, config, vllm_config, prefix, aux_stream, run_gemm_rs_ar)  # noqa: F821
    self.model_config = vllm_config.model_config
    self.local_num_heads = config.num_heads // 8
    self.head_dim = 128
    backend = resolve_kda_prefill_backend(  # noqa: F821
        config.backend, 128, "bf16", "fp32", -5.0
    )
    self.kda_prefill_backend = backend
    self._flashkda_buffer_specs = None
    if backend == "flashinfer" and is_workspace_manager_initialized():  # noqa: F821
        current_workspace_manager().get_simultaneous(  # noqa: F821
            (
                (
                    1,
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    self.local_num_heads,
                    self.head_dim,
                ),
                self.model_config.dtype,
            )
        )
    record_norm(self)  # noqa: F821
    after_resolve(self, config)  # noqa: F821
    if config.fail_ctor:
        raise RuntimeError("external constructor failed")
    vllm_config.compilation_config.static_forward_context[prefix] = self


@pytest.fixture
def kda_constructor(modules):
    def load(*, initialized=True, ppu=True, fail_reserve=False, after=None):
        events, calls = [], []
        platform = NS(is_ppu=lambda: ppu)
        modules("torch", bfloat16="bf16", float32="fp32", uint8="uint8")
        modules("vllm.platforms", current_platform=platform)
        modules("vllm.config", VllmConfig=type("VllmConfig", (), {}))
        modules(
            "vllm.forward_context", get_forward_context=lambda: NS(attn_metadata=None)
        )
        modules(
            "vllm.transformers_utils.configs.kimi_linear",
            KimiLinearConfig=type("KimiLinearConfig", (), {}),
        )
        modules("vllm_sail.envs", VLLM_SAIL_USE_PLA=True)

        def reserve(*specs):
            if fail_reserve:
                raise RuntimeError("external reservation failed")
            events.append(("reserve", specs))

        def manager():
            assert initialized
            return NS(get_simultaneous=reserve)

        def resolve(backend, head_dim, dtype, state_dtype, lower_bound):
            assert (head_dim, dtype, state_dtype, lower_bound) == (
                128,
                "bf16",
                "fp32",
                -5.0,
            )
            return backend

        class Attention:
            def _forward(self, *args):
                raise AssertionError("forward is not part of constructor test")

            def get_kv_cache_spec(self, config):
                return CacheSpec()

            def get_state_dtype(self):
                return "bf16", "fp32"

        provider = modules(
            "vllm.models.kimi_k3.nvidia.kda",
            KimiK3DeltaAttention=Attention,
            logger=NS(info_once=lambda *args: None),
            is_conv_state_dim_first=lambda: False,
            is_flashkda_supported=lambda *args: False,
            is_fused_kda_decode_supported=lambda *args: False,
            _flashkda_prefill=lambda *args: None,
            resolve_kda_prefill_backend=resolve,
            is_workspace_manager_initialized=lambda: initialized,
            current_workspace_manager=manager,
            FlashKDAPrefillCheckpointExporter=lambda: "checkpoint-exporter",
            eager_break_during_capture=lambda fn: fn,
            record_ctor=lambda *args: calls.append(args),
            record_norm=lambda self: events.append(("norm",)),
            after_resolve=after or (lambda self, config: None),
        )
        Attention.__init__ = bind_body(upstream_constructor, provider)
        load_patch("vllm_sail/patch/enhancement/attention/kda.py")
        return NS(provider=provider, events=events, calls=calls, resolver=resolve)

    return load


def configs(backend="flashkda", *, tokens=32, seqs=4, heads=16, fail_ctor=False):
    return NS(backend=backend, num_heads=heads, fail_ctor=fail_ctor), NS(
        model_config=NS(dtype="bf16"),
        scheduler_config=NS(max_num_seqs=seqs, max_num_batched_tokens=tokens),
        compilation_config=NS(static_forward_context={}),
    )


@pytest.mark.parametrize("initialized", [False, True])
@pytest.mark.parametrize("backend", ["flashkda", "flashinfer", "triton"])
def test_registered_constructor_reserves_selected_workspace(
    kda_constructor, initialized, backend
):
    contract = kda_constructor(initialized=initialized)
    config, vllm_config = configs(backend)
    layer = contract.provider.KimiK3DeltaAttention(config, vllm_config, prefix="layer0")
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
        assert contract.events == [("reserve", expected), ("norm",)]
    else:
        assert contract.events == [("norm",)]
    assert layer.kda_prefill_backend == backend
    if backend == "flashkda":
        assert layer._checkpoint_exporter == "checkpoint-exporter"
        assert layer._flashkda_buffer_specs[-1] == ((0,), "uint8")
    assert contract.provider.resolve_kda_prefill_backend is contract.resolver


@pytest.mark.parametrize("reserve_fails", [False, True])
def test_constructor_failure_never_changes_live_resolver(
    kda_constructor, reserve_fails
):
    c = kda_constructor(fail_reserve=reserve_fails)
    config, vllm_config = configs(fail_ctor=not reserve_fails)
    with pytest.raises(RuntimeError, match="external"):
        c.provider.KimiK3DeltaAttention(config, vllm_config)
    assert c.provider.resolve_kda_prefill_backend is c.resolver


def test_concurrent_constructor_namespaces_do_not_cross_layers(kda_constructor):
    barrier = Barrier(2)
    c = kda_constructor(after=lambda self, config: barrier.wait(timeout=10))
    inputs = [
        configs(tokens=32, seqs=4, heads=16),
        configs(tokens=64, seqs=8, heads=32),
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        layers = list(
            pool.map(lambda args: c.provider.KimiK3DeltaAttention(*args), inputs)
        )
    for layer, (_, cfg) in zip(layers, inputs, strict=False):
        assert layer._flashkda_buffer_specs[0] == (
            (
                1,
                cfg.scheduler_config.max_num_batched_tokens,
                layer.local_num_heads,
                128,
            ),
            "bf16",
        )
    assert c.provider.resolve_kda_prefill_backend is c.resolver
    assert sorted(e[1][0][0][1] for e in c.events if e[0] == "reserve") == [32, 64]


def test_non_ppu_constructor_delegates_arguments_unchanged(kda_constructor):
    c = kda_constructor(ppu=False)
    config, cfg = configs("triton")
    layer = c.provider.KimiK3DeltaAttention(config, cfg, "model.kda", None, True)
    assert c.calls == [(layer, config, cfg, "model.kda", None, True)]
    assert c.events == [("norm",)]


@pytest.mark.parametrize("backend", ["flashkda", "triton"])
def test_cache_spec_keeps_unrelated_fields(kda_constructor, backend):
    c = kda_constructor()
    config, cfg = configs(backend)
    layer = c.provider.KimiK3DeltaAttention(config, cfg)
    spec = layer.get_kv_cache_spec(cfg)
    assert spec.retained == "unchanged"
    assert (spec.num_prefill_checkpoint_blocks, spec.prefill_checkpoint_alignment) == (
        (0, None) if backend == "flashkda" else (3, 128)
    )
