# SPDX-License-Identifier: Apache-2.0
"""Landing and behaviour tests for the Phase-3 compute-core patches.

Covers the patch modules that wire the compute core into stock vLLM:

* ``deep_gemm_redirect``  — fp8_utils / fp8 / input_quant_fp8 helper rebinding
* ``kernel_warmup``       — PPU DeepGEMM warmup appended to kernel_warmup
* ``quant_keys``          — int8 QuantKey constants for the Acext path
* ``tuned_config_lookup`` — block-quant configs found in the plugin directory
* ``int8_quant``          — Triton int8 quant with rounding + forced-Triton knobs

(The MoE Marlin default-off gate is covered by
``patch/enhancement/models/moe_marlin_gate.py`` and tested in
``test_model_patches.py``.)

Every test asserts the patch *landed* (marker present) before asserting
behaviour, mirroring ``test_patch_install.py``.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def installed():
    pytest.importorskip("torch", reason="requires torch")
    pytest.importorskip("vllm", reason="requires vLLM")

    import vllm_sail.patch as patch_pkg

    patch_pkg.install()
    return patch_pkg


def _patched(module_name: str, attribute: str):
    import importlib

    from vllm_sail.patch.utils import PATCH_MARKER

    replacement = getattr(importlib.import_module(module_name), attribute)
    marker = getattr(replacement, PATCH_MARKER, None)
    assert marker is not None, f"{module_name}.{attribute} is not patched"
    return replacement


# ---------------------------------------------------------------------------
# deep_gemm_redirect
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("module_name", "attribute"),
    [
        (
            "vllm.model_executor.layers.quantization.utils.fp8_utils",
            "get_tma_aligned_size",
        ),
        (
            "vllm.model_executor.layers.quantization.utils.fp8_utils",
            "is_deep_gemm_e8m0_used",
        ),
        (
            "vllm.model_executor.layers.quantization.utils.fp8_utils",
            "transform_sf_into_required_layout",
        ),
        ("vllm.model_executor.layers.quantization.fp8", "is_deep_gemm_supported"),
        (
            "vllm.model_executor.layers.quantization.input_quant_fp8",
            "DeepGemmQuantScaleFMT",
        ),
        (
            "vllm.model_executor.layers.quantization.input_quant_fp8",
            "is_deep_gemm_e8m0_used",
        ),
        (
            "vllm.model_executor.layers.quantization.input_quant_fp8",
            "is_deep_gemm_supported",
        ),
    ],
)
def test_deep_gemm_helpers_are_rebound_to_the_ppu_wrapper(
    installed, module_name, attribute
) -> None:
    import vllm_sail.utils.deep_gemm as ppu_deep_gemm
    from vllm_sail.patch.utils import PATCH_MARKER

    target = _patched(module_name, attribute)
    assert target is getattr(ppu_deep_gemm, attribute)
    # The marker must record this exact target, proving the rebinding replaced
    # the module's own binding rather than landing somewhere else.
    assert f"{module_name}.{attribute}" in getattr(target, PATCH_MARKER)


# ---------------------------------------------------------------------------
# kernel_warmup
# ---------------------------------------------------------------------------


def test_kernel_warmup_patch_landed(installed) -> None:
    import vllm_sail.patch.enhancement.kernel_warmup as kw_patch

    if not hasattr(kw_patch, "_upstream_kernel_warmup"):
        pytest.skip("kernel_warmup target module is not importable here")
    _patched("vllm.model_executor.warmup.kernel_warmup", "kernel_warmup")


def test_ppu_deep_gemm_warmup_gate(monkeypatch: pytest.MonkeyPatch, installed) -> None:
    """The fork's gate: skip wins, support required, backends must allow it."""
    import vllm_sail.patch.enhancement.kernel_warmup as kw_patch
    import vllm_sail.utils.deep_gemm as deep_gemm

    monkeypatch.setattr(deep_gemm, "is_deep_gemm_supported", lambda: True)

    monkeypatch.setenv("VLLM_DEEP_GEMM_WARMUP", "skip")
    assert kw_patch._ppu_deep_gemm_warmup_enabled() is False
    monkeypatch.delenv("VLLM_DEEP_GEMM_WARMUP", raising=False)

    monkeypatch.setattr(deep_gemm, "is_deep_gemm_supported", lambda: False)
    assert kw_patch._ppu_deep_gemm_warmup_enabled() is False
    monkeypatch.setattr(deep_gemm, "is_deep_gemm_supported", lambda: True)

    assert kw_patch._ppu_deep_gemm_warmup_enabled() is True

    monkeypatch.setenv("VLLM_PPU_MOE_BACKEND", "deepgemm")
    assert kw_patch._ppu_deep_gemm_warmup_enabled() is True

    monkeypatch.setenv("VLLM_PPU_MOE_BACKEND", "acext")
    monkeypatch.setenv("VLLM_PPU_DENSE_BACKEND", "acext")
    assert kw_patch._ppu_deep_gemm_warmup_enabled() is False

    # Either selector allowing DeepGEMM is enough.
    monkeypatch.setenv("VLLM_PPU_DENSE_BACKEND", "deepgemm")
    assert kw_patch._ppu_deep_gemm_warmup_enabled() is True


def test_kernel_warmup_delegates_off_ppu_and_runs_ppu_warmup_on_ppu(
    monkeypatch: pytest.MonkeyPatch, installed
) -> None:
    import sys
    import types

    import vllm_sail.patch.enhancement.kernel_warmup as kw_patch

    if not hasattr(kw_patch, "_upstream_kernel_warmup"):
        pytest.skip("kernel_warmup target module is not importable here")

    calls: list[str] = []
    monkeypatch.setattr(
        kw_patch,
        "_upstream_kernel_warmup",
        lambda worker, **kwargs: calls.append("upstream"),
    )
    worker = types.SimpleNamespace(
        use_v2_model_runner=True,
        model_runner=types.SimpleNamespace(
            jit_warmup_registry=None, is_pooling_model=True, attn_groups=[]
        ),
        vllm_config=types.SimpleNamespace(
            model_config=types.SimpleNamespace(),
            kernel_config=types.SimpleNamespace(
                enable_jit_warmup=False,
                enable_cutedsl_warmup=False,
                enable_flashinfer_autotune=False,
            ),
            compilation_config=types.SimpleNamespace(cudagraph_capture_sizes=[]),
        ),
        get_model=lambda: calls.append("get_model") or object(),
        scheduler_config=types.SimpleNamespace(max_num_batched_tokens=512),
    )

    from vllm.platforms import current_platform

    monkeypatch.setattr(type(current_platform), "is_ppu", lambda self: False)
    kw_patch.kernel_warmup(worker)
    assert calls == ["upstream"]

    import vllm.model_executor.warmup.kernel_warmup as provider
    import vllm.model_executor.warmup.step5_dsa_warmup as step5

    # Keep the real PPU body: isolate only expensive external callbacks.
    monkeypatch.setattr(
        provider, "qwen_triton_warmup", lambda *args: calls.append("upstream")
    )
    for name in (
        "qwen_vl_triton_warmup",
        "mamba_triton_warmup",
        "_warmup_gemm_rs_ar",
        "flashinfer_sparse_mla_decode_autotune_warmup",
        "deepseek_v4_sparse_mla_attention_warmup",
        "b12x_warmup",
    ):
        monkeypatch.setattr(provider, name, lambda *args: None)
    monkeypatch.setattr(step5, "step5_dsa_warmup", lambda worker: None)
    monkeypatch.setattr(provider, "is_deep_gemm_supported", lambda: False)
    monkeypatch.setattr(provider.envs, "VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC", False)
    monkeypatch.setattr(type(current_platform), "is_ppu", lambda self: True)
    monkeypatch.setenv("VLLM_DEEP_GEMM_WARMUP", "skip")
    calls.clear()
    kw_patch.kernel_warmup(worker)
    assert calls == ["upstream"]

    monkeypatch.delenv("VLLM_DEEP_GEMM_WARMUP", raising=False)
    monkeypatch.delenv("VLLM_PPU_MOE_BACKEND", raising=False)
    monkeypatch.delenv("VLLM_PPU_DENSE_BACKEND", raising=False)
    import vllm_sail.utils.deep_gemm as deep_gemm

    monkeypatch.setattr(deep_gemm, "is_deep_gemm_supported", lambda: True)

    warmup_module = types.ModuleType("vllm_sail.model_executor.warmup.deep_gemm_warmup")
    warmup_module.deep_gemm_warmup = lambda model, max_tokens: calls.append(
        ("ppu_warmup", max_tokens)
    )
    monkeypatch.setitem(
        sys.modules, "vllm_sail.model_executor.warmup.deep_gemm_warmup", warmup_module
    )
    calls.clear()
    kw_patch.kernel_warmup(worker)
    assert calls == ["upstream", "get_model", ("ppu_warmup", 512)]

    # process_local_only skips everything after the upstream body.
    calls.clear()
    kw_patch.kernel_warmup(worker, process_local_only=True)
    assert calls == ["upstream"]


def test_kernel_warmup_filters_real_jit_registry(monkeypatch, installed):
    from types import SimpleNamespace

    import vllm.distributed
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry, VllmJitKernel
    from vllm.model_executor.warmup.jit_warmup_cutedsl_helper import (
        VllmCuTeDSLJitKernel,
    )

    from vllm_sail.patch.enhancement.kernel_warmup import (
        _exclude_ppu_cutedsl_registrations,
    )

    calls = []

    class UnsupportedKernel(VllmCuTeDSLJitKernel):
        def get_warmup_keys(self):
            pytest.fail("PPU must exclude CuTeDSL before expanding compile keys")

        def warmup_inputs(self, compile_key):
            pytest.fail("PPU must never prepare CuTeDSL compilation")

    class SupportedKernel(VllmJitKernel):
        def get_warmup_keys(self, *, head_dim):
            return [head_dim]

        def compile(self, compile_key):
            calls.append(compile_key)

    registry = JitWarmupRegistry(SimpleNamespace())
    with registry.activate():
        UnsupportedKernel().register_warmup()
        SupportedKernel().register_warmup(head_dim=128)
    worker = SimpleNamespace(model_runner=SimpleNamespace(jit_warmup_registry=registry))
    _exclude_ppu_cutedsl_registrations(worker)
    monkeypatch.setattr(vllm.distributed, "is_global_first_rank", lambda: False)
    registry.warmup()
    assert calls == [128]


def test_register_ppu_sampler_warmups_gating(monkeypatch, installed):
    """The sampler registration mirrors upstream's ROCm gate, on PPU."""
    from types import SimpleNamespace

    import vllm.v1.sample.ops.topk_topp_sampler as sampler_mod
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry, VllmJitKernel

    from vllm_sail.patch.enhancement.kernel_warmup import (
        _register_ppu_sampler_warmups,
    )

    calls: list[str] = []

    class ProbeKernel(VllmJitKernel):
        def get_warmup_keys(self):
            return ["k"]

        def compile(self, compile_key):
            calls.append(compile_key)

    def fake_register():
        calls.append("registered")
        # Landing in the registry proves the activate() context was live.
        ProbeKernel().register_warmup()

    monkeypatch.setattr(
        sampler_mod, "register_top_k_top_p_warmups", fake_register, raising=False
    )

    def make_worker(**runner_kwargs):
        registry = JitWarmupRegistry(SimpleNamespace())
        runner = SimpleNamespace(jit_warmup_registry=registry, **runner_kwargs)
        return SimpleNamespace(model_runner=runner), registry

    # A runner without a registry (e.g. pooling-only or older vLLM) is a no-op.
    worker, _ = make_worker(is_last_pp_rank=True, is_pooling_model=False)
    worker.model_runner.jit_warmup_registry = None
    _register_ppu_sampler_warmups(worker)
    assert calls == []

    # Non-last pipeline-parallel ranks never sample.
    worker, registry = make_worker(is_last_pp_rank=False, is_pooling_model=False)
    _register_ppu_sampler_warmups(worker)
    assert calls == []
    assert not registry._registrations

    # Pooling models never sample.
    worker, registry = make_worker(is_last_pp_rank=True, is_pooling_model=True)
    _register_ppu_sampler_warmups(worker)
    assert calls == []
    assert not registry._registrations

    worker, registry = make_worker(is_last_pp_rank=True, is_pooling_model=False)
    _register_ppu_sampler_warmups(worker)
    assert calls == ["registered"]
    assert len(registry._registrations) == 1


# ---------------------------------------------------------------------------
# quant_keys
# ---------------------------------------------------------------------------


def test_int8_quant_keys_constants_exist(installed) -> None:
    import torch
    from vllm.model_executor.layers.quantization.utils import quant_utils

    assert (
        quant_utils.kStaticChannelScale.group_shape
        is quant_utils.GroupShape.PER_CHANNEL
    )
    assert (
        quant_utils.kDynamicTokenScale.group_shape is quant_utils.GroupShape.PER_TOKEN
    )
    assert quant_utils.kInt8StaticChannelSym == quant_utils.QuantKey(
        torch.int8, quant_utils.kStaticChannelScale, symmetric=True
    )
    assert quant_utils.kInt8DynamicTokenSym == quant_utils.QuantKey(
        torch.int8, quant_utils.kDynamicTokenScale, symmetric=True
    )


# ---------------------------------------------------------------------------
# tuned_config_lookup
# ---------------------------------------------------------------------------


def test_block_fp8_configs_upstream_wins(
    monkeypatch: pytest.MonkeyPatch, installed
) -> None:
    import vllm_sail.patch.enhancement.tuned_config_lookup as lookup

    sentinel = {1: {"BLOCK_SIZE_M": 16}}
    monkeypatch.setattr(lookup, "_upstream_fp8", lambda *args: sentinel)
    assert lookup.get_w8a8_block_fp8_configs(1536, 7168, 128, 128) is sentinel


def test_block_fp8_configs_fall_back_to_plugin_directory(
    monkeypatch: pytest.MonkeyPatch, installed
) -> None:
    """Pins the duplicated filename format against a real shipped config."""
    import vllm_sail.patch.enhancement.tuned_config_lookup as lookup

    monkeypatch.setattr(lookup, "_upstream_fp8", lambda *args: None)
    monkeypatch.setattr(lookup, "get_device_name_as_file_name", lambda: "ZW-M890P")
    config = lookup.get_w8a8_block_fp8_configs(1536, 7168, 128, 128)
    assert config is not None
    assert all(isinstance(key, int) for key in config)


def test_fp8_block_config_filename_matches_upstream_format(
    monkeypatch: pytest.MonkeyPatch, installed
) -> None:
    """Retain FP8's compact block_shape after removal of the INT8 loader."""
    import vllm_sail.patch.enhancement.tuned_config_lookup as lookup

    seen: list[str] = []

    def _record(json_file_name: str, kind: str):
        seen.append(json_file_name)
        return None

    monkeypatch.setattr(lookup, "_upstream_fp8", lambda *args: None)
    monkeypatch.setattr(lookup, "_load_plugin_config", _record)
    monkeypatch.setattr(lookup, "get_device_name_as_file_name", lambda: "PPU-ZW810E")

    lookup.get_w8a8_block_fp8_configs(1024, 2048, 128, 128)
    assert seen[0] == (
        "N=1024,K=2048,device_name=PPU-ZW810E,dtype=fp8_w8a8,block_shape=[128,128].json"
    )
    assert len(seen) == 1


# ---------------------------------------------------------------------------
# int8_quant
# ---------------------------------------------------------------------------


def test_int8_quant_wrapper_gained_ppu_knobs(installed) -> None:
    import inspect

    wrapper = _patched(
        "vllm.model_executor.layers.quantization.utils.int8_utils",
        "per_token_group_quant_int8",
    )
    params = inspect.signature(wrapper).parameters
    assert "use_triton" in params
    assert "use_rounding" in params
    assert params["use_triton"].default is False
    assert params["use_rounding"].default is False


# ---------------------------------------------------------------------------
# modular_kernel (packed-MXFP4 K fix)
# ---------------------------------------------------------------------------


def test_moe_problem_size_doubles_k_for_packed_mxfp4_on_ppu(
    monkeypatch: pytest.MonkeyPatch, installed
) -> None:
    import torch
    from vllm.model_executor.layers.fused_moe import modular_kernel as mk_module
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEExpertsModular,
    )
    from vllm.platforms import current_platform

    from vllm_sail.patch.utils import PATCH_MARKER

    marker = getattr(FusedMoEExpertsModular.moe_problem_size, PATCH_MARKER, None)
    assert marker is not None, "FusedMoEExpertsModular.moe_problem_size not patched"
    target = f"{mk_module.__name__}.FusedMoEExpertsModular.moe_problem_size"
    assert target in marker

    E, M, N = 2, 4, 32
    packed_k = 64
    w1 = torch.zeros((E, N, packed_k), dtype=torch.uint8)
    w2 = torch.zeros((E, 2 * packed_k, N), dtype=torch.uint8)
    topk_ids = torch.zeros((M, 2), dtype=torch.int64)

    def _solve(a1):
        e, m, n, k, topk = FusedMoEExpertsModular.moe_problem_size(
            None, a1, w1, w2, topk_ids
        )
        return (e, m, n, k, topk)

    mxfp4_act = torch.zeros((M, packed_k), dtype=torch.uint8)
    bf16_act = torch.zeros((M, 2 * packed_k), dtype=torch.bfloat16)

    monkeypatch.setattr(type(current_platform), "is_ppu", lambda self: True)
    assert _solve(mxfp4_act) == (E, M, N, 2 * packed_k, 2)
    assert _solve(bf16_act) == (E, M, N, 2 * packed_k, 2)

    monkeypatch.setattr(type(current_platform), "is_ppu", lambda self: False)
    assert _solve(mxfp4_act) == (E, M, N, packed_k, 2)
