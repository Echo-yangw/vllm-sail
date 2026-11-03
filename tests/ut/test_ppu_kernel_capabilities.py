# SPDX-License-Identifier: Apache-2.0
"""Exercise imported providers and real compressor dispatch without device deps."""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

import pytest

from tests.support.source import function

ROOT = Path(__file__).parents[2]


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "patch_file,inventory_name,provider",
    [
        ("import_gates.py", "_CONSUMERS", "vllm.utils.import_utils"),
        (
            "attention/flashmla_ops.py",
            "_FLASHMLA_ALIAS_CONSUMERS",
            "vllm.v1.attention.ops.flashmla",
        ),
    ],
)
def test_optional_gate_consumer_inventory_matches_upstream(
    upstream_source_root, patch_file, inventory_name, provider
):
    tree = ast.parse((ROOT / "vllm_sail/patch/enhancement" / patch_file).read_text())
    inventory = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == inventory_name
            for target in node.targets
        )
    )
    discovered = {name: set() for name in inventory}
    source = upstream_source_root / "vllm"
    for path in source.rglob("*.py"):
        parts = path.relative_to(source).with_suffix("").parts
        module = ".".join(("vllm", *(parts[:-1] if parts[-1] == "__init__" else parts)))
        pending = list(ast.parse(path.read_text()).body)
        while pending:
            node = pending.pop()
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                continue
            if isinstance(node, ast.ImportFrom) and node.module == provider:
                for alias in node.names:
                    if alias.name in discovered:
                        assert alias.asname is None, "renamed alias needs rebinding"
                        discovered[alias.name].add(module)
            pending.extend(ast.iter_child_nodes(node))
    assert discovered == {name: set(consumers) for name, consumers in inventory.items()}


@pytest.fixture
def modules(monkeypatch, patch_utils_module):
    loaded = {}

    def add(name, **attrs):
        if name not in loaded:
            mod = types.ModuleType(name)
            mod.__path__ = []
            mod.__package__ = name.rpartition(".")[0]
            loaded[name] = mod
            monkeypatch.setitem(sys.modules, name, mod)
            parent, _, leaf = name.rpartition(".")
            if parent:
                setattr(add(parent), leaf, mod)
        loaded[name].__dict__.update(attrs)
        return loaded[name]

    add(
        "vllm_sail.patch.utils",
        **{
            name: getattr(patch_utils_module, name)
            for name in ("patch", "PATCH_MARKER")
        },
    )
    spec = importlib.util.spec_from_file_location(
        "vllm_sail.patch.bodies", ROOT / "vllm_sail/patch/bodies.py"
    )
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    add("vllm_sail.patch.bodies", bind_body=helpers.bind_body)
    spec = importlib.util.spec_from_file_location(
        "_ppu_warmup_body",
        ROOT / "vllm_sail/model_executor/warmup/ppu_kernel_warmup.py",
    )
    warmup_body = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(warmup_body)
    add(
        "vllm_sail.model_executor.warmup.ppu_kernel_warmup",
        kernel_warmup=warmup_body.kernel_warmup,
    )
    spec = importlib.util.spec_from_file_location(
        "_ppu_minimax_mlp_body", ROOT / "vllm_sail/models/minimax_m3_mlp.py"
    )
    mlp_body = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mlp_body)
    add("vllm_sail.models.minimax_m3_mlp", init_mlp=mlp_body.init_mlp)
    return add


def load_patch(relative):
    spec = importlib.util.spec_from_file_location("_capability_patch", ROOT / relative)
    mod = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(spec.name)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        if previous is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous
    return mod


def complete_warmup_contract(modules, provider, worker, platform, *, entered=None):
    """Strict external callbacks for the real copied warmup body."""
    import time

    ns = types.SimpleNamespace
    entered = entered or (lambda: None)
    platform.is_cuda = lambda: True
    platform.is_rocm = lambda: False
    platform.is_device_capability_family = lambda _: False
    worker.use_v2_model_runner = True
    worker.model_runner.attn_groups = []
    worker.vllm_config.model_config = ns()
    worker.vllm_config.compilation_config = ns(cudagraph_capture_sizes=[])
    config = worker.vllm_config.kernel_config
    if not hasattr(config, "enable_cutedsl_warmup"):
        config.enable_cutedsl_warmup = False
    config.enable_flashinfer_autotune = False

    def registry_warmup():
        entered()
        for kernel, keys in list(
            worker.model_runner.jit_warmup_registry._registrations.items()
        ):
            kernel.compile_many(keys)

    worker.model_runner.jit_warmup_registry.warmup = registry_warmup
    callbacks = {
        "qwen_triton_warmup": lambda runner, config: entered(),
        "qwen_vl_triton_warmup": lambda runner: None,
        "mamba_triton_warmup": lambda runner: None,
        "kimi_k3_triton_warmup": lambda worker: None,
        "watermark_sample_warmup": lambda worker: None,
        "qwen4_exp_qsa_triton_warmup": lambda worker: None,
        "_warmup_bf16x3_router_gemm": lambda model, tokens: None,
        "_warmup_ll_bf16_router_gemm": lambda model: None,
        "_warmup_gemm_rs_ar": lambda: None,
        "flashinfer_sparse_mla_decode_autotune_warmup": lambda worker: None,
        "deepseek_v4_sparse_mla_attention_warmup": lambda worker: None,
        "b12x_warmup": lambda worker, sizes: None,
        "mixed_batch_attention_warmup": lambda worker: None,
        "is_deep_gemm_supported": lambda: False,
        "has_flashinfer": lambda: False,
    }
    provider.__dict__.update(
        callbacks,
        current_platform=platform,
        time=time,
        envs=ns(
            VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=False, VLLM_DEEP_GEMM_WARMUP="skip"
        ),
        logger=ns(
            info=lambda *a: None, info_once=lambda *a: None, exception=lambda *a: None
        ),
    )
    modules(
        "vllm.model_executor.warmup.step5_dsa_warmup",
        step5_dsa_warmup=lambda worker: None,
    )


@pytest.mark.parametrize(
    "cute_consumer",
    [
        "vllm.models.deepseek_v4.common.ops.cache_utils",
        "vllm.models.glm5next.nvidia.sparse_indexer",
    ],
)
def test_optional_gates_rebind_preloaded_and_future_consumers(modules, cute_consumer):
    platform = types.SimpleNamespace(ppu=True, is_ppu=lambda: platform.ppu)
    modules("vllm.platforms", current_platform=platform)

    def probe():
        return True

    provider = modules("vllm.utils.import_utils", has_cutedsl=probe, has_humming=probe)
    cute = modules(cute_consumer, has_cutedsl=probe)
    humming = modules(
        "vllm.model_executor.layers.quantization.utils.humming.moe", has_humming=probe
    )
    patched = load_patch("vllm_sail/patch/enhancement/import_gates.py")
    assert cute.has_cutedsl is provider.has_cutedsl
    assert humming.has_humming is provider.has_humming
    assert not cute.has_cutedsl() and not humming.has_humming()
    future = modules("future", has_cutedsl=provider.has_cutedsl)
    assert not future.has_cutedsl()
    platform.ppu = False
    assert cute.has_cutedsl() and humming.has_humming()
    patched._rebind_loaded_consumers()
    humming.has_humming = lambda: "other override"
    patched._rebind_loaded_consumers()
    assert humming.has_humming() == "other override"


@pytest.mark.parametrize(
    "ppu,head,two_stage,expected",
    [
        (True, 512, False, "triton"),
        (True, 128, False, "triton"),
        (False, 512, False, "cute"),
        (False, 512, True, "two_stage"),
    ],
)
def test_compressor_uses_supported_kernel(modules, ppu, head, two_stage, expected):
    base = "vllm.models.deepseek_v4"
    calls = []
    platform = types.SimpleNamespace(
        is_ppu=lambda: ppu,
        is_cuda=lambda: not two_stage,
        is_rocm=lambda: two_stage,
        is_xpu=lambda: False,
    )

    def kernel(name):
        def call(**kwargs):
            calls.append((name, kwargs))

        return call

    meta = types.SimpleNamespace(
        token_to_req_indices=object(),
        slot_mapping=types.SimpleNamespace(shape=(3,)),
        block_table=object(),
        block_size=64,
        c128_boundary=True,
        num_decode_tokens=1,
    )
    context = types.SimpleNamespace(
        attn_metadata={"state": meta, "kv": object()},
        cudagraph_runtime_mode="none",
        capturing=False,
    )
    cache = types.SimpleNamespace(shape=(2, 64, 2048), dtype="u8")
    cls = type("DeepseekCompressor", (), {"_compress_norm_rope_store": lambda *a: None})
    module = modules(
        f"{base}.compressor",
        DeepseekCompressor=cls,
        current_platform=platform,
        get_forward_context=lambda: context,
        cast=lambda kind, value: value,
        CompressorMetadata=object,
        Any=object,
        CUDAGraphMode=types.SimpleNamespace(FULL="full"),
        torch=types.SimpleNamespace(
            uint8="u8",
            float8_e4m3fn="fp8",
            cuda=types.SimpleNamespace(
                is_current_stream_capturing=lambda: context.capturing
            ),
        ),
        _SAVE_PARTIAL_STATES_KERNEL=kernel("save"),
        compress_norm_rope_store_triton=kernel("triton"),
        compress_norm_rope_store_two_stage_triton=kernel("two_stage"),
    )
    original_forward = function(
        "tests/fixtures/deepseek_v4_compressor_forward.py", "forward", module.__dict__
    )
    cls.forward = original_forward
    if not ppu:
        modules(
            f"{base}.nvidia.ops.sparse_attn_compress_cutedsl",
            _SPARSE_ATTN_COMPRESSOR_CUTEDSL_KERNEL=kernel("cute"),
        )
    load_patch("vllm_sail/patch/enhancement/models/deepseek_v4_compressor.py")
    obj = cls.__new__(cls)
    obj.__dict__.update(
        coff=2,
        head_dim=head,
        state_cache=types.SimpleNamespace(prefix="state", kv_cache=cache),
        ape=object(),
        compress_ratio=4,
        k_cache_prefix="kv",
        _static_forward_context={"kv": types.SimpleNamespace(kv_cache=cache)},
        overlap=False,
        eager_scratch_pool=None,
        _use_two_stage_fused_compressor=two_stage,
        _compress_scratch=object(),
        rope_head_dim=64,
        use_fp4_cache=False,
        norm=types.SimpleNamespace(weight=object()),
        rms_norm_eps=1e-6,
        _quant_block=64,
        _token_stride=576,
        _scale_dim=8,
    )
    scores = types.SimpleNamespace(split=lambda *args, **kwargs: (object(), object()))
    obj.forward(scores, object(), types.SimpleNamespace(cos_sin_cache=object()))
    assert [name for name, _ in calls] == ["save", expected]
    assert ("store_full_kv" in calls[-1][1]) == (expected == "cute")
    assert obj.forward.__func__.__globals__ is module.__dict__
    assert obj.forward.__func__ is original_forward
    assert obj._compress_norm_rope_store.__func__.__globals__ is module.__dict__
    # Non-boundary C128 steps still save partial state, except full graph capture.
    if not two_stage and head == 512:
        obj.compress_ratio = 128
        meta.c128_boundary = False
        calls.clear()
        obj.forward(scores, object(), types.SimpleNamespace(cos_sin_cache=object()))
        assert [name for name, _ in calls] == ["save"]
        context.cudagraph_runtime_mode = "full"
        context.capturing = True
        calls.clear()
        obj.forward(scores, object(), types.SimpleNamespace(cos_sin_cache=object()))
        assert [name for name, _ in calls] == ["save", expected]
    context.attn_metadata = None
    calls.clear()
    obj.forward(scores, object(), object())
    assert not calls


@pytest.mark.parametrize("ppu", [True, False])
@pytest.mark.parametrize("head,ratio", [(512, 4), (512, 128), (128, 4)])
@pytest.mark.parametrize("warmup", [True, False])
@pytest.mark.parametrize("cache_dtype", ["fp8_ds_mla", "bf16"])
def test_compressor_constructor_registers_supported_warmup(
    modules, ppu, head, ratio, warmup, cache_dtype
):
    base = "vllm.models.deepseek_v4"
    calls = []

    def kernel(name):
        return types.SimpleNamespace(register_warmup=lambda *a, **k: calls.append(name))

    def noop(*a, **k):
        return object()

    cls = type(
        "DeepseekCompressor", (), {"forward": noop, "_compress_norm_rope_store": noop}
    )
    provider = modules(
        f"{base}.compressor",
        DeepseekCompressor=cls,
        current_platform=types.SimpleNamespace(
            device_type="cuda", is_cuda=lambda: True, is_ppu=lambda: ppu
        ),
        torch=types.SimpleNamespace(empty=noop, float32="fp32"),
        nn=types.SimpleNamespace(Parameter=noop),
        _prefer_two_stage_compressor=lambda: False,
        MergedColumnParallelLinear=noop,
        RMSNorm=noop,
        CompressorStateCache=noop,
        _SAVE_PARTIAL_STATES_KERNEL=kernel("save"),
    )
    modules(
        f"{base}.common.ops.fused_compress_quant_cache",
        _FUSED_KV_COMPRESS_NORM_ROPE_INSERT_INDEXER_TRITON_KERNEL=kernel("triton"),
    )
    # On PPU the CuTeDSL module is deliberately unavailable: even importing it
    # during construction is a regression, regardless of the forward guard.
    if not ppu:
        modules(
            f"{base}.nvidia.ops.sparse_attn_compress_cutedsl",
            _SPARSE_ATTN_COMPRESS_C128_BLOCK8_KERNEL=kernel("cute128"),
            _SPARSE_ATTN_COMPRESS_NORM_ROPE_STORE_C4_KERNEL=kernel("cute4"),
            _SPARSE_ATTN_COMPRESS_NORM_ROPE_STORE_FULL_C4_KERNEL=kernel("cute4full"),
            _SPARSE_ATTN_NORM_ROPE_STORE_FULL_KERNEL=kernel("cutestorefull"),
            _SPARSE_ATTN_NORM_ROPE_STORE_KERNEL=kernel("cutestore"),
        )
    load_patch("vllm_sail/patch/enhancement/models/deepseek_v4_compressor.py")
    ns = types.SimpleNamespace
    config = ns(
        model_config=ns(
            hf_config=ns(qk_rope_head_dim=64, rms_norm_eps=1e-6), max_model_len=1024
        ),
        scheduler_config=ns(max_num_seqs=4, max_num_batched_tokens=64),
        compilation_config=ns(static_forward_context={}),
        kernel_config=ns(enable_jit_warmup=warmup),
        cache_config=ns(cache_dtype=cache_dtype),
    )
    obj = cls(config, ratio, 256, head)
    assert obj.__init__.__func__.__globals__ is provider.__dict__
    if not warmup:
        assert calls == []
    elif ppu or head != 512:
        assert calls == ["save", "triton"]
    elif ratio == 4:
        assert calls == [
            "save",
            "cute4" if cache_dtype == "fp8_ds_mla" else "cute4full",
        ]
    else:
        assert calls == [
            "save",
            "cute128",
            "cutestore" if cache_dtype == "fp8_ds_mla" else "cutestorefull",
        ]


@pytest.mark.parametrize("ppu", [True, False])
@pytest.mark.parametrize("process_local_only", [True, False])
@pytest.mark.parametrize("jit_enabled", [True, False])
def test_warmup_excludes_cutedsl_only_on_ppu(
    modules, ppu, process_local_only, jit_enabled
):
    ns = types.SimpleNamespace
    import contextlib

    calls = []
    modules(
        "vllm.platforms",
        current_platform=ns(is_ppu=lambda: ppu, has_device_capability=lambda _: True),
    )
    modules("vllm.logger", init_logger=lambda _: ns(warning=lambda *a: None))

    class VllmCuTeDSLJitKernel:
        def compile_many(self, keys):
            assert not ppu, "CuTeDSL compilation reached PPU"
            calls.append("cute")

    class CuTeKernel(VllmCuTeDSLJitKernel):
        pass

    class TritonKernel:
        def compile_many(self, keys):
            calls.append("triton")

    class SamplerKernel:
        def compile_many(self, keys):
            calls.append("sampler_compiled")

    modules(
        "vllm.model_executor.warmup.jit_warmup_cutedsl_helper",
        VllmCuTeDSLJitKernel=VllmCuTeDSLJitKernel,
    )
    cute, triton = CuTeKernel(), TritonKernel()
    registrations = {cute: [((), {})], triton: [((), {"head_dim": 128})]}
    retained = registrations[triton]

    def register_top_k_top_p_warmups():
        calls.append("sampler_registered")
        registrations[SamplerKernel()] = [((), {})]

    modules(
        "vllm.v1.sample.ops.topk_topp_sampler",
        register_top_k_top_p_warmups=register_top_k_top_p_warmups,
    )
    worker = ns(
        model_runner=ns(
            jit_warmup_registry=ns(
                _registrations=registrations, activate=lambda: contextlib.nullcontext()
            ),
            is_last_pp_rank=True,
            is_pooling_model=False,
        ),
        vllm_config=ns(
            kernel_config=ns(enable_jit_warmup=jit_enabled, enable_cutedsl_warmup=True)
        ),
        get_model=lambda: "model",
        scheduler_config=ns(max_num_batched_tokens=64),
    )

    def legacy():
        assert not ppu, "Legacy CuTeDSL warmup reached PPU"
        calls.append("legacy")

    legacy_module = modules(
        "vllm.model_executor.warmup.cutedsl_warmup", cutedsl_warmup=legacy
    )

    def upstream(actual_worker, *, process_local_only=False):
        assert actual_worker is worker
        calls.append(("upstream", process_local_only))
        if worker.vllm_config.kernel_config.enable_jit_warmup:
            for kernel, keys in registrations.items():
                kernel.compile_many(keys)
        provider.cutedsl_warmup()
        return "result"

    # The consumer imported the function by value before the patch was loaded.
    provider = modules(
        "vllm.model_executor.warmup.kernel_warmup",
        kernel_warmup=upstream,
        cutedsl_warmup=legacy,
    )
    entered = False

    def enter_body():
        nonlocal entered
        if not entered:
            calls.append(("upstream", process_local_only))
            entered = True

    complete_warmup_contract(
        modules,
        provider,
        worker,
        sys.modules["vllm.platforms"].current_platform,
        entered=enter_body,
    )
    modules(
        "vllm_sail.model_executor.warmup.deep_gemm_warmup",
        deep_gemm_warmup=lambda model, size: calls.append((model, size)),
    )
    patched = load_patch("vllm_sail/patch/enhancement/kernel_warmup.py")
    patched._ppu_deep_gemm_warmup_enabled = lambda: True
    assert provider.kernel_warmup(worker, process_local_only=process_local_only) == (
        None if ppu else "result"
    )
    expected = []
    if ppu:
        expected += ["sampler_registered"]
    expected += [("upstream", process_local_only)]
    if jit_enabled:
        expected += ["triton"] if ppu else ["cute", "triton"]
        if ppu:
            expected += ["sampler_compiled"]
    if not ppu:
        expected += ["legacy"]
    elif not process_local_only:
        expected += [("model", 64)]
    assert calls == expected
    assert registrations[triton] is retained
    assert (cute in registrations) is (not ppu)
    assert worker.vllm_config.kernel_config.enable_jit_warmup is jit_enabled
    assert worker.vllm_config.kernel_config.enable_cutedsl_warmup is True
    assert legacy_module.cutedsl_warmup is provider.cutedsl_warmup
    calls.clear()
    legacy_module.cutedsl_warmup()
    assert calls == ([] if ppu else ["legacy"])


class _TritonLang:
    """Nameable stand-in for ``triton.language`` inside never-executed fixtures."""

    float8e4nv = "fp8e4nv"
    bfloat16 = "bfloat16"


tl = _TritonLang()


def _fp8_direct(x):
    return x.to(tl.float8e4nv)


def _e4m3_helper(x):
    return x.to(tl.float8e4nv)


# ``@triton.jit`` helpers are inlined at trace time, so only their ``fn`` is read.
_INLINE_E4M3 = types.SimpleNamespace(fn=_e4m3_helper)


def _fp8_inlined(x):
    return _INLINE_E4M3(x)


def _bf16_only(x):
    return x.to(tl.bfloat16)


_FP8_COMPILE_ERROR = ValueError(
    "type fp8e4nv not supported in this architecture. "
    "The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')"
)


def _fixture_kernel(name, jit_fn, compiled, *, error=None):
    """Build a warmup registration shaped like a vLLM Triton kernel wrapper."""

    def compile_many(self, keys):
        compiled.append(name)
        if error is not None:
            raise error

    attrs = {"kernel": staticmethod(jit_fn), "compile_many": compile_many}
    return type(name, (), attrs)()


@pytest.mark.parametrize(
    "ppu,arch,fp8_skipped",
    [
        (True, (8, 0), True),
        (True, (8, 9), False),
        (False, (8, 0), False),
    ],
)
def test_warmup_excludes_fp8_kernels_without_native_fp8(
    modules, ppu, arch, fp8_skipped
):
    """Triton rejects tl.float8e4nv on sm80-class devices, so drop those warmups."""
    compiled = []
    direct = _fixture_kernel("FusedInvRopeFP8QuantKernel", _fp8_direct, compiled)
    inlined = _fixture_kernel(
        "FusedIndexerQRopeQuantTritonKernel", _fp8_inlined, compiled
    )
    clean = _fixture_kernel("PackSeqTritonKernel", _bf16_only, compiled)
    registrations = {item: [((), {})] for item in (direct, inlined, clean)}

    assert _run_ppu_warmup(modules, ppu, arch, registrations) == (
        None if ppu else "result"
    )

    skipped = fp8_skipped and ppu
    assert (direct in registrations) is not skipped
    assert (inlined in registrations) is not skipped
    assert clean in registrations
    assert len(compiled) == 3 - 2 * skipped


def test_warmup_tolerates_signature_level_fp8_on_sm80(modules):
    """A compile key that passes an E4M3 tensor only fails at trace time."""
    compiled = []
    signature_only = _fixture_kernel(
        "SignatureFp8Kernel", _bf16_only, compiled, error=_FP8_COMPILE_ERROR
    )
    clean = _fixture_kernel("PackSeqTritonKernel", _bf16_only, compiled)

    registrations = {signature_only: [((), {})], clean: [((), {})]}
    assert _run_ppu_warmup(modules, True, (8, 0), registrations) is None

    # Source inspection cannot see the dtype, so the registration is kept and
    # the guard turns the resulting compile error into a warning.
    assert compiled == ["SignatureFp8Kernel", "PackSeqTritonKernel"]
    assert signature_only in registrations


def test_warmup_propagates_other_compile_errors(modules):
    """Only the unsupported-fp8 rejection is tolerated."""
    broken = _fixture_kernel(
        "BrokenKernel", _bf16_only, [], error=RuntimeError("fused build failed")
    )

    with pytest.raises(RuntimeError, match="fused build failed"):
        _run_ppu_warmup(modules, True, (8, 0), {broken: [((), {})]})


def _run_ppu_warmup(modules, ppu, arch, registrations):
    """Install the PPU patch and run the warmup over ``registrations`` in order."""
    ns = types.SimpleNamespace
    import contextlib

    def has_device_capability(capability):
        wanted = (
            (capability // 10, capability % 10)
            if isinstance(capability, int)
            else capability
        )
        return arch >= wanted

    modules(
        "vllm.platforms",
        current_platform=ns(
            is_ppu=lambda: ppu, has_device_capability=has_device_capability
        ),
    )
    modules("vllm.logger", init_logger=lambda _: ns(warning=lambda *a: None))
    modules(
        "vllm.model_executor.warmup.jit_warmup_cutedsl_helper",
        VllmCuTeDSLJitKernel=type("VllmCuTeDSLJitKernel", (), {}),
    )
    modules(
        "vllm.v1.sample.ops.topk_topp_sampler",
        register_top_k_top_p_warmups=lambda: None,
    )
    worker = ns(
        model_runner=ns(
            jit_warmup_registry=ns(
                _registrations=registrations, activate=lambda: contextlib.nullcontext()
            ),
            is_last_pp_rank=True,
            is_pooling_model=False,
        ),
        vllm_config=ns(kernel_config=ns(enable_jit_warmup=True)),
        get_model=lambda: "model",
        scheduler_config=ns(max_num_batched_tokens=64),
    )

    def upstream(actual_worker, *, process_local_only=False):
        for actual_kernel, keys in list(registrations.items()):
            actual_kernel.compile_many(keys)
        return "result"

    provider = modules(
        "vllm.model_executor.warmup.kernel_warmup",
        kernel_warmup=upstream,
        cutedsl_warmup=lambda: None,
    )
    complete_warmup_contract(
        modules, provider, worker, sys.modules["vllm.platforms"].current_platform
    )
    modules("vllm.model_executor.warmup.cutedsl_warmup", cutedsl_warmup=lambda: None)
    patched = load_patch("vllm_sail/patch/enhancement/kernel_warmup.py")
    patched._ppu_deep_gemm_warmup_enabled = lambda: False
    return provider.kernel_warmup(worker)


@pytest.mark.parametrize(
    "arch,ppu,alibi,pinned,available,expected",
    [
        (80, True, False, None, (2, 3), 3),
        (89, True, False, None, (2, 3), 3),
        (89, True, True, None, (2, 3), 2),
        (89, True, False, 2, (2, 3), 2),
        (89, True, False, None, (2,), 2),
        (89, True, False, 3, (2,), None),
        (80, False, False, None, (2, 3), 2),
        (90, False, False, None, (2, 3), 3),
    ],
)
def test_fa_selection_and_fp8_contract(
    modules, arch, ppu, alibi, pinned, available, expected
):
    platform = types.SimpleNamespace(
        is_ppu=lambda: ppu,
        is_rocm=lambda: False,
        is_xpu=lambda: False,
        supports_fp8=lambda: arch >= 89,
        get_device_capability=lambda: types.SimpleNamespace(major=arch // 10),
        is_device_capability_family=lambda family: arch // 10 == family // 10,
    )

    def old(*a, **k):
        return "old"

    provider = modules(
        "vllm.v1.attention.backends.fa_utils",
        get_flash_attn_version=old,
        flash_attn_supports_kv_cache_dtype=old,
        current_platform=platform,
        envs=types.SimpleNamespace(VLLM_BATCH_INVARIANT=False),
        logger=types.SimpleNamespace(
            warning_once=lambda *a: None, error=lambda *a: None
        ),
    )
    consumer = modules(
        "vllm.v1.attention.backends.flash_attn",
        get_flash_attn_version=old,
        flash_attn_supports_kv_cache_dtype=old,
    )
    modules(
        "vllm.config",
        get_current_vllm_config_or_none=lambda: types.SimpleNamespace(
            attention_config=types.SimpleNamespace(flash_attn_version=pinned)
        ),
    )
    modules(
        "vllm.vllm_flash_attn.flash_attn_interface",
        is_fa_version_supported=lambda version: version in available,
        fa_version_unsupported_reason=lambda version: "missing",
    )
    load_patch("vllm_sail/patch/enhancement/attention/fa_utils.py")
    assert consumer.get_flash_attn_version is provider.get_flash_attn_version
    assert consumer.get_flash_attn_version(requires_alibi=alibi) == expected
    assert consumer.flash_attn_supports_kv_cache_dtype(requires_alibi=alibi) == (
        expected == 3 and arch >= 89
    )
    assert not consumer.flash_attn_supports_kv_cache_dtype("fp8_e5m2")


def test_indexer_fields_survive_runner_replace_and_merge(modules):
    from dataclasses import asdict, dataclass, fields, replace

    @dataclass(frozen=True)
    class AttentionSpec:
        block_size: int
        indexes_kv_by_block_stride: bool = False

    @dataclass(frozen=True)
    class MLAAttentionSpec(AttentionSpec):
        alignment: int = 576

        @classmethod
        def merge(cls, specs):
            first = specs[0]
            return cls(
                first.block_size, first.indexes_kv_by_block_stride, first.alignment
            )

    @dataclass(frozen=True)
    class HiddenStateCacheSpec(MLAAttentionSpec):
        pass

    HiddenStateCacheSpec.__module__ = "vllm.v1.kv_cache_interface"
    modules(
        "vllm.v1.kv_cache_interface",
        MLAAttentionSpec=MLAAttentionSpec,
        HiddenStateCacheSpec=HiddenStateCacheSpec,
    )
    load_patch("vllm_sail/patch/enhancement/attention/kv_cache_interface.py")
    spec = MLAAttentionSpec(64, indexer_n_head=64, indexer_q_head_dim=128)
    rebuilt = replace(spec, indexes_kv_by_block_stride=True)
    merged = MLAAttentionSpec.merge([rebuilt, rebuilt])
    assert (merged.indexer_n_head, merged.indexer_q_head_dim) == (64, 128)
    assert merged.indexes_kv_by_block_stride
    assert len(fields(AttentionSpec)) == 2  # inherited metadata remains untouched
    assert {"indexer_n_head", "indexer_q_head_dim"} <= {f.name for f in fields(spec)}
    assert asdict(merged)["indexer_n_head"] == 64
    assert spec == replace(spec) and hash(spec) == hash(replace(spec))
    assert len({spec, replace(spec, indexer_n_head=32)}) == 2
    hidden = HiddenStateCacheSpec(64, indexer_n_head=16, indexer_q_head_dim=128)
    assert isinstance(hidden.merge([hidden]), HiddenStateCacheSpec)
    assert replace(hidden).indexer_n_head == 16
    assert hidden == replace(hidden) and hash(hidden) == hash(replace(hidden))


def test_body_binding_preserves_keyword_defaults_and_live_globals(modules):
    helper = sys.modules["vllm_sail.patch.bodies"].bind_body
    namespace = types.ModuleType("target")
    namespace.value = 3
    exec(
        "def body(*, offset: int = 2) -> int:\n    return value + offset",
        namespace.__dict__,
    )
    other = types.ModuleType("other")
    other.value = 10
    bound = helper(namespace.body, other)
    assert bound() == 12 and bound(offset=1) == 11
    other.value = 20
    assert bound() == 22
    assert bound.__annotations__ == namespace.body.__annotations__


def test_input_batch_reinitializes_when_only_context_length_changes(modules):
    platform = types.SimpleNamespace(is_ppu=lambda: True)
    modules("vllm.platforms", current_platform=platform)
    cls = type("GPUModelRunner", (), {"may_reinitialize_input_batch": lambda *a: None})
    modules(
        "vllm.v1.worker.gpu_model_runner",
        GPUModelRunner=cls,
        InputBatch=lambda **kwargs: types.SimpleNamespace(**kwargs),
    )
    modules(
        "vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils",
        rejection_sample=lambda *a: None,
    )
    load_patch("vllm_sail/patch/enhancement/runtime.py")
    runner = cls()
    runner.__dict__.update(
        max_model_len=8192,
        cp_kv_cache_interleave_size=1,
        jit_warmup_registry=types.SimpleNamespace(
            activate=__import__("contextlib").nullcontext
        ),
        max_encoder_len=0,
        _init_block_sizes=[],
        _init_max_num_blocks=[],
        _init_slot_mapping_modes=[],
        input_batch=types.SimpleNamespace(
            max_model_len=4096,
            logitsprocs=object(),
            logitsprocs_need_output_token_ids=False,
        ),
        max_num_reqs=16,
        max_num_tokens=2048,
        device="ppu",
        model_config=types.SimpleNamespace(get_vocab_size=lambda: 1024),
        num_spec_tokens=2,
        is_pooling_model=False,
        parallel_config=types.SimpleNamespace(cp_kv_cache_interleave_size=1),
        vllm_config=types.SimpleNamespace(reasoning_config=None),
        cache_config=types.SimpleNamespace(use_replayssm=False),
    )
    old = runner.input_batch
    runner.may_reinitialize_input_batch(types.SimpleNamespace(kv_cache_groups=[]))
    assert runner.input_batch is not old and runner.input_batch.max_model_len == 8192
    new = runner.input_batch
    runner.may_reinitialize_input_batch(types.SimpleNamespace(kv_cache_groups=[]))
    assert runner.input_batch is new


def test_deepseek_config_preserves_dense_channelwise_patterns_for_mtp(modules):
    class Fp8Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.packed_modules_mapping = {}

        def apply_vllm_mapper(self, mapper):
            pass

    class LinearBase:
        pass

    modules("vllm.config", get_current_vllm_config=lambda: None)
    modules(
        "vllm.platforms", current_platform=types.SimpleNamespace(is_ppu=lambda: True)
    )
    modules(
        "vllm.model_executor.layers.fused_moe",
        RoutedExperts=type("RoutedExperts", (), {}),
        UnquantizedFusedMoEMethod=object,
    )
    modules("vllm.model_executor.layers.linear", LinearBase=LinearBase)
    modules("vllm.model_executor.layers.quantization", QuantizationMethods=str)
    modules("vllm.model_executor.layers.quantization.fp8", Fp8Config=Fp8Config)
    modules("vllm.model_executor.layers.quantization.mxfp4", Mxfp4MoEMethod=object)
    modules(
        "vllm.model_executor.layers.quantization.utils.quant_utils",
        is_layer_skipped=lambda **kwargs: any(
            p in kwargs["prefix"] for p in kwargs["ignored_layers"]
        ),
    )
    cfg_module = load_patch("vllm_sail/models/deepseek_v4/quant_config.py")
    cls = cfg_module.DeepseekV4FP8Config
    checkpoint = {
        "quant_method": "mxfp4",
        "fp8_channelwise_layers": ["layers.0.attn.wkv"],
    }
    mtp = types.SimpleNamespace(
        model_type="deepseek_mtp", architectures=["DeepSeekV4MTPModel"]
    )
    assert cls.override_quantization_method(checkpoint, None, mtp) == "deepseek_v4_fp8"
    config = cls.from_config(checkpoint)
    assert config.is_checkpoint_fp8_serialized
    config.apply_vllm_mapper(
        types.SimpleNamespace(apply_list=lambda values: ["model.layers.0.attn.wkv"])
    )
    assert config.fp8_channelwise_layers == ["model.layers.0.attn.wkv", "attn.wkv"]
    modules(
        "compressed_tensors.quantization",
        QuantizationArgs=lambda **kwargs: kwargs,
        QuantizationStrategy=types.SimpleNamespace(CHANNEL="channel"),
        QuantizationType=types.SimpleNamespace(FLOAT="float"),
    )
    modules(
        "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors",
        CompressedTensorsLinearMethod=lambda cfg: ("channel-linear", cfg),
    )
    modules(
        "vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8",
        CompressedTensorsW8A8Fp8=lambda **kwargs: kwargs,
    )
    layer = LinearBase()
    assert config.get_quant_method(layer, "model.mtp.3.attn.wkv") == (
        "channel-linear",
        config,
    )
    assert layer.scheme["weight_quant"]["strategy"] == "channel"
    assert not layer.scheme["is_static_input_scheme"]


@pytest.mark.parametrize(
    "quant_dtype,block,expected",
    [
        ("fp8", None, (True, False, False)),
        ("fp8", [128, 128], (True, False, False)),
        ("fp8", [64, 64], (False, False, False)),
        ("int8", None, (False, True, False)),
        ("int8", [128, 128], (False, False, False)),
        ("mxfp4", None, (False, False, True)),
        (None, None, (False, False, False)),
    ],
)
@pytest.mark.parametrize("explicit_manager", [False, True])
@pytest.mark.parametrize(
    "consumer_name",
    [
        "vllm.model_executor.layers.fused_moe.oracle.fp8",
        "vllm.model_executor.layers.quantization.modelopt_block_quant",
    ],
)
def test_deepep_factory_uses_bound_globals_and_quantized_protocol(
    modules, quant_dtype, block, expected, explicit_manager, consumer_name
):
    """Execute the installed factory, including its ordinary non-EP branch."""
    ns = types.SimpleNamespace
    platform = ns(
        ppu=True,
        is_ppu=lambda: platform.ppu,
        is_xpu=lambda: False,
        fp8_dtype=lambda: "fp8",
    )
    modules("vllm.platforms", current_platform=platform)
    manager = ns(world_size=4, get_handle=lambda args: ("handle", args))

    def original(*args, **kwargs):
        return "original"

    def get_manager():
        assert not explicit_manager, "an explicitly supplied manager must be retained"
        return manager

    provider = modules(
        "vllm.model_executor.layers.fused_moe.all2all_utils",
        current_platform=platform,
        torch=ns(int8="int8"),
        maybe_make_prepare_finalize=original,
        maybe_roundup_layer_hidden_size=lambda *args: "upstream-roundup",
        make_moe_prepare_and_finalize_no_dp_ep=lambda flag: ("no-ep", flag),
        get_ep_all2all_manager=get_manager,
        DEEPEP_QUANT_BLOCK_SHAPE=[128, 128],
    )
    consumer = modules(
        consumer_name,
        maybe_make_prepare_finalize=original,
    )
    modules(
        "vllm_sail.model_executor.layers.fused_moe.prepare_finalize.deepep_ll",
        DeepEPLLPrepareAndFinalize=lambda *args, **kwargs: ns(args=args, **kwargs),
    )
    load_patch("vllm_sail/patch/enhancement/deepep.py")
    assert consumer.maybe_make_prepare_finalize is provider.maybe_make_prepare_finalize
    moe = ns(
        moe_parallel_config=ns(
            use_all2all_kernels=False, use_passthrough_all2all=False, dp_size=1
        )
    )
    assert provider.maybe_make_prepare_finalize(moe, None) is None
    assert provider.maybe_make_prepare_finalize(
        moe, None, allow_new_interface=True
    ) == ("no-ep", False)
    moe.moe_parallel_config.use_all2all_kernels = True
    moe.use_deepep_ht_kernels = False
    moe.use_deepep_ll_kernels = True
    moe.max_num_tokens = 32
    moe.hidden_dim = 3584
    moe.num_experts = 16
    result = consumer.maybe_make_prepare_finalize(
        moe,
        ns(quant_dtype=quant_dtype, block_shape=block),
        ("g2p", "p2g", "ids"),
        all2all_manager=manager if explicit_manager else None,
    )
    assert (
        result.use_fp8_dispatch,
        result.use_int8_dispatch,
        result.use_mxfp4_dispatch,
    ) == expected
    assert result.global_to_physical == "g2p"
    assert result.args[0][1]["token_hidden_size"] == 3584
    platform.ppu = False
    assert provider.maybe_make_prepare_finalize(moe, None) == "original"


def test_pla_prefill_resolution_remains_lazy_and_gate_is_live(modules):
    ns = types.SimpleNamespace
    platform = ns(ppu=True, is_ppu=lambda: platform.ppu)
    modules("vllm.platforms", current_platform=platform)
    modules(
        "vllm.logger",
        init_logger=lambda _: ns(
            info_once=lambda *a: None, warning_once=lambda *a: None
        ),
    )
    env = modules("vllm_sail.envs", VLLM_SAIL_USE_PLA=False)
    resolver = load_patch("vllm_sail/attention/pla_prefill.py")
    assert resolver.get_sail_cuda_pla_prefill_fwd() is None
    assert not resolver._resolved

    def kernel():
        return None

    modules("pla.prefill.flashqla", chunk_gated_delta_rule_fwd=kernel)
    modules("pla.prefill.flashqla.ops", SUPPORTED_HEAD_CONFIGS=frozenset({(8, 4)}))
    env.VLLM_SAIL_USE_PLA = True
    assert resolver.get_sail_cuda_pla_prefill_fwd() is kernel
    assert resolver.get_sail_cuda_pla_prefill_head_configs() == frozenset({(8, 4)})
    env.VLLM_SAIL_USE_PLA = False
    assert resolver.get_sail_cuda_pla_prefill_fwd() is None
    assert resolver.get_sail_cuda_pla_prefill_head_configs() == frozenset()
    env.VLLM_SAIL_USE_PLA = True
    platform.ppu = False
    assert resolver.get_sail_cuda_pla_prefill_fwd() is None


@pytest.mark.parametrize(
    "ppu,arch,version,fp8,expected",
    [
        (True, (8, 0), 3, False, None),
        (True, (8, 9), 3, True, None),
        (True, (8, 0), 2, False, "sink"),
        (False, (8, 9), 3, False, "sink"),
        (True, (8, 0), 3, True, "FP8"),
    ],
)
def test_fa_sink_combination_respects_ppu_fa3_and_fp8_gates(
    modules, ppu, arch, version, fp8, expected
):
    class Backend:
        @classmethod
        def supports_combination(cls, *args, **kwargs):
            return "original"

    modules(
        "vllm.v1.attention.backends.flash_attn",
        FlashAttentionBackend=Backend,
        current_platform=types.SimpleNamespace(is_ppu=lambda: ppu),
        DeviceCapability=lambda major, minor: (major, minor),
        get_flash_attn_version=lambda **kw: version,
        is_quantized_kv_cache=lambda dtype: dtype == "fp8",
        flash_attn_supports_kv_cache_dtype=lambda *args, **kw: arch == (8, 9),
    )
    load_patch("vllm_sail/patch/enhancement/attention/flash_attn.py")
    result = Backend.supports_combination(
        128, "bf16", "fp8" if fp8 else "auto", 16, False, True, False, False, arch
    )
    if expected is None:
        assert result is None
    else:
        assert expected in result
