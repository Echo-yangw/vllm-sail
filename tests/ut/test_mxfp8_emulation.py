# SPDX-License-Identifier: Apache-2.0
"""Generic MXFP8 selection and upstream emulation lifecycle, without torch."""

import runpy
from types import SimpleNamespace

import pytest

from tests.support.modules import load_patch
from tests.support.source import function


@pytest.mark.parametrize("is_ppu", [True, False])
def test_emulation_reuses_upstream_only_on_ppu(modules, is_ppu):
    class UpstreamEmulation:
        process_weights_after_loading = object()
        apply_weights = object()

    modules(
        "vllm.model_executor.kernels.linear.mxfp8.emulation",
        EmulationMxfp8LinearKernel=UpstreamEmulation,
    )
    modules("vllm.platforms", current_platform=SimpleNamespace(is_ppu=lambda: is_ppu))
    module = load_patch("vllm_sail/model_executor/kernels/linear/mxfp8.py")
    cls = module.PPUEmulationMxfp8LinearKernel
    assert cls.is_supported()[0] is is_ppu
    assert (
        cls.process_weights_after_loading
        is UpstreamEmulation.process_weights_after_loading
    )
    assert cls.apply_weights is UpstreamEmulation.apply_weights


def test_registration_prepends_once_and_preserves_emulation_filter(
    modules, monkeypatch
):
    registry = load_patch("vllm_sail/registry/linear_kernels/__init__.py")
    ppu, cuda, upstream_emulation = (
        type(name, (), {}) for name in ("PPU", "CUDA", "Emulation")
    )
    upstream = modules(
        "vllm.model_executor.kernels.linear",
        _POSSIBLE_MXFP8_KERNELS={"cuda": [cuda, upstream_emulation]},
        _LINEAR_BACKEND_KERNEL_MAP={"emulation": {upstream_emulation}},
    )
    modules("vllm.platforms.interface", PlatformEnum=SimpleNamespace(CUDA="cuda"))
    modules(
        "vllm.logger", init_logger=lambda name: SimpleNamespace(debug=lambda *a: None)
    )
    monkeypatch.setattr(registry, "_selection_plan", lambda: [("mxfp8", [ppu])])
    registry.register()
    registry.register()
    assert upstream._POSSIBLE_MXFP8_KERNELS["cuda"] == [ppu, cuda, upstream_emulation]
    assert upstream._LINEAR_BACKEND_KERNEL_MAP["emulation"] == {ppu, upstream_emulation}


@pytest.mark.upstream_source
@pytest.mark.parametrize("is_ppu", [True, False])
@pytest.mark.parametrize("backend", ["auto", "emulation"])
@pytest.mark.parametrize("bmm_batch_size", [None, 8])
def test_upstream_selection_preserves_dense_bmm_and_cuda_paths(
    modules, upstream_source_root, is_ppu, backend, bmm_batch_size
):
    platform = SimpleNamespace(
        _enum="cuda",
        is_ppu=lambda: is_ppu,
        is_cuda=lambda: True,
        is_device_capability_family=lambda family: not is_ppu and family == 100,
        get_device_capability=lambda: SimpleNamespace(
            to_int=lambda: 89 if is_ppu else 100
        ),
    )

    class Emulation:
        def __init__(self, config):
            self.config = config

        @classmethod
        def is_supported(cls):
            return True, None

        @classmethod
        def can_implement(cls, config):
            return True, None

    class NativeDense(Emulation):
        pass

    modules("vllm.platforms", current_platform=platform)
    modules(
        "vllm.model_executor.kernels.linear.mxfp8.emulation",
        EmulationMxfp8LinearKernel=Emulation,
    )
    ppu = load_patch(
        "vllm_sail/model_executor/kernels/linear/mxfp8.py"
    ).PPUEmulationMxfp8LinearKernel
    native_path = (
        upstream_source_root / "vllm/model_executor/kernels/linear/mxfp8/deep_gemm.py"
    )
    native_bmm = type(
        "NativeBmm",
        (Emulation,),
        {
            "is_supported": classmethod(
                function(
                    native_path,
                    "DeepGemmMxfp8BmmLinearKernel.is_supported",
                    {
                        "current_platform": platform,
                        "is_deep_gemm_supported": lambda: True,
                    },
                )
            ),
            "can_implement": classmethod(
                function(native_path, "DeepGemmMxfp8BmmLinearKernel.can_implement", {})
            ),
        },
    )
    logger = SimpleNamespace(
        info_once=lambda *a: None, warning_once=lambda *a, **k: None
    )
    namespace = {
        "current_platform": platform,
        "Mxfp8LinearLayerConfig": SimpleNamespace,
        "_POSSIBLE_MXFP8_KERNELS": {"cuda": [ppu, NativeDense, Emulation]},
        "_LINEAR_BACKEND_KERNEL_MAP": {"emulation": {ppu, Emulation}},
        "_get_linear_backend": lambda **kw: backend,
        "DeepGemmMxfp8BmmLinearKernel": native_bmm,
        "EmulationMxfp8LinearKernel": Emulation,
        "envs": SimpleNamespace(VLLM_DISABLED_KERNELS=[]),
        "logger": logger,
    }
    priority = runpy.run_path(
        str(
            upstream_source_root
            / "vllm/model_executor/layers/quantization/utils/humming/priority.py"
        )
    )
    namespace["prioritize_humming"] = priority["prioritize_humming"]
    path = upstream_source_root / "vllm/model_executor/kernels/linear/__init__.py"
    for name in (
        "_filter_kernels_by_backend",
        "_resolve_backend_kernels",
        "init_mxfp8_linear_kernel",
    ):
        function(path, name, namespace)
    selected = namespace["init_mxfp8_linear_kernel"](
        weight_shape=(64, 576), bmm_batch_size=bmm_batch_size
    )
    if bmm_batch_size is not None:
        expected = Emulation if is_ppu or backend == "emulation" else native_bmm
    elif is_ppu:
        expected = ppu
    else:
        expected = Emulation if backend == "emulation" else NativeDense
    assert type(selected) is expected
    assert selected.config.bmm_batch_size == bmm_batch_size
    assert selected.config.weight_shape == (64, 576)


@pytest.mark.upstream_source
@pytest.mark.parametrize("capability", [80, 89, 90, 100])
def test_upstream_humming_priority_keeps_ppu_registration_first(
    upstream_source_root, capability
):
    priority = runpy.run_path(
        str(
            upstream_source_root
            / "vllm/model_executor/layers/quantization/utils/humming/priority.py"
        )
    )
    ppu, marlin, humming = (type(name, (), {}) for name in ("PPU", "Marlin", "Humming"))
    candidates = [ppu, marlin, humming]
    selected = priority["prioritize_humming"](candidates, capability)
    assert selected == (
        [ppu, humming, marlin] if capability == 90 else [ppu, marlin, humming]
    )
    assert candidates == [ppu, marlin, humming]


class Tensor:
    """Shape/dtype double; numerical dequantization is tested on device."""

    # Physical tensor metadata is not a custom reload attribute. Keep it out
    # of __dict__, as on a real torch.Tensor, for the real replace_parameter.
    __slots__ = ("shape", "dtype", "ndim", "data", "device", "is_meta", "__dict__")

    def __init__(self, shape, dtype):
        self.shape, self.dtype = shape, dtype
        self.ndim = len(shape)
        self.data = self
        self.device = "host-double"
        self.is_meta = False

    def stride(self):
        import math

        return tuple(math.prod(self.shape[index + 1 :]) for index in range(self.ndim))

    def copy_(self, value):
        assert self.shape == value.shape and self.dtype == value.dtype
        return self

    def contiguous(self):
        return self

    def numel(self):
        import math

        return math.prod(self.shape)

    def untyped_storage(self):
        return SimpleNamespace(data_ptr=lambda: id(self))

    def storage_offset(self):
        return 0

    def __getitem__(self, slices):
        return Tensor(tuple(s.stop for s in slices), self.dtype)

    def element_size(self):
        return 1 if self.dtype in ("fp8", "uint8") else 2

    def to(self, dtype):
        return Tensor(self.shape, dtype)


@pytest.mark.upstream_source
@pytest.mark.parametrize("dequant_at_load", [True, False])
def test_upstream_dequantizes_at_load_or_per_forward(
    modules, upstream_source_root, dequant_at_load
):
    calls = []

    def dequant(weight, scale):
        calls.append(("dequant", weight, scale))
        return Tensor(weight.shape, "bf16")

    def linear(x, weight, bias):
        calls.append(("linear", weight.dtype, bias))
        assert weight.shape == (64, 576)
        assert weight.dtype == x.dtype
        return x

    class Parameter:
        def __init__(self, tensor, **kwargs):
            self.data = tensor

        def __getattr__(self, name):
            return getattr(self.data, name)

    modules("vllm.envs", VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD=dequant_at_load)
    namespace = {
        "torch": SimpleNamespace(
            nn=SimpleNamespace(
                Parameter=Parameter, functional=SimpleNamespace(linear=linear)
            )
        ),
        "MXFP8_BLOCK_SIZE": 32,
        "MXFP8_SCALE_DTYPE": "uint8",
        "dequant_mxfp8_to_bf16": dequant,
    }
    utils = upstream_source_root / "vllm/model_executor/utils.py"
    for name in ("_extent", "_in_own_storage", "replace_parameter"):
        function(utils, name, namespace)
    path = (
        upstream_source_root / "vllm/model_executor/kernels/linear/mxfp8/emulation.py"
    )
    process = function(
        path, "EmulationMxfp8LinearKernel.process_weights_after_loading", namespace
    )
    apply = function(path, "EmulationMxfp8LinearKernel.apply_weights", namespace)
    layer = SimpleNamespace(
        weight=Parameter(Tensor((64, 576), "fp8")),
        weight_scale=Parameter(Tensor((64, 18), "uint8")),
    )
    weight, scale = layer.weight, layer.weight_scale
    weight.weight_loader, scale.weight_loader = object(), object()
    loaders = (weight.weight_loader, scale.weight_loader)
    process(None, layer)
    assert layer.weight is weight and layer.weight_scale is scale
    assert (layer.weight.weight_loader, layer.weight_scale.weight_loader) == loaders
    assert layer.weight.dtype == ("bf16" if dequant_at_load else "fp8")
    assert len(calls) == int(dequant_at_load)
    x, bias = Tensor((3, 576), "fp16"), object()
    for _ in range(2):
        apply(None, layer, x, bias)
    assert sum(c[0] == "dequant" for c in calls) == (1 if dequant_at_load else 2)
    assert [c for c in calls if c[0] == "linear"] == [("linear", "fp16", bias)] * 2
    assert layer.weight_scale.shape == (64, 18)
