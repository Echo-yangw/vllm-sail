# SPDX-License-Identifier: Apache-2.0
"""The imported router preserves Triton dispatch and cached consumer aliases."""

import importlib.util
import inspect
import sys
from types import SimpleNamespace as NS

import pytest

from tests.ut.test_ppu_kernel_capabilities import load_patch
from tests.ut.test_ppu_kernel_capabilities import modules as _modules

modules = _modules


@pytest.mark.upstream_source
@pytest.mark.parametrize("ppu", [False, True])
@pytest.mark.parametrize("experts", [128, 256])
@pytest.mark.parametrize("normalization", ["sigmoid", "softmax"])
@pytest.mark.parametrize("bias_dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("padding", [False, True])
def test_registered_router_preserves_fallback_and_preloaded_aliases(
    modules,
    monkeypatch,
    upstream_source_root,
    ppu,
    experts,
    normalization,
    bias_dtype,
    padding,
):
    class Tensor:
        device = "host-double"

        def __init__(self, shape, dtype="float32", root=None):
            self.shape, self.dtype = shape, dtype
            self.root = self if root is None else root

        def dim(self):
            return len(self.shape)

        def is_contiguous(self):
            return True

        def stride(self, axis):
            return self.shape[1] if axis == 0 and self.dim() == 2 else 1

        def float(self):
            return Tensor(self.shape)

        def __getitem__(self, index):
            assert index == (slice(None, 2), slice(None, 8))
            return Tensor((2, 8), self.dtype, root=self.root)

    calls = []

    class Kernel:
        def __init__(self, fn):
            self.fn = fn

        def __getitem__(self, grid):
            assert grid == (2,)

            def launch(*args, **kwargs):
                options = {k: v for k, v in kwargs.items() if k != "num_warps"}
                bound = inspect.signature(self.fn).bind(*args, **options).arguments
                calls.append(("triton", bound, kwargs))

            return launch

    def native(*args):
        calls.append(("native", args))
        if ppu:
            raise NotImplementedError("unsupported PPU compute stub")

    def no_allocation(*args, **kwargs):
        pytest.fail("provided output buffers must preserve allocation-free dispatch")

    modules(
        "torch",
        Tensor=Tensor,
        float32="float32",
        int32="int32",
        bool="bool",
        empty=no_allocation,
    )
    modules("vllm._custom_ops", muse_spark_routing=native)
    modules(
        "vllm.platforms",
        current_platform=NS(
            is_ppu=lambda: ppu,
            is_cuda=lambda: True,
            is_rocm=lambda: False,
            is_arch_support_pdl=lambda: False,
        ),
    )
    modules(
        "vllm.triton_utils",
        tl=NS(constexpr=object),
        triton=NS(jit=Kernel, next_power_of_2=lambda n: 1 << (n - 1).bit_length()),
    )
    name = "vllm.models.muse_spark.common.ops.fused_routing"
    modules(name)
    path = upstream_source_root / "vllm/models/muse_spark/common/ops/fused_routing.py"
    spec = importlib.util.spec_from_file_location(name, path)
    provider = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, provider)
    monkeypatch.setattr(sys.modules[name.rpartition(".")[0]], "fused_routing", provider)
    spec.loader.exec_module(provider)
    original = provider.fused_musespark1x_routing
    consumers = [
        modules(n, fused_musespark1x_routing=original)
        for n in (
            "vllm.models.muse_spark.common.moe",
            "vllm.models.muse_spark.nvidia.model",
        )
    ]
    load_patch("vllm_sail/patch/enhancement/attention/musespark_routing.py")
    assert all(
        c.fused_musespark1x_routing is provider.fused_musespark1x_routing
        for c in consumers
    )
    assert inspect.signature(provider.fused_musespark1x_routing) == inspect.signature(
        original
    )
    weights, ids = Tensor((2, 8)), Tensor((2, 8), "int32")
    mask = Tensor((2,), "bool") if padding else None
    bias = Tensor((experts,), bias_dtype)
    outputs = consumers[0].fused_musespark1x_routing(
        Tensor((2, experts)),
        bias,
        8,
        0.5,
        1.25,
        normalization,
        1e-6,
        weights,
        ids,
        is_padding=mask,
    )
    assert outputs[0].root is weights and outputs[1].root is ids
    assert len(calls) == 1
    if not ppu and experts == 256 and normalization == "sigmoid":
        assert calls[0][0] == "native"
        assert calls[0][1][-1] is mask
    else:
        kind, bound, options = calls[0]
        assert kind == "triton" and bound["is_padding"] is mask
        assert bound["bias_ptr"].dtype == "float32"
        assert bound["TOPK"] == 8 and bound["NUM_EXPERTS_PADDED"] == experts
        assert bound["NORMALIZATION_MODE"] == int(normalization == "softmax")
        assert bound["router_output_multiplier"] == 1.25
        assert bound["division_eps"] == 1e-6 and bound["route_scale"] == 0.5
        assert bound["USE_PDL"] is False and "launch_pdl" not in options
        assert options["num_warps"] == (1 if experts == 256 else 4)


def test_unimportable_provider_skips_with_a_warning(modules, monkeypatch):
    """An empty vLLM build must skip the patch, not abort ``patch.install()``.

    ``VLLM_TARGET_DEVICE=empty`` leaves ``torch.ops._C`` unclaimed, and importing
    ``vllm.models.muse_spark`` reads that namespace at module scope, so the provider
    import raises ``AttributeError``. Registering nothing is the honest outcome
    there: the native router the patch guards against cannot dispatch either.
    """

    class ExplodingLoader:
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            raise AttributeError(
                "'_OpNamespace' '_C' object has no attribute 'rotary_embedding'"
            )

    class ExplodingFinder:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "vllm.models.muse_spark":
                return importlib.util.spec_from_loader(fullname, ExplodingLoader())
            return None

    warnings = []
    modules("vllm.models")
    modules(
        "vllm.logger",
        init_logger=lambda name: NS(
            warning=lambda message, *args: warnings.append(message % args)
        ),
    )
    modules("vllm.platforms", current_platform=NS(is_ppu=lambda: False))
    # An earlier test can leave the real package cached (its fake compiled ops make
    # the import succeed once), which would bypass the finder below.
    for cached in [
        name
        for name in sys.modules
        if name == "vllm.models.muse_spark"
        or name.startswith("vllm.models.muse_spark.")
    ]:
        monkeypatch.delitem(sys.modules, cached)
    monkeypatch.setattr(sys, "meta_path", [ExplodingFinder(), *sys.meta_path])

    patch_module = load_patch(
        "vllm_sail/patch/enhancement/attention/musespark_routing.py"
    )

    assert patch_module._unavailable is True
    assert not hasattr(patch_module, "fused_musespark1x_routing")
    assert len(warnings) == 1 and "MuseSpark routing patch" in warnings[0]
