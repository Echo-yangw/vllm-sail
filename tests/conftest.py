# SPDX-License-Identifier: Apache-2.0
"""Shared test harness with two explicit tiers.

``tests/ut`` is CPU-only and must pass anywhere, including environments without
vLLM, torch, or an accelerator. ``tests/e2e`` requires a real PPU and is gated
by the ``ppu`` marker. SDK fakes are opt-in fixtures so no module-level stub can
leak into another test and hide an unexpected dependency.

Device-gated items are further filtered by chip capability: ``cap80`` items run
only on ZW-810E ``(8, 0)`` hardware and ``cap89`` items only on ZW-890P
``(8, 9)`` hardware. ``fake_ppu_platform`` is parameterized over both chip
profiles so CPU-only tests cover each capability branch.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
import types
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, NamedTuple
from uuid import uuid4

import pytest

from tests.support.source import source_root


class DeviceCapability(NamedTuple):
    """Small CUDA-compatible capability value used by platform unit tests."""

    major: int
    minor: int


class ChipProfile(NamedTuple):
    """Static capability matrix of one supported PPU chip."""

    name: str
    capability: DeviceCapability
    fp8: bool
    acext: bool
    bf16_dense_deepgemm: bool


CHIP_PROFILES: dict[str, ChipProfile] = {
    # ZW-810E (PPU 1.0): int8/acext path, no fp8.
    "810e": ChipProfile(
        name="PPU-ZW810E",
        capability=DeviceCapability(8, 0),
        fp8=False,
        acext=True,
        bf16_dense_deepgemm=False,
    ),
    # ZW-890P (PPU 1.5): fp8/DeepGEMM path, no acext.
    "890p": ChipProfile(
        name="PPU-ZW890P",
        capability=DeviceCapability(8, 9),
        fp8=True,
        acext=False,
        bf16_dense_deepgemm=True,
    ),
}


class FakePPUPlatform:
    """Minimal platform double exposing only the contract tests may rely on.

    The chip profile is injected at construction so a single test body can be
    parameterized across both supported cards. ``is_cuda()`` stays ``True``:
    vllm-sail disguises the platform as CUDA to reuse upstream code paths, and
    ``is_ppu()`` is the only PPU discriminator.
    """

    def __init__(self, chip: str = "810e") -> None:
        self.chip = chip
        self.profile = CHIP_PROFILES[chip]

    def get_device_name(self, device_id: int = 0) -> str:
        del device_id
        return self.profile.name

    def get_device_capability(self, device_id: int = 0) -> DeviceCapability:
        del device_id
        return self.profile.capability

    def is_ppu(self) -> bool:
        return True

    def is_cuda(self) -> bool:
        return True


class StrictStubModule(types.ModuleType):
    """A real module that rejects every attribute not explicitly provided."""

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(
            f"PPU SDK test stub {self.__name__!r} has no attribute {name!r}; "
            "declare the test-facing API explicitly instead of mocking it."
        )


def _stub_module(name: str, *, package: bool = False) -> StrictStubModule:
    module = StrictStubModule(name)
    module.__spec__ = importlib.util.spec_from_loader(name, loader=None)
    module.__package__ = name.rpartition(".")[0]
    if package:
        module.__path__ = []  # type: ignore[attr-defined]
    return module


def _real_device_available() -> bool:
    """Probe CUDA/PPU availability without making torch a test dependency."""
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch

            if bool(torch.cuda.is_available()):
                return True
        except (AttributeError, ImportError, OSError, RuntimeError):
            pass
    try:
        probe = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0 and bool(probe.stdout.strip())


HAS_REAL_DEVICE = _real_device_available()

_REAL_CAPABILITY: DeviceCapability | None = None
_REAL_CAPABILITY_PROBED = False


def _real_device_capability() -> DeviceCapability | None:
    """Probe the real device capability once; ``None`` without a device.

    Only called when ``HAS_REAL_DEVICE`` is already true, so importing torch
    here adds no new collection-time dependency on CPU-only hosts.
    """
    global _REAL_CAPABILITY, _REAL_CAPABILITY_PROBED
    if not _REAL_CAPABILITY_PROBED:
        _REAL_CAPABILITY_PROBED = True
        if HAS_REAL_DEVICE:
            try:
                import torch

                major, minor = torch.cuda.get_device_capability(0)
                _REAL_CAPABILITY = DeviceCapability(major, minor)
            except (AttributeError, ImportError, OSError, RuntimeError, ValueError):
                _REAL_CAPABILITY = None
    return _REAL_CAPABILITY


_REAL_DEVICE_BYTES: int | None = None
_REAL_DEVICE_BYTES_PROBED = False


def _real_device_total_bytes() -> int | None:
    """Total memory of device 0 in bytes; ``None`` when it cannot be probed.

    Feeds the checkpoint capacity guard in ``_offline_runner.checkpoint_skip_
    reason``: a weight shard larger than a whole device cannot load at any
    ``gpu_memory_utilization``, and saying so up front beats an opaque
    ``Engine core initialization failed``.  Probed once, lazily, and only when
    ``HAS_REAL_DEVICE`` is already true -- same discipline as the capability
    probe above, so a CPU-only host still collects without torch.
    """
    global _REAL_DEVICE_BYTES, _REAL_DEVICE_BYTES_PROBED
    if not _REAL_DEVICE_BYTES_PROBED:
        _REAL_DEVICE_BYTES_PROBED = True
        if HAS_REAL_DEVICE:
            try:
                import torch

                _REAL_DEVICE_BYTES = int(
                    torch.cuda.get_device_properties(0).total_memory
                )
            except (AttributeError, ImportError, OSError, RuntimeError, ValueError):
                _REAL_DEVICE_BYTES = None
    return _REAL_DEVICE_BYTES


def _chip_for_capability(capability: DeviceCapability) -> str:
    """Map a real device capability onto the closest known chip profile."""
    for chip, profile in CHIP_PROFILES.items():
        if profile.capability == capability:
            return chip
    return "810e"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--require-upstream-source",
        action="store_true",
        help="Fail early unless VLLM_SOURCE_ROOT points to upstream source",
    )
    parser.addoption(
        "--update-golden",
        action="store_true",
        help="Rewrite golden outputs under tests/e2e/models/_golden instead of comparing",
    )
    parser.addoption(
        "--baseline-root",
        action="store",
        default=None,
        help=(
            "Override the read-only perf golden root "
            "(default: tests/e2e/perf/baselines)"
        ),
    )
    parser.addoption(
        "--baseline-output-root",
        action="store",
        default=None,
        help=(
            "Write pending candidates under this root instead of the golden root; "
            "CI sets this to its artifact directory"
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "ppu: requires a real PPU/CUDA device")
    config.addinivalue_line("markers", "cap80: ZW-810E (8,0) only")
    config.addinivalue_line("markers", "cap89: ZW-890P (8,9) only")
    config.addinivalue_line("markers", "perf: performance benchmark")
    config.addinivalue_line("markers", "model_e2e: offline model E2E")
    # Registered here as well so the harness stays self-contained; the
    # canonical declaration lives in pyproject.toml [tool.pytest.ini_options].
    config.addinivalue_line(
        "markers",
        "upstream_source: checks the source selected by VLLM_SOURCE_ROOT "
        "without importing vLLM",
    )
    if config.getoption("--require-upstream-source"):
        source_root(os.environ.get("VLLM_SOURCE_ROOT"), required=True)


@pytest.fixture(scope="session")
def upstream_source_root(request: pytest.FixtureRequest) -> Path:
    """Use the explicitly selected source without importing upstream packages."""
    return source_root(
        os.environ.get("VLLM_SOURCE_ROOT"),
        required=request.config.getoption("--require-upstream-source"),
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if HAS_REAL_DEVICE:
        capability = _real_device_capability()
        for item in items:
            keywords = item.keywords
            if "cap80" in keywords and capability != DeviceCapability(8, 0):
                item.add_marker(
                    pytest.mark.skip(reason="cap80: requires a ZW-810E (8,0) device")
                )
            elif "cap89" in keywords and capability != DeviceCapability(8, 9):
                item.add_marker(
                    pytest.mark.skip(reason="cap89: requires a ZW-890P (8,9) device")
                )
        return
    skip = pytest.mark.skip(reason="requires a real PPU/CUDA device")
    for item in items:
        keywords = item.keywords
        if "ppu" in keywords or "cap80" in keywords or "cap89" in keywords:
            item.add_marker(skip)


@pytest.fixture(params=sorted(CHIP_PROFILES))
def fake_ppu_platform(request: pytest.FixtureRequest) -> FakePPUPlatform:
    """Platform double parameterized over every supported chip profile."""
    return FakePPUPlatform(request.param)


@pytest.fixture(scope="session")
def chip_capability() -> DeviceCapability | None:
    """Capability of the real device, or ``None`` on CPU-only hosts."""
    return _real_device_capability()


class BackendRegistryContext:
    """Isolation helper for MoE/Linear/Attention backend-selection tests.

    Intended usage (category-2 tests), combined with ``ppu_sdk_stubs`` and the
    parameterized ``fake_ppu_platform``::

        def test_select(backend_registry, fake_ppu_platform, ppu_sdk_stubs):
            backend_registry.set_env("VLLM_SAIL_DENSE_BF16_DEEPGEMM", "1")
            module = backend_registry.load(
                "vllm_sail.registry.moe_backends.unquantized"
            )
            assert module.select(...) is expected

    ``load`` executes the target module in a fresh private ``sys.modules``
    entry with ``vllm.platforms.current_platform`` bound to the fake platform,
    so module-level ``current_platform.is_ppu()`` gates see the selected chip
    and repeated loads never share cached state. Everything is monkeypatched
    through the test's own ``MonkeyPatch`` and unwinds automatically.

    The helper itself imports neither torch nor vLLM; ``load`` needs an
    installed ``vllm`` because vllm_sail registry modules import it. CPU-only
    callers must ``pytest.importorskip("vllm")`` before calling ``load``.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, chip: str) -> None:
        self.monkeypatch = monkeypatch
        self.chip = chip
        self.profile = CHIP_PROFILES[chip]

    def set_env(self, name: str, value: str) -> None:
        """Set an environment variable for the requesting test only."""
        self.monkeypatch.setenv(name, value)

    def load(self, module_name: str) -> types.ModuleType:
        """(Re)import ``module_name`` in isolation with the fake platform."""
        import vllm.platforms

        self.monkeypatch.setattr(
            vllm.platforms,
            "current_platform",
            FakePPUPlatform(self.chip),
            raising=False,
        )
        # Remove the cached module so import_module re-executes it from
        # scratch. delitem (not setitem-to-None) is correct: setting the
        # entry to None tells the import system to *block* the name with a
        # ModuleNotFoundError, whereas deleting it lets the normal finder
        # re-discover and re-execute the module. monkeypatch.delitem
        # restores the original entry (if any) on teardown.
        self.monkeypatch.delitem(sys.modules, module_name, raising=False)
        return importlib.import_module(module_name)


@pytest.fixture
def backend_registry(
    monkeypatch: pytest.MonkeyPatch, fake_ppu_platform: FakePPUPlatform
) -> Iterator[BackendRegistryContext]:
    """Yield a backend-selection isolation context bound to the active chip."""
    yield BackendRegistryContext(monkeypatch, fake_ppu_platform.chip)


class GoldenStore:
    """Locates golden model outputs and knows whether to refresh them."""

    def __init__(self, root: Path, update: bool) -> None:
        self.root = root
        self.update = update

    def path(self, name: str) -> Path:
        """Return the golden file path for one model/case name."""
        return self.root / name

    @property
    def reports_root(self) -> Path:
        """Fixed root for the per-run prompt/response reports (git-ignored).

        Sibling of the golden directory; ``_offline_runner.REPORTS_ROOT`` is
        the same path, exposed here so tests can locate reports through the
        store they already use.
        """
        return self.root.parent / "_reports"


@pytest.fixture(scope="session")
def golden_store(request: pytest.FixtureRequest) -> GoldenStore:
    """Golden-output directory for model E2E comparisons.

    Tests compare against ``golden_store.path(name)`` unless
    ``--update-golden`` was passed, in which case they rewrite it.
    """
    root = Path(__file__).parent / "e2e" / "models" / "_golden"
    return GoldenStore(root, request.config.getoption("--update-golden"))


@pytest.fixture(scope="session")
def offline_llm() -> Callable[..., Any]:
    """Thin factory delegating to ``_offline_runner.build_llm``.

    Resolves the real device capability into a chip profile (TP=8 on ZW-810E,
    8 on ZW-890P) and hands it to the shared offline runner, which applies the
    fixed model-E2E parameters. Without a real device the factory skips, and
    the runner is imported lazily so collection stays CPU-only.

    The resolved weight tree is checked here, once, for every caller: a
    checkpoint directory that is absent or holds no ``config.json`` skips with
    ``weights unavailable: <path>``, and one whose per-rank weight shard is
    larger than a whole device skips with ``weights do not fit: ...`` instead of
    OOM-ing inside ``initialize_model``.  The paths are recorded constants of the
    suite rather than test inputs, and a machine that cannot serve the weights --
    because it does not mount the model NAS or because it has too few
    accelerators for this model -- has nothing to assert about numerics.

    At most **one** engine is live at a time.  ``gpu_memory_utilization`` is
    0.90, so a second reservation cannot coexist with the first; every build
    therefore tears the previous engine down first (see
    ``_offline_runner.release_llm``).  Callers that want to reuse an engine
    across tests cache it themselves -- ``test_model_perf.perf_llm_factory``
    does exactly that, once per model key.
    """
    live: list[Any] = []

    def factory(model: str, config: Any = None) -> Any:
        if not HAS_REAL_DEVICE:
            pytest.skip("requires a real PPU/CUDA device")
        capability = _real_device_capability()
        if capability is None:
            pytest.skip("cannot detect the capability of the real device")
        chip = _chip_for_capability(capability)
        profile = CHIP_PROFILES[chip]
        from tests.e2e.models._offline_runner import (
            build_llm,
            checkpoint_skip_reason,
            release_llm,
            tensor_parallel_for_chip,
        )

        reason = checkpoint_skip_reason(
            model,
            tensor_parallel_size=tensor_parallel_for_chip(chip),
            device_total_bytes=_real_device_total_bytes(),
        )
        if reason is not None:
            pytest.skip(reason)
        while live:
            release_llm(live.pop())
        llm = build_llm(model, profile, config)
        live.append(llm)
        return llm

    return factory


@pytest.fixture(scope="session")
def perf_baseline(request: pytest.FixtureRequest) -> Iterator[dict[str, Any]]:
    """Perf-baseline handle backed by ``tests/e2e/perf/_baseline_store``.

    Skipped without a real device. Resolves the active chip key and the read-only
    golden root (``--baseline-root`` override, else ``tests/e2e/perf/baselines``)
    plus an optional writable candidate root (``--baseline-output-root``), then
    exposes a ``store(suite)`` factory that returns one shared
    :class:`BaselineStore` per suite. At session teardown each candidate is
    persisted -- the first run on a chip writes its ``status="pending"``
    candidate -- and, when ``VLLM_SAIL_PERF_REPORT_DIR`` is set, the comparison
    report (``perf_report.json`` / ``perf_report.md``) is emitted for CI upload.
    The store module is pure standard library and is imported lazily here, so
    CPU-only collection never reaches it.
    """
    if not HAS_REAL_DEVICE:
        pytest.skip("requires a real PPU/CUDA device")
    capability = _real_device_capability()
    chip = _chip_for_capability(capability) if capability is not None else "810e"
    override = request.config.getoption("--baseline-root")
    root = (
        Path(override)
        if override
        else Path(__file__).parent / "e2e" / "perf" / "baselines"
    )
    output_override = request.config.getoption("--baseline-output-root")
    output_root = Path(output_override) if output_override else root
    update = False

    from tests.e2e.perf._baseline_store import BaselineStore

    stores: dict[str, Any] = {}

    def store(suite: str) -> Any:
        if suite not in stores:
            stores[suite] = BaselineStore(root, chip, suite, output_root=output_root)
        return stores[suite]

    yield {
        "update": update,
        "root": root,
        "output_root": output_root,
        "chip": chip,
        "store": store,
    }

    report_dir = os.environ.get("VLLM_SAIL_PERF_REPORT_DIR")
    for suite, baseline_store in stores.items():
        baseline_store.save()
        if report_dir:
            baseline_store.emit(Path(report_dir) / suite)


@pytest.fixture
def ppu_sdk_stubs(monkeypatch: pytest.MonkeyPatch) -> dict[str, StrictStubModule]:
    """Install strict SDK package stubs for one requesting test only."""
    stubs = {
        name: _stub_module(name, package=name in {"flash_attn", "flash_attn_3"})
        for name in (
            "acext",
            "deep_gemm",
            "tokenspeed_mla",
            "flash_attn",
            "flash_attn_3",
        )
    }
    flash_attn_3_c = _stub_module("flash_attn_3._C")
    stubs["flash_attn_3._C"] = flash_attn_3_c
    stubs["flash_attn_3"]._C = flash_attn_3_c  # type: ignore[attr-defined]
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    return stubs


@pytest.fixture
def patch_utils_module() -> Iterator[types.ModuleType]:
    """Load patch/utils.py without executing the runtime patch package init."""
    name = f"_vllm_sail_patch_utils_test_{uuid4().hex}"
    path = Path(__file__).parents[1] / "vllm_sail" / "patch" / "utils.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)
