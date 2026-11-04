# SPDX-License-Identifier: Apache-2.0
"""Tests for the opt-in NVTX profiling subpackage (Phase-1 decision Q6).

Everything here runs on a bare CPU runner: no vLLM, no ``nvtx`` and no
``model_prof`` are required. The fork's engine/worker/scheduler NVTX hunks
became delegating ``@patch`` modules under ``vllm_sail/profiling/`` that
install nothing unless ``VLLM_PPU_NVTX_PROFILE`` is set, so the tests cover:

* gate off -> ``install()`` patches nothing;
* gate on  -> every patch lands on its (faked) upstream target;
* the shared helpers degrade to no-ops without the optional dependencies.
"""

from __future__ import annotations

import functools
import importlib
import importlib.util
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vllm_sail.patch.utils import PATCH_MARKER, PATCH_REGISTRY, original_of

GATE_VARS = ("VLLM_SAIL_NVTX_PROFILE", "VLLM_PPU_NVTX_PROFILE", "SAIL_NVTX_PROFILE")

PROFILING_TARGETS = (
    "vllm.model_executor.layers.linear.ReplicatedLinear.forward",
    "vllm.model_executor.layers.linear.ColumnParallelLinear.forward",
    "vllm.model_executor.layers.linear.RowParallelLinear.forward",
    "vllm.model_executor.layers.fused_moe.modular_kernel.FusedMoEKernelModularImpl.apply",
    "vllm.v1.core.sched.scheduler.Scheduler.schedule",
    "vllm.v1.core.sched.scheduler.Scheduler._update_after_schedule",
    "vllm.v1.engine.core.EngineCore.__init__",
    "vllm.v1.engine.core.EngineCore.step",
    "vllm.v1.engine.core.EngineCore.step_with_batch_queue",
    "vllm.v1.worker.gpu_model_runner.GPUModelRunner._prepare_inputs",
    "vllm.v1.worker.gpu_model_runner.GPUModelRunner._sample",
    "vllm.v1.worker.gpu_model_runner.GPUModelRunner._model_forward",
    "vllm.v1.worker.gpu_model_runner.GPUModelRunner.execute_model",
    "vllm.v1.worker.gpu.model_runner.GPUModelRunner.prepare_inputs",
    "vllm.v1.worker.gpu.model_runner.GPUModelRunner.sample",
    "vllm.v1.worker.gpu.model_runner.GPUModelRunner.execute_model",
)

PROFILING_SUBMODULES = (
    "vllm_sail.profiling.linear",
    "vllm_sail.profiling.modular_moe",
    "vllm_sail.profiling.moe",
    "vllm_sail.profiling.nvtx",
    "vllm_sail.profiling.engine_core",
    "vllm_sail.profiling.model_runner",
    "vllm_sail.profiling.model_runner_v2",
    "vllm_sail.profiling.scheduler",
)


def _sched_output(**overrides: Any) -> SimpleNamespace:
    defaults = {
        "scheduled_new_reqs": [],
        "total_num_scheduled_tokens": 5,
        "num_scheduled_tokens": {"r1": 5},
        "scheduled_spec_decode_tokens": {},
        "finished_req_ids": set(),
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.fixture
def cleared_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in GATE_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fake_vllm(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A minimal fake vLLM tree carrying the three patch targets."""

    def package(name: str) -> types.ModuleType:
        module = types.ModuleType(name)
        module.__path__ = []  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, name, module)
        return module

    def plain(name: str) -> types.ModuleType:
        module = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    vllm = package("vllm")
    vllm_logger = plain("vllm.logger")
    vllm_logger.init_logger = logging.getLogger
    vllm.logger = vllm_logger

    package("vllm.v1")
    package("vllm.v1.worker")
    package("vllm.v1.worker.gpu")
    package("vllm.v1.engine")
    package("vllm.v1.core")
    package("vllm.v1.core.sched")

    class GPUModelRunner:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.init_args = (args, kwargs)

        def _prepare_inputs(self, scheduler_output, num_scheduled_tokens):
            return ("prepare_inputs", scheduler_output, num_scheduled_tokens)

        def _sample(self, logits, spec_decode_metadata):
            return ("sample", logits, spec_decode_metadata)

        def _model_forward(self, *args: Any, **kwargs: Any):
            return ("forward", args, kwargs)

        def execute_model(self, scheduler_output, intermediate_tensors=None):
            return ("execute_model", scheduler_output, intermediate_tensors)

    class GPUModelRunnerV2:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.init_args = (args, kwargs)

        def prepare_inputs(
            self, scheduler_output, batch_req_state, batch_desc, num_active_loras
        ):
            return ("prepare_inputs", scheduler_output, batch_req_state)

        def sample(self, hidden_states, input_batch, grammar_output):
            return ("sample", hidden_states, input_batch)

        def execute_model(
            self, scheduler_output, intermediate_tensors=None, dummy_run=False, **kw
        ):
            return ("execute_model", scheduler_output, dummy_run)

    class EngineCore:
        # The profiling wrappers ask the scheduler whether this step dispatches a
        # batch (``EngineCore.scheduler.has_requests()``). The fake always does;
        # tests override the attribute to model an idle step.
        scheduler = SimpleNamespace(has_requests=lambda: True)

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.init_args = (args, kwargs)

        def step(self):
            return ("step",)

        def step_with_batch_queue(self):
            return ("step_with_batch_queue",)

    class Scheduler:
        def schedule(self, throttle_prefills: bool = False):
            return ("schedule", throttle_prefills)

        def _update_after_schedule(self, scheduler_output):
            self.last_update = scheduler_output

    package("vllm.model_executor")
    package("vllm.model_executor.layers")
    package("vllm.model_executor.layers.fused_moe")
    linear = plain("vllm.model_executor.layers.linear")
    for name in ("ReplicatedLinear", "ColumnParallelLinear", "RowParallelLinear"):
        setattr(linear, name, type(name, (), {"forward": lambda *a: None}))
    modular = plain("vllm.model_executor.layers.fused_moe.modular_kernel")
    modular.FusedMoEKernelModularImpl = type(
        "FusedMoEKernelModularImpl", (), {"apply": lambda *a: None}
    )

    runner_module = plain("vllm.v1.worker.gpu_model_runner")
    runner_module.GPUModelRunner = GPUModelRunner
    runner_v2_module = plain("vllm.v1.worker.gpu.model_runner")
    runner_v2_module.GPUModelRunner = GPUModelRunnerV2
    core_module = plain("vllm.v1.engine.core")
    core_module.EngineCore = EngineCore
    scheduler_module = plain("vllm.v1.core.sched.scheduler")
    scheduler_module.Scheduler = Scheduler

    return SimpleNamespace(
        GPUModelRunner=GPUModelRunner,
        GPUModelRunnerV2=GPUModelRunnerV2,
        EngineCore=EngineCore,
        Scheduler=Scheduler,
        originals={
            "GPUModelRunner._prepare_inputs": GPUModelRunner._prepare_inputs,
            "GPUModelRunner._sample": GPUModelRunner._sample,
            "GPUModelRunner._model_forward": GPUModelRunner._model_forward,
            "GPUModelRunner.execute_model": GPUModelRunner.execute_model,
            "gpu.model_runner.GPUModelRunner.prepare_inputs": (
                GPUModelRunnerV2.prepare_inputs
            ),
            "gpu.model_runner.GPUModelRunner.sample": GPUModelRunnerV2.sample,
            "gpu.model_runner.GPUModelRunner.execute_model": (
                GPUModelRunnerV2.execute_model
            ),
            "EngineCore.__init__": EngineCore.__init__,
            "EngineCore.step": EngineCore.step,
            "EngineCore.step_with_batch_queue": EngineCore.step_with_batch_queue,
            "Scheduler.schedule": Scheduler.schedule,
            "Scheduler._update_after_schedule": Scheduler._update_after_schedule,
        },
    )


@pytest.fixture
def nvtx_stub(monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:
    """Fake `nvtx` package recording mark/annotate usage."""
    calls: dict[str, list] = {"marks": [], "annotated": []}
    stub = types.ModuleType("nvtx")

    def mark(message: str, **kwargs: Any) -> None:
        calls["marks"].append(message)

    def annotate(name: str, **kwargs: Any):
        calls["annotated"].append(name)

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kw):
                return func(*args, **kw)

            return wrapper

        return decorator

    stub.mark = mark
    stub.annotate = annotate
    monkeypatch.setitem(sys.modules, "nvtx", stub)
    return calls


@pytest.fixture
def profiling_pkg(fake_vllm: SimpleNamespace, monkeypatch: pytest.MonkeyPatch):
    """Import vllm_sail.profiling fresh, with the gate flag reset."""
    module = importlib.import_module("vllm_sail.profiling")
    for name in PROFILING_SUBMODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    # `from vllm_sail.profiling import engine_core, ...` resolves via these
    # attributes first; drop them too so install() re-imports the submodules
    # against this test's fakes instead of reusing a stale cached copy.
    for attr in (
        "nvtx",
        "engine_core",
        "model_runner",
        "model_runner_v2",
        "scheduler",
        "linear",
        "modular_moe",
        "moe",
    ):
        monkeypatch.delattr(module, attr, raising=False)
    monkeypatch.setattr(module, "_installed", False)
    return module


@pytest.fixture
def installed(
    profiling_pkg,
    nvtx_stub: dict[str, list],
    monkeypatch: pytest.MonkeyPatch,
    fake_vllm: SimpleNamespace,
):
    """Run install() with the gate on and return the faked targets."""
    monkeypatch.setenv("VLLM_PPU_NVTX_PROFILE", "1")
    registry_before = len(PATCH_REGISTRY)
    profiling_pkg.install()
    return SimpleNamespace(
        pkg=profiling_pkg,
        fakes=fake_vllm,
        nvtx_calls=nvtx_stub,
        registry_before=registry_before,
    )


def _records_for(targets) -> dict[str, Any]:
    return {r.target: r for r in PATCH_REGISTRY if r.target in targets}


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_gate_off_installs_nothing(
    profiling_pkg, fake_vllm: SimpleNamespace, cleared_gate: None
) -> None:
    records_before = _records_for(PROFILING_TARGETS)
    profiling_pkg.install()

    assert profiling_pkg._installed is False
    for name in PROFILING_SUBMODULES:
        assert name not in sys.modules
    for original in fake_vllm.originals.values():
        assert getattr(original, PATCH_MARKER, None) is None
    assert _records_for(PROFILING_TARGETS) == records_before


def test_gate_on_installs_every_patch(installed) -> None:
    assert installed.pkg._installed is True

    records = _records_for(PROFILING_TARGETS)
    assert set(records) == set(PROFILING_TARGETS)
    assert len(PATCH_REGISTRY) == installed.registry_before + len(PROFILING_TARGETS)

    fakes = installed.fakes
    replacements = {
        "GPUModelRunner._prepare_inputs": fakes.GPUModelRunner._prepare_inputs,
        "GPUModelRunner._sample": fakes.GPUModelRunner._sample,
        "GPUModelRunner._model_forward": fakes.GPUModelRunner._model_forward,
        "GPUModelRunner.execute_model": fakes.GPUModelRunner.execute_model,
        "gpu.model_runner.GPUModelRunner.prepare_inputs": (
            fakes.GPUModelRunnerV2.prepare_inputs
        ),
        "gpu.model_runner.GPUModelRunner.sample": fakes.GPUModelRunnerV2.sample,
        "gpu.model_runner.GPUModelRunner.execute_model": (
            fakes.GPUModelRunnerV2.execute_model
        ),
        "EngineCore.__init__": fakes.EngineCore.__init__,
        "EngineCore.step": fakes.EngineCore.step,
        "EngineCore.step_with_batch_queue": fakes.EngineCore.step_with_batch_queue,
        "Scheduler.schedule": fakes.Scheduler.schedule,
        "Scheduler._update_after_schedule": fakes.Scheduler._update_after_schedule,
    }
    for key, replacement in replacements.items():
        assert getattr(replacement, PATCH_MARKER, None) is not None, key
        assert replacement is not fakes.originals[key], key
        target = records[[t for t in records if t.endswith(key)][0]].target
        assert original_of(replacement, target) is fakes.originals[key], key


def test_install_is_idempotent(installed) -> None:
    count = len(PATCH_REGISTRY)
    installed.pkg.install()
    installed.pkg.install()
    assert len(PATCH_REGISTRY) == count


@pytest.mark.parametrize("gate_var", GATE_VARS)
def test_gate_aliases_enable_profiling(
    profiling_pkg,
    nvtx_stub,
    monkeypatch: pytest.MonkeyPatch,
    cleared_gate: None,
    gate_var: str,
) -> None:
    monkeypatch.setenv(gate_var, "1")
    profiling_pkg.install()
    assert profiling_pkg._installed is True


@pytest.mark.parametrize("value", ["0", ""])
def test_sail_false_prevents_install_with_truthy_legacy_aliases(
    profiling_pkg,
    nvtx_stub,
    monkeypatch: pytest.MonkeyPatch,
    cleared_gate: None,
    value: str,
) -> None:
    monkeypatch.setenv("VLLM_SAIL_NVTX_PROFILE", value)
    monkeypatch.setenv("VLLM_PPU_NVTX_PROFILE", "1")
    monkeypatch.setenv("SAIL_NVTX_PROFILE", "1")
    profiling_pkg.install()
    assert profiling_pkg._installed is False


def test_missing_nvtx_package_disables_with_warning(
    profiling_pkg,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("VLLM_PPU_NVTX_PROFILE", "1")
    monkeypatch.setitem(sys.modules, "nvtx", None)
    with caplog.at_level(logging.WARNING):
        profiling_pkg.install()
    assert profiling_pkg._installed is False
    assert any("nvtx" in message for message in caplog.messages)


def test_missing_model_prof_still_installs(
    profiling_pkg,
    nvtx_stub,
    monkeypatch: pytest.MonkeyPatch,
    cleared_gate: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("VLLM_PPU_NVTX_PROFILE", "1")
    monkeypatch.setitem(sys.modules, "model_prof", None)
    with caplog.at_level(logging.INFO):
        profiling_pkg.install()
    assert profiling_pkg._installed is True
    assert any(
        "without `model_prof`" in record.getMessage() for record in caplog.records
    )


def test_drifted_target_degrades_to_warning(
    profiling_pkg,
    nvtx_stub,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("VLLM_PPU_NVTX_PROFILE", "1")
    monkeypatch.delitem(sys.modules, "vllm.v1.worker.gpu_model_runner")
    with caplog.at_level(logging.WARNING):
        profiling_pkg.install()
    assert profiling_pkg._installed is False
    assert any("could not be" in message for message in caplog.messages)


# ---------------------------------------------------------------------------
# Wrapper behaviour against the faked targets
# ---------------------------------------------------------------------------


@pytest.fixture
def recording_nvtx(installed, monkeypatch: pytest.MonkeyPatch):
    """Swap the helper backends for recorders after install imported them.

    ``owned`` mirrors what ``model_prof.prof_iter`` reports: True means the SDK
    already opened ``[prof_range]: iter N``, so the wrapper must not open a
    second one.
    """
    prof_nvtx = importlib.import_module("vllm_sail.profiling.nvtx")
    pushes: list[str] = []
    pops: list[int] = []
    iters: list[int] = []
    recording = SimpleNamespace(pushes=pushes, pops=pops, iters=iters, owned=False)

    def model_prof_iter(iteration: int) -> bool:
        iters.append(iteration)
        return recording.owned

    monkeypatch.setattr(
        prof_nvtx, "_torch_range_push", lambda label: pushes.append(label)
    )
    monkeypatch.setattr(prof_nvtx, "_torch_range_pop", lambda: pops.append(1))
    monkeypatch.setattr(prof_nvtx, "_model_prof_iter", model_prof_iter)
    return recording


def test_execute_model_wraps_iteration_range(installed, recording_nvtx) -> None:
    runner = installed.fakes.GPUModelRunner()
    assert not hasattr(runner, "iteration")

    output = _sched_output()
    result = runner.execute_model(output)

    assert result == ("execute_model", output, None)
    assert recording_nvtx.pushes == ["[prof_range]: iter 0"]
    assert len(recording_nvtx.pops) == 1
    assert recording_nvtx.iters == [0]
    assert runner.iteration == 1
    assert "total_tokens=5,req_id:num_tokens={'r1': 5}" in installed.nvtx_calls["marks"]

    runner.execute_model(output)
    assert recording_nvtx.pushes[1] == "[prof_range]: iter 1"


def test_execute_model_defers_to_model_prof(installed, recording_nvtx) -> None:
    """`model_prof.prof_iter` opens `[prof_range]`; a second push nests it."""
    recording_nvtx.owned = True
    runner = installed.fakes.GPUModelRunner()

    runner.execute_model(_sched_output())
    runner.execute_model(_sched_output())

    assert recording_nvtx.pushes == []
    assert recording_nvtx.pops == []
    assert recording_nvtx.iters == [0, 1]
    assert runner.iteration == 2


def test_sample_wraps_sampler_range(installed, recording_nvtx) -> None:
    runner = installed.fakes.GPUModelRunner()
    logits = SimpleNamespace(shape=(4, 8))

    result = runner._sample(logits, None)

    assert result == ("sample", logits, None)
    assert recording_nvtx.pushes == ["[FW_NATIVE] op:sampler,logits:(4, 8)"]
    assert len(recording_nvtx.pops) == 1

    runner._sample(None, "spec")
    assert recording_nvtx.pushes[1] == "[FW_NATIVE] op:sampler,logits:None"


def test_model_forward_range_uses_stashed_scheduler_output(
    installed, recording_nvtx
) -> None:
    runner = installed.fakes.GPUModelRunner()

    assert runner._model_forward(input_ids=1) == ("forward", (), {"input_ids": 1})
    assert recording_nvtx.pushes == []

    runner._ppu_nvtx_prof_scheduler_output = _sched_output(
        num_scheduled_tokens={"r1": 3, "r2": 1},
        scheduled_spec_decode_tokens={"r1": [10]},
    )
    runner._model_forward(input_ids=2)
    assert recording_nvtx.pushes == ["total bs=2, P bs=1"]
    assert len(recording_nvtx.pops) == 1

    # execute_model clears the stash, so later forwards are uninstrumented.
    count = len(recording_nvtx.pushes)
    runner.execute_model(_sched_output())
    assert recording_nvtx.pushes[count:] == ["[prof_range]: iter 0"]
    count = len(recording_nvtx.pushes)
    runner._model_forward(input_ids=3)
    assert len(recording_nvtx.pushes) == count


def test_prepare_inputs_is_annotated(installed) -> None:
    runner = installed.fakes.GPUModelRunner()
    assert runner._prepare_inputs("so", "nst") == ("prepare_inputs", "so", "nst")
    assert "prepare_inputs" in installed.nvtx_calls["annotated"]


# ---------------------------------------------------------------------------
# Model Runner V2 (the 0.31 default) carries the same ranges
# ---------------------------------------------------------------------------


def test_v2_execute_model_wraps_iteration_and_batch_ranges(
    installed, recording_nvtx
) -> None:
    runner = installed.fakes.GPUModelRunnerV2()
    # No __init__ patch: the counter is created on first use, like on V1.
    assert not hasattr(runner, "iteration")

    output = _sched_output()
    assert runner.execute_model(output) == ("execute_model", output, False)

    assert recording_nvtx.pushes == ["[prof_range]: iter 0", "total bs=1, P bs=1"]
    assert len(recording_nvtx.pops) == 2
    assert recording_nvtx.iters == [0]
    assert runner.iteration == 1
    assert any(m.startswith("total_tokens=5") for m in installed.nvtx_calls["marks"])

    runner.execute_model(output)
    assert recording_nvtx.pushes[2:] == [
        "[prof_range]: iter 1",
        "total bs=1, P bs=1",
    ]


def test_v2_execute_model_forwards_dummy_runs_unprofiled(
    installed, recording_nvtx
) -> None:
    runner = installed.fakes.GPUModelRunnerV2()
    output = _sched_output()

    assert runner.execute_model(output, dummy_run=True) == (
        "execute_model",
        output,
        True,
    )
    assert recording_nvtx.pushes == []
    assert recording_nvtx.iters == []
    assert not hasattr(runner, "iteration")

    # The first real batch still lands on iteration 0: a warmup run must not
    # offset this process's labels (nor the SDK's profile window) from the
    # engine's, which never counted it either.
    runner.execute_model(output)
    assert recording_nvtx.iters == [0]
    assert recording_nvtx.pushes == ["[prof_range]: iter 0", "total bs=1, P bs=1"]


def test_v2_sample_wraps_sampler_range(installed, recording_nvtx) -> None:
    runner = installed.fakes.GPUModelRunnerV2()
    hidden_states = SimpleNamespace(shape=(7, 16))

    assert runner.sample(hidden_states, "batch", None) == (
        "sample",
        hidden_states,
        "batch",
    )
    assert recording_nvtx.pushes == ["[FW_NATIVE] op:sampler,hidden_states:(7, 16)"]
    assert len(recording_nvtx.pops) == 1


def test_v2_prepare_inputs_is_annotated(installed) -> None:
    runner = installed.fakes.GPUModelRunnerV2()

    assert runner.prepare_inputs("so", "state", "desc", 0) == (
        "prepare_inputs",
        "so",
        "state",
    )
    assert "prepare_inputs" in installed.nvtx_calls["annotated"]


def test_v2_instruments_compute_logits_once(installed, recording_nvtx) -> None:
    runner = installed.fakes.GPUModelRunnerV2()
    model = SimpleNamespace(compute_logits=lambda hidden_states: hidden_states)
    runner.model = model

    runner.execute_model(_sched_output())
    traced = model.compute_logits
    runner.execute_model(_sched_output())
    assert model.compute_logits is traced, "compute_logits must be wrapped once"

    hidden_states = SimpleNamespace(shape=(2, 4))
    assert model.compute_logits(hidden_states) is hidden_states
    assert recording_nvtx.pushes[-1] == (
        "[FW_GEMM] op:compute_logits,hidden_states:(2, 4)"
    )


def test_engine_core_steps_fall_back_to_own_ranges(installed, recording_nvtx) -> None:
    """Without SDK ownership the engine wrapper opens/closes the range itself."""
    core = installed.fakes.EngineCore("arg")
    assert core.iteration == 0
    assert core.init_args == (("arg",), {})

    assert core.step() == ("step",)
    assert core.step_with_batch_queue() == ("step_with_batch_queue",)

    assert recording_nvtx.pushes == [
        "[prof_range]: iter 0",
        "[prof_range]: iter 1",
    ]
    assert len(recording_nvtx.pops) == 2
    assert recording_nvtx.iters == [0, 1]
    assert core.iteration == 2


def test_engine_core_steps_defer_to_model_prof(installed, recording_nvtx) -> None:
    """`model_prof.prof_iter` opens the range; a second push would nest it."""
    recording_nvtx.owned = True

    core = installed.fakes.EngineCore()
    assert core.step() == ("step",)
    assert core.step_with_batch_queue() == ("step_with_batch_queue",)

    assert recording_nvtx.pushes == []
    assert recording_nvtx.pops == []
    assert recording_nvtx.iters == [0, 1]
    assert core.iteration == 2


@pytest.mark.parametrize("step_name", ["step", "step_with_batch_queue"])
def test_engine_core_idle_steps_consume_no_iteration(
    installed, recording_nvtx, step_name
) -> None:
    """A step that dispatches nothing must not advance the iteration counter."""
    core = installed.fakes.EngineCore()
    core.scheduler = SimpleNamespace(has_requests=lambda: False)

    assert getattr(core, step_name)() == (step_name,)

    assert recording_nvtx.iters == []
    assert recording_nvtx.pushes == []
    assert core.iteration == 0

    core.scheduler = SimpleNamespace(has_requests=lambda: True)
    getattr(core, step_name)()
    assert recording_nvtx.iters == [0]
    assert core.iteration == 1


@pytest.mark.parametrize("owned", [False, True], ids=["sdk-declines", "sdk-owns"])
@pytest.mark.parametrize("step_name", ["step", "step_with_batch_queue"])
def test_engine_core_loads_profiling_during_init(
    profiling_pkg, nvtx_stub, fake_vllm, cleared_gate, monkeypatch, step_name, owned
) -> None:
    monkeypatch.setenv("VLLM_SAIL_NVTX_PROFILE", "1")

    def upstream_init(self):
        profiling_pkg.install()
        self.step_fn = getattr(self, step_name)

    monkeypatch.setattr(fake_vllm.EngineCore, "__init__", upstream_init)

    class EngineCoreProc(fake_vllm.EngineCore):
        def __init__(self):
            super().__init__()

    core = EngineCoreProc()
    assert profiling_pkg._installed
    assert not hasattr(core, "iteration")

    prof_nvtx = importlib.import_module("vllm_sail.profiling.nvtx")
    events = []
    monkeypatch.setattr(
        prof_nvtx, "_model_prof_iter", lambda i: events.append(("iter", i)) or owned
    )
    monkeypatch.setattr(
        prof_nvtx, "_torch_range_push", lambda label: events.append(("push", label))
    )
    monkeypatch.setattr(prof_nvtx, "_torch_range_pop", lambda: events.append(("pop",)))

    for iteration in range(2):
        assert core.step_fn() == (step_name,)
        assert core.iteration == iteration + 1

    expected: list[tuple] = []
    for iteration in range(2):
        expected.append(("iter", iteration))
        if not owned:
            expected += [("push", f"[prof_range]: iter {iteration}"), ("pop",)]
    assert events == expected


def test_scheduler_schedule_is_annotated_and_update_marks(installed) -> None:
    scheduler = installed.fakes.Scheduler()

    assert scheduler.schedule(True) == ("schedule", True)
    assert "schedule" in installed.nvtx_calls["annotated"]

    output = _sched_output(
        scheduled_new_reqs=[SimpleNamespace(req_id="r1")],
        finished_req_ids={"r9"},
    )
    scheduler._update_after_schedule(output)
    assert scheduler.last_update is output
    marks = installed.nvtx_calls["marks"]
    assert "new_reqs: ['r1']" in marks
    assert "finish_req: {'r9'}" in marks
    assert any(m.startswith("total_tokens=5") for m in marks)


# ---------------------------------------------------------------------------
# Shared helpers without the optional dependencies
# ---------------------------------------------------------------------------


@pytest.fixture(params=[False, True], ids=["bare", "installed-profiler"])
def bare_nvtx_helpers(
    fake_vllm: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    request: pytest.FixtureRequest,
):
    """Import profiling/nvtx.py fresh with nvtx/model_prof genuinely absent."""
    if request.param:
        # Deleting sys.modules is insufficient: installed packages can be
        # imported again from disk. Reproduce an identity NVTX decorator.
        (tmp_path / "nvtx.py").write_text(
            "def annotate(name):\n    return lambda func: func\n"
            "def mark(message):\n    pass\n"
        )
        (tmp_path / "model_prof.py").write_text("def prof_iter(i):\n    pass\n")
        monkeypatch.syspath_prepend(str(tmp_path))
    # None blocks import even if the SDK/PyPI package is installed on disk.
    monkeypatch.setitem(sys.modules, "nvtx", None)
    monkeypatch.setitem(sys.modules, "model_prof", None)
    monkeypatch.delitem(sys.modules, "vllm_sail.profiling.nvtx", raising=False)
    module = importlib.import_module("vllm_sail.profiling.nvtx")
    yield module
    monkeypatch.delitem(sys.modules, "vllm_sail.profiling.nvtx", raising=False)


def test_helpers_import_cleanly_without_optional_deps(bare_nvtx_helpers) -> None:
    assert bare_nvtx_helpers.has_nvtx is False
    assert bare_nvtx_helpers.has_model_prof is False

    bare_nvtx_helpers.mark("ignored")
    bare_nvtx_helpers.range_push("ignored")
    bare_nvtx_helpers.range_pop()
    # No SDK ownership to report, so callers must fall back to their own range.
    assert bare_nvtx_helpers.prof_iter(3) is False
    bare_nvtx_helpers.sche_mark(_sched_output())


def test_annotate_fallback_is_identity(bare_nvtx_helpers) -> None:
    @bare_nvtx_helpers.annotate("whatever")
    def documented(x: int) -> int:
        """Keeps its name."""
        return x + 1

    assert documented(1) == 2
    assert documented.__name__ == "documented"
    assert documented.__wrapped__ is not None


def test_range_push_swallows_cpu_torch_errors(bare_nvtx_helpers) -> None:
    # torch is installed in the test venv but has no CUDA NVTX support, so the
    # real torch.cuda.nvtx.range_push raises; a profiler must not propagate it.
    if not bare_nvtx_helpers.has_torch_nvtx:
        pytest.skip("torch.cuda.nvtx not importable here")
    bare_nvtx_helpers.range_push("probe")
    bare_nvtx_helpers.range_pop()


def test_sche_mark_format_with_nvtx(
    fake_vllm: SimpleNamespace,
    nvtx_stub: dict[str, list],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "vllm_sail.profiling.nvtx", raising=False)
    prof_nvtx = importlib.import_module("vllm_sail.profiling.nvtx")
    output = _sched_output(
        scheduled_new_reqs=[SimpleNamespace(req_id="a"), SimpleNamespace(req_id="b")],
        finished_req_ids={"z"},
    )
    prof_nvtx.sche_mark(output)
    assert nvtx_stub["marks"] == [
        "new_reqs: ['a', 'b']",
        "total_tokens=5,req_id:num_tokens={'r1': 5}",
        "finish_req: {'z'}",
    ]
