# SPDX-License-Identifier: Apache-2.0
"""NVTX instrumentation for ``vllm/v1/engine/core.py``.

Port of the fork's engine-core hunks (+59 lines) as delegating patches, per
``docs/developer_guide/architecture.md``:

* ``EngineCore.__init__`` gains the ``self.iteration`` counter.
* ``step`` / ``step_with_batch_queue`` are wrapped in the
  ``[prof_range]: iter N`` range plus the ``prof_iter`` hook.

Deviations from the fork, all deliberate and documented in the patch
metadata: the fork opened each range *after* ``scheduler.schedule()`` and
skipped it when no requests were pending; the wrappers open it around the
whole method instead. The range therefore spans scheduling as well (nested
inside the ``schedule`` NVTX range anyway) and empty steps get a range too.
The fork's ``sche_mark`` call is covered by the scheduler patch module,
which marks every scheduler output in this process exactly once.

The engine core and each model runner emit ``[prof_range]: iter N`` for the
same ``N``. Two rules keep them in step:

* one iteration per *batch the engine actually dispatched*, so no process
  consumes a number on a step or call that never executed a batch: the engine
  skips its ``not scheduler.has_requests()`` steps (and the batch-queue pops that
  dispatch nothing), and the model runners skip warmup dummy runs;
* ``model_prof.prof_iter`` already opens that exact range and pops it on the
  next call, so a wrapper only falls back to its own push/pop when the SDK
  declines (no ``model_prof``, or no ``MODEL_PROFILE``). Pushing anyway nests
  two identically named ranges per iteration, which doubles every iteration in
  a range summary and makes the outer range end at the wrong place.

Installed only by ``vllm_sail.profiling.install()`` when
``VLLM_SAIL_NVTX_PROFILE`` is set; never imported otherwise.
"""

from __future__ import annotations

from vllm.v1.engine.core import EngineCore

from vllm_sail.patch.utils import patch
from vllm_sail.profiling import nvtx

_AFFECTED = ">=0.32.0,<0.33.0"

_upstream_init = EngineCore.__init__
_upstream_step = EngineCore.step
_upstream_step_with_batch_queue = EngineCore.step_with_batch_queue


@patch(
    "vllm.v1.engine.core",
    "EngineCore.__init__",
    reason=(
        "[profiling, opt-in] Adds the self.iteration counter that labels the "
        "engine-level NVTX iteration ranges (fork engine/core.py hunk "
        "@@ -134). Installed only when VLLM_SAIL_NVTX_PROFILE is set."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "the opt-in NVTX subpackage (docs/developer_guide/architecture.md) "
        "is retired, or upstream adds native NVTX/tracing hooks to the v1 "
        "engine core."
    ),
)
def engine_core_init(self, *args, **kwargs):
    _upstream_init(self, *args, **kwargs)
    self.iteration = 0


@patch(
    "vllm.v1.engine.core",
    "EngineCore.step",
    reason=(
        "[profiling, opt-in] Wraps one dispatched engine step in the fork's "
        "`[prof_range]: iter N` range and prof_iter hook. The fork opened the "
        "range after scheduler.schedule(); the wrapper spans the whole method, "
        "but only for steps that dispatch a batch, so this counter keeps "
        "tracking the model runners' (see module docstring)."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "the opt-in NVTX subpackage (docs/developer_guide/architecture.md) "
        "is retired, or upstream adds native NVTX/tracing hooks to the v1 "
        "engine core."
    ),
)
def step(self):
    # Upstream returns ``({}, False)`` here without touching the model, so the
    # workers never see this step. Counting it would advance this process's
    # iteration (and the SDK's profile window) ahead of every worker's.
    scheduler = getattr(self, "scheduler", None)
    if scheduler is not None and not scheduler.has_requests():
        return _upstream_step(self)

    iteration = getattr(self, "iteration", 0)
    owns_range = nvtx.prof_iter(iteration)
    if not owns_range:
        nvtx.range_push(f"[prof_range]: iter {iteration}")
    self.iteration = iteration + 1
    try:
        return _upstream_step(self)
    finally:
        if not owns_range:
            nvtx.range_pop()


@patch(
    "vllm.v1.engine.core",
    "EngineCore.step_with_batch_queue",
    reason=(
        "[profiling, opt-in] Same `[prof_range]: iter N` wrapping as "
        "EngineCore.step but for the async batch-queue loop, and only for the "
        "calls that dispatch a new batch (draining the queue executes nothing). "
        "The fork popped right after execute_model; the wrapper spans the whole "
        "method (see module docstring)."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "the opt-in NVTX subpackage (docs/developer_guide/architecture.md) "
        "is retired, or upstream adds native NVTX/tracing hooks to the v1 "
        "engine core."
    ),
)
def step_with_batch_queue(self):
    # Same guard as ``step``: only the branch that schedules and dispatches a new
    # batch is an iteration. When the queue is merely drained (or still empty)
    # the workers execute nothing new, so the counter must not move.
    scheduler = getattr(self, "scheduler", None)
    if scheduler is not None and not scheduler.has_requests():
        return _upstream_step_with_batch_queue(self)

    iteration = getattr(self, "iteration", 0)
    owns_range = nvtx.prof_iter(iteration)
    if not owns_range:
        nvtx.range_push(f"[prof_range]: iter {iteration}")
    self.iteration = iteration + 1
    try:
        return _upstream_step_with_batch_queue(self)
    finally:
        if not owns_range:
            nvtx.range_pop()
