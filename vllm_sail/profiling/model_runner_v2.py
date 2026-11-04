# SPDX-License-Identifier: Apache-2.0
"""NVTX instrumentation for the V2 model runner.

vLLM 0.31 selects ``vllm.v1.worker.gpu.model_runner.GPUModelRunner`` (Model
Runner V2) by default and only falls back to Model Runner V1 when the config
needs a feature V2 does not implement. The V1 instrumentation in
:mod:`vllm_sail.profiling.model_runner` patches
``vllm.v1.worker.gpu_model_runner.GPUModelRunner``, a class V2 never
instantiates, so with V2 active the worker process emitted no sampler /
compute-logits / prepare-inputs ranges and no batch range at all.

This module reattaches the same ranges to the V2 class, at whole-method
boundaries and without copying any upstream body:

* ``sche_mark`` + ``[prof_range]: iter N`` + ``total bs=..., P bs=...`` ->
  wrapper around ``execute_model``, as in V1. ``model_prof.prof_iter`` opens
  ``[prof_range]`` itself, so the wrapper only opens its own when the SDK
  declines. V2 inlines the forward into ``execute_model`` (there is no
  ``_model_forward`` to wrap), so the batch range spans the forward exactly.
* ``[FW_NATIVE] op:sampler,...`` -> wrapper around ``sample``, V2's ``_sample``
  equivalent. V2 computes the logits inside ``sample`` and never exposes them,
  so the label reports the sampling input's shape; the logits shape is still
  recorded by the ``[FW_GEMM] op:compute_logits`` range.
* ``@annotate("prepare_inputs")`` on ``prepare_inputs`` -- exact equivalent of
  V1's ``_prepare_inputs`` annotation.
* ``compute_logits`` ranges are installed lazily from the ``execute_model``
  wrapper, exactly as the V1 module does.

V2 also splits the step in two: ``execute_model`` runs the forward and
``sample_tokens`` runs the sampling. The batch range covers the forward; the
sampling work stays visible through the sampler and compute-logits ranges
above, and the per-step ``[prof_range]`` emitted by the engine core still spans
both.

Dummy runs (memory profiling and CUDA-graph capture) call ``execute_model``
with ``dummy_run=True``. V1's ``_dummy_run`` bypasses every instrumented method,
so they are skipped here too: warmup must neither emit ranges nor consume the
iteration number. This is what keeps ``iter N`` identical to the engine's:
``N`` counts batches the engine dispatched, and a warmup run was never one of
them, so counting it would offset this process's labels (and the SDK's
``cudaProfilerStart``/``Stop`` window) by the number of warmup runs.
"""

from __future__ import annotations

from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_sail.patch.utils import patch
from vllm_sail.profiling import nvtx

_AFFECTED = ">=0.32.0,<0.33.0"

_UPSTREAM_MODULE = "vllm.v1.worker.gpu.model_runner"

_REMOVE_WHEN = (
    "the opt-in NVTX subpackage (docs/developer_guide/architecture.md) is "
    "retired, or upstream adds native NVTX/tracing hooks to the V2 model "
    "runner."
)

_upstream_prepare_inputs = GPUModelRunner.prepare_inputs
_upstream_sample = GPUModelRunner.sample
_upstream_execute_model = GPUModelRunner.execute_model


@patch(
    _UPSTREAM_MODULE,
    "GPUModelRunner.prepare_inputs",
    reason=(
        "[profiling, opt-in] NVTX `prepare_inputs` range for Model Runner V2, "
        "the counterpart of the V1 `_prepare_inputs` annotation. The wrapper "
        "delegates to upstream unchanged."
    ),
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
@nvtx.annotate("prepare_inputs")
def prepare_inputs(self, *args, **kwargs):
    return _upstream_prepare_inputs(self, *args, **kwargs)


@patch(
    _UPSTREAM_MODULE,
    "GPUModelRunner.sample",
    reason=(
        "[profiling, opt-in] Wraps V2 sampling in the fork's "
        "`[FW_NATIVE] op:sampler,...` range, the counterpart of the V1 "
        "`_sample` wrapper. V2 does not expose the logits tensor, so the label "
        "carries the hidden-state shape instead."
    ),
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def sample(self, hidden_states, *args, **kwargs):
    shape = getattr(hidden_states, "shape", None)
    nvtx.range_push(f"[FW_NATIVE] op:sampler,hidden_states:{shape}")
    try:
        return _upstream_sample(self, hidden_states, *args, **kwargs)
    finally:
        nvtx.range_pop()


@patch(
    _UPSTREAM_MODULE,
    "GPUModelRunner.execute_model",
    reason=(
        "[profiling, opt-in] Wraps one executed batch in the fork's "
        "`[prof_range]: iter N` range, the `total bs=..., P bs=...` range, the "
        "prof_iter hook and sche_mark. Without it, V2 runs (the 0.31 default) "
        "emit no worker NVTX at all. `prof_iter` already opens `[prof_range]`, "
        "so the wrapper only opens its own when the SDK declines. Dummy runs "
        "consume no iteration number and are forwarded untouched."
    ),
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def execute_model(
    self,
    scheduler_output,
    intermediate_tensors=None,
    dummy_run=False,
    *args,
    **kwargs,
):
    if dummy_run:
        return _upstream_execute_model(
            self, scheduler_output, intermediate_tensors, dummy_run, *args, **kwargs
        )

    nvtx.instrument_compute_logits(getattr(self, "model", None))
    nvtx.sche_mark(scheduler_output)
    # One iteration per batch the engine dispatched, so this counter stays in
    # step with ``EngineCore.step`` (which skips steps that dispatch nothing).
    iteration = getattr(self, "iteration", 0)
    owns_range = nvtx.prof_iter(iteration)
    if not owns_range:
        nvtx.range_push(f"[prof_range]: iter {iteration}")
    self.iteration = iteration + 1
    nvtx.range_push(nvtx.batch_label(scheduler_output))
    try:
        return _upstream_execute_model(
            self, scheduler_output, intermediate_tensors, dummy_run, *args, **kwargs
        )
    finally:
        nvtx.range_pop()
        if not owns_range:
            nvtx.range_pop()
