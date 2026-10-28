# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-level E2E performance regression suite (CI plan PART 6, category 5).

Where ``test_kernel_perf.py`` anchors the per-operator ceilings, this suite
anchors the *end-to-end* serving metrics a release actually regresses against:
output-token throughput, request throughput, wall-clock latency and (best
effort) TTFT / TPOT.  It drives vLLM's offline :class:`~vllm.LLM` exactly like
the numeric model suite (``tests/e2e/models``) -- no server, no OpenAI client --
so the two lanes stay comparable, but it measures *speed* instead of numerics.

Fixed-config contract (mirrors ``_offline_runner``):
  * models are identified by HuggingFace model IDs (e.g.
    ``Qwen/Qwen3.8-Flash-Next``); weights are resolved locally through
    ``HF_HOME`` + ``HF_HUB_OFFLINE=1`` + ``TRANSFORMERS_OFFLINE=1`` (same
    offline cache mechanism as the PR 88 model-test lane);
  * tensor parallelism follows the detected chip profile (ZW-810E -> 8,
    ZW-890P -> 8) via the shared ``offline_llm`` factory;
  * every other knob -- ``max_model_len``, dtype, greedy sampling, the prompt
    set and the per-workload ``max_tokens`` -- is a module constant below, so a
    run is reproducible and a delta can only come from the wheel/hardware.

Baselines live in ``tests/e2e/perf/baselines/<chip>/model.json`` and follow the
same store / tolerance / promotion contract as the kernel suite (see
``_baseline_store.py``): the first run on a chip writes ``status="pending"`` and
never fails; a regression beyond ``FAIL_THRESHOLD`` (25%) fails the lane; drift
beyond ``WARN_THRESHOLD`` (10%) only warns.

Measurement protocol (two-phase warmup):
  1. A tiny 8-token generation settles lazy init (KV cache allocator, weight
     loading, basic compilation paths).
  2. A full-workload untimed generation with the actual batch size and
     ``max_tokens`` triggers any remaining JIT kernel compilations so the
     subsequent timed runs reflect steady-state performance.
After warmup, ``MODEL_REPEATS`` timed generations follow; the authoritative
value is the median wall clock and the tail is the nearest-rank p90.

TTFT / TPOT are derived from vLLM's per-request metrics object when the
running vLLM exposes it and are silently omitted otherwise, so a metrics-API
change never fails the throughput lane.

Adding a model is a one-line change to :data:`PERF_MODELS` (the LLM is built
once per model key and cached by ``perf_llm_factory``).
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import pytest

# --- Guard chain: skip the whole module without a real PPU device. ------------
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("requires a real PPU/CUDA device", allow_module_level=True)
pytest.importorskip("vllm")

import vllm_sail  # noqa: E402

vllm_sail.register_out_of_tree()

from vllm import SamplingParams  # noqa: E402
from vllm.platforms import current_platform  # noqa: E402

if not current_platform.is_ppu():
    pytest.skip("requires a PPU platform (is_ppu() is False)", allow_module_level=True)

from tests.e2e.models import _offline_runner  # noqa: E402
from tests.e2e.models._offline_runner import ModelConfig  # noqa: E402
from tests.e2e.perf._perf_harness import PerfHarness, summarize  # noqa: E402

pytestmark = [pytest.mark.ppu, pytest.mark.perf]

SUITE = "model"

#: Timed generations per workload; the median is authoritative (plan PART 6.1).
MODEL_REPEATS = 3
#: Warmup generation length (tokens) before the timed repeats.
WARMUP_TOKENS = 8


@dataclass(frozen=True)
class PerfModel:
    """One model whose serving speed is regression-tracked.

    ``model_id`` is a HuggingFace repository ID (e.g. ``Qwen/Qwen3.8-Flash-Next``).
    The offline runtime environment (``HF_HOME`` + ``HF_HUB_OFFLINE=1`` +
    ``TRANSFORMERS_OFFLINE=1``) resolves it to a local snapshot, matching the
    PR 88 model-loading contract.

    ``env`` carries chip-independent runtime knobs applied before ``LLM`` is
    built (e.g. ``VLLM_SAIL_USE_PLA``); everything else uses the shared offline
    defaults so the perf build matches the numeric build.
    """

    key: str
    model_id: str
    max_model_len: int = _offline_runner.DEFAULT_MAX_MODEL_LEN
    trust_remote_code: bool = True
    env: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PerfWorkload:
    """A fixed generation workload (prompt set + decode length)."""

    key: str
    prompts: tuple
    max_tokens: int


#: The models tracked by the nightly perf lane. Qwen3.8-Next is the primary
#: upstream-architecture-plus-patches model already guarded numerically; adding
#: a MoE model here reuses the whole flow unchanged.
PERF_MODELS = [
    pytest.param(
        PerfModel(
            key="qwen3_next",
            model_id="Qwen/Qwen3.8-Flash-Next",
        ),
        id="qwen3_next",
    ),
]

#: A single long-context prompt exercises prefill + a long decode tail, the
#: regime where TTFT and TPOT are most visible.
LONG_PROMPT = (
    "You are auditing a distributed inference engine. The scheduler batches "
    "requests, the KV cache is paged into fixed-size blocks, and tensor "
    "parallelism splits each attention head across devices. Walk through, step "
    "by step, what happens from the moment a 2048-token prompt arrives until "
    "the last of 256 generated tokens is emitted: queueing, prefill, block "
    "allocation, the decode loop, and detokenisation. Then list three sources "
    "of tail latency and one mitigation for each. Be concrete and technical."
)

#: Workloads run against every model (throughput-oriented + latency-oriented).
PERF_WORKLOADS = [
    pytest.param(
        PerfWorkload(
            key="batch8_tok128",
            prompts=_offline_runner.PROMPTS,
            max_tokens=128,
        ),
        id="batch8_tok128",
    ),
    pytest.param(
        PerfWorkload(
            key="long1_tok256",
            prompts=(LONG_PROMPT,),
            max_tokens=256,
        ),
        id="long1_tok256",
    ),
]


@pytest.fixture
def perf_harness(perf_baseline: dict) -> PerfHarness:
    """Model-suite harness bound to ``baselines/<chip>/model.json``."""
    return PerfHarness(perf_baseline["store"](SUITE))


@pytest.fixture(scope="module")
def perf_llm_factory(offline_llm):
    """Build (once per model key) and cache the offline ``LLM`` under test."""
    cache: dict = {}

    def build(model: PerfModel) -> Any:
        if model.key not in cache:
            config = ModelConfig(
                key=model.key,
                checkpoint=model.model_id,
                max_model_len=model.max_model_len,
                trust_remote_code=model.trust_remote_code,
                env=model.env,
            )
            cache[model.key] = offline_llm(model.model_id, config)
        return cache[model.key]

    return build


def _sampling(workload: PerfWorkload) -> SamplingParams:
    """Fixed greedy sampling for one workload (deterministic decode)."""
    return SamplingParams(
        temperature=0.0,
        max_tokens=workload.max_tokens,
        seed=_offline_runner.DEFAULT_SEED,
    )


def _request_timing(
    request_output: Any,
) -> tuple[Optional[float], Optional[float]]:  # noqa: UP045
    """Best-effort ``(ttft_ms, tpot_ms)`` from vLLM's per-request metrics.

    Returns ``(None, None)`` when the running vLLM does not attach a metrics
    object (or lacks the timing fields), so the throughput lane never depends
    on an internal stats API.
    """
    metrics = getattr(request_output, "metrics", None)
    if metrics is None:
        return None, None
    arrival = getattr(metrics, "arrival_time", None)
    first = getattr(metrics, "first_token_time", None)
    finished = getattr(metrics, "finished_time", None)
    ttft_ms = (first - arrival) * 1000.0 if (arrival and first) else None
    tokens = len(request_output.outputs[0].token_ids)
    tpot_ms = None
    if first and finished and tokens > 1:
        tpot_ms = (finished - first) * 1000.0 / (tokens - 1)
    return ttft_ms, tpot_ms


def _warmup(llm: Any) -> None:
    """One tiny generation to settle lazy init before the timed repeats."""
    llm.generate(
        [_offline_runner.PROMPTS[0]],
        SamplingParams(
            temperature=0.0, max_tokens=WARMUP_TOKENS, seed=_offline_runner.DEFAULT_SEED
        ),
    )


@pytest.mark.parametrize("workload", PERF_WORKLOADS)
@pytest.mark.parametrize("model", PERF_MODELS)
def test_model_e2e_perf(perf_llm_factory, perf_harness, model, workload):
    """Throughput / latency / TTFT / TPOT for one (model, workload) pair."""
    llm = perf_llm_factory(model)
    params = _sampling(workload)
    prompts = list(workload.prompts)

    _warmup(llm)

    # Full-workload warmup: triggers JIT compilation for the actual
    # batch size / max_tokens shape so timed runs reflect steady state.
    llm.generate(prompts, params)

    wall_times: list = []
    output_tokens: list = []
    ttfts: list = []
    tpots: list = []
    for _ in range(MODEL_REPEATS):
        start = time.perf_counter()
        outputs = llm.generate(prompts, params)
        wall_ms = (time.perf_counter() - start) * 1000.0
        wall_times.append(wall_ms)
        generated = sum(len(o.outputs[0].token_ids) for o in outputs)
        output_tokens.append(generated)
        for request_output in outputs:
            ttft_ms, tpot_ms = _request_timing(request_output)
            if ttft_ms is not None:
                ttfts.append(ttft_ms)
            if tpot_ms is not None:
                tpots.append(tpot_ms)

    median_ms, p90_ms, _cv = summarize(wall_times)
    median_wall_s = median_ms / 1000.0
    # Output tokens are identical across greedy repeats; use the median count.
    median_tokens = statistics.median(output_tokens)
    prefix = f"model.{model.key}.{workload.key}"
    config = {
        "model": model.key,
        "workload": workload.key,
        "num_prompts": len(prompts),
        "max_tokens": workload.max_tokens,
        "repeats": MODEL_REPEATS,
        "output_tokens": median_tokens,
    }

    perf_harness.record(
        f"{prefix}.e2e_ms",
        median_ms,
        "lower_better",
        unit="ms",
        p90=p90_ms,
        samples=MODEL_REPEATS,
        config=config,
    )
    perf_harness.record(
        f"{prefix}.output_tok_s",
        median_tokens / median_wall_s,
        "higher_better",
        unit="tok/s",
        samples=MODEL_REPEATS,
        config=config,
    )
    perf_harness.record(
        f"{prefix}.request_per_s",
        len(prompts) / median_wall_s,
        "higher_better",
        unit="req/s",
        samples=MODEL_REPEATS,
        config=config,
    )
    # TTFT / TPOT are recorded only when vLLM exposed per-request timing.
    if ttfts:
        perf_harness.record(
            f"{prefix}.ttft_ms",
            statistics.median(ttfts),
            "lower_better",
            unit="ms",
            samples=len(ttfts),
            config=config,
        )
    if tpots:
        perf_harness.record(
            f"{prefix}.tpot_ms",
            statistics.median(tpots),
            "lower_better",
            unit="ms",
            samples=len(tpots),
            config=config,
        )
