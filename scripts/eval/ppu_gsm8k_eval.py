#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PPU GSM8K accuracy E2E test via evalscope (server-client mode).

Single-file, standalone runnable::

    python scripts/eval/ppu_gsm8k_eval.py \\
        --model-key deepseek_v4_mxfp4 [--update-golden] [--limit 20]

Steps
-----
1. Parse arguments
2. Resolve model_id + chip → TP + guards
3. Start ``vllm serve`` (model resolved via HF_HOME + HF_HUB_OFFLINE)
4. Wait for readiness (poll ``/v1/models`` + scan fatal regex)
5. Run evalscope ``TaskConfig(eval_type='openai_api', datasets=['gsm8k'])``
6. Extract accuracy score (version-tolerant)
7. Compare golden / write pending / stop server
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

# --- Project root for cross-module imports ---------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tests.e2e.models._accuracy_gate import (  # noqa: E402
    DEFAULT_THRESHOLD,
    DEFAULT_TOLERANCE,
    GSM8K_FULL_SIZE,
    build_accuracy_golden,
    check_accuracy_golden,
    check_generation_config,
    check_golden_identity,
    check_score_threshold,
    describe_serving,
    golden_filename,
    pending_golden_filename,
    read_golden,
    write_golden,
)
from tests.e2e.models._offline_runner import (  # noqa: E402
    CAPABILITY_TO_CHIP,
    DEFAULT_CHIP,
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_SEED,
    REPORTS_ROOT,
    _checkpoint_tail,
    tensor_parallel_for_chip,
)

# --- Server readiness / fatal detection ------------------------------------
# Cold start on PPU is dominated by CUDA graph capture. Left at the checkpoint
# default of 1,048,576 tokens the engine captures 51 piecewise graphs plus 2
# full ones, measured at 997s, on top of ~317s of weight loading and ~200s of
# JIT kernel warmup. Every serve command therefore pins the window explicitly:
# GSM8K prompts and answers are far below it, and the shorter context collapses
# the capture schedule. The old 1800s budget expired mid-warmup.
DEFAULT_MAX_MODEL_LEN = 16384
SERVER_READY_TIMEOUT = 3600
SERVER_POST_READY_SLEEP = 30
FATAL_RE = re.compile(
    r"Traceback \(most recent"
    r"|EngineCore failed"
    r"|Engine core initialization failed"
    r"|core dumped"
    r"|Failed to infer device type"
    r"|Error: model architectures"
)

# --- GSM8K dataset path (from environment) --------------------------------
GSM8K_DATASET_PATH = os.environ.get("GSM8K_DATASET_PATH", "")
if not GSM8K_DATASET_PATH:
    raise OSError(
        "GSM8K_DATASET_PATH environment variable is required but not set. "
        "Set it to the local path of the GSM8K dataset "
        "(e.g. /path/to/EvalScope/data/gsm8k/v2/gsm8k)."
    )

# --- Golden / reports directories ------------------------------------------
GOLDEN_DIR = _PROJECT_ROOT / "tests" / "e2e" / "models" / "_golden"

# --- Generation parameters -------------------------------------------------
# Sampling parameters belong to the weights, not to a case: two cases over one
# checkpoint are compared against that checkpoint's single golden, so they must
# be measured under one contract or their scores are not comparable.  A case
# that needs a different layout changes its serve argv, never these values.
#
# ``max_tokens`` stays below ``DEFAULT_MAX_MODEL_LEN``: a model that starts
# repeating never emits EOS, and one sample would otherwise burn the whole
# served window.
DEFAULT_MAX_TOKENS = 10240
# Fallback only: every weight in ``MODEL_CONFIGS`` carries its own contract, and
# a weight added without one falls back to greedy -- the reproducible choice.
DEFAULT_GENERATION_CONFIG: dict[str, Any] = {
    "temperature": 0.0,
    "top_p": 0.95,
    "max_tokens": DEFAULT_MAX_TOKENS,
}


def _generation(
    *,
    temperature: float,
    top_p: float,
    **extra: Any,
) -> dict[str, Any]:
    """Build the sampling parameters one weight is measured under.

    ``temperature`` / ``top_p`` follow the team's reference generation config
    for that weight family; ``max_tokens`` is ours and stays uniform.  Extra
    keys (`top_k`, `min_p`, the penalties) are passed through as-is: evalscope
    moves whatever the OpenAI SDK does not accept into ``extra_body``, and
    vLLM reads it from there.
    """
    return {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": DEFAULT_MAX_TOKENS,
        **extra,
    }


# --- Model configurations --------------------------------------------------
MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "deepseek_v4_int8": {
        "model_id": "T-HEAD/DeepSeek-V4-Flash-0731-W8A8-INT8",
        "served_name": "deepseek-v4-int8",
        "extra_flags": ["--trust-remote-code"],
        "chip": "810e",
        # o_groups=8 in the model config requires TP <= 8; the
        # upstream FlashMLA _o_proj computes
        #   heads_per_group = n_local_heads // n_local_groups
        # which divides by zero when TP > o_groups because
        #   n_local_groups = o_groups // tp_size = 8 // 16 = 0.
        "ppu_count": 8,
        # reference_generation_config.json: deepseek_v4_flash
        "generation_config": _generation(temperature=1.0, top_p=1.0),
    },
    "deepseek_v4_mxfp4": {
        "model_id": (
            "T-HEAD/DeepSeek-V4-Flash-0731-MoE-"
            "Quant-W-MXFP4-A-MXFP4-Dense-"
            "Quant-W-FP8-PerChannel-A-FP8-PerToken"
        ),
        "served_name": "deepseek-v4-mxfp4",
        "extra_flags": ["--trust-remote-code"],
        "chip": "890p",
        # Same weights family as the int8 case: one contract, one golden.
        "generation_config": _generation(temperature=1.0, top_p=1.0),
    },
    "minimax-m3-fp8": {
        "model_id": ("T-HEAD/MiniMax-M3-Quant-W-FP8-PerChannel-A-FP8-PerToken"),
        "served_name": "MiniMax-M3-Quant-W-FP8-PerChannel-A-FP8-PerToken",
        "extra_flags": ["--trust-remote-code"],
        # The catalog runs this case on ZW-M890P; without the key the chip
        # falls back to 810e and the golden would be filed under the wrong
        # board.
        "chip": "890p",
        # reference_generation_config.json: minimax_m3
        "generation_config": _generation(temperature=1.0, top_p=0.95),
    },
    "glm_5_3_mxfp4": {
        "model_id": (
            "T-HEAD/GLM-5.3-MoE-Quant-W-MXFP4-A-MXFP4-Dense-"
            "Quant-W-FP8-PerChannel-A-FP8-PerToken"
        ),
        "served_name": "glm-5.3-mxfp4",
        "extra_flags": ["--trust-remote-code"],
        "chip": "890p",
        # Half of a 890P node: the prefill recipe only needs four cards.
        "ppu_count": 4,
        # reference_generation_config.json: glm_5_3
        "generation_config": _generation(temperature=1.0, top_p=1.0),
    },
    "glm_5_3_flash_mxfp4": {
        "model_id": (
            "T-HEAD/GLM-5.3-Flash-MoE-Quant-W-MXFP4-A-MXFP4-Dense-"
            "Quant-W-FP8-PerChannel-A-FP8-PerToken"
        ),
        "served_name": "glm-5.3-flash-mxfp4",
        "extra_flags": ["--trust-remote-code"],
        "chip": "890p",
        "ppu_count": 2,
        # No row of its own in the reference config: same family as glm_5_3.
        "generation_config": _generation(temperature=1.0, top_p=1.0),
    },
    "kimi_k2_6_mxfp4": {
        "model_id": "T-HEAD/Kimi-K2.6-MXFP4-FP8",
        "served_name": "kimi-k2.6-mxfp4",
        "extra_flags": ["--trust-remote-code"],
        "chip": "890p",
        "ppu_count": 8,
        # reference_generation_config.json: kimi_k2_6
        "generation_config": _generation(temperature=1.0, top_p=0.95),
    },
    "qwen3.8-flash-next": {
        "model_id": (
            "T-HEAD/Qwen3.8-Flash-Next-MoE-Quant-W-MXFP4-A-MXFP4-Dense-"
            "Quant-W-FP8-PerChannel-A-FP8-PerToken"
        ),
        "served_name": "Qwen3.8-Flash-Next",
        "extra_flags": ["--trust-remote-code"],
        # The checkpoint README's recipe keeps every card in one process, so
        # the requested count and the tensor parallel size are the same.
        "chip": "890p",
        "ppu_count": 4,
        # No row of its own: reference_generation_config.json gives every
        # qwen3_8 weight the same sampling parameters.
        "generation_config": _generation(
            temperature=1.0,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
        ),
    },
}


# ===================================================================
# GSM8K dataset resolution
# ===================================================================


def resolve_gsm8k_dataset() -> str:
    """Return the GSM8K dataset path from ``GSM8K_DATASET_PATH``."""
    return GSM8K_DATASET_PATH


# ===================================================================
# Server lifecycle
# ===================================================================


def _expand_server_placeholders(
    token: str,
    *,
    model: str,
    port: int,
    served_name: str,
) -> str:
    """Substitute the case-level placeholders inside one argv token."""
    return (
        token.replace("{model}", model)
        .replace("{port}", str(port))
        .replace("{served_name}", served_name)
    )


def _parse_server_cmd(raw: str | None) -> list[str] | None:
    """Decode ``--server-cmd`` into an argv, or ``None`` for the built-in one."""
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except ValueError as error:
        raise SystemExit(f"--server-cmd is not valid JSON: {error}") from error
    if not isinstance(value, list) or not all(
        isinstance(token, str) for token in value
    ):
        raise SystemExit("--server-cmd must be a JSON array of strings")
    return value


def _parse_server_env(raw: str | None) -> dict[str, str] | None:
    """Decode ``--server-env`` into a string map, or ``None`` when unused."""
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except ValueError as error:
        raise SystemExit(f"--server-env is not valid JSON: {error}") from error
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise SystemExit("--server-env must be a JSON object of string values")
    return value


def build_serve_argv(
    model: str,
    *,
    served_model_name: str,
    port: int = 9985,
    tp: int = 8,
    extra_flags: list[str] | None = None,
    server_cmd: list[str] | None = None,
) -> list[str]:
    """Return the ``vllm serve`` argv this case will actually run.

    ``server_cmd`` replaces the built-in argument list entirely when present.
    It is a pure function so the golden can describe the layout the process
    was started with instead of guessing from ``MODEL_CONFIGS``.

    ``tp`` only shapes the built-in list: a recipe that ships its own argv
    picks its own parallelism there, and this value then serves as nothing but
    the card count the job requested.
    """
    if server_cmd:
        return [
            _expand_server_placeholders(
                token,
                model=model,
                port=port,
                served_name=served_model_name,
            )
            for token in server_cmd
        ]
    cmd = [
        "vllm",
        "serve",
        model,
        "--served-model-name",
        served_model_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tp),
        "--gpu-memory-utilization",
        str(DEFAULT_GPU_MEMORY_UTILIZATION),
        "--max-model-len",
        str(DEFAULT_MAX_MODEL_LEN),
        "--trust-remote-code",
        "--seed",
        str(DEFAULT_SEED),
    ]
    if extra_flags:
        cmd.extend(extra_flags)
    return cmd


def start_vllm_server(
    model: str,
    *,
    served_model_name: str,
    port: int = 9985,
    tp: int = 8,
    extra_flags: list[str] | None = None,
    server_cmd: list[str] | None = None,
    server_env: dict[str, str] | None = None,
    log_path: Path | None = None,
) -> tuple[subprocess.Popen[bytes], Path]:
    """Start ``vllm serve`` in a new session; return (proc, log).

    ``server_cmd`` is a per-case argv carrying ``{model}``, ``{port}`` and
    ``{served_name}`` placeholders.  When present it replaces the built-in
    argument list entirely, so a case can drop a default flag or pick a
    different parallel layout without the caller second-guessing it.  Cases
    that leave it unset keep the shared defaults below.

    ``server_env`` adds variables for this server process only.  It is merged
    over the inherited environment rather than replacing it, so ``HF_HOME``,
    the wheel paths and the rest of the CI contract survive.
    """
    if log_path is None:
        log_path = Path("/tmp/vllm_serve_gsm8k.log")
    cmd = build_serve_argv(
        model,
        served_model_name=served_model_name,
        port=port,
        tp=tp,
        extra_flags=extra_flags,
        server_cmd=server_cmd,
    )
    env = None
    if server_env:
        # A DP+EP launch needs its own engine-ready budget; the value given by
        # the docs is per case, so it belongs here and not in the wrapper.
        env = {**os.environ, **{k: str(v) for k, v in server_env.items()}}
    log_fh = open(log_path, "w")  # noqa: SIM115
    proc = subprocess.Popen(
        cmd,
        start_new_session=True,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        env=env,
    )
    return proc, log_path


def unique_served_model_name(base: str) -> str:
    """Return a run-unique served model name derived from ``base``.

    The suffix is drawn from :func:`secrets.token_hex` so a lingering vLLM
    process from a previous run cannot forge the identity even if it keeps
    answering on the same port with the same catalog name.
    """
    return f"{base}-run-{secrets.token_hex(4)}"


def _probe_models_endpoint(url: str, expected_served_name: str) -> bool:
    """Return True iff ``/v1/models`` lists ``expected_served_name`` as ``id``.

    HTTP 200 alone is not sufficient: a stale vLLM instance left running on
    the same port -- even one serving the same catalog base name -- will
    happily answer 200, so readiness is bound to the per-run nonce by
    inspecting the JSON ``data`` list for an exact ``id`` match.
    Malformed JSON, a missing ``data`` field, or any other mismatch returns
    False so the caller keeps waiting instead of mis-attributing readiness.
    """
    try:
        with urlopen(url, timeout=5) as resp:
            if resp.status != 200:
                return False
            raw = resp.read()
    except (URLError, OSError):
        return False
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError):
        return False
    if not isinstance(payload, dict):
        return False
    data = payload.get("data")
    if not isinstance(data, list):
        return False
    for item in data:
        if isinstance(item, dict) and item.get("id") == expected_served_name:
            return True
    return False


def _server_ready_timeout(server_env: dict[str, str] | None) -> int:
    """Readiness budget for one case.

    A ``--data-parallel-size 8 --enable-expert-parallel`` launch raises vLLM's
    own engine-ready budget (the recipes set ``VLLM_ENGINE_READY_TIMEOUT_S`` to
    6000); polling for less than that would abandon a start the engine still
    considers healthy.
    """
    if not server_env:
        return SERVER_READY_TIMEOUT
    try:
        engine_budget = int(server_env.get("VLLM_ENGINE_READY_TIMEOUT_S", ""))
    except (TypeError, ValueError):
        return SERVER_READY_TIMEOUT
    return max(SERVER_READY_TIMEOUT, engine_budget)


def wait_server_ready(
    proc: subprocess.Popen[bytes],
    log_path: Path,
    *,
    expected_served_name: str,
    port: int = 9985,
    timeout: int = SERVER_READY_TIMEOUT,
) -> None:
    """Poll ``/v1/models`` + scan the log for fatal patterns.

    Readiness is bound to the launched process *and* its unique served model
    name: the loop keeps waiting until ``/v1/models`` returns 200 **and** the
    JSON ``data[*].id`` list contains ``expected_served_name``.  A stale
    service on the same port that still returns 200 (or returns the catalog
    base name only) cannot pass this gate.
    """
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = _log_tail(log_path)
            raise RuntimeError(
                "vllm serve exited with code "
                f"{proc.returncode} before becoming ready."
                f"\nLog tail:\n{tail}"
            )
        try:
            log_text = log_path.read_text()
            if FATAL_RE.search(log_text):
                raise RuntimeError(
                    f"fatal error detected in vllm serve log:\n{log_text[-2000:]}"
                )
        except OSError:
            pass
        if _probe_models_endpoint(url, expected_served_name):
            return
        time.sleep(5)
    raise RuntimeError(
        f"server not ready within {timeout}s (expected served model "
        f"{expected_served_name!r}).\nLog tail:\n{_log_tail(log_path)}"
    )


def _ensure_launched_server_alive(
    proc: subprocess.Popen[bytes],
    log_path: Path,
    phase: str,
) -> None:
    """Fail loudly if the launched vLLM process has exited.

    Called after readiness and the warm-up sleep, immediately before the
    first evalscope request, so a crash during the post-ready window is
    turned into an explicit error instead of silently falling through to
    whatever lingering service happens to answer the port.
    """
    if proc.poll() is None:
        return
    tail = _log_tail(log_path)
    raise RuntimeError(
        f"vllm serve exited with code {proc.returncode} "
        f"after readiness ({phase}); refusing to evaluate."
        f"\nLog tail:\n{tail}"
    )


def stop_server(proc: subprocess.Popen[bytes]) -> None:
    """Gracefully stop the vLLM server and all workers in its process group."""
    # start_new_session=True makes the server PID its process-group ID.
    # Workers may outlive the leader, including after proc.poll() reaps it.
    pgid = proc.pid

    # Always try SIGTERM on the process group first
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        pass  # Group already gone

    # Wait for graceful shutdown
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass

    # SIGKILL any survivors
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass  # Group already gone

    # Final reap
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def _log_tail(path: Path, chars: int = 2000) -> str:
    try:
        text = path.read_text()
        return text[-chars:]
    except Exception:
        return "<unreadable>"


# ===================================================================
# Evalscope runner
# ===================================================================


def run_evalscope(
    served_name: str,
    port: int,
    gsm8k_path: str,
    *,
    generation_config: dict[str, Any] | None = None,
    limit: int | None = None,
    work_dir: Path | None = None,
) -> Any:
    """Run evalscope GSM8K (runtime import).

    ``generation_config`` is the sampling contract of the weights under test;
    it is passed through to ``TaskConfig`` untouched so every case over one
    checkpoint samples identically.  The benchmark settings around it --
    dataset, few-shot, batch size, seed -- stay shared by all cases.
    """
    from evalscope import TaskConfig, run_task  # type: ignore[import-untyped]

    generation = dict(
        DEFAULT_GENERATION_CONFIG if generation_config is None else generation_config
    )
    cfg = TaskConfig(
        model=served_name,
        api_url=f"http://127.0.0.1:{port}/v1",
        api_key="EMPTY",
        eval_type="openai_api",
        datasets=["gsm8k"],
        dataset_args={
            "gsm8k": {
                "dataset_id": gsm8k_path,
                "few_shot_num": 4,
                "few_shot_random": False,
                "subset_list": ["main"],
            }
        },
        generation_config=generation,
        eval_batch_size=128,
        limit=limit,
        seed=42,
        work_dir=str(work_dir) if work_dir else None,
    )
    return run_task(task_cfg=cfg)


def extract_score(result: Any) -> tuple[float, int]:
    """Version-tolerant extraction of (accuracy, num_samples)."""
    rep = result["gsm8k"]
    score = getattr(rep, "score", None)
    if score is None:
        score = rep.metrics[0].score
    num = getattr(rep, "num", None)
    if num is None:
        num = rep.metrics[0].num
    return float(score), int(num)


# ===================================================================
# Main pipeline
# ===================================================================


def run_accuracy_eval(  # noqa: C901
    model_key: str,
    *,
    model_override: str | None = None,
    case_id: str | None = None,
    chip: str = "auto",
    update_golden: bool = False,
    limit: int | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    work_dir: Path | None = None,
    port: int = 9985,
    served_model_name: str | None = None,
    extra_serve_flags: list[str] | None = None,
    server_cmd: list[str] | None = None,
    server_env: dict[str, str] | None = None,
    golden_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the full GSM8K accuracy eval pipeline.

    ``model_key`` selects both the evaluation settings and the golden
    identity: several cases may share one checkpoint, and they are all
    compared against that checkpoint's single golden.  ``case_id`` only
    labels the run.

    Returns a dict with keys: ``status``, ``score``, ``num``,
    ``golden_path``, ``error``, ``case_id``, ``record``.
    """
    if model_key not in MODEL_CONFIGS:
        raise ValueError(
            f"unknown model key {model_key!r}; available: {sorted(MODEL_CONFIGS)}"
        )
    model_cfg = MODEL_CONFIGS[model_key]
    # Sampling parameters belong to the weights: the DP+EP case and the plain
    # case over one checkpoint share them, and therefore share one golden.
    generation_cfg = dict(
        model_cfg.get("generation_config") or DEFAULT_GENERATION_CONFIG
    )

    # -- resolve model ------------------------------------------------
    model_id = model_override or model_cfg["model_id"]

    # -- resolve chip → TP --------------------------------------------
    if chip == "auto":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA device available")
        cap = torch.cuda.get_device_capability(0)
        chip = CAPABILITY_TO_CHIP.get((cap[0], cap[1]), DEFAULT_CHIP)
    # Per-model card count override (e.g. DeepSeek-V4 o_groups=8 caps TP at
    # 8).  A recipe whose parallelism stays inside one process spends all of
    # these cards on tensor parallel; a DP+EP recipe splits them across data
    # parallel ranks instead.
    ppu_count = model_cfg.get("ppu_count") or tensor_parallel_for_chip(chip)

    # -- guard: enough GPUs? ------------------------------------------
    import torch

    if torch.cuda.device_count() < ppu_count:
        raise RuntimeError(
            f"need {ppu_count} GPUs for chip {chip!r} but only "
            f"{torch.cuda.device_count()} available"
        )

    # -- resolve GSM8K dataset ----------------------------------------
    gsm8k_path = resolve_gsm8k_dataset()

    # -- paths --------------------------------------------------------
    golden_root = golden_dir or GOLDEN_DIR
    gf = golden_root / golden_filename(model_key, chip)
    pending_gf = golden_root / pending_golden_filename(model_key, chip)
    wd = Path(work_dir) if work_dir else REPORTS_ROOT
    wd.mkdir(parents=True, exist_ok=True)

    served_base = served_model_name or model_cfg["served_name"]
    # Bind this run to a unique served-model identity so readiness and the
    # eval client both reject any stale vLLM instance that lingers on the
    # same port with a stable catalog name.  The golden identity below is
    # keyed by chip + checkpoint_tail + benchmark, so this per-run suffix
    # never leaks into the persisted golden contract.
    unique_served = unique_served_model_name(served_base)
    flags = list(extra_serve_flags or model_cfg.get("extra_flags", []))
    log_path = wd / f"vllm_serve_{case_id or model_key}.log"

    # -- start server -------------------------------------------------
    # Built here as well as inside ``start_vllm_server`` so the golden records
    # the args the process really ran with; a pure function yields the same
    # argv from the same inputs.
    serve_argv = build_serve_argv(
        model_id,
        served_model_name=unique_served,
        port=port,
        tp=ppu_count,
        extra_flags=flags,
        server_cmd=server_cmd,
    )
    server, log_path = start_vllm_server(
        model_id,
        served_model_name=unique_served,
        port=port,
        tp=ppu_count,
        extra_flags=flags,
        server_cmd=server_cmd,
        server_env=server_env,
        log_path=log_path,
    )
    try:
        wait_server_ready(
            server,
            log_path,
            port=port,
            expected_served_name=unique_served,
            timeout=_server_ready_timeout(server_env),
        )
        time.sleep(SERVER_POST_READY_SLEEP)
        # A readiness probe only proves the server was alive at the time it
        # answered; re-check right before the first eval request so a crash
        # during the warm-up window fails the run instead of silently
        # routing traffic to whatever still answers the port.
        _ensure_launched_server_alive(server, log_path, phase="post-readiness warm-up")

        # -- run evalscope --------------------------------------------
        result_raw = run_evalscope(
            unique_served,
            port,
            gsm8k_path,
            generation_config=generation_cfg,
            limit=limit,
            work_dir=wd / "evalscope",
        )
        score, num = extract_score(result_raw)

        # -- evalscope version ----------------------------------------
        try:
            import evalscope as _es  # type: ignore[import-untyped]

            es_ver = getattr(_es, "__version__", "unknown")
        except ImportError:
            es_ver = "unknown"

        # -- build golden record --------------------------------------
        ckpt_tail = _checkpoint_tail(model_id)
        ds_cfg: dict[str, Any] = {
            "dataset_id": gsm8k_path,
            "few_shot_num": 4,
            "few_shot_random": False,
            "subset_list": ["main"],
            "num_expected": GSM8K_FULL_SIZE,
        }
        srv_cfg: dict[str, Any] = describe_serving(serve_argv, ppu_count=ppu_count)
        tol = dict(DEFAULT_TOLERANCE)
        record = build_accuracy_golden(
            model_key=model_key,
            chip=chip,
            checkpoint=model_id,
            evalscope_version=es_ver,
            dataset_cfg=ds_cfg,
            serving_cfg=srv_cfg,
            generation_cfg=generation_cfg,
            score=score,
            # Correct answers, not samples asked: ``num`` is the sample count
            # and the score is ``score_num / num``.
            score_num=round(score * num),
            tolerance=tol,
        )

        # -- golden comparison ----------------------------------------
        result = _golden_compare(
            record=record,
            score=score,
            num=num,
            chip=chip,
            ckpt_tail=ckpt_tail,
            gf=gf,
            pending_gf=pending_gf,
            golden_root=golden_root,
            update_golden=update_golden,
            model_key=model_key,
            threshold=threshold,
        )
        result["case_id"] = case_id or model_key
        return result
    finally:
        stop_server(server)


def _golden_compare(
    *,
    record: dict[str, Any],
    score: float,
    num: int,
    chip: str,
    ckpt_tail: str,
    gf: Path,
    pending_gf: Path,
    golden_root: Path,
    update_golden: bool,
    model_key: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Perform the golden band check, or the threshold fallback.

    ``threshold`` only governs a checkpoint that has no baseline yet. Once a
    golden is committed it *is* the reference: a checkpoint whose golden lands
    at 0.60 must gate against 0.60, not against a fallback number that was only
    ever meant to stop a first run from being recorded far below anything
    acceptable.
    """
    status: str
    error_msg: str | None = None

    # --- Completeness always precedes creation, update, or comparison.
    expected_num = record["dataset"]["num_expected"]
    if num != expected_num:
        return {
            "status": "fail_completeness",
            "score": score,
            "num": num,
            "golden_path": str(gf),
            "error": (
                f"completeness gate failed: evaluated {num} samples, "
                f"expected {expected_num}"
            ),
            "record": record,
        }

    # --- Threshold, used only where no golden can arbitrate --------------
    ok_thr, thr_err = check_score_threshold(score, threshold)

    if update_golden:
        if not ok_thr:
            status = "fail_threshold"
            error_msg = thr_err
        else:
            write_golden(record, gf)
            if pending_gf.exists():
                pending_gf.unlink()
            status = "updated"
    elif gf.exists():
        golden = read_golden(gf)
        ok_id, id_err = check_golden_identity(
            golden, chip=chip, checkpoint_tail=ckpt_tail
        )
        if not ok_id:
            status = "fail_identity"
            error_msg = id_err
        else:
            # A score recorded under different sampling parameters cannot
            # arbitrate this run, so the contract is settled before the band:
            # fail_config names the drifted knob instead of reporting the
            # accuracy delta it caused.
            ok_cfg, cfg_err = check_generation_config(golden, record["generation"])
            if not ok_cfg:
                status = "fail_config"
                error_msg = cfg_err
            else:
                # The band replaces the threshold gate entirely here.
                ok_g, acc_err = check_accuracy_golden(score, num, golden)
                status = "pass" if ok_g else "fail_accuracy"
                error_msg = acc_err
    elif not ok_thr:
        # No golden yet; threshold is the only gate.
        write_golden(record, pending_gf)
        status = "fail_threshold"
        error_msg = thr_err
    else:
        # Passes threshold but no golden — write pending.
        write_golden(record, pending_gf)
        status = "created"
        warnings.warn(
            f"no golden for {model_key}/{chip}; "
            f"wrote pending to {pending_gf.name}. "
            f"score={score:.4f} passes threshold "
            f"{threshold}.",
            stacklevel=3,
        )

    return {
        "status": status,
        "score": score,
        "num": num,
        "golden_path": str(gf),
        "error": error_msg,
        "record": record,
    }


# ===================================================================
# CLI entry point
# ===================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="PPU GSM8K accuracy E2E test")
    parser.add_argument(
        "--model-key",
        required=True,
        choices=sorted(MODEL_CONFIGS),
        help="Model configuration key",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the default HF model_id or local path",
    )
    parser.add_argument(
        "--chip",
        default="auto",
        choices=["auto", "810e", "890p"],
    )
    parser.add_argument(
        "--case-id",
        default=None,
        help=(
            "Catalog case identifier; labels the serve log and the run "
            "report only. Golden identity stays keyed on --model-key, so "
            "several cases over one checkpoint share one golden."
        ),
    )
    parser.add_argument(
        "--server-cmd",
        default=None,
        help=(
            "JSON array holding the complete 'vllm serve' argv for this "
            "case, with {model} / {port} / {served_name} placeholders. "
            "Omit to use the built-in argument list."
        ),
    )
    parser.add_argument(
        "--server-env",
        default=None,
        help=(
            "JSON object of extra environment variables for this case's "
            "server process, e.g. a larger engine-ready timeout. Merged "
            "over the inherited environment."
        ),
    )
    parser.add_argument(
        "--update-golden",
        action="store_true",
        help="Rewrite the formal golden baseline",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Score >= threshold to pass (default {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of evaluation samples",
    )
    parser.add_argument("--work-dir", default=None)
    parser.add_argument(
        "--golden-dir",
        default=None,
        help="Writable directory for formal or pending golden records",
    )
    parser.add_argument("--port", type=int, default=9985)
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--extra-serve-flags", default=None)
    args = parser.parse_args()

    extra = args.extra_serve_flags.split() if args.extra_serve_flags else None
    wd = Path(args.work_dir) if args.work_dir else None
    server_cmd = _parse_server_cmd(args.server_cmd)
    server_env = _parse_server_env(args.server_env)

    result = run_accuracy_eval(
        args.model_key,
        model_override=args.model,
        case_id=args.case_id,
        chip=args.chip,
        update_golden=args.update_golden,
        limit=args.limit,
        threshold=args.threshold,
        work_dir=wd,
        port=args.port,
        served_model_name=args.served_model_name,
        extra_serve_flags=extra,
        server_cmd=server_cmd,
        server_env=server_env,
        golden_dir=Path(args.golden_dir) if args.golden_dir else None,
    )

    out = {k: v for k, v in result.items() if k != "record"}
    print(json.dumps(out, indent=2))
    if result["status"].startswith("fail"):
        sys.exit(1)


if __name__ == "__main__":
    main()
