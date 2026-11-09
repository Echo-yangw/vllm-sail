#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Serve the 810E sanity presets sequentially using the installed vLLM SAIL.

Run inside a prepared device container. This script neither installs packages
nor allocates machines. Results are a serving sanity check, not an accuracy or
performance baseline. Keep checkpoint overrides and the exact argv in results.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

CATALOG = Path(__file__).with_name("ppu_810e_sanity_models.json")


def serve_command(preset: dict, checkpoint: Path, port: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        str(checkpoint),
        "--trust-remote-code",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        preset["key"],
        *preset["args"],
    ]


def request_json(url: str, *, payload: dict | None = None, timeout: float = 5):
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    # Corporate proxy settings must not redirect loopback requests.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        raw = response.read()
        return json.loads(raw) if raw else None


def validate_completion(response: dict, model: str) -> None:
    if response.get("error"):
        raise ValueError(f"API error: {response['error']}")
    if response.get("model") != model:
        raise ValueError(f"unexpected response model: {response.get('model')!r}")
    choices = response.get("choices") or []
    if len(choices) != 1:
        raise ValueError(f"expected one choice, got {len(choices)}")
    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("empty final answer (reasoning alone does not pass sanity)")
    if choices[0].get("finish_reason") != "stop":
        raise ValueError(f"unexpected finish reason: {choices[0].get('finish_reason')}")
    if response.get("usage", {}).get("completion_tokens", 0) <= 0:
        raise ValueError("response has no generated tokens")


def runtime_environment() -> dict:
    # The script directory is outside the package; cwd is changed before imports.
    import torch
    import vllm
    from vllm.platforms import current_platform

    import vllm_sail
    from vllm_sail.compat import check_vllm_compatibility

    check_vllm_compatibility(force=True)
    if not current_platform.is_ppu() or not torch.cuda.is_available():
        raise RuntimeError("sanity requires the PPU platform and a real device")
    for module in (vllm, vllm_sail):
        if "site-packages" not in Path(module.__file__).parts:
            raise RuntimeError(f"expected installed package, got {module.__file__}")
    devices = []
    for index in range(torch.cuda.device_count()):
        capability = tuple(torch.cuda.get_device_capability(index))
        if capability != (8, 0):
            raise RuntimeError(f"device {index} is not 810E: {capability}")
        devices.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": capability,
            }
        )
    return {
        "host": socket.gethostname(),
        "python": sys.version,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "vllm", "vllm-sail")
        },
        "imports": {"vllm": vllm.__file__, "vllm_sail": vllm_sail.__file__},
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "catalog_sha256": hashlib.sha256(CATALOG.read_bytes()).hexdigest(),
        "devices": devices,
        "env": {
            key: os.environ[key]
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "VLLM_PLUGINS",
                "VLLM_SAIL_MOE_BACKEND",
                "VLLM_PPU_MOE_BACKEND",
                "NCCL_SOCKET_IFNAME",
                "NCCL_IB_HCA",
                "NCCL_P2P_DISABLE",
                "NCCL_IB_DISABLE",
            )
            if key in os.environ
        },
    }


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def stop_server(process: subprocess.Popen) -> None:
    # Workers share this dedicated process group, including after parent exit.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


@contextmanager
def cleanup_signals():
    """Finish owned-process cleanup and result writes after cancellation."""
    previous = {}
    if threading.current_thread() is threading.main_thread():
        previous = {
            signum: signal.signal(signum, signal.SIG_IGN)
            for signum in (signal.SIGTERM, signal.SIGINT)
        }
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def run_preset(preset: dict, checkpoint: Path, args, available_cards: int) -> dict:
    output = args.results_dir / preset["key"]
    output.mkdir()  # A prior run must not be overwritten.
    argv = serve_command(preset, checkpoint, args.port)
    result = {
        "key": preset["key"],
        "checkpoint": str(checkpoint),
        "argv": argv,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
        "stage": "preflight",
    }
    process = None
    pool = None
    started = time.monotonic()
    try:
        config = checkpoint / "config.json"
        if not config.is_file():
            raise FileNotFoundError(f"checkpoint config missing: {config}")
        result["config_sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
        if available_cards < preset["cards"]:
            raise RuntimeError(
                f"requires {preset['cards']} cards; got {available_cards}"
            )
        # Reject an occupied port before starting our own server.
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", args.port))
        env = dict(os.environ)
        for key in ("PYTHONPATH", "VLLM_VERSION", "VLLM_VERSION_OVERRIDE"):
            env.pop(key, None)
        env.update(
            VLLM_PLUGINS="ppu",
            VLLM_USE_RUST_FRONTEND="0",
            VLLM_USE_RUST_BENCH="0",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            VLLM_ENGINE_READY_TIMEOUT_S=str(args.ready_timeout),
        )
        result["serve_env"] = {
            key: env[key]
            for key in (
                "VLLM_PLUGINS",
                "VLLM_USE_RUST_FRONTEND",
                "VLLM_USE_RUST_BENCH",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "VLLM_ENGINE_READY_TIMEOUT_S",
            )
        }
        result["stage"] = "startup"
        with (output / "serve.log").open("w") as log:
            process = subprocess.Popen(
                argv,
                cwd="/tmp",
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

            def check_launcher():
                code = process.poll()
                if code is not None:
                    result["launcher_exit_code"] = code
                    raise RuntimeError(f"server exited with code {code}")

            deadline = time.monotonic() + args.ready_timeout
            endpoint = f"http://127.0.0.1:{args.port}"
            while True:
                check_launcher()
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"server not ready within {args.ready_timeout}s")
                try:
                    request_json(endpoint + "/health")
                    models = request_json(endpoint + "/v1/models")
                    if preset["key"] in {m["id"] for m in models["data"]}:
                        break
                except (urllib.error.URLError, TimeoutError, OSError):
                    pass
                time.sleep(5)
            result["ready_seconds"] = time.monotonic() - started
            write_json(output / "models.json", models)
            result["stage"] = "generation"
            check_launcher()

            def generate(index: int) -> dict:
                payload = {
                    "model": preset["key"],
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Reply briefly: what is 1 + 1?"
                                if index % 2 == 0
                                else "请用一句简短的话介绍你自己。"
                            ),
                        }
                    ],
                    "temperature": 0,
                    "seed": 0,
                    "max_completion_tokens": args.max_tokens,
                }
                write_json(output / f"request-{index}.json", payload)
                response = request_json(
                    endpoint + "/v1/chat/completions",
                    payload=payload,
                    timeout=args.request_timeout,
                )
                write_json(output / f"response-{index}.json", response)
                validate_completion(response, preset["key"])
                return response

            # Exercise a first request and a repeat/burst after warm-up. DP16
            # uses a 16-request burst; this does not assert per-rank routing.
            generate(0)
            concurrency = preset.get("concurrency", 1)
            # Joining the pool belongs after server cleanup: a blocked HTTP
            # request must not delay termination of the process group it uses.
            pool = concurrent.futures.ThreadPoolExecutor(concurrency)
            list(pool.map(generate, range(1, concurrency + 1)))
            request_json(endpoint + "/health")
            check_launcher()
            result.update(status="passed", stage="complete", requests=concurrency + 1)
    except KeyboardInterrupt as error:
        result.update(status="cancelled", error=f"KeyboardInterrupt: {error}")
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        with cleanup_signals():
            try:
                if process is not None:
                    stop_server(process)
            finally:
                try:
                    if pool is not None:
                        pool.shutdown(wait=True, cancel_futures=True)
                finally:
                    result["elapsed_seconds"] = time.monotonic() - started
                    write_json(output / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", help="preset keys; default: all")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--checkpoint-root", type=Path, default=Path("/ppusw/datasets/checkpoints")
    )
    parser.add_argument("--checkpoint", action="append", default=[], metavar="KEY=PATH")
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--port", type=int, default=9985)
    parser.add_argument("--ready-timeout", type=int, default=3600)
    parser.add_argument("--request-timeout", type=int, default=600)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
    catalog = {item["key"]: item for item in json.loads(CATALOG.read_text())}
    if args.list:
        for key, preset in catalog.items():
            print(f"{key}\t{preset['cards']} cards\t{preset['checkpoint']}")
        return 0
    keys = args.models or list(catalog)
    if len(set(keys)) != len(keys) or set(keys) - catalog.keys():
        parser.error("model keys must be unique and present in --list")
    overrides = {}
    for value in args.checkpoint:
        key, separator, path = value.partition("=")
        if not separator or key not in keys or not path:
            parser.error(f"invalid checkpoint override: {value}")
        overrides[key] = Path(path).expanduser().resolve()
    args.checkpoint_root = args.checkpoint_root.expanduser().resolve()
    checkpoints = {
        key: overrides.get(key, args.checkpoint_root / catalog[key]["checkpoint"])
        for key in keys
    }
    if args.dry_run:
        for key in keys:
            print(shlex.join(serve_command(catalog[key], checkpoints[key], args.port)))
        return 0
    if args.results_dir is None:
        parser.error("--results-dir is required")
    if min(args.ready_timeout, args.request_timeout, args.max_tokens) <= 0:
        parser.error("timeouts and max-tokens must be positive")
    args.results_dir = args.results_dir.resolve()
    args.results_dir.mkdir(parents=True, exist_ok=False)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    os.chdir("/tmp")
    environment = runtime_environment()
    write_json(args.results_dir / "environment.json", environment)
    results = []
    for key in keys:
        print(f"START {key}", flush=True)
        result = run_preset(
            catalog[key], checkpoints[key], args, len(environment["devices"])
        )
        results.append(result)
        write_json(args.results_dir / "summary.json", results)
        print(
            f"{result['status'].upper()} {key}: {result.get('error', '')}", flush=True
        )
        if result["status"] == "cancelled":
            return 130
    return int(any(result["status"] != "passed" for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
