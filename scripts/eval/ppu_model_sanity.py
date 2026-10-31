#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Launch one installed model, check a complete answer, and stop owned workers.

Standard library only. This is a short inference gate, not an accuracy benchmark
or the complete checkpoint acceptance contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

PROFILES_PATH = Path(__file__).with_name("model_sanity_profiles.json")
BASELINE_PROFILE = "deepseek-v4-community-tp8"
PROMPT = "计算 17 + 25。最终答案只输出十进制整数，不要单位或解释。"
FATAL_RE = re.compile(
    r"Traceback \(most recent|EngineCore failed|Engine core initialization failed"
    r"|core dumped|Failed to infer device type"
)
ERROR_RE = re.compile(r"\bERROR\b")


def load_profiles() -> dict[str, Any]:
    return json.loads(PROFILES_PATH.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def process_identity(pid: int) -> dict:
    """Observe the actual PID/group and Linux start-time record during execution."""
    stat = Path(f"/proc/{pid}/stat")
    return {
        "pid": pid,
        "pgid": os.getpgid(pid),
        "observed_unix": time.time(),
        "proc_stat": stat.read_text() if stat.is_file() else None,
    }


def build_command(
    profile: dict[str, Any], checkpoint_root: Path, served_name: str, port: int
) -> list[str]:
    """Keep structured JSON flags as individual argv entries; never shell-split."""
    return [
        sys.executable,
        "-I",
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        str(checkpoint_root / profile["checkpoint"]),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        served_name,
        *profile["serve_args"],
    ]


def request_json(url: str, timeout: float, body: dict | None = None) -> dict:
    request = Request(
        url,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    # Local serving must not pass through image HTTP(S) proxies.
    with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
        return json.load(response)


def validate_answer(response: dict, served_name: str) -> str:
    """Require a complete, correct final answer, rather than reasoning or a match."""
    if response.get("model") != served_name:
        raise ValueError("response model does not match this run's served identity")
    choices = response.get("choices", [])
    if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
        raise ValueError("answer must finish naturally with finish_reason=stop")
    message = choices[0].get("message", {})
    content = message.get("content")
    if message.get("tool_calls") or not isinstance(content, str):
        raise ValueError("expected a final textual answer")
    if content.strip() != "42":
        raise ValueError(f"incorrect final answer: {content!r}; expected '42'")
    return content


def stream_answer(
    url: str, body: dict, timeout: float, raw_path: Path
) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps({**body, "stream": True}).encode(),
        headers={"Content-Type": "application/json"},
    )
    content: list[str] = []
    model, finish, done = None, None, False
    deadline = time.monotonic() + timeout
    with (
        build_opener(ProxyHandler({})).open(request, timeout=timeout) as response,
        raw_path.open("wb") as raw,
    ):
        for line in response:
            raw.write(line)
            raw.flush()
            if done:
                if line.strip():
                    raise ValueError("unexpected data after stream [DONE]")
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("stream exceeded request timeout")
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                done = True
                continue
            event = json.loads(data)
            if "error" in event:
                raise ValueError(f"stream returned an error: {event['error']}")
            if event.get("model") is not None:
                if model is not None and model != event["model"]:
                    raise ValueError("stream changed model identity")
                model = event["model"]
            for choice in event.get("choices", []):
                if choice.get("index") != 0:
                    raise ValueError("unexpected stream choice")
                delta = choice.get("delta", {})
                if delta.get("tool_calls"):
                    raise ValueError("unexpected streamed tool call")
                content.append(delta.get("content") or "")
                if choice.get("finish_reason") is not None:
                    finish = choice["finish_reason"]
    if not done:
        raise ValueError("stream ended without [DONE]")
    return {
        "model": model,
        "choices": [
            {"message": {"content": "".join(content)}, "finish_reason": finish}
        ],
    }


def wait_ready(
    process: subprocess.Popen,
    base_url: str,
    served_name: str,
    log: Path,
    timeout: float,
) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited before readiness: {process.returncode}")
        if FATAL_RE.search(log.read_text(errors="replace")):
            raise RuntimeError("fatal startup error; inspect serve.log")
        try:
            models = request_json(base_url + "/v1/models", min(5, timeout))
            if any(item.get("id") == served_name for item in models.get("data", [])):
                if process.poll() is not None:
                    raise RuntimeError("owned server exited during readiness")
                return models
        except (URLError, TimeoutError, ValueError, OSError):
            pass
        time.sleep(min(1, max(0, deadline - time.monotonic())))
    raise TimeoutError(f"server not ready after {timeout}s")


def stop_owned_server(process: subprocess.Popen, timeout: float = 60) -> dict:
    """Only signal the new session created by this runner, including its workers."""
    pgid = process.pid
    was_running = process.poll() is None

    def group_alive() -> bool:
        process.poll()  # Reap the leader before probing a possibly empty group.
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            return False

    forced = False
    if group_alive():
        try:
            # Let the API server drain and stop its engines before signalling
            # workers; killing them together can manufacture shutdown errors.
            if process.poll() is None:
                process.terminate()
            else:
                os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + timeout
        while group_alive() and time.monotonic() < deadline:
            process.poll()  # Reap an exited group leader before testing its group.
            time.sleep(0.1)
        if group_alive():
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 5
            while group_alive() and time.monotonic() < deadline:
                process.poll()
                time.sleep(0.1)
        if group_alive():
            forced = True
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    process.wait(timeout=10)
    deadline = time.monotonic() + 5
    while group_alive() and time.monotonic() < deadline:
        time.sleep(0.1)
    return {
        "pgid": pgid,
        "was_running": was_running,
        "returncode": process.returncode,
        "forced": forced,
        "group_absent": not group_alive(),
    }


def installed_identity(directory: Path, env: dict[str, str]) -> dict:
    """Probe installed paths in an isolated interpreter outside the checkout."""
    code = """
import hashlib, importlib.metadata as md, importlib.util, json, sys
from pathlib import Path
packages = {}
for dist_name, module in [('vllm', 'vllm'), ('vllm-sail', 'vllm_sail'), ('torch', 'torch')]:
    dist = md.distribution(dist_name)
    spec = importlib.util.find_spec(module)
    expected = Path(dist.locate_file(module + '/__init__.py')).resolve()
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != expected:
        raise RuntimeError(dist_name + ' is shadowed or editable; use installed wheels')
    direct_url = json.loads(dist.read_text('direct_url.json') or '{}')
    if direct_url.get('dir_info', {}).get('editable'):
        raise RuntimeError(dist_name + ' is editable; use installed wheels')
    record = dist.read_text('RECORD')
    packages[dist_name] = dict(version=dist.version, origin=spec.origin,
        record_sha256=hashlib.sha256((record or '').encode()).hexdigest())
print(json.dumps(dict(python=sys.version, executable=sys.executable, packages=packages)))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=directory,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return json.loads(result.stdout)


def verify_execution_proof(path: Path, inventory: dict) -> dict:
    """Recheck a reviewed wheel-byte receipt against this interpreter's packages."""
    proof = json.loads(path.read_text())
    if proof.get("status") != "complete-package-byte-maps-match-wheels":
        raise ValueError("execution proof is not a complete package-byte receipt")
    names = {"vllm": "vllm", "vllm-sail": "vllm_sail"}
    for field, length in (("commits", 40), ("wheel_sha256", 64)):
        values = proof.get(field, {})
        if (
            not isinstance(values, dict)
            or set(values) != set(names)
            or any(
                not isinstance(value, str)
                or re.fullmatch(r"[0-9a-f]{" + str(length) + "}", value) is None
                for value in values.values()
            )
        ):
            raise ValueError(f"invalid {field} in execution proof")
    for name, module in names.items():
        record = proof["packages"][name]
        root = Path(record["root"]).resolve()
        current = inventory["packages"][name]
        if (
            Path(current["origin"]).resolve() != root / "__init__.py"
            or current["version"] != record["version"]
        ):
            raise ValueError(f"execution proof belongs to another {name} installation")
        actual = {}
        for file in root.rglob("*"):
            if (
                not file.is_file()
                or "__pycache__" in file.parts
                or file.suffix in (".pyc", ".pyo")
            ):
                continue
            digest = hashlib.sha256()
            with file.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual[module + "/" + file.relative_to(root).as_posix()] = (
                digest.hexdigest()
            )
        if not actual or actual != record["files"]:
            raise ValueError(f"installed {name} bytes changed since execution proof")
    return {
        "commits": proof["commits"],
        "wheel_sha256": proof["wheel_sha256"],
        "execution_proof_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def run_sanity(
    profile_key: str,
    *,
    output: Path,
    checkpoint_root: Path = Path("/ppusw/datasets"),
    port: int = 9985,
    ready_timeout: float = 3600,
    request_timeout: float = 300,
    max_tokens: int = 2048,
    with_stream: bool = False,
    execution_proof: Path | None = None,
) -> dict:
    """Write evidence in a fresh directory; any failed gate returns failed status."""
    profile = load_profiles()[profile_key]
    checkpoint_root = checkpoint_root.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "runner-process.txt", process_identity(os.getpid()))
    served_name = profile_key + "-sanity-" + secrets.token_hex(8)
    command = build_command(profile, checkpoint_root, served_name, port)
    env = {**os.environ, **profile.get("env", {})}
    env.pop("PYTHONPATH", None)
    body = {
        "model": served_name,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0,
        "max_completion_tokens": max_tokens,
    }
    invocation = {
        "profile": profile_key,
        "argv": command,
        "env_overrides": profile.get("env", {}),
        "profile_sha256": hashlib.sha256(PROFILES_PATH.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "request": body,
        "with_stream": with_stream,
        "ready_timeout": ready_timeout,
        "request_timeout": request_timeout,
    }
    write_json(output / "invocation.json", invocation)
    started = time.monotonic()
    result: dict[str, Any] = {
        "status": "failed",
        "profile": profile_key,
        "case": profile_key,
    }
    process = None
    log = output / "serve.log"
    try:
        if min(ready_timeout, request_timeout, max_tokens) <= 0:
            raise ValueError("timeouts and token budget must be positive")
        checkpoint = checkpoint_root / profile["checkpoint"]
        if not (checkpoint / "config.json").is_file():
            raise FileNotFoundError(f"missing checkpoint config: {checkpoint}")
        with socket.socket() as probe:
            # Match HTTP servers: permit the previous service's TIME_WAIT
            # sockets while still rejecting an active listener.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", port))
        inventory = installed_identity(output, env)
        write_json(output / "environment.json", inventory)
        if execution_proof is not None:
            identities = verify_execution_proof(execution_proof, inventory)
            invocation.update(identities)
            result.update(identities)
            write_json(output / "invocation.json", invocation)
        with log.open("wb") as stream:
            process = subprocess.Popen(
                command,
                cwd=output,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        write_json(
            output / "owned-process-group.json",
            {"pid": process.pid, "pgid": process.pid},
        )
        write_json(output / "owned-process-group.txt", process_identity(process.pid))
        base_url = f"http://127.0.0.1:{port}"
        models = wait_ready(process, base_url, served_name, log, ready_timeout)
        write_json(output / "models.json", models)
        result["ready_seconds"] = time.monotonic() - started
        response = request_json(
            base_url + "/v1/chat/completions", request_timeout, body
        )
        write_json(output / "request-0.json", response)
        result["answer"] = validate_answer(response, served_name)
        if with_stream:
            response = stream_answer(
                base_url + "/v1/chat/completions",
                body,
                request_timeout,
                output / "stream.sse",
            )
            write_json(output / "stream.json", response)
            result["stream_answer"] = validate_answer(response, served_name)
        with build_opener(ProxyHandler({})).open(
            base_url + "/metrics", timeout=10
        ) as response:
            (output / "metrics.prom").write_bytes(response.read())
        if process.poll() is not None:
            raise RuntimeError("owned server exited during requests")
        result["pre_shutdown_errors"] = len(
            ERROR_RE.findall(log.read_text(errors="replace"))
        )
        if result["pre_shutdown_errors"]:
            raise RuntimeError("ERROR entries in serving log")
        result["status"] = "passed"
    except (Exception, KeyboardInterrupt) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if process is not None:
            try:
                result["cleanup"] = stop_owned_server(process)
                cleanup = result["cleanup"]
                if (
                    not cleanup["group_absent"]
                    or cleanup["forced"]
                    or not cleanup["was_running"]
                    or cleanup["returncode"] not in (0, -signal.SIGTERM, -signal.SIGINT)
                ):
                    result.update(
                        status="failed",
                        cleanup_error="owned group did not stop gracefully",
                    )
            except Exception as exc:
                result.update(
                    status="failed", cleanup_error=f"{type(exc).__name__}: {exc}"
                )
                result["cleanup_traceback"] = traceback.format_exc()
        if log.exists():
            result["post_shutdown_errors"] = len(
                ERROR_RE.findall(log.read_text(errors="replace"))
            )
            if result["post_shutdown_errors"]:
                result.update(
                    status="failed", log_error="ERROR entries in final serve.log"
                )
        result["state"] = result["status"]
        result["server_exit_after_cleanup"] = result.get("cleanup", {}).get(
            "returncode"
        )
        metrics = output / "metrics.prom"
        if metrics.is_file():
            result["metrics_sha256"] = hashlib.sha256(metrics.read_bytes()).hexdigest()
        result["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--profile", choices=load_profiles(), default=BASELINE_PROFILE)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("/ppusw/datasets"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--port", type=int, default=9985)
    parser.add_argument("--ready-timeout", type=float, default=3600)
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--with-stream", action="store_true")
    parser.add_argument("--execution-proof", type=Path)
    args = parser.parse_args()
    if args.list:
        print("\n".join(load_profiles()))
        return 0
    if args.output is None:
        parser.error("--output is required (a new directory for each attempt)")

    def interrupted(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    result = run_sanity(
        args.profile,
        output=args.output,
        checkpoint_root=args.checkpoint_root,
        port=args.port,
        ready_timeout=args.ready_timeout,
        request_timeout=args.request_timeout,
        max_tokens=args.max_tokens,
        with_stream=args.with_stream,
        execution_proof=args.execution_proof,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
