# SPDX-License-Identifier: Apache-2.0
"""The sanity runner's public command and response contracts need no SDK."""

import concurrent.futures
import importlib.util
import json
import os
import signal
import socket
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "ppu_810e_sanity", ROOT / "scripts/ci/ppu_810e_sanity.py"
)
sanity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sanity)


def completion(**overrides):
    return {
        "model": "demo",
        "choices": [{"message": {"content": "2"}, "finish_reason": "stop"}],
        "usage": {"completion_tokens": 1},
        **overrides,
    }


@pytest.fixture
def external_launcher(monkeypatch):
    """Strict process/OS doubles; retain the real runner and thread pool."""
    stopped = threading.Event()
    signals = []

    class Launcher:
        pid = 987654321
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            assert stopped.is_set()
            return self.returncode

    process = Launcher()

    def launch(argv, *, cwd, env, stdout, stderr, start_new_session):
        assert start_new_session
        return process

    def kill_group(pid, signum):
        assert pid == process.pid
        signals.append(signum)
        if process.returncode is None:
            process.returncode = -signum
        stopped.set()

    monkeypatch.setattr(sanity.subprocess, "Popen", launch)
    monkeypatch.setattr(sanity.os, "killpg", kill_group)
    return SimpleNamespace(process=process, stopped=stopped, signals=signals)


def test_exited_launcher_fails_even_when_leftover_workers_answer(
    tmp_path, monkeypatch, external_launcher
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    args = SimpleNamespace(
        results_dir=tmp_path, port=0, ready_timeout=5, request_timeout=5, max_tokens=16
    )
    health_checks = 0

    def request(url, *, payload=None, timeout=5):
        nonlocal health_checks
        if url.endswith("/health"):
            health_checks += 1
            if health_checks == 2:
                external_launcher.process.returncode = 23
            return None
        if url.endswith("/v1/models"):
            return {"data": [{"id": "demo"}]}
        assert url.endswith("/v1/chat/completions")
        return completion()

    monkeypatch.setattr(sanity, "request_json", request)
    result = sanity.run_preset(
        {"key": "demo", "cards": 2, "args": []}, checkpoint, args, 2
    )
    assert result["status"] == "failed"
    assert result["stage"] == "generation"
    assert result["launcher_exit_code"] == 23
    assert "server exited with code 23" in result["error"]
    assert json.loads((tmp_path / "demo/result.json").read_text()) == result
    assert (tmp_path / "demo/response-1.json").is_file()


@pytest.mark.parametrize("repeat_signal", [False, True])
def test_sigterm_stops_server_before_joining_blocked_burst_and_saves_results(
    tmp_path, monkeypatch, external_launcher, repeat_signal
):
    key = "qwen3.5-35b_bf16_tp2"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    results = tmp_path / "results"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ppu_810e_sanity.py",
            "--models",
            key,
            "--checkpoint",
            f"{key}={checkpoint}",
            "--results-dir",
            str(results),
            "--port",
            "0",
        ],
    )
    monkeypatch.setattr(
        sanity, "runtime_environment", lambda: {"devices": [None, None]}
    )
    generations = 0
    unblocked_by_cleanup = []
    waiting_for_result = threading.Event()
    future_result = concurrent.futures.Future.result

    def observed_result(future, timeout=None):
        if threading.current_thread() is threading.main_thread():
            waiting_for_result.set()
        return future_result(future, timeout=timeout)

    monkeypatch.setattr(concurrent.futures.Future, "result", observed_result)

    def request(url, *, payload=None, timeout=5):
        nonlocal generations
        if url.endswith("/health"):
            return None
        if url.endswith("/v1/models"):
            return {"data": [{"id": key}]}
        assert url.endswith("/v1/chat/completions")
        generations += 1
        if generations == 1:
            return completion(model=key)
        # Future.result is reached after executor.submit has registered the
        # worker. This keeps the negative control inside the old join window.
        assert waiting_for_result.wait(timeout=2)
        os.kill(os.getpid(), signal.SIGTERM)
        # The finite wait makes the old pool-context bug fail deterministically
        # instead of hanging the test process indefinitely.
        unblocked_by_cleanup.append(external_launcher.stopped.wait(timeout=2))
        raise OSError("owned server connection closed")

    kill_group = sanity.os.killpg

    def repeated_cancel(pid, signum):
        kill_group(pid, signum)
        if repeat_signal and signum == signal.SIGTERM:
            os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(sanity, "request_json", request)
    monkeypatch.setattr(sanity.os, "killpg", repeated_cancel)
    previous = signal.getsignal(signal.SIGTERM)
    try:
        try:
            code = sanity.main()
        except KeyboardInterrupt:
            # Keep the negative control on the old implementation from
            # aborting pytest before its cleanup-order assertions can run.
            code = 130
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert code == 130
    assert unblocked_by_cleanup == [True]
    assert external_launcher.signals == [signal.SIGTERM, signal.SIGKILL]
    result = json.loads((results / key / "result.json").read_text())
    assert result["status"] == "cancelled" and result["stage"] == "generation"
    assert "KeyboardInterrupt" in result["error"]
    assert json.loads((results / "summary.json").read_text()) == [result]


def test_completion_requires_a_final_answer_and_generated_tokens():
    sanity.validate_completion(completion(), "demo")
    for response in (
        completion(error={"message": "engine failed"}),
        completion(model="another-server"),
        completion(choices=[]),
        completion(choices=[{"message": {"content": None, "reasoning": "thinking"}}]),
        completion(choices=[{"message": {"content": " "}}]),
        completion(choices=[{"message": {"content": "2"}, "finish_reason": "error"}]),
        completion(
            choices=[{"message": {"content": "thinking"}, "finish_reason": "length"}]
        ),
        completion(usage={"completion_tokens": 0}),
    ):
        with pytest.raises(ValueError):
            sanity.validate_completion(response, "demo")


def test_serve_preserves_json_and_checkpoint_arguments():
    presets = json.loads(sanity.CATALOG.read_text())
    assert len(presets) == 8
    for preset in presets:
        argv = sanity.serve_command(preset, Path("/models/space in name"), 9985)
        assert argv[4] == "/models/space in name"
        assert argv[argv.index("--host") + 1] == "127.0.0.1"
        assert argv[argv.index("--served-model-name") + 1] == preset["key"]
        assert argv[-len(preset["args"]) :] == preset["args"]
        if "--speculative-config" in argv:
            config = json.loads(argv[argv.index("--speculative-config") + 1])
            assert config == {
                "method": "dspark",
                "num_speculative_tokens": 7,
                "draft_sample_method": "probabilistic",
            }


def test_missing_checkpoint_fails_without_starting_a_server(tmp_path):
    args = type("Options", (), {"results_dir": tmp_path, "port": 9985})()
    preset = {"key": "missing", "cards": 16, "args": []}
    result = sanity.run_preset(preset, tmp_path / "absent", args, available_cards=16)
    assert result["status"] == "failed"
    assert result["stage"] == "preflight"
    assert "config missing" in result["error"]
    assert not (tmp_path / "missing" / "serve.log").exists()


def test_insufficient_devices_fail_without_starting_a_server(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    args = type("Options", (), {"results_dir": tmp_path, "port": 9985})()
    result = sanity.run_preset(
        {"key": "too-few", "cards": 16, "args": []},
        checkpoint,
        args,
        available_cards=8,
    )
    assert result["status"] == "failed"
    assert "requires 16 cards; got 8" in result["error"]


@pytest.mark.parametrize("answer", ["hello", ""])
def test_real_http_lifecycle_records_generation_and_cleans_up(
    tmp_path, monkeypatch, answer
):
    """A strict HTTP double stands in for the external vLLM server only."""
    with socket.socket() as free_port:
        free_port.bind(("127.0.0.1", 0))
        port = free_port.getsockname()[1]
    server = tmp_path / "server.py"
    server.write_text("""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

assert os.environ.get("VLLM_ENGINE_READY_TIMEOUT_S") == "20"

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        assert self.path in {"/health", "/v1/models"}
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({"data": [{"id": "demo"}]}).encode())

    def do_POST(self):
        assert self.path == "/v1/chat/completions"
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert payload["model"] == "demo" and payload["messages"]
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({
            "model": "demo",
            "choices": [{"message": {"content": sys.argv[2]}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 1},
        }).encode())

HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
""")
    monkeypatch.setattr(
        sanity,
        "serve_command",
        lambda *unused: [sys.executable, str(server), str(port), answer],
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    args = SimpleNamespace(
        results_dir=tmp_path,
        port=port,
        ready_timeout=20,
        request_timeout=5,
        max_tokens=16,
    )
    result = sanity.run_preset(
        {"key": "demo", "cards": 2, "args": []},
        checkpoint,
        args,
        available_cards=2,
    )
    assert result["status"] == ("passed" if answer else "failed")
    assert json.loads((tmp_path / "demo" / "result.json").read_text()) == result
    assert (tmp_path / "demo" / "response-0.json").is_file()
    if answer:
        assert result["requests"] == 2
        assert (tmp_path / "demo" / "response-1.json").is_file()
    else:
        assert result["stage"] == "generation"
        assert "empty final answer" in result["error"]
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", port)) != 0
    # Reuse the same port immediately: TIME_WAIT must not be treated as an
    # active server when advancing to the next preset.
    args.results_dir = tmp_path / "repeat"
    args.results_dir.mkdir()
    repeated = sanity.run_preset(
        {"key": "demo", "cards": 2, "args": []},
        checkpoint,
        args,
        available_cards=2,
    )
    assert repeated["status"] == ("passed" if answer else "failed")
    assert repeated["stage"] == ("complete" if answer else "generation")


def test_listening_port_is_rejected(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        args = SimpleNamespace(results_dir=tmp_path, port=listener.getsockname()[1])
        result = sanity.run_preset(
            {"key": "occupied", "cards": 2, "args": []},
            checkpoint,
            args,
            available_cards=2,
        )
    assert result["status"] == "failed"
    assert result["stage"] == "preflight"
    assert "Address already in use" in result["error"]
    assert not (tmp_path / "occupied" / "serve.log").exists()
