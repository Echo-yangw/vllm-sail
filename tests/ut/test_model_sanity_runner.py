# SPDX-License-Identifier: Apache-2.0
"""Exercise real HTTP, SSE, answer validation and owned-process cleanup on CPU."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.eval import ppu_model_sanity as sanity


def completion(content="42", finish="stop", model="this-run"):
    return {
        "model": model,
        "choices": [{"message": {"content": content}, "finish_reason": finish}],
    }


@pytest.mark.parametrize(
    "response",
    [
        completion(""),
        completion("43"),
        completion("142"),
        completion("42, or perhaps 43"),
        completion("42", "length"),
        completion(None),
        completion(model="previous-run"),
        {"error": {"message": "failed"}},
        {
            "model": "this-run",
            "choices": [{"message": {"reasoning": "42"}, "finish_reason": "stop"}],
        },
    ],
)
def test_rejects_incorrect_incomplete_or_unrelated_answers(response):
    with pytest.raises(ValueError):
        sanity.validate_answer(response, "this-run")


def test_accepts_complete_correct_final_answer():
    assert sanity.validate_answer(completion(" 42\n"), "this-run") == " 42\n"


@pytest.mark.parametrize(
    "content",
    [
        "We compute the sum.</think>42",
        "<think>We compute the sum.</think>42",
        "</think> 42\n",
        "42",
    ],
)
def test_inline_reasoning_uses_only_the_complete_final_answer(content):
    result = sanity.validate_answer(
        completion(content), "this-run", answer_format="inline_think"
    )
    assert result.strip() == "42"


@pytest.mark.parametrize(
    "content",
    [
        "The answer is 42.</think>43",
        "The answer is 42.</think>",
        "<think>42",
        "42</think>42</think>42",
        "Reasoning</think>42 followed by an explanation",
        "<think>nested<think>42</think>42",
    ],
)
def test_inline_reasoning_cannot_hide_wrong_missing_or_ambiguous_finals(content):
    with pytest.raises(ValueError):
        sanity.validate_answer(
            completion(content), "this-run", answer_format="inline_think"
        )


def test_plain_answers_do_not_implicitly_strip_inline_reasoning():
    with pytest.raises(ValueError):
        sanity.validate_answer(completion("Reasoning</think>42"), "this-run")


def test_structured_profiles_preserve_parallel_quantization_and_speculation():
    profiles = sanity.load_profiles()
    assert len(profiles) == 5
    assert profiles["glm5.2-mxfp4-tp8"]["answer_format"] == "inline_think"
    for key, profile in profiles.items():
        command = sanity.build_command(profile, Path("/data"), "this-run", 9985)
        assert command[command.index("--served-model-name") + 1] == "this-run"
        if key.startswith("deepseek"):
            speculative = json.loads(command[command.index("--speculative-config") + 1])
            assert speculative == {
                "method": "dspark",
                "num_speculative_tokens": 7,
                "draft_sample_method": "probabilistic",
            }
        if "dp8" in key:
            assert "--tensor-parallel-size" not in command
            assert command[command.index("--data-parallel-size") + 1] == "8"
            assert (
                command[command.index("--all2all-backend") + 1] == "deepep_low_latency"
            )
            assert (
                json.loads(command[command.index("--compilation-config") + 1])[
                    "cudagraph_mode"
                ]
                == "FULL_DECODE_ONLY"
            )
            assert profile["env"]["VLLM_ENGINE_READY_TIMEOUT_S"] == "3600"
            for flag in (
                "--distributed-timeout-seconds",
                "--cpu-distributed-timeout-seconds",
            ):
                assert command[command.index(flag) + 1] == "1800"
            assert "VLLM_SAIL_PROCESS_GROUP_TIMEOUT_M" not in profile["env"]
        else:
            assert command[command.index("--tensor-parallel-size") + 1] == "8"


@pytest.fixture
def launch_external_double(tmp_path, monkeypatch):
    server_script = Path(__file__).parents[1] / "support" / "sanity_http_server.py"

    def launch(
        mode="ok",
        with_stream=True,
        execution_proof=None,
        inventory=None,
        port=None,
        profile_key=sanity.BASELINE_PROFILE,
    ):
        selected = sanity.load_profiles()[profile_key]
        selected_checkpoint = tmp_path / selected["checkpoint"]
        selected_checkpoint.mkdir(parents=True, exist_ok=True)
        (selected_checkpoint / "config.json").write_text("{}")
        if port is None:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]

        def command(profile, root, name, port):
            return [
                sys.executable,
                str(server_script),
                "--port",
                str(port),
                "--model",
                name,
                "--mode",
                mode,
            ]

        # Replace only the unavailable external program and package inventory.
        # The runner's lifecycle, HTTP and validation implementations execute.
        monkeypatch.setattr(sanity, "build_command", command)
        monkeypatch.setattr(
            sanity,
            "installed_identity",
            lambda directory, env: inventory or {"external_double": True},
        )
        output = tmp_path / (profile_key + "-" + mode + "-run")
        result = sanity.run_sanity(
            profile_key,
            output=output,
            checkpoint_root=tmp_path,
            port=port,
            ready_timeout=3 if mode == "stale" else 10,
            request_timeout=2,
            with_stream=with_stream,
            execution_proof=execution_proof,
        )
        return result, output

    return launch


def test_glm_inline_reasoning_http_and_split_sse_round_trip(launch_external_double):
    result, output = launch_external_double(
        mode="inline-think", profile_key="glm5.2-mxfp4-tp8"
    )
    assert result["status"] == "passed", result
    assert result["answer"] == result["stream_answer"] == "42"
    assert result["answer_format"] == "inline_think"
    ordinary = json.loads((output / "request-0.json").read_text())
    stream = json.loads((output / "stream.json").read_text())
    assert (
        ordinary["choices"][0]["message"]["content"]
        == stream["choices"][0]["message"]["content"]
    )
    assert "</think>" in ordinary["choices"][0]["message"]["content"]


@pytest.fixture
def package_proof(tmp_path):
    proof = {
        "status": "complete-package-byte-maps-match-wheels",
        "commits": {"vllm": "a" * 40, "vllm-sail": "b" * 40},
        "wheel_sha256": {"vllm": "c" * 64, "vllm-sail": "d" * 64},
        "packages": {},
    }
    inventory = {"packages": {}}
    for name, module in (("vllm", "vllm"), ("vllm-sail", "vllm_sail")):
        root = tmp_path / "fixture-site" / module
        root.mkdir(parents=True)
        (root / "__init__.py").write_text("value = 42\n")
        proof["packages"][name] = {
            "root": str(root),
            "version": "0.1",
            "files": {
                module + "/__init__.py": hashlib.sha256(
                    (root / "__init__.py").read_bytes()
                ).hexdigest()
            },
        }
        inventory["packages"][name] = {
            "origin": str(root / "__init__.py"),
            "version": "0.1",
        }
    path = tmp_path / "fixture-proof.json"
    path.write_text(json.dumps(proof))
    return path, proof, inventory


def test_execution_identities_are_bound_before_serving(
    launch_external_double, package_proof
):
    path, proof, inventory = package_proof
    result, output = launch_external_double(execution_proof=path, inventory=inventory)
    assert result["status"] == "passed", result
    assert result["state"] == "passed"
    assert result["case"] == sanity.BASELINE_PROFILE
    assert result["server_exit_after_cleanup"] == 0
    invocation = json.loads((output / "invocation.json").read_text())
    for record in (result, invocation):
        assert record["commits"] == proof["commits"]
        assert record["wheel_sha256"] == proof["wheel_sha256"]
        assert (
            record["execution_proof_sha256"]
            == hashlib.sha256(path.read_bytes()).hexdigest()
        )


def test_changed_installed_bytes_block_launch(launch_external_double, package_proof):
    path, proof, inventory = package_proof
    (Path(proof["packages"]["vllm"]["root"]) / "__init__.py").write_text("changed\n")
    result, output = launch_external_double(execution_proof=path, inventory=inventory)
    assert result["status"] == "failed"
    assert "bytes changed" in result["error"]
    assert not (output / "owned-process-group.json").exists()


@pytest.mark.parametrize(
    "change", ["extra", "missing", "origin", "version", "identity"]
)
def test_rejects_stale_or_mismatched_execution_proof(package_proof, change):
    path, proof, inventory = package_proof
    root = Path(proof["packages"]["vllm"]["root"])
    if change == "extra":
        (root / "stale.so").write_bytes(b"stale")
    elif change == "missing":
        (root / "__init__.py").unlink()
    elif change == "origin":
        inventory["packages"]["vllm"]["origin"] = "/another/install/vllm/__init__.py"
    elif change == "version":
        inventory["packages"]["vllm"]["version"] = "0.2"
    else:
        proof["commits"]["vllm"] = "not-a-full-sha"
        path.write_text(json.dumps(proof))
    with pytest.raises(ValueError):
        sanity.verify_execution_proof(path, inventory)


def test_real_http_and_sse_round_trip_cleanup_and_evidence(launch_external_double):
    result, output = launch_external_double()
    assert result["status"] == "passed", result
    assert result["answer"] == result["stream_answer"] == "42"
    assert result["cleanup"]["group_absent"]
    assert not result["cleanup"]["forced"]
    assert json.loads((output / "result.json").read_text()) == result
    assert (output / "metrics.prom").read_text().strip().endswith("2")
    assert (
        result["metrics_sha256"]
        == hashlib.sha256((output / "metrics.prom").read_bytes()).hexdigest()
    )
    runner = json.loads((output / "runner-process.txt").read_text())
    owned = json.loads((output / "owned-process-group.txt").read_text())
    assert runner["pid"] == os.getpid()
    assert owned["pid"] == owned["pgid"] == result["cleanup"]["pgid"]
    assert runner["observed_unix"] <= owned["observed_unix"]
    if sys.platform.startswith("linux"):
        assert runner["proc_stat"] and owned["proc_stat"]
    assert (output / "stream.sse").read_text().endswith("data: [DONE]\n\n")


@pytest.mark.parametrize(
    "mode",
    [
        "wrong",
        "truncated",
        "http-error",
        "missing-done",
        "stale",
        "log-error",
        "shutdown-error",
    ],
)
def test_failed_serving_attempt_cannot_pass_and_still_cleans_up(
    launch_external_double, mode
):
    result, output = launch_external_double(mode)
    assert result["status"] == "failed", result
    assert result["cleanup"]["group_absent"]
    assert json.loads((output / "result.json").read_text())["status"] == "failed"
    if mode == "shutdown-error":
        assert result["pre_shutdown_errors"] == 0
        assert result["post_shutdown_errors"] == 1
    elif mode == "log-error":
        assert result["post_shutdown_errors"] > 0
    else:
        expected = {
            "wrong": "incorrect final answer",
            "truncated": "finish_reason=stop",
            "http-error": "HTTP Error 500",
            "missing-done": "without [DONE]",
            "stale": "not ready",
        }
        assert expected[mode] in result["error"]


def test_missing_checkpoint_is_a_failure_with_a_durable_result(tmp_path):
    output = tmp_path / "missing"
    result = sanity.run_sanity(
        sanity.BASELINE_PROFILE, output=output, checkpoint_root=tmp_path
    )
    assert result["status"] == "failed"
    assert "missing checkpoint" in result["error"]
    assert json.loads((output / "result.json").read_text()) == result


def test_refuses_to_overwrite_attempt_evidence(tmp_path):
    with pytest.raises(FileExistsError):
        sanity.run_sanity(sanity.BASELINE_PROFILE, output=tmp_path)


def test_busy_port_fails_without_starting_or_signalling_another_service(tmp_path):
    checkpoint = (
        tmp_path / sanity.load_profiles()[sanity.BASELINE_PROFILE]["checkpoint"]
    )
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        result = sanity.run_sanity(
            sanity.BASELINE_PROFILE,
            output=tmp_path / "busy-port",
            checkpoint_root=tmp_path,
            port=listener.getsockname()[1],
        )
        assert result["status"] == "failed"
        assert "Address already in use" in result["error"]
        assert "cleanup" not in result
        assert listener.getsockname()[1] > 0


def test_previous_service_time_wait_does_not_block_next_launch(launch_external_double):
    with socket.socket() as retired:
        retired.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        retired.bind(("127.0.0.1", 0))
        retired.listen()
        address = retired.getsockname()
        with socket.create_connection(address, timeout=2) as client:
            accepted, _ = retired.accept()
            accepted.close()  # Active close leaves the server port in TIME_WAIT.
            assert client.recv(1) == b""
    with (
        socket.socket() as probe,
        pytest.raises(OSError, match="Address already in use"),
    ):
        probe.bind(address)
    result, _ = launch_external_double(port=address[1], with_stream=False)
    assert result["status"] == "passed"
    assert result["cleanup"]["group_absent"]


def test_cli_failure_exits_nonzero_and_lists_profiles_without_runtime_deps(tmp_path):
    script = Path(sanity.__file__)
    listing = subprocess.run(
        [sys.executable, str(script), "--list"], capture_output=True, text=True
    )
    assert listing.returncode == 0
    assert set(listing.stdout.splitlines()) == set(sanity.load_profiles())
    failed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output",
            str(tmp_path / "attempt"),
            "--checkpoint-root",
            str(tmp_path / "absent"),
        ],
        capture_output=True,
        text=True,
    )
    assert failed.returncode == 1
    assert json.loads(failed.stdout)["status"] == "failed"
