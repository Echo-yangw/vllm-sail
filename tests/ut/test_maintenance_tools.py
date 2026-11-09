# SPDX-License-Identifier: Apache-2.0
"""Maintenance CLI contracts, without loading vLLM or a device toolchain."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

from tools import _common, check_patch_drift, patch_manifest
from tools import port_upstream_kernels as upstream


def test_patch_report_is_sorted_and_escapes_cells():
    def record(target, reason):
        return SimpleNamespace(
            target=target,
            kind="module",
            reason=reason,
            affected_versions=">=1,<2",
            remove_when="upstream adds hook",
        )

    report = patch_manifest.render(
        [record("vllm.z.run", "why | this\nnext"), record("vllm.a.run", "simple")]
    )
    assert report.index("vllm.a.run") < report.index("vllm.z.run")
    assert "why &#124; this<br>next" in report
    assert "actual installation results" in report
    assert "do not cover every conditional" in report
    assert "| Target |" in patch_manifest.render([])


def test_patch_report_defaults_to_stdout_and_saves_only_explicit_paths(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(patch_manifest, "load_patch_records", lambda command: [])
    monkeypatch.chdir(tmp_path)
    assert patch_manifest.main([]) == 0
    expected = capsys.readouterr().out
    assert expected.startswith("# Runtime patch manifest")
    assert list(tmp_path.iterdir()) == []
    destination = tmp_path / "report.md"
    assert patch_manifest.main(["--output", str(destination)]) == 0
    assert destination.read_text() == expected
    assert capsys.readouterr().out == ""
    assert patch_manifest.main(["--output", str(tmp_path / "absent/report.md")]) == 2
    assert "cannot write report" in capsys.readouterr().err


@pytest.mark.parametrize("state", ["match", "missing", "drift"])
def test_port_check_reports_drift(tmp_path, monkeypatch, capsys, state):
    destination = tmp_path / "committed"
    source = tmp_path / "source"
    (source / "csrc").mkdir(parents=True)
    if state != "missing":
        (destination / "detail").mkdir(parents=True)
        (destination / "detail/kernel.cu").write_text(
            "old" if state == "drift" else "new"
        )
        (destination / "manifest.toml").write_text("handwritten input")
    if state == "drift":
        (destination / "stale.h").write_text("stale")

    def generate(fresh, manifest):
        (fresh / "detail").mkdir(parents=True)
        (fresh / "detail/kernel.cu").write_text("new")
        if state == "drift":
            (fresh / "added.h").write_text("added")
            # Equal size and timestamp must not hide a content change.
            stamp = (destination / "detail/kernel.cu").stat().st_mtime_ns
            os.utime(fresh / "detail/kernel.cu", ns=(stamp, stamp))

    monkeypatch.setattr(upstream, "DEST", destination)
    monkeypatch.setattr(upstream, "_check_ref", lambda *args: None)
    monkeypatch.setattr(
        upstream.manifest_reader,
        "load",
        lambda: SimpleNamespace(upstream_ref="test"),
    )
    monkeypatch.setattr(
        upstream.porting, "port", lambda src, dest, m: generate(dest, m)
    )
    arguments = ["--vllm-src", str(source), "--check"]

    assert upstream.main(arguments) == (0 if state == "match" else 1)
    output = capsys.readouterr()
    if state == "match":
        assert "matches" in output.out
        assert not output.err
    elif state == "missing":
        assert f"{destination} does not exist" in output.err
    else:
        expected = ["only in a fresh port: added.h"]
        expected += ["only in the tree: stale.h", "differs: detail/kernel.cu"]
        assert output.err.splitlines()[1:] == [f"  {line}" for line in expected]
        assert (destination / "detail/kernel.cu").read_text() == "old"


@pytest.mark.parametrize(
    ("command", "label"),
    [(patch_manifest, "patch manifest"), (check_patch_drift, "patch drift checking")],
)
def test_patch_commands_name_missing_dependencies(monkeypatch, capsys, command, label):
    monkeypatch.setattr(_common.importlib.util, "find_spec", lambda name: None)
    assert command.main([]) == 2
    assert capsys.readouterr().err.startswith(
        f"ERROR: {label} requires a real vLLM environment; missing torch, vllm."
    )


@pytest.mark.parametrize(
    "failure", [None, ImportError("SDK unavailable"), OSError("ABI")]
)
def test_patch_loader_keeps_errors_and_returns_a_snapshot(monkeypatch, failure):
    monkeypatch.setattr(_common.importlib.util, "find_spec", lambda name: object())
    records = [object()]
    calls = []

    def register():
        calls.append("register")
        if failure:
            raise failure

    monkeypatch.setitem(
        sys.modules, "vllm_sail", SimpleNamespace(register_out_of_tree=register)
    )
    monkeypatch.setitem(
        sys.modules, "vllm_sail.patch", SimpleNamespace(PATCH_REGISTRY=records)
    )
    if failure:
        with pytest.raises(_common.PatchEnvironmentError) as error:
            _common.load_patch_records("test")
        assert error.value.__cause__ is failure
    else:
        loaded = _common.load_patch_records("test")
        assert loaded == records
        assert loaded is not records
    assert calls == ["register"]
