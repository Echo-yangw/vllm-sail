# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the upstream-reuse harness's environment probes.

``tests/upstream_reuse`` executes *upstream* vLLM test files as isolated pytest
subprocesses. Two of its outcomes must never be confused:

* the ambient runtime cannot even load upstream's own ``tests/conftest.py``
  (an upstream test-only dependency is missing) -- nothing was collected, no
  vllm-sail code ran, and reporting a regression would be noise, so the entry
  skips with the verbatim pytest error;
* our own manifest narrowed the run into oblivion, or an upstream test module
  drifted -- a real defect that must keep failing.

:func:`tests.upstream_reuse._reuse_harness.conftest_import_error` is the whole
distinction, so it is pinned here rather than only exercised end to end.

Importing the harness pulls in neither torch nor vLLM (its module docstring
promises collection stays CPU-only), which is what lets these tests run on a
bare host.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from tests.upstream_reuse import _reuse_harness as harness
from tests.upstream_reuse._reuse_harness import (
    CONFTEST_IMPORT_MARKER,
    PYTEST_USAGE_ERROR,
    ReuseEntry,
    RunResult,
    UpstreamRunner,
    cached_repo_ids,
    conftest_import_error,
    hub_endpoint,
    hub_repo_ids,
    offline_hub_gap,
    resolve_source_root,
)

#: The failure a container without upstream's test-only dependencies produces,
#: transcribed from a real ZW-890P CI run (exit status 4, nothing collected).
TBLIB_OUTPUT = """\
============================= test session starts ==============================
collected 0 items

ERROR: ImportError while loading conftest '/src/upstream-vllm/tests/conftest.py'.
Traceback (most recent call last):
  File "/src/upstream-vllm/tests/conftest.py", line 7, in <module>
    from tblib import pickling_support
ModuleNotFoundError: No module named 'tblib'
"""


def make_result(
    *,
    returncode: int = PYTEST_USAGE_ERROR,
    collected: int = 0,
    tail: str = TBLIB_OUTPUT,
) -> RunResult:
    """A ``RunResult`` with the fields :func:`conftest_import_error` reads."""
    return RunResult(
        returncode=returncode,
        collected=collected,
        passed=0,
        failures=0,
        errors=0,
        skipped=0,
        failed_names=(),
        tail=tail,
    )


# ---------------------------------------------------------------------------
# conftest_import_error: the case it exists for
# ---------------------------------------------------------------------------
def test_missing_upstream_test_dependency_is_reported_verbatim() -> None:
    """The skip reason must name the module, so the fix is obvious."""
    reason = conftest_import_error(make_result())
    assert reason is not None
    assert CONFTEST_IMPORT_MARKER in reason
    assert "ModuleNotFoundError: No module named 'tblib'" in reason
    assert "/src/upstream-vllm/tests/conftest.py" in reason


def test_reason_quotes_the_conftest_path_that_failed() -> None:
    """A conftest deeper in the tree is named, not just the top-level one."""
    tail = (
        "ERROR: ImportError while loading conftest "
        "'/src/upstream-vllm/tests/kernels/conftest.py'.\n"
        "ImportError: cannot import name 'thing'\n"
    )
    reason = conftest_import_error(make_result(tail=tail))
    assert reason is not None
    assert "tests/kernels/conftest.py" in reason
    assert "ImportError: cannot import name 'thing'" in reason


def test_reason_is_bounded_even_for_a_long_traceback() -> None:
    """A skip reason goes into ``-rs`` output; it must stay readable."""
    noise = "\n".join(f'  File "/x/mod{i}.py", line {i}, in f{i}' for i in range(80))
    tail = (
        f"ERROR: {CONFTEST_IMPORT_MARKER} '/src/tests/conftest.py'.\n{noise}\n"
        "ModuleNotFoundError: No module named 'tblib'\n"
    )
    reason = conftest_import_error(make_result(tail=tail))
    assert reason is not None
    assert reason.count('File "/x/mod') <= 8
    # The actual cause survives the truncation, which is the point of quoting
    # the trailing exception line separately from the first few.
    assert "No module named 'tblib'" in reason


# ---------------------------------------------------------------------------
# conftest_import_error: what it must never absorb
# ---------------------------------------------------------------------------
def test_a_bad_manifest_marker_expression_still_fails() -> None:
    """Exit 4 from our own ``-m`` typo is a manifest bug, not an environment gap.

    pytest reports it without the conftest marker, so the entry keeps failing
    and the typo gets fixed instead of silently skipped forever.
    """
    tail = (
        "Usage: pytest [options] [file_or_dir] [file_or_dir] [...]\n"
        "ERROR: Wrong expression passed to '-m': 'cpu and ('. "
        "Please verify.\n"
    )
    assert conftest_import_error(make_result(tail=tail)) is None


def test_a_test_module_import_error_is_not_a_conftest_error() -> None:
    """An upstream *test* file that fails to import is drift, and stays a failure.

    Even if pytest happens to exit with a usage status, the tell is the absent
    conftest marker: this is ``tests/basic/test_sampling.py`` that broke, not
    the runtime, so the manifest or the pin needs review.
    """
    tail = (
        "collected 0 items / 1 errors\n"
        "ERROR tests/basic/test_sampling.py\n"
        "= ERRORS =\n"
        "E   ModuleNotFoundError: No module named 'some_upstream_thing'\n"
        "!!!! Interrupted: 1 error during collection !!!!\n"
    )
    assert conftest_import_error(make_result(tail=tail)) is None


def test_a_clean_run_is_never_reclassified() -> None:
    """Green subprocesses carry no environment excuse."""
    tail = "collected 12 items\n............\n12 passed in 3.21s\n"
    assert conftest_import_error(
        make_result(returncode=0, collected=12, tail=tail)
    ) is (None)


def test_a_failing_run_is_never_reclassified() -> None:
    """Real upstream regressions keep failing even though output mentions errors."""
    tail = (
        "collected 4 items\n.F.F\n2 failed, 2 passed in 1.00s\n"
        f"{CONFTEST_IMPORT_MARKER} is unrelated noise printed by a test\n"
    )
    result = make_result(returncode=1, collected=4, tail=tail)
    assert conftest_import_error(result) is None


def test_a_usage_error_that_collected_tests_is_not_an_environment_gap() -> None:
    """``collected`` wins: tests ran, so the runtime plainly works."""
    result = make_result(collected=3)
    assert conftest_import_error(result) is None


def test_empty_output_is_not_an_environment_gap() -> None:
    """A subprocess killed before printing anything has no evidence to quote."""
    assert conftest_import_error(make_result(tail="")) is None


def test_the_marker_alone_without_an_exception_line_is_still_enough() -> None:
    """pytest always prints the marker; the cause line is a bonus."""
    tail = f"ERROR: {CONFTEST_IMPORT_MARKER} '/src/tests/conftest.py'.\n"
    reason = conftest_import_error(make_result(tail=tail))
    assert reason is not None
    assert reason.startswith("ERROR:")


# ---------------------------------------------------------------------------
# resolve_source_root
# ---------------------------------------------------------------------------
def test_resolve_source_root_rejects_an_unset_variable() -> None:
    assert resolve_source_root(None) is None
    assert resolve_source_root("") is None


def test_resolve_source_root_rejects_a_checkout_without_vllm(tmp_path: Path) -> None:
    """A half-fetched clone must not be mistaken for a usable source root.

    A ``git clone`` interrupted after the pack download leaves ``.git`` and an
    empty ``vllm/`` behind; without this probe every AST-based test would fail
    on ``FileNotFoundError`` instead of skipping.
    """
    (tmp_path / "vllm").mkdir()
    assert resolve_source_root(str(tmp_path)) is None


def test_resolve_source_root_accepts_a_real_checkout(tmp_path: Path) -> None:
    (tmp_path / "vllm").mkdir()
    (tmp_path / "vllm" / "__init__.py").write_text("", encoding="utf-8")
    assert resolve_source_root(str(tmp_path)) == tmp_path.resolve()


@pytest.mark.parametrize("raw", ["~/does-not-exist-upstream", "/definitely/not/here"])
def test_resolve_source_root_rejects_missing_paths(raw: str) -> None:
    assert resolve_source_root(raw) is None


# ---------------------------------------------------------------------------
# The hub guard: an air-gapped runner must skip, not hang
# ---------------------------------------------------------------------------
#: Transcribed from the abort of a real ZW-890P run of
#: ``tests/config/test_multimodal_config.py``, which blocked in
#: ``socket.create_connection`` under ``vllm/transformers_utils/repo_utils.py``.
MULTIMODAL_SOURCE = """\
from tests.models.utils import build_model_context
from vllm.config.model import ModelConfig

@pytest.mark.parametrize("model_id", [
    ("Qwen/Qwen2-0.5B-Instruct", {}, False),
    ("Qwen/Qwen2.5-VL-3B-Instruct", {}, True),
])
def test_supports_multimodal_inputs(model_id):
    ctx = build_model_context(model_id)

def test_language_model_only_affects_model_hash():
    model = "llava-hf/llava-1.5-7b-hf"
    base_hash = ModelConfig(model).compute_hash()

def test_mm_encoder_attn_backend_str_conversion():
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    assert "tests/config/test_multimodal_config.py".endswith(".py")
    assert open("config.json").read()
"""

#: A module that never resolves a checkpoint: pure logic, no repo ids.
OFFLINE_SAFE_SOURCE = """\
def test_sampler_rejects_empty_batches():
    assert "vllm/v1/sample" != "tests/v1/sample"
"""


def make_entry(path: str, **overrides: object) -> ReuseEntry:
    """A manifest row with only ``path`` worth varying in these tests."""
    fields: dict[str, object] = {
        "id": Path(path).stem,
        "path": path,
        "mode": "direct",
        "chips": ("cpu",),
        "category": "core",
        "reason": "",
        "min_tests": 1,
        "markers": "",
        "keywords": "",
        "deselect": (),
        "env": {},
        "xfail": (),
        "skip_reason": "",
    }
    fields.update(overrides)
    return ReuseEntry(**fields)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def no_probe_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the session-scoped hub probe from leaking between tests."""
    monkeypatch.setattr(harness, "_HUB_PROBE_CACHE", {})
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)


@pytest.fixture
def upstream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake checkout holding the two modules above, with an empty hub cache."""
    root = tmp_path / "upstream"
    (root / "tests" / "config").mkdir(parents=True)
    (root / "tests" / "config" / "test_multimodal_config.py").write_text(
        MULTIMODAL_SOURCE, encoding="utf-8"
    )
    (root / "tests" / "v1" / "sample").mkdir(parents=True)
    (root / "tests" / "v1" / "sample" / "test_sampler.py").write_text(
        OFFLINE_SAFE_SOURCE, encoding="utf-8"
    )
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    return root


def test_hub_repo_ids_finds_checkpoint_literals() -> None:
    """The ids upstream passes to ``build_model_context`` / ``ModelConfig``."""
    assert hub_repo_ids(MULTIMODAL_SOURCE) == (
        "Qwen/Qwen2-0.5B-Instruct",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "llava-hf/llava-1.5-7b-hf",
    )


def test_hub_repo_ids_ignores_paths_and_filenames() -> None:
    """``tests/config/...`` and ``config.json`` are not checkpoints.

    Without that filter every import string in upstream would look like a repo
    id and the guard would skip the whole whitelist.
    """
    assert "tests/config" not in hub_repo_ids(MULTIMODAL_SOURCE)
    assert "vllm/v1" not in hub_repo_ids(MULTIMODAL_SOURCE)
    assert hub_repo_ids('open("config.json")') == ()
    assert hub_repo_ids('from "tests/models/utils.py" import x') == ()


def test_hub_repo_ids_dedupes_preserving_order() -> None:
    source = 'a = "facebook/opt-125m"\nb = "facebook/opt-125m"\nc = "Qwen/Qwen3-0.6B"'
    assert hub_repo_ids(source) == ("facebook/opt-125m", "Qwen/Qwen3-0.6B")


def test_hub_repo_ids_of_a_module_that_resolves_nothing() -> None:
    assert hub_repo_ids(OFFLINE_SAFE_SOURCE) == ()


def test_cached_repo_ids_reads_the_models_layout(tmp_path: Path) -> None:
    """``models--<org>--<name>`` is how ``huggingface_hub`` stores a snapshot."""
    cache = tmp_path / "hub"
    (cache / "models--facebook--opt-125m" / "snapshots").mkdir(parents=True)
    (cache / "models--Qwen--Qwen2-0.5B-Instruct").mkdir(parents=True)
    (cache / "not-a-model").mkdir()
    (cache / "models--incomplete").mkdir()
    assert cached_repo_ids(cache) == frozenset(
        {"facebook/opt-125m", "Qwen/Qwen2-0.5B-Instruct"}
    )


def test_cached_repo_ids_of_a_missing_cache_is_empty(tmp_path: Path) -> None:
    assert cached_repo_ids(tmp_path / "nope") == frozenset()


def test_hub_endpoint_honours_hf_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """A mirror is what an air-gapped site would point the hub at."""
    assert hub_endpoint() == ("huggingface.co", 443)
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.internal:8443")
    assert hub_endpoint() == ("hf-mirror.internal", 8443)
    monkeypatch.setenv("HF_ENDPOINT", "http://hf-mirror.internal")
    assert hub_endpoint() == ("hf-mirror.internal", 80)


def test_an_offline_switch_needs_no_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``HF_HUB_OFFLINE=1`` already says the answer; dialling would be a lie."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the probe must not touch the network here")

    monkeypatch.setattr(harness.socket, "create_connection", refuse)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    reachable, reason = harness.hub_reachable()
    assert not reachable
    assert "HF_HUB_OFFLINE" in reason


def test_the_probe_runs_once_per_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Twenty entries share one connect attempt, so a dead route costs 5 s total."""
    calls: list[object] = []

    def blocked(*args: object, **kwargs: object) -> socket.socket:
        calls.append(args)
        raise OSError(101, "Network is unreachable")

    monkeypatch.setattr(harness.socket, "create_connection", blocked)
    first = harness.hub_reachable()
    second = harness.hub_reachable()
    assert first == second
    assert not first[0]
    assert "Network is unreachable" in first[1]
    assert len(calls) == 1


def test_the_guard_is_inert_when_the_hub_answers(
    upstream: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CI has a network, so CI runs every entry -- no coverage is given up."""
    monkeypatch.setattr(harness, "hub_reachable", lambda *a, **k: (True, ""))
    entry = make_entry("tests/config/test_multimodal_config.py")
    assert offline_hub_gap(entry, upstream) is None


def test_the_guard_names_the_checkpoint_and_the_probe(
    upstream: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The skip reason must carry the evidence, not just the verdict."""
    monkeypatch.setattr(
        harness,
        "hub_reachable",
        lambda *a, **k: (False, "cannot reach huggingface.co:443 (OSError: no route)"),
    )
    entry = make_entry("tests/config/test_multimodal_config.py")
    reason = offline_hub_gap(entry, upstream)
    assert reason is not None
    assert "Qwen/Qwen2-0.5B-Instruct" in reason
    assert "llava-hf/llava-1.5-7b-hf" in reason
    assert "no route" in reason


def test_the_guard_lets_a_cached_checkpoint_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Offline and warm is a runnable combination, so it must not be skipped."""
    root = tmp_path / "upstream"
    (root / "tests" / "config").mkdir(parents=True)
    (root / "tests" / "config" / "test_multimodal_config.py").write_text(
        'model = "facebook/opt-125m"\n', encoding="utf-8"
    )
    cache = tmp_path / "hub"
    (cache / "models--facebook--opt-125m").mkdir(parents=True)
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(harness, "hub_reachable", lambda *a, **k: (False, "offline"))
    entry = make_entry("tests/config/test_multimodal_config.py")
    assert offline_hub_gap(entry, root) is None


def test_the_guard_lets_an_entry_without_checkpoints_run(
    upstream: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A module that resolves nothing cannot be blamed on the network.

    If it fails offline, that failure is real and must stay a failure.
    """
    monkeypatch.setattr(harness, "hub_reachable", lambda *a, **k: (False, "offline"))
    entry = make_entry("tests/v1/sample/test_sampler.py")
    assert offline_hub_gap(entry, upstream) is None


def test_the_guard_does_not_excuse_a_missing_upstream_file(
    upstream: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pin drift is reported as an error by ``run``, never softened to a skip."""
    monkeypatch.setattr(harness, "hub_reachable", lambda *a, **k: (False, "offline"))
    entry = make_entry("tests/config/test_deleted_upstream.py")
    assert offline_hub_gap(entry, upstream) is None


def test_the_runner_decides_before_spawning_a_subprocess(
    upstream: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point is to avoid ``RUN_TIMEOUT_SECONDS`` of blocked wall clock."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("no subprocess may be started for a hub-blocked entry")

    monkeypatch.setattr(harness.subprocess, "run", refuse)
    monkeypatch.setattr(harness, "hub_reachable", lambda *a, **k: (False, "offline"))
    result = UpstreamRunner(upstream).run(
        make_entry("tests/config/test_multimodal_config.py")
    )
    assert result.environment_gap
    assert result.collected == 0
    assert result.errors == 0
    assert result.tail == result.environment_gap


def passing_result(**changes: object) -> RunResult:
    result = RunResult(0, 2, 2, 0, 0, 0, (), "")
    return result._replace(**changes)


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"returncode": 5, "collected": 0, "passed": 0}, "status 5"),
        ({"passed": 0, "skipped": 2}, "no upstream test passed"),
        ({"errors": 1}, "import/collection errors"),
        ({"returncode": 1}, "without matching test evidence"),
        ({"environment_gap": "missing tokenizer cache"}, "missing tokenizer cache"),
        ({"skipped": 1}, "lack matching node/reason evidence"),
    ],
)
def test_strict_results_fail_without_real_coverage(changes, message):
    result = passing_result(**changes)
    error = harness.strict_result_error(make_entry("tests/test_example.py"), result)
    assert error is not None and message in error


def test_strict_missing_dependency_is_not_a_successful_skip():
    error = harness.strict_result_error(
        make_entry("tests/test_example.py"), make_result()
    )
    assert error is not None and "tblib" in error


def test_strict_clean_execution_passes():
    assert (
        harness.strict_result_error(
            make_entry("tests/test_example.py"), passing_result()
        )
        is None
    )


def test_strict_collection_minimum_is_retained():
    error = harness.strict_result_error(
        make_entry("tests/test_example.py", min_tests=3), passing_result()
    )
    assert error is not None and "expected at least 3" in error


def test_strict_allowed_skip_requires_node_reason_and_bounded_count():
    entry = make_entry(
        "tests/test_example.py",
        allowed_skips=(
            {"nodeid": "tests.example::test_*", "reason": "capability", "max_count": 1},
        ),
    )
    result = passing_result(
        collected=2,
        passed=1,
        skipped=1,
        skipped_cases=(
            ("tests.example::test_shape", "missing capability", "pytest.skip"),
        ),
    )
    assert harness.strict_result_error(entry, result) is None
    different = result._replace(
        skipped_cases=(
            ("tests.example::test_shape", "missing dependency", "pytest.skip"),
        )
    )
    assert "undeclared" in harness.strict_result_error(entry, different)
    overflow = result._replace(
        collected=3,
        skipped=2,
        skipped_cases=result.skipped_cases + result.skipped_cases,
    )
    assert "count exceeded" in harness.strict_result_error(entry, overflow)


def test_strict_all_skip_fails_even_if_skip_rules_match():
    entry = make_entry(
        "tests/test_example.py",
        allowed_skips=({"nodeid": "*", "reason": "capability", "max_count": 2},),
    )
    result = passing_result(
        passed=0,
        skipped=2,
        skipped_cases=(
            ("test_a", "capability", "pytest.skip"),
            ("test_b", "capability", "pytest.skip"),
        ),
    )
    assert "no upstream test passed" in harness.strict_result_error(entry, result)


def test_strict_declared_xfail_does_not_allow_an_ordinary_skip():
    entry = make_entry("tests/test_example.py", mode="adapted", xfail=("known",))
    result = passing_result(
        passed=1,
        skipped=1,
        skipped_cases=(("tests.example::test_known", "known bug", "pytest.xfail"),),
    )
    assert harness.strict_result_error(entry, result) is None
    ordinary = result._replace(
        skipped_cases=(("tests.example::test_known", "missing library", "pytest.skip"),)
    )
    assert "undeclared" in harness.strict_result_error(entry, ordinary)
    assert "import/collection" in harness.strict_result_error(
        entry, result._replace(errors=1)
    )


def test_strict_declared_failure_still_requires_a_passing_case():
    entry = make_entry("tests/test_example.py", mode="adapted", xfail=("known",))
    result = passing_result(
        returncode=1, passed=1, failures=1, failed_names=("test_known",)
    )
    assert harness.strict_result_error(entry, result) is None
    assert "no upstream test passed" in harness.strict_result_error(
        entry, result._replace(passed=0)
    )
    assert "unexpected" in harness.strict_result_error(
        entry, result._replace(failed_names=("test_other",))
    )


def test_junit_preserves_skip_evidence(tmp_path):
    report = tmp_path / "report.xml"
    report.write_text(
        '<testsuites><testsuite tests="3" skipped="1">'
        '<testcase classname="tests.example" name="test_pass"/>'
        '<testcase classname="tests.example" name="test_skip">'
        '<skipped type="pytest.skip" message="unsupported capability"/></testcase>'
        '<testcase classname="tests.example" name="test_fail"><failure/></testcase>'
        "</testsuite></testsuites>"
    )
    collected, failures, errors, skipped, names, cases = harness._parse_junit(report)
    assert (collected, failures, errors, skipped) == (3, 1, 0, 1)
    assert names == ["tests.example::test_fail"]
    assert cases == [
        ("tests.example::test_skip", "unsupported capability", "pytest.skip")
    ]


def test_strict_report_rejects_unexecuted_entries_and_separates_parked(tmp_path):
    entry = make_entry("tests/test_example.py")
    parked = make_entry(
        "tests/test_parked.py", skip_reason="offline tokenizer cache is not provisioned"
    )
    runner = UpstreamRunner(tmp_path, strict=True, report_dir=tmp_path / "reports")
    report = runner.write_report([entry, parked])
    assert not report["qualified"]
    assert report["active_entries"] == 1
    assert [row["status"] for row in report["entries"]] == ["not_run", "parked"]
    runner.complete(entry, passing_result())
    report = runner.write_report([entry, parked])
    assert report["qualified"] and report["passed_entries"] == 1
    assert json.loads((tmp_path / "reports/report.json").read_text()) == report
    assert "810e" in report["deferred_targets"]


def test_local_optional_results_never_establish_ci_qualification(tmp_path):
    entry = make_entry("tests/test_example.py")
    runner = UpstreamRunner(tmp_path)
    runner.complete(entry, passing_result())
    assert not runner.write_report([entry])["qualified"]


def test_report_filters_entries_by_actual_target(tmp_path):
    cpu = make_entry("tests/test_cpu.py")
    m890 = make_entry("tests/test_m890.py", chips=("890p",))
    runner = UpstreamRunner(tmp_path, strict=True, execution_target="890p")
    runner.complete(m890, passing_result())
    report = runner.write_report([cpu, m890])
    assert report["qualified"] and report["active_entries"] == 1
    assert report["entries"][0]["status"] == "not_applicable"


def test_strict_environment_gap_is_recorded_before_failing(
    upstream, tmp_path, monkeypatch
):
    monkeypatch.setattr(harness, "hub_reachable", lambda *a, **k: (False, "offline"))
    runner = UpstreamRunner(upstream, strict=True, report_dir=tmp_path / "reports")
    entry = make_entry("tests/config/test_multimodal_config.py")
    with pytest.raises(AssertionError, match="Hugging Face"):
        runner.run(entry)
    report = runner.write_report([entry])
    assert not report["qualified"] and report["entries"][0]["status"] == "failed"


def test_strict_runtime_view_cannot_shadow_installed_packages(upstream, monkeypatch):
    (upstream / "vllm").mkdir()
    (upstream / "vllm/__init__.py").write_text(
        "raise RuntimeError('source package imported')"
    )
    monkeypatch.setenv("PYTHONPATH", str(upstream))
    runner = UpstreamRunner(upstream, strict=True, execution_target="890p")
    entry = make_entry(
        "tests/v1/sample/test_sampler.py", env={"PYTHONPATH": str(upstream)}
    )
    try:
        root = runner.runtime_root()
        assert (root / entry.path).is_file()
        assert not (root / "vllm").exists()
        env = runner.build_env(entry)
        assert env["PYTHONPATH"] == str(root)
        assert env["VLLM_UPSTREAM_REUSE_TARGET"] == "890p"
        command = runner.build_command(entry, root / "report.xml")
        assert command[:2] == [harness.sys.executable, "-c"]
        assert "installed wheel" in command[2]
    finally:
        runner.close()


def test_reviewed_apc_skip_retains_source_location_and_count_limit(tmp_path):
    entry = next(e for e in harness.load_manifest() if e.id == "v1_sample_logprobs")
    report = tmp_path / "report.xml"
    report.write_text(
        '<testsuites><testsuite tests="2" skipped="1">'
        '<testcase name="test_pass"/>'
        '<testcase classname="tests.v1.sample.test_logprobs" '
        'name="test_get_logprobs_and_prompt_logprobs[True-0.0-NONE]">'
        '<skipped type="pytest.skip" message="Skipped">'
        "/runtime/tests/v1/sample/test_logprobs.py:360: Skipped"
        "</skipped></testcase></testsuite></testsuites>"
    )
    collected, failures, errors, skipped, names, cases = harness._parse_junit(report)
    result = RunResult(
        0,
        collected,
        1,
        failures,
        errors,
        skipped,
        tuple(names),
        "",
        skipped_cases=tuple(cases),
    )
    assert harness.strict_result_error(entry, result) is None
    overflow = result._replace(
        collected=9, skipped=8, skipped_cases=result.skipped_cases * 8
    )
    assert "count exceeded" in harness.strict_result_error(entry, overflow)
    wrong_reason = result._replace(
        skipped_cases=((cases[0][0], "missing library", "pytest.skip"),)
    )
    assert "undeclared" in harness.strict_result_error(entry, wrong_reason)


def test_strict_runner_retains_xml_and_logs(upstream, tmp_path, monkeypatch):
    entry = make_entry("tests/v1/sample/test_sampler.py")

    def execute(command, **kwargs):
        xml = Path(
            next(
                arg.split("=", 1)[1] for arg in command if arg.startswith("--junitxml=")
            )
        )
        xml.write_text(
            '<testsuites><testsuite tests="1" skipped="0"><testcase name="test_pass"/></testsuite></testsuites>'
        )
        assert kwargs["cwd"] != str(upstream)
        return harness.subprocess.CompletedProcess(command, 0, "executed", "")

    monkeypatch.setattr(harness, "hub_reachable", lambda *a, **k: (True, ""))
    monkeypatch.setattr(harness.subprocess, "run", execute)
    runner = UpstreamRunner(upstream, strict=True, report_dir=tmp_path / "reports")
    try:
        result = runner.run(entry)
        assert result.passed == 1
        assert (tmp_path / f"reports/{entry.id}.xml").is_file()
        assert (tmp_path / f"reports/{entry.id}.log").read_text() == "executed\n"
    finally:
        runner.close()
