# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "scripts" / "ci" / "ppu_e2e_smoke.sh"
ACCURACY = ROOT / "scripts" / "ci" / "ppu_e2e_accuracy.sh"
RUNTIME = ROOT / "scripts" / "ci" / "ppu_runtime_container.sh"
SELECT = ROOT / "scripts" / "ci" / "ppu_e2e_select.py"
MODELS = ROOT / "scripts" / "ci" / "ppu_e2e_models.json"
EVAL_SCRIPT = ROOT / "scripts" / "eval" / "ppu_gsm8k_eval.py"
GOLDEN = ROOT / "tests" / "e2e" / "models" / "_golden"
MODEL_WORKFLOW = ROOT / ".github" / "workflows" / "_ppu-model-tests.yaml"
ACCURACY_WORKFLOW = ROOT / ".github" / "workflows" / "_ppu-accuracy-tests.yaml"
NIGHTLY_WORKFLOW = ROOT / ".github" / "workflows" / "ppu-accuracy-nightly.yaml"
KERNEL_PERF_WORKFLOW = ROOT / ".github" / "workflows" / "_ppu-kernel-perf.yaml"
KERNEL_PERF_LABEL_WORKFLOW = ROOT / ".github" / "workflows" / "kernel-perf-label.yaml"
PERF_REGRESSION_WORKFLOW = ROOT / ".github" / "workflows" / "perf-regression.yaml"
PERF_REGRESSION_SCRIPT = ROOT / "scripts" / "ci" / "perf_regression.sh"
MODEL_PERF_TEST = ROOT / "tests" / "e2e" / "perf" / "test_model_perf.py"
DEBUG_WORKFLOW = ROOT / ".github" / "workflows" / "ppu-debug.yaml"


def _load_select():
    spec = importlib.util.spec_from_file_location("ppu_e2e_select", SELECT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_evaluator():
    """Import ``ppu_gsm8k_eval`` so the catalog can be checked against it."""
    spec = importlib.util.spec_from_file_location("ppu_gsm8k_eval_catalog", EVAL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sentinel = "GSM8K_DATASET_PATH" not in os.environ
    if sentinel:
        os.environ["GSM8K_DATASET_PATH"] = "/unused/gsm8k"
    try:
        spec.loader.exec_module(module)
    finally:
        if sentinel:
            del os.environ["GSM8K_DATASET_PATH"]
    return module


def _catalog() -> list[dict]:
    return json.loads(MODELS.read_text())


def _workflow(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


# --- model catalog -----------------------------------------------------------


def test_models_catalog_is_partitioned_by_category_and_contract() -> None:
    catalog = _catalog()
    assert isinstance(catalog, list) and catalog
    common = {"key", "served_name", "board", "ppu_count"}
    for entry in catalog:
        assert common <= set(entry), entry
        category = entry.get("category", "smoke")
        assert category in {"smoke", "accuracy"}
        assert "model_id" in entry and "checkpoint" not in entry
    keys = [entry["key"] for entry in catalog]
    assert len(keys) == len(set(keys)), "model keys must be unique"


def test_catalog_keeps_smoke_and_accuracy_subsets_exact() -> None:
    by_category = {
        category: {
            entry["key"]
            for entry in _catalog()
            if entry.get("category", "smoke") == category
        }
        for category in ("smoke", "accuracy")
    }
    assert by_category["smoke"] == {"qwen3.8-27b", "qwen3.8-27b-m890p"}
    assert by_category["accuracy"] == {
        "deepseek-v4-int8",
        "deepseek-v4-mxfp4",
        "minimax-m3-fp8",
        "deepseek-v4-int8-dp-ep",
        "deepseek-v4-mxfp4-dp-ep",
        "glm-5.3-mxfp4-tp4",
        "glm-5.3-flash-mxfp4-tp2",
        "kimi-k2.6-mxfp4-tp8",
        "qwen3.8-flash-next-tp4",
        "qwen3.8-flash-next-dp-ep",
    }


@pytest.mark.parametrize(
    ("key", "board"),
    [
        ("qwen3.8-27b", "OAM-810E"),
        ("qwen3.8-27b-m890p", "ZW-M890P"),
    ],
)
def test_qwen_smoke_catalog_targets_both_boards(key: str, board: str) -> None:
    qwen = {entry["key"]: entry for entry in _catalog()}[key]
    assert qwen["model_id"] == "Qwen/Qwen3.8-27B"
    assert qwen["board"] == board
    selected = _load_select().build_matrix(_catalog(), key)["include"][0]
    assert selected["checkpoint"] == selected["model_id"]
    assert int(qwen["ppu_count"]) == 2
    assert qwen["served_name"] == "qwen3.8-27b"
    assert qwen["extra_serve_flags"] == "--trust-remote-code"


def test_accuracy_catalog_keeps_model_ids_and_per_model_card_count() -> None:
    accuracy = {
        entry["key"]: entry for entry in _catalog() if entry["category"] == "accuracy"
    }
    assert accuracy["deepseek-v4-int8"]["model_id"] == (
        "T-HEAD/DeepSeek-V4-Flash-0731-W8A8-INT8"
    )
    assert accuracy["deepseek-v4-int8"]["board"] == "OAM-810E"
    assert accuracy["deepseek-v4-mxfp4"]["board"] == "ZW-M890P"
    # The card count handed to the container is per case, not per chip:
    # GLM-5.3 needs a TP4 prefill recipe and GLM-5.3-Flash a TP2 one, while
    # every DeepSeek/Kimi case takes all eight cards.  A DP+EP case splits
    # those cards across data parallel ranks, so this is not its TP size.
    assert accuracy["glm-5.3-mxfp4-tp4"]["ppu_count"] == 4
    assert accuracy["glm-5.3-flash-mxfp4-tp2"]["ppu_count"] == 2
    # Qwen3.8-Flash-Next keeps every card in a single tensor parallel group.
    assert accuracy["qwen3.8-flash-next-tp4"]["ppu_count"] == 4
    others = {
        key: entry["ppu_count"]
        for key, entry in accuracy.items()
        if key
        not in {
            "glm-5.3-mxfp4-tp4",
            "glm-5.3-flash-mxfp4-tp2",
            "qwen3.8-flash-next-tp4",
        }
    }
    assert set(others.values()) == {8}, others


# --- matrix selector ---------------------------------------------------------


def test_selector_defaults_to_smoke_and_legacy_entries_are_smoke() -> None:
    select = _load_select()
    matrix = select.build_matrix(_catalog(), "")
    assert [entry["key"] for entry in matrix["include"]] == [
        "qwen3.8-27b",
        "qwen3.8-27b-m890p",
    ]
    legacy = [{"key": "legacy", "checkpoint": "/tmp/model"}]
    assert select.build_matrix(legacy, "")["include"] == legacy


def test_selector_filters_smoke_without_leaking_accuracy() -> None:
    select = _load_select()
    matrix = select.build_matrix(_catalog(), "deepseek-v4-int8,qwen3.8-27b-m890p")
    assert [entry["key"] for entry in matrix["include"]] == ["qwen3.8-27b-m890p"]


def test_smoke_matrix_answers_the_default_branch_card_count_name() -> None:
    """A PR run reads its workflow files from the default branch and its
    catalog from the PR head.  The rename of the catalog's card count to
    `ppu_count` is only visible on the PR side, so the matrix has to carry the
    old name too or the PR's own smoke lane dies on an empty device count."""
    matrix = _load_select().build_matrix(_catalog(), "")
    for entry in matrix["include"]:
        assert entry["tp"] == entry["ppu_count"]


def test_accuracy_matrix_uses_only_accuracy_and_eval_keys() -> None:
    select = _load_select()
    matrix = select.build_accuracy_matrix(
        _catalog(), "deepseek-v4-mxfp4,minimax-m3-fp8"
    )
    assert matrix == {
        "include": [
            {
                "case_id": "deepseek-v4-mxfp4",
                "model_key": "deepseek_v4_mxfp4",
                "model_id": (
                    "T-HEAD/DeepSeek-V4-Flash-0731-MoE-Quant-W-MXFP4-A-MXFP4-"
                    "Dense-Quant-W-FP8-PerChannel-A-FP8-PerToken"
                ),
                "board": "ZW-M890P",
                "ppu_count": 8,
                "server_cmd": [],
                "server_env": {},
            },
            {
                "case_id": "minimax-m3-fp8",
                "model_key": "minimax-m3-fp8",
                "model_id": ("T-HEAD/MiniMax-M3-Quant-W-FP8-PerChannel-A-FP8-PerToken"),
                "board": "ZW-M890P",
                "ppu_count": 8,
                "server_cmd": [],
                "server_env": {},
            },
        ]
    }


def test_accuracy_matrix_default_contains_every_accuracy_model() -> None:
    select = _load_select()
    matrix = select.build_accuracy_matrix(_catalog(), "")
    # Derived from the catalog so adding a case does not need a test edit; the
    # explicit set pins the weights that already have a reviewed golden.
    expected_keys = {
        entry["model_key"] for entry in _catalog() if entry["category"] == "accuracy"
    }
    assert {entry["model_key"] for entry in matrix["include"]} == expected_keys
    assert expected_keys == {
        "deepseek_v4_int8",
        "deepseek_v4_mxfp4",
        "minimax-m3-fp8",
        "glm_5_3_mxfp4",
        "glm_5_3_flash_mxfp4",
        "kimi_k2_6_mxfp4",
        "qwen3.8-flash-next",
    }


def test_accuracy_catalog_entries_declare_model_key_and_server_cmd() -> None:
    accuracy = [entry for entry in _catalog() if entry["category"] == "accuracy"]
    for entry in accuracy:
        # model_key picks the evaluator config and keys the golden file; it
        # must be present even though the catalog key is what labels a job.
        assert entry.get("model_key"), entry["key"]
        # server_cmd is opt-in: null means "use the evaluator's built-in argv".
        assert "server_cmd" in entry, entry["key"]


def test_accuracy_matrix_keeps_multiple_cases_over_one_checkpoint() -> None:
    """Two cases over one checkpoint stay distinct and keep their own argv."""
    select = _load_select()
    catalog = [
        {
            "key": "deepseek-v4-mxfp4",
            "category": "accuracy",
            "model_key": "deepseek_v4_mxfp4",
            "model_id": "T-HEAD/ckpt",
            "board": "ZW-M890P",
            "ppu_count": 8,
            "server_cmd": None,
        },
        {
            "key": "deepseek-v4-mxfp4-ep",
            "category": "accuracy",
            "model_key": "deepseek_v4_mxfp4",
            "model_id": "T-HEAD/ckpt",
            "board": "ZW-M890P",
            "ppu_count": 8,
            "server_cmd": ["vllm", "serve", "{model}", "--enable-expert-parallel"],
        },
    ]
    include = select.build_accuracy_matrix(catalog, "")["include"]
    assert [entry["case_id"] for entry in include] == [
        "deepseek-v4-mxfp4",
        "deepseek-v4-mxfp4-ep",
    ]
    # Both cases compare against the same golden, so the identity must agree.
    assert {entry["model_key"] for entry in include} == {"deepseek_v4_mxfp4"}
    assert include[0]["server_cmd"] == []
    assert include[1]["server_cmd"] == [
        "vllm",
        "serve",
        "{model}",
        "--enable-expert-parallel",
    ]


def test_every_case_server_cmd_keeps_the_readiness_contract() -> None:
    """A case argv must still let the readiness probe identify the server.

    The probe compares ``/v1/models`` ids against the run-unique served name
    and polls ``--port``; a catalog argv that forgets either flag hangs the
    job until the engine-ready timeout, so pin both here.
    """
    for entry in _catalog():
        cmd = entry.get("server_cmd") or []
        if not cmd:
            continue
        key = entry["key"]
        assert "{model}" in cmd, key
        assert "--served-model-name" in cmd, key
        served = cmd[cmd.index("--served-model-name") + 1]
        assert served == "{served_name}", key
        assert "--host" in cmd, key
        assert cmd[cmd.index("--host") + 1] == "127.0.0.1", key
        assert "--port" in cmd, key
        assert cmd[cmd.index("--port") + 1] == "{port}", key


def test_no_accuracy_case_serves_in_eager_mode() -> None:
    """CI must measure the production path, which captures CUDA graphs.

    Eager mode would hide a graph-capture regression and shift the numbers
    every golden was recorded from.
    """
    offenders = [
        entry["key"]
        for entry in _catalog()
        if any(
            token in {"--enforce-eager", "--enforce_eager"}
            for token in (entry.get("server_cmd") or [])
        )
    ]
    assert offenders == []


def test_every_accuracy_case_pins_the_served_window() -> None:
    """Without --max-model-len the engine captures graphs for a 1M window.

    That measured 997s before the engine was even ready, so every case recipe
    pins the same 16k window this lane evaluates in.
    """
    for entry in _catalog():
        cmd = entry.get("server_cmd") or []
        if not cmd:
            continue
        key = entry["key"]
        assert "--max-model-len" in cmd, key
        assert cmd[cmd.index("--max-model-len") + 1] == "16384", key


def test_builtin_argv_also_pins_the_served_window() -> None:
    """Cases that leave server_cmd null fall back to the built-in argv."""
    evaluator = _load_evaluator()
    assert evaluator.DEFAULT_MAX_MODEL_LEN == 16384


def test_accuracy_matrix_forwards_server_env() -> None:
    select = _load_select()
    catalog = [
        {
            "key": "dp-ep",
            "category": "accuracy",
            "model_key": "deepseek_v4_mxfp4",
            "model_id": "T-HEAD/ckpt",
            "board": "ZW-M890P",
            "ppu_count": 8,
            "server_env": {"VLLM_ENGINE_READY_TIMEOUT_S": "6000"},
        },
        {
            "key": "plain",
            "category": "accuracy",
            "model_key": "minimax-m3-fp8",
            "model_id": "T-HEAD/ckpt2",
            "board": "ZW-M890P",
            "ppu_count": 8,
        },
    ]
    by_case = {
        entry["case_id"]: entry
        for entry in select.build_accuracy_matrix(catalog, "")["include"]
    }
    assert by_case["dp-ep"]["server_env"] == {"VLLM_ENGINE_READY_TIMEOUT_S": "6000"}
    # Absent means "inherit", which must serialise as an empty object so the
    # workflow can toJSON() it without a null branch.
    assert by_case["plain"]["server_env"] == {}


def test_accuracy_catalog_resolves_against_the_evaluator_configs() -> None:
    """Every case must name a model_key the evaluator knows, for that model."""
    evaluator = _load_evaluator()
    configs = evaluator.MODEL_CONFIGS
    board_to_chip = {"OAM-810E": "810e", "ZW-M890P": "890p"}
    accuracy = [entry for entry in _catalog() if entry["category"] == "accuracy"]
    assert accuracy
    for entry in accuracy:
        key = entry["key"]
        assert entry["model_key"] in configs, key
        config = configs[entry["model_key"]]
        # One weight, one config: the id may not drift between the two.
        assert config["model_id"] == entry["model_id"], key
        assert config["served_name"] == entry["served_name"], key
        # A config may leave ``chip``/``ppu_count`` implicit and take the
        # defaults.  The default is the card count of the single-process
        # recipe; a DP+EP case asks for as many cards as it has data parallel
        # ranks, so the case may exceed it.
        chip = config.get("chip") or evaluator.DEFAULT_CHIP
        assert chip == board_to_chip[entry["board"]], key
        expected = config.get("ppu_count") or evaluator.tensor_parallel_for_chip(chip)
        cmd = entry.get("server_cmd") or []
        dp = 1
        if "--data-parallel-size" in cmd:
            dp = int(cmd[cmd.index("--data-parallel-size") + 1])
        assert int(entry["ppu_count"]) == max(expected, dp), key


def test_committed_goldens_describe_their_own_checkpoint() -> None:
    """A hand-written golden must still be a valid schema-v2 record.

    The gate trusts the score in these files, so a typo in the checkpoint tail
    or a stale tolerance would silently mis-score the nightly.
    """
    evaluator = _load_evaluator()
    configs = evaluator.MODEL_CONFIGS
    files = sorted(GOLDEN.glob("accuracy_*.json"))
    assert files, "no committed golden found"
    for path in files:
        golden = json.loads(path.read_text())
        model_key = golden["model"]
        assert model_key in configs, path.name
        assert path.name == f"accuracy_{model_key}_{golden['chip']}.json"
        assert golden["schema_version"] == 2, path.name
        assert golden["benchmark"] == "gsm8k", path.name
        assert golden["tolerance"] == {"mode": "band", "abs": 0.02}, path.name
        expected = golden["dataset"]["num_expected"]
        assert expected == 1319, path.name
        # ``score_num`` is the correct-answer count behind the score.
        assert golden["score_num"] == round(golden["score"] * expected), path.name
        # checkpoint_tail is what the identity gate compares against the run.
        # The lane serves by Hugging Face id, so the tail is that id's last
        # two segments -- i.e. the whole "<org>/<name>" string.
        model_id = configs[model_key]["model_id"]
        assert golden["checkpoint"] == model_id, path.name
        assert golden["checkpoint_tail"] == model_id, path.name


def test_describe_serving_reads_the_layout_off_the_argv() -> None:
    """A DP+EP recipe must not be recorded as tensor parallel 8.

    The layout only exists in the per-case argv, so anything derived from
    ``MODEL_CONFIGS`` would publish a parallel size no process ever used.
    """
    evaluator = _load_evaluator()
    layout = evaluator.describe_serving(
        [
            "vllm",
            "serve",
            "T-HEAD/ckpt",
            "--data-parallel-size",
            "8",
            "--enable-expert-parallel",
            "--gpu-memory-utilization",
            "0.9",
            "--max-model-len",
            "16384",
        ],
        ppu_count=8,
    )
    assert layout == {
        "ppu_count": 8,
        "tensor_parallel_size": 1,
        "data_parallel_size": 8,
        "enable_expert_parallel": True,
        "gpu_memory_utilization": 0.9,
        "max_model_len": 16384,
        "seed": evaluator.DEFAULT_SEED,
    }


def test_describe_serving_accepts_the_equals_form_and_records_an_unset_window() -> None:
    """``--flag=value`` is valid argv; an unset window stays visibly unset."""
    evaluator = _load_evaluator()
    layout = evaluator.describe_serving(
        ["vllm", "serve", "ckpt", "--tensor-parallel-size=4"], ppu_count=4
    )
    assert layout["tensor_parallel_size"] == 4
    assert layout["data_parallel_size"] == 1
    assert layout["max_model_len"] is None


def test_committed_goldens_report_the_recipe_that_produced_them() -> None:
    """``serving`` has to stay a real layout, not a copy of the defaults."""
    required = {
        "ppu_count",
        "tensor_parallel_size",
        "data_parallel_size",
        "enable_expert_parallel",
        "gpu_memory_utilization",
        "max_model_len",
        "seed",
    }
    for path in sorted(GOLDEN.glob("accuracy_*.json")):
        serving = json.loads(path.read_text())["serving"]
        assert set(serving) == required, path.name
        # Cards asked for can never be fewer than the parallelism they host.
        assert serving["ppu_count"] >= serving["tensor_parallel_size"], path.name
        assert serving["ppu_count"] >= serving["data_parallel_size"], path.name


def test_committed_goldens_carry_the_weight_sampling_contract() -> None:
    """A golden is only comparable under the contract it was measured in.

    The block is keyed on the weights, so every case over one checkpoint has
    to produce the same one -- that is what lets the DP+EP case gate against
    the baseline the plain case established.
    """
    evaluator = _load_evaluator()
    configs = evaluator.MODEL_CONFIGS
    for path in sorted(GOLDEN.glob("accuracy_*.json")):
        golden = json.loads(path.read_text())
        contract = configs[golden["model"]]["generation_config"]
        assert golden["generation"] == contract, path.name
        # Below the served window: see the budget test in the readiness suite.
        assert contract["max_tokens"] < evaluator.DEFAULT_MAX_MODEL_LEN, path.name


def test_every_model_config_declares_one_output_budget() -> None:
    """``max_tokens`` is uniform, so a timeout means a bad answer, not luck."""
    evaluator = _load_evaluator()
    assert evaluator.DEFAULT_MAX_TOKENS == 10240
    assert evaluator.DEFAULT_MAX_MODEL_LEN == 16384
    for key, config in evaluator.MODEL_CONFIGS.items():
        generation = config["generation_config"]
        assert generation["max_tokens"] == evaluator.DEFAULT_MAX_TOKENS, key
        assert 0.0 <= generation["temperature"] <= 2.0, key
        assert 0.0 < generation["top_p"] <= 1.0, key


def test_sibling_cases_over_one_weight_share_the_contract() -> None:
    """deepseek-v4-mxfp4 and its DP+EP twin must not diverge."""
    evaluator = _load_evaluator()
    cases = [
        entry
        for entry in _catalog()
        if entry.get("category") == "accuracy"
        and entry["model_key"] == "deepseek_v4_mxfp4"
    ]
    assert len(cases) >= 2
    contracts = {
        json.dumps(
            evaluator.MODEL_CONFIGS[entry["model_key"]]["generation_config"],
            sort_keys=True,
        )
        for entry in cases
    }
    assert len(contracts) == 1


def test_golden_coverage_names_the_weight_without_a_baseline() -> None:
    """Every accuracy weight has a golden except the one with no reference.

    GLM-5.3-Flash is deliberately absent: the shared GSM8K summary has no row
    for it, so the case runs on the threshold gate and uploads a
    pending-review artifact until a real number exists.
    """
    covered = {
        json.loads(path.read_text())["model"] for path in GOLDEN.glob("accuracy_*.json")
    }
    catalog_keys = {
        entry["model_key"] for entry in _catalog() if entry["category"] == "accuracy"
    }
    assert covered == catalog_keys - {"glm_5_3_flash_mxfp4"}


def test_accuracy_matrix_rejects_entry_without_model_key() -> None:
    select = _load_select()
    catalog = [
        {
            "key": "no-model-key",
            "category": "accuracy",
            "model_id": "T-HEAD/ckpt",
            "board": "ZW-M890P",
            "ppu_count": 8,
        }
    ]
    with pytest.raises(ValueError, match="missing 'model_key'"):
        select.build_accuracy_matrix(catalog, "")


def test_selector_rejects_unknown_key() -> None:
    with pytest.raises(ValueError, match="unknown model key"):
        _load_select().build_matrix(_catalog(), "does-not-exist")


def test_selector_cli_emits_separate_smoke_and_accuracy_outputs() -> None:
    result = subprocess.run(
        ["python3", str(SELECT), ""],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    outputs = dict(line.split("=", 1) for line in result.stdout.splitlines())
    smoke = json.loads(outputs["matrix"])
    accuracy = json.loads(outputs["accuracy_matrix"])
    catalog = _catalog()
    assert len(smoke["include"]) == sum(
        1 for entry in catalog if entry.get("category", "smoke") == "smoke"
    )
    assert len(accuracy["include"]) == sum(
        1 for entry in catalog if entry.get("category") == "accuracy"
    )


# --- scripts -----------------------------------------------------------------


@pytest.mark.parametrize("script", [SMOKE, ACCURACY, RUNTIME])
def test_e2e_shell_scripts_have_valid_bash_syntax(script: Path) -> None:
    result = subprocess.run(
        ["bash", "-n", str(script)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_smoke_script_preserves_required_e2e_contract() -> None:
    text = SMOKE.read_text()
    for marker in (
        "set -Eeuo pipefail",
        "--no-deps",
        "--force-reinstall",
        "from vllm_sail import collect_env",
        "vllm serve",
        "/v1/chat/completions",
        "Application startup complete",
        "kill -0",
        "ppu-smi",
        "check_vllm_compatibility(force=True)",
        "ppu_wheel_manifest.py",
    ):
        assert marker in text, marker


def test_smoke_resolves_model_id_only_from_offline_hf_cache() -> None:
    text = SMOKE.read_text()
    assert 'MODEL_ID="${MODEL_ID:?MODEL_ID is required}"' in text
    assert 'HF_HOME="${HF_HOME:?HF_HOME is required}"' in text
    assert "local_files_only=True" in text
    assert "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1" in text
    assert "snapshot_download(repo_id=sys.argv[1]" in text


def test_accuracy_script_uses_writable_results_with_read_only_source() -> None:
    text = ACCURACY.read_text()
    assert '--golden-dir "${RESULTS_DIR}/golden"' in text
    assert "cd /tmp" in text
    assert "unset PYTHONPATH" in text
    assert 'cp -fv "${wheel_so[@]}" "${ROOT_DIR}/vllm_sail/"' not in text
    assert 'python "${ROOT_DIR}/scripts/eval/ppu_gsm8k_eval.py"' in text
    # A repo missing from the offline cache must fail before the server boots
    # instead of twelve minutes into engine startup.
    probe = "snapshot_download(repo_id=sys.argv[1]"
    assert probe in text
    assert "local_files_only=True" in text
    assert text.index(probe) < text.index(
        'python "${ROOT_DIR}/scripts/eval/ppu_gsm8k_eval.py"'
    )


def test_accuracy_script_seeds_reviewed_goldens_into_the_writable_tree() -> None:
    """Without the seed every run reports "created" and never compares."""
    text = ACCURACY.read_text()
    seed = 'GOLDEN_SEED_DIR="${GOLDEN_SEED_DIR:-/source/tests/e2e/models/_golden}"'
    assert seed in text
    # The checked-out source is mounted read-only, so the copy must land in
    # RESULTS_DIR and the evaluator must read the copy.
    assert 'cp -f "${golden}" "${golden_dir}/"' in text
    assert text.index(seed) < text.index('--golden-dir "${RESULTS_DIR}/golden"')
    assert "threshold-only run" in text


def test_runtime_wrapper_exposes_accuracy_mode_and_environment() -> None:
    text = RUNTIME.read_text()
    assert "pin-ut|model-test|model-accuracy|kernel-perf" in text
    assert "/source/scripts/ci/ppu_e2e_accuracy.sh" in text
    for marker in (
        "HF_HOME",
        "HF_HUB_OFFLINE=1",
        "GSM8K_DATASET_PATH",
        "MODEL_KEY",
        "RESULTS_DIR=/artifacts",
    ):
        assert marker in text


def test_evaluator_requires_dataset_environment_without_hardcoded_paths() -> None:
    text = (ROOT / "scripts" / "eval" / "ppu_gsm8k_eval.py").read_text()
    assert "GSM8K_DATASET_PATH environment variable is required" in text
    assert "/ppusw/" not in text
    assert "/nas_aisw/" not in text


# --- workflows ---------------------------------------------------------------


def test_model_workflow_is_call_only_and_uses_docker_runner() -> None:
    workflow = _workflow(MODEL_WORKFLOW)
    assert set(workflow[True]) == {"workflow_call"}
    assert workflow["jobs"]["prepare"]["runs-on"] == "ubuntu-latest"
    assert workflow["jobs"]["e2e"]["runs-on"] == "8-ppu-m890p"
    assert "container" not in workflow["jobs"]["e2e"]
    text = MODEL_WORKFLOW.read_text()
    assert 'ppu_runtime_container.sh" model-test' in text
    assert "ppu-scheduler-action" not in text


@pytest.mark.parametrize(
    ("filename", "event", "expected"),
    [
        ("_ppu-model-tests.yaml", "workflow_call", "qwen3.8-27b-m890p"),
        ("ppu-debug.yaml", "workflow_dispatch", "qwen3.8-27b-m890p"),
    ],
)
def test_smoke_workflow_defaults_target_available_m890_runner(
    filename: str, event: str, expected: str
) -> None:
    config = _workflow(MODEL_WORKFLOW.parent / filename)
    models = config[True][event]["inputs"]["models"]["default"]
    assert models == expected
    matrix = _load_select().build_matrix(_catalog(), models)
    assert [(entry["key"], entry["board"]) for entry in matrix["include"]] == [
        ("qwen3.8-27b-m890p", "ZW-M890P")
    ]


def test_accuracy_workflow_is_call_only_and_uses_docker_runner() -> None:
    workflow = _workflow(ACCURACY_WORKFLOW)
    assert workflow["name"] == "PPU accuracy tests"
    assert set(workflow[True]) == {"workflow_call"}
    assert set(workflow["jobs"]) == {"prepare", "accuracy"}
    assert workflow["jobs"]["prepare"]["runs-on"] == "ubuntu-latest"
    accuracy = workflow["jobs"]["accuracy"]
    assert accuracy["runs-on"] == "8-ppu-m890p"
    assert "container" not in accuracy
    text = ACCURACY_WORKFLOW.read_text()
    assert 'ppu_runtime_container.sh" model-accuracy' in text
    assert "ppu-scheduler-action" not in text
    assert text.count("actions/download-artifact@v4") == 1


def test_accuracy_entrypoints_are_the_manual_debug_and_the_nightly_runs() -> None:
    debug = _workflow(DEBUG_WORKFLOW)
    assert set(debug[True]) == {"workflow_dispatch"}
    inputs = debug[True]["workflow_dispatch"]["inputs"]
    assert set(inputs) == {
        "build_run_id",
        "wheel_source_sha",
        "target",
        "models",
        "accuracy_models",
    }
    assert "accuracy-tests" in inputs["target"]["options"]
    assert "kernel-perf" in inputs["target"]["options"]
    assert debug["jobs"]["accuracy-tests"]["uses"] == (
        "./.github/workflows/_ppu-accuracy-tests.yaml"
    )

    nightly = _workflow(NIGHTLY_WORKFLOW)
    assert set(nightly[True]) == {"workflow_dispatch", "schedule"}
    assert nightly[True]["schedule"] == [{"cron": "0 12 * * *"}]
    assert set(nightly["jobs"]["accuracy"]["with"]) >= {
        "build_run_id",
        "models",
        "pr_head_sha",
        "wheel_source_sha",
    }
    assert nightly["jobs"]["accuracy"]["uses"] == (
        "./.github/workflows/_ppu-accuracy-tests.yaml"
    )

    assert debug["jobs"]["kernel-perf"]["uses"] == (
        "./.github/workflows/_ppu-kernel-perf.yaml"
    )

    # kernel-perf is no longer part of the per-PR gate.
    automatic = (ROOT / ".github/workflows/ppu-build-and-test.yaml").read_text()
    assert "_ppu-accuracy-tests.yaml" not in automatic
    assert "model-accuracy" not in automatic
    assert "_ppu-kernel-perf.yaml" not in automatic


def test_nightly_select_step_labels_jobs_with_the_case_id() -> None:
    """The select step must key on case_id, not on the shared model_key.

    Two cases over one checkpoint share a model_key, so a model_key-keyed
    mapping collapses them into one matrix row and the other case silently
    never runs.
    """
    text = NIGHTLY_WORKFLOW.read_text()
    assert '{"model": entry["case_id"]}' in text
    assert "ACCURACY_MODEL_KEYS" not in text


def test_accuracy_nightly_is_serial_and_wheel_sourced_from_the_build_commit() -> None:
    nightly = _workflow(NIGHTLY_WORKFLOW)
    accuracy = nightly["jobs"]["accuracy"]
    # Every lane needs all eight devices and the runtime wrapper holds an
    # exclusive lock, so a parallel matrix would abort instead of queueing.
    assert accuracy["strategy"]["max-parallel"] == 1
    assert accuracy["strategy"]["fail-fast"] is False
    assert accuracy["strategy"]["matrix"] == (
        "${{ fromJson(needs.select.outputs.matrix) }}"
    )
    # The nightly builds its own wheels, so github.sha is both the source
    # checkout and the commit whose artifact names carry this run.
    assert nightly["jobs"]["build"]["uses"] == (
        "./.github/workflows/ppu-build-and-test.yaml"
    )
    assert nightly["jobs"]["build"]["with"] == {
        "pr_head_sha": "${{ github.sha }}",
        "skip_device_tests": True,
    }
    assert accuracy["needs"] == ["build", "select"]
    # Checkout and wheels must describe the same SAIL tree.
    assert accuracy["with"]["pr_head_sha"] == "${{ github.sha }}"
    assert accuracy["with"]["wheel_source_sha"] == "${{ github.sha }}"
    assert accuracy["with"]["build_run_id"] == "${{ github.run_id }}"
    assert accuracy["with"]["pr_mode"] is False
    # The accuracy work runs inside the reusable call, not on a hosted shell.
    assert "runs-on" not in accuracy
    text = NIGHTLY_WORKFLOW.read_text()
    # Each case must reach the reusable workflow with one catalog key.
    catalog = _catalog()
    matrix = _load_select().build_accuracy_matrix(catalog, "")
    assert len(matrix["include"]) == sum(
        1 for entry in catalog if entry.get("category") == "accuracy"
    )
    assert "models: ${{ matrix.model }}" in text


def test_workflows_pull_wheels_by_run_id_and_keep_python_frontend() -> None:
    for path in (MODEL_WORKFLOW, ACCURACY_WORKFLOW):
        text = path.read_text()
        assert "build_run_id:" in text
        assert "actions/download-artifact@v4" in text
        assert "run-id: ${{ inputs.build_run_id }}" in text
        assert "ppu-wheels-${{ inputs.build_run_id }}" in text
        assert "VLLM_VERSION=" not in text
    smoke_text = SMOKE.read_text()
    assert "VLLM_USE_RUST_FRONTEND=0" in smoke_text
    assert "VLLM_USE_RUST_BENCH=0" in smoke_text


def test_kernel_perf_label_workflow_structure() -> None:
    """kernel-perf-label.yaml triggers independently via the kernel-perf label."""
    workflow = _workflow(KERNEL_PERF_LABEL_WORKFLOW)
    assert workflow["name"] == "Kernel Perf Gate"
    assert set(workflow[True]) == {"pull_request"}
    pr = workflow[True]["pull_request"]
    assert set(pr["types"]) == {"labeled", "synchronize", "reopened"}

    # Fresh-build design: build + kernel-perf (no resolve-build)
    assert set(workflow["jobs"]) == {"build", "kernel-perf"}

    build = workflow["jobs"]["build"]
    # Build job is a reusable workflow call to _ppu-build.yaml
    assert build["uses"] == "./.github/workflows/_ppu-build.yaml"
    # Guard: label + not draft + correct label on labeled event
    assert "kernel-perf" in build["if"]
    assert "!github.event.pull_request.draft" in build["if"]
    assert "github.event.label.name == 'kernel-perf'" in build["if"]
    # Build checks out and packages the PR head
    assert build["with"]["pr_head_sha"] == ("${{ github.event.pull_request.head.sha }}")

    kp = workflow["jobs"]["kernel-perf"]
    assert kp["needs"] == "build"
    assert kp["uses"] == "./.github/workflows/_ppu-kernel-perf.yaml"
    assert kp["with"]["pr_mode"] is True
    assert kp["with"]["pr_head_sha"] == ("${{ github.event.pull_request.head.sha }}")
    # build_run_id uses the caller's own run_id (fresh build, not external)
    assert kp["with"]["build_run_id"] == "${{ github.run_id }}"
    assert kp["with"]["wheel_source_sha"] == (
        "${{ github.event.pull_request.head.sha }}"
    )

    # This workflow is NOT part of ppu-build-and-test gate
    gate_text = (ROOT / ".github/workflows/ppu-build-and-test.yaml").read_text()
    assert "kernel-perf-label" not in gate_text


def test_weekly_perf_builds_current_commit_and_never_resolves_old_runs() -> None:
    workflow = _workflow(PERF_REGRESSION_WORKFLOW)
    assert set(workflow[True]) == {"workflow_dispatch", "schedule"}
    assert workflow[True]["schedule"] == [{"cron": "0 20 * * 1"}]
    assert "build_run_id" not in workflow[True]["workflow_dispatch"]["inputs"]
    assert workflow["permissions"] == {"contents": "read", "actions": "read"}
    assert workflow["cache-mode"] == "none"
    assert set(workflow["jobs"]) == {"build", "perf"}

    build = workflow["jobs"]["build"]
    assert build["uses"] == "./.github/workflows/_ppu-build.yaml"
    assert build["with"] == {"pr_head_sha": "${{ github.sha }}"}

    perf = workflow["jobs"]["perf"]
    assert perf["needs"] == "build"
    assert {entry["chip"] for entry in perf["strategy"]["matrix"]["include"]} == {
        "810e",
        "890p",
    }
    download = next(
        step
        for step in perf["steps"]
        if step.get("uses") == "actions/download-artifact@v4"
    )
    assert download["with"]["name"] == "ppu-wheels-${{ github.run_id }}"
    assert download["with"]["run-id"] == "${{ github.run_id }}"
    verify = next(
        step
        for step in perf["steps"]
        if step.get("name") == "Verify wheel manifest against current sources"
    )
    assert verify["env"]["EXPECTED_SAIL_COMMIT"] == "${{ github.sha }}"
    assert ".github/vllm-main-verified.commit" in verify["run"]
    assert "--vllm-commit" in verify["run"]
    assert "--sail-commit" in verify["run"]

    text = PERF_REGRESSION_WORKFLOW.read_text()
    assert "ppu-build-and-test.yaml/runs" not in text
    assert "api.github.com" not in text
    assert "latest successful" not in text
    assert "EXPECTED_SAIL_COMMIT=${{ github.sha }}" in text


def test_kernel_perf_workflow_is_call_only_and_uses_docker_runner() -> None:
    workflow = _workflow(KERNEL_PERF_WORKFLOW)
    assert workflow["name"] == "PPU kernel performance"
    assert set(workflow[True]) == {"workflow_call"}
    assert set(workflow["jobs"]) == {"kernel-perf"}
    kp = workflow["jobs"]["kernel-perf"]
    assert kp["runs-on"] == "8-ppu-m890p"
    assert kp["timeout-minutes"] == 60
    assert "container" not in kp
    inputs = workflow[True]["workflow_call"]["inputs"]
    assert inputs["wheel_source_sha"]["required"] is True
    assert inputs["wheel_source_sha"]["type"] == "string"
    runtime = next(
        step
        for step in kp["steps"]
        if "ppu_runtime_container.sh" in step.get("run", "")
    )
    assert runtime["env"]["PPU_CI_WHEEL_SOURCE_SHA"] == "${{ inputs.wheel_source_sha }}"
    text = KERNEL_PERF_WORKFLOW.read_text()
    assert 'ppu_runtime_container.sh" kernel-perf' in text
    assert "ppu-scheduler-action" not in text


def test_runtime_container_kernel_perf_mode() -> None:
    text = RUNTIME.read_text()
    assert "kernel-perf)" in text
    assert "/source/scripts/ci/perf_regression.sh" in text
    # kernel-perf mode injects SUITE and does not require MODEL_PATH/HF_HOME
    lines = text.split("\n")
    in_kernel_block = False
    kernel_block = []
    for line in lines:
        if line.strip() == "kernel-perf)":
            in_kernel_block = True
        elif in_kernel_block and line.strip() == ";;":
            break
        elif in_kernel_block:
            kernel_block.append(line)
    kernel_text = "\n".join(kernel_block)
    assert "SUITE" in kernel_text
    assert "TEST_PATH=tests/e2e/perf/test_kernel_perf.py" in kernel_text
    assert "RESULTS_DIR=/artifacts" in kernel_text
    assert "BASELINE_ROOT=/source/tests/e2e/perf/baselines" in kernel_text
    assert "BASELINE_OUTPUT_ROOT=/artifacts/baselines" in kernel_text
    assert "perf_regression.sh" in kernel_text
    assert "MODEL_PATH" not in kernel_text
    assert "HF_HOME" not in kernel_text
    assert "GSM8K_DATASET_PATH" not in kernel_text


# --- perf regression (model) -------------------------------------------------


def test_perf_regression_workflow_no_hardcoded_hf_paths() -> None:
    """The perf workflow must inherit its cache instead of hardcoding paths."""
    text = PERF_REGRESSION_WORKFLOW.read_text()
    workflow = _workflow(PERF_REGRESSION_WORKFLOW)
    matrix = workflow["jobs"]["perf"]["strategy"]["matrix"]
    for entry in matrix["include"]:
        assert "hf_home" not in entry, f"hf_home leaked into matrix entry: {entry}"
        assert "hf_cache" not in entry, f"hf_cache leaked into matrix entry: {entry}"
    assert "/nas_aisw/datasets/hf_home" not in text
    assert "/ppusw/datasets/hf_home" not in text


def test_perf_regression_workflow_model_lanes_have_offline_protection() -> None:
    """Model lanes must prevent Hugging Face and transformers network access."""
    text = PERF_REGRESSION_WORKFLOW.read_text()
    assert "HF_HUB_OFFLINE=1" in text
    assert "TRANSFORMERS_OFFLINE=1" in text


def test_perf_regression_script_exports_transformers_offline() -> None:
    """The model perf runner must export both offline controls."""
    text = PERF_REGRESSION_SCRIPT.read_text()
    assert "TRANSFORMERS_OFFLINE" in text
    assert "export HF_HUB_OFFLINE TRANSFORMERS_OFFLINE" in text


def test_perf_regression_script_does_not_hardcode_hf_paths() -> None:
    """The model perf runner must not contain hardcoded cache paths."""
    text = PERF_REGRESSION_SCRIPT.read_text()
    assert "/nas_aisw/datasets/hf_home" not in text
    assert "/ppusw/datasets/hf_home" not in text


def test_model_perf_uses_model_id_not_checkpoint() -> None:
    """PerfModel must use the PR 88 model_id loading contract."""
    import ast

    text = MODEL_PERF_TEST.read_text()
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "PerfModel":
            field_names = [
                stmt.target.id
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
            ]
            assert "model_id" in field_names, "PerfModel must have a model_id field"
            assert "checkpoint" not in field_names, (
                "PerfModel must not have a checkpoint field (use model_id)"
            )
            break
    else:
        raise AssertionError("PerfModel class not found in test_model_perf.py")
