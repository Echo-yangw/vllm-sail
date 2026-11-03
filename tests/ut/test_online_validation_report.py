# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_report_checks_numerics_even_when_tokens_agree(tmp_path):
    path = Path(__file__).resolve().parents[2] / "tools/validate_online_quantization.py"
    spec = importlib.util.spec_from_file_location("validation_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    record = {
        "id": "one",
        "prompt_token_ids": [1, 2],
        "token_ids": [3],
        "finish_reason": "stop",
        "stop_reason": None,
        "logprobs": [{"3": -0.1}],
    }
    artifact = {
        "sampling": {"seed": 0},
        "records": [record],
        "workers": [{"weights": {"proj.weight": "digest"}}],
        "startup_seconds": 1.0,
    }
    reference, candidate, output = [
        tmp_path / p for p in ("reference.json", "candidate.json", "report.json")
    ]
    reference.write_text(json.dumps(artifact))
    candidate.write_text(json.dumps(artifact))
    args = SimpleNamespace(
        reference=reference,
        candidate=candidate,
        output=output,
        require_weight_match=True,
        logprob_atol=1e-3,
    )
    module.compare(args)
    assert json.loads(output.read_text())["generation_differences"] == []
    artifact["workers"][0]["weights"]["proj.weight"] = "different-digest"
    candidate.write_text(json.dumps(artifact))
    with pytest.raises(SystemExit):
        module.compare(args)
    artifact["workers"][0]["weights"]["proj.weight"] = "digest"
    record["logprobs"][0]["3"] = -0.2
    candidate.write_text(json.dumps(artifact))
    with pytest.raises(SystemExit):
        module.compare(args)
    assert json.loads(output.read_text())["generation_differences"] == [
        {"id": "one", "field": "logprobs"}
    ]
    record["prompt_token_ids"] = [2, 3]
    candidate.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="Prompt tokens"):
        module.compare(args)
