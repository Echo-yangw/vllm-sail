# SPDX-License-Identifier: Apache-2.0
"""Capture comparable model runs in separate processes, then compare artifacts.

Run with an installed PPU environment. Input prompts are JSON objects containing
id and prompt_token_ids, optionally expected_text for exact-match scoring.
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path


def worker_snapshot(worker):
    import torch

    torch.cuda.synchronize()
    records = {}
    model = worker.model_runner.model
    for name, param in model.named_parameters():
        if param.dtype == torch.int8 or name.endswith(
            ("weight", "weight_scale", "bias")
        ):
            tensor = param.detach().cpu().contiguous()
            records[name] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": hashlib.sha256(
                    tensor.view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
            }
    return {
        "device": torch.cuda.get_device_name(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "weights": records,
    }


def run(args):
    import importlib.metadata

    from vllm import LLM, SamplingParams

    kwargs = json.loads(args.engine_config.read_text())
    prompts = json.loads(args.prompts.read_text())
    ids = [p["id"] for p in prompts]
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError("Prompt IDs must be unique and nonempty")
    sampling = dict(
        temperature=0, seed=0, max_tokens=args.max_tokens, logprobs=5, prompt_logprobs=5
    )
    started = time.perf_counter()
    llm = LLM(**kwargs)
    startup_seconds = time.perf_counter() - started
    # Snapshot before generation: peak includes loading, KV allocation and any
    # graph capture, not only the quantization operation.
    startup = llm.collective_rpc(worker_snapshot)
    started = time.perf_counter()
    outputs = llm.generate(
        [{"prompt_token_ids": p["prompt_token_ids"]} for p in prompts],
        SamplingParams(**sampling),
    )
    inference_seconds = time.perf_counter() - started

    def logprobs(value):
        if value is None:
            return None
        return [
            None if entry is None else {str(k): v.logprob for k, v in entry.items()}
            for entry in value
        ]

    records = []
    for prompt, output in zip(prompts, outputs, strict=True):
        candidate = output.outputs[0]
        records.append(
            {
                "id": prompt["id"],
                "prompt_token_ids": output.prompt_token_ids,
                "prompt_logprobs": logprobs(output.prompt_logprobs),
                "token_ids": list(candidate.token_ids),
                "text": candidate.text,
                "logprobs": logprobs(candidate.logprobs),
                "finish_reason": candidate.finish_reason,
                "stop_reason": candidate.stop_reason,
                "exact_match": candidate.text.strip() == prompt["expected_text"].strip()
                if "expected_text" in prompt
                else None,
            }
        )
    versions = {}
    for package in ("vllm", "vllm-sail", "torch", "qlean"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    result = {
        "engine_config": kwargs,
        "sampling": sampling,
        "versions": versions,
        "startup_seconds": startup_seconds,
        "inference_seconds": inference_seconds,
        "workers": startup,
        "records": records,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


def compare(args):
    left = json.loads(args.reference.read_text())
    right = json.loads(args.candidate.read_text())
    if left["sampling"] != right["sampling"]:
        raise ValueError("Sampling settings differ")
    references = {r["id"]: r for r in left["records"]}
    candidates = {r["id"]: r for r in right["records"]}
    if references.keys() != candidates.keys():
        raise ValueError("Prompt ID sets differ")
    differences = []
    for key, ref in references.items():
        candidate = candidates[key]
        if ref["prompt_token_ids"] != candidate["prompt_token_ids"]:
            raise ValueError(f"Prompt tokens differ for {key}")
        for field in ("token_ids", "finish_reason", "stop_reason"):
            if ref[field] != candidate[field]:
                differences.append({"id": key, "field": field})
        for field in ("prompt_logprobs", "logprobs"):
            a, b = ref.get(field), candidate.get(field)
            if a is None or b is None:
                if a != b:
                    differences.append({"id": key, "field": field})
                continue
            equal = len(a) == len(b)
            if equal:
                for ap, bp in zip(a, b, strict=True):
                    if ap is None or bp is None:
                        equal = ap == bp
                    else:
                        equal = ap.keys() == bp.keys() and all(
                            math.isclose(
                                ap[token],
                                bp[token],
                                rel_tol=0,
                                abs_tol=args.logprob_atol,
                            )
                            for token in ap
                        )
                    if not equal:
                        break
            if not equal:
                differences.append({"id": key, "field": field})
    reference_weights = [w["weights"] for w in left["workers"]]
    candidate_weights = [w["weights"] for w in right["workers"]]
    weights_equal = (
        bool(reference_weights)
        and all(reference_weights)
        and reference_weights == candidate_weights
    )
    result = {
        "requests": len(references),
        "generation_differences": differences,
        "runtime_weight_hashes_equal": weights_equal,
        "reference_startup_seconds": left["startup_seconds"],
        "candidate_startup_seconds": right["startup_seconds"],
        "candidate_exact_matches": sum(
            r.get("exact_match") is True for r in candidates.values()
        ),
        "scored_requests": sum(
            r.get("exact_match") is not None for r in candidates.values()
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if differences or (args.require_weight_match and not weights_equal):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("run")
    capture.add_argument("--engine-config", type=Path, required=True)
    capture.add_argument("--prompts", type=Path, required=True)
    capture.add_argument("--max-tokens", type=int, default=256)
    diff = sub.add_parser("compare")
    diff.add_argument("--reference", type=Path, required=True)
    diff.add_argument("--candidate", type=Path, required=True)
    diff.add_argument(
        "--require-weight-match", action=argparse.BooleanOptionalAction, default=True
    )
    diff.add_argument("--logprob-atol", type=float, default=1e-3)
    for command in (capture, diff):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (run if args.command == "run" else compare)(args)


if __name__ == "__main__":
    main()
