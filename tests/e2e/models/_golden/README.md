# Golden baselines — accuracy model E2E (schema v2)

Per-chip evalscope GSM8K accuracy baselines for the model-precision suite
(`tests/e2e/models`). A baseline normally comes from a run on real PPU
hardware; the seeded records listed under
[Model configurations](#model-configurations) are the exception, and they say
so in their own `baseline_origin`.

## What changed (schema v1 → v2)

Schema v1 stored per-token logprob goldens; schema v2 stores a single accuracy
score from evalscope GSM8K evaluation. A score is judged against its own
checkpoint's golden: the run passes when it lands inside
**`golden ± 0.02`**. The former `score >= 0.9` threshold is no longer a gate —
it only applies to a checkpoint that has no golden yet, so a first run cannot
be reported as "created" while scoring far below anything acceptable.

## File naming

```
accuracy_<model_key>_<chip>.json                   formal baseline
accuracy_<model_key>_<chip>.pending-review.json    unreviewed first-run output
```

`chip` is derived from the detected device capability (`(8,0) → 810e`,
`(8,9) → 890p`).

## Cases vs. goldens

One checkpoint may be exercised by several **cases** in
`scripts/ci/ppu_e2e_models.json`, each with its own `vllm serve` argv
(`server_cmd`) and therefore its own run configuration. Two identifiers keep
the two concepts apart:

| Field | Meaning | Used for |
|-------|---------|----------|
| `key` | catalog case label | job name, artifact name, serve log |
| `model_key` | evaluator model-config key | **golden identity** |

Goldens are keyed on `model_key`, **not** on the case: cases over the same
weights share one baseline, so a second case cannot quietly establish a
second, laxer reference for the same checkpoint. Adding a case therefore needs
no new golden file — only a `model_key` and an optional `server_cmd`.

A golden created for a checkpoint whose common failure mode is a *high*
score is still bounded on the upper side: the band is symmetric, so drift in
either direction fails.

### Adding an accuracy case

```jsonc
{
  "key": "deepseek-v4-mxfp4-dp-ep",      // unique; becomes the job label
  "category": "accuracy",
  "model_key": "deepseek_v4_mxfp4",       // must exist in MODEL_CONFIGS
  "model_id": "T-HEAD/...",               // resolved against the offline cache
  "served_name": "deepseek-v4-mxfp4",
  "board": "ZW-M890P",
  "ppu_count": 8,                         // cards the container gets
  "extra_serve_flags": "--trust-remote-code",
  // null / omitted: use the evaluator's built-in argv.  A non-empty array
  // replaces that argv wholesale, so include every flag this case needs.
  "server_cmd": [
    "vllm", "serve", "{model}",
    "--served-model-name", "{served_name}",
    "--host", "127.0.0.1", "--port", "{port}",
    "--trust-remote-code",
    "--enable-auto-tool-choice",
    "--data-parallel-size", "8", "--enable-expert-parallel",
    "--all2all-backend", "deepep_low_latency",
    "--max-model-len", "16384"
  ],
  // Optional: extra environment for the server process only, merged over the
  // inherited environment (HF_HOME etc. survive). Omit for the default env.
  "server_env": {"VLLM_ENGINE_READY_TIMEOUT_S": "6000"}
}
```

`{model}` expands to the resolved checkpoint, `{port}` to `VLLM_PORT`, and
`{served_name}` to the per-run nonce (`<served_name>-run-<hex>`), so the
readiness probe and evalscope keep addressing the process this case started.
Unknown braces are left untouched. Whenever `server_cmd` is set it must keep
`--served-model-name {served_name}` and `--host`/`--port`, otherwise the
readiness probe can never match the process and the job burns the full
engine-ready budget. Mirror these keys in `MODEL_CONFIGS` of
`scripts/eval/ppu_gsm8k_eval.py` if the case needs a new `model_key`.

The parallel layout belongs to the case recipe, not to this example: a case
pins whatever `--tensor-parallel-size` / `--data-parallel-size` its own recipe
calls for, and `ppu_count` above only says how many cards the container gets.
Never add `--enforce-eager`: CI measures the production path, which runs with
CUDA graphs captured, and eager mode would both hide a graph-capture
regression and change the numbers the golden was recorded from.
`--max-model-len 16384` is likewise part of every recipe — without it the
server falls back to the checkpoint's 1M-token window and spends ~997s
capturing graphs that no GSM8K sample can use.

`server_env` exists because some launch recipes need variables that no CLI flag
expresses — `VLLM_ENGINE_READY_TIMEOUT_S` for a `--data-parallel-size 8
--enable-expert-parallel` cold start, `DG_USE_MOE_DYNAMIC_TILE` for the Kimi
MXFP4 path. It is scoped to the `vllm serve` child, so two cases over the same
weights can run with different variables in the same job.

## Checkpoint resolution

The accuracy lane never resolves a path: it records and serves the model id,
and vLLM finds the snapshot through the offline cache. The path-mirroring rule
below applies to the offline lane, whose cases may record a checkpoint path
under either `/nas_aisw/datasets` or `/ppusw/datasets`. Both mount the same
NFS export. `resolve_checkpoint` tries the recorded path first and then each
mirrored root (`_offline_runner.CHECKPOINT_ROOT_MIRRORS`).

## Schema (`schema_version: 2`)

```jsonc
{
  "schema_version": 2,
  "model": "deepseek_v4_int8",        // model key
  "chip": "890p",                      // must equal the running device chip
  "checkpoint": "T-HEAD/DeepSeek-V4-Flash-0731-W8A8-INT8",  // identifier served
  "checkpoint_tail": "T-HEAD/DeepSeek-V4-Flash-0731-W8A8-INT8",  // identity
  "benchmark": "gsm8k",
  "evalscope_version": "1.14.1",
  "dataset": {
    "dataset_id": "/ppusw/datasets/vision/EvalScope/data/gsm8k/v2/gsm8k",
    "few_shot_num": 4,
    "few_shot_random": false,
    "subset_list": ["main"],
    "num_expected": 1319
  },
  "serving": {
    "ppu_count": 8,                  // cards the recorded run asked for
    "tensor_parallel_size": 8,       // read off the argv that ran
    "data_parallel_size": 1,
    "enable_expert_parallel": false,
    "gpu_memory_utilization": 0.9,   // null = the flag was not set
    "max_model_len": 16384,          // null = the checkpoint's own window
    "seed": 0                        // null = the engine's default seed
  },
  "generation": {
    "temperature": 1.0,              // owned by the weights, not by the case
    "top_p": 1.0,
    "max_tokens": 10240              // below the served 16384 window
  },
  "score": 0.9742,                     // evalscope accuracy (0.0–1.0)
  "score_num": 1284,                   // correct answers, = round(score * 1319)
  "tolerance": {
    "mode": "band",                    // symmetric band around score
    "abs": 0.02                        // ±2 points
  },
  "measured_at": "2025-07-15T06:00:00+00:00"
}
```

`serving` records the layout the score was measured under. It is read off the
argv that actually ran (`describe_serving`), not off `MODEL_CONFIGS`: a DP+EP
recipe serves tensor parallel 1 and takes its memory fraction from its own
flag, so copying the config defaults would publish numbers no process ran
with. A field stays `null` when the argv does not pin it, which is why the
committed seeded goldens carry `null` for the three flags their source row
never recorded.

For a seeded baseline it describes the production row named in
`baseline_origin`, not today's CI case, so `ppu_count` and
`tensor_parallel_size` may legitimately differ from the `Cards` column below --
that difference is what `tolerance` is there to absorb. No gate reads this
block; it exists so a human can tell what the number came from.

Two optional fields mark a baseline that was not produced by a run of this
suite. Neither takes part in comparison:

* `measured_at: null` — the score was not measured by the evaluator that will
gate against it, so it carries no timestamp.
* `baseline_origin` — free text naming where the number came from. The
currently committed goldens point at rows of the team's shared
`H20_vllm_evalscope_gsm8k_精度汇总_v2.csv`, using the newest framework version
recorded for that weight.

The first `--update-golden` run against a seeded baseline replaces it with a
measured record (timestamp filled in, `baseline_origin` gone).

## Sampling contract (`generation`)

`generation` is the evalscope sampling contract the score was measured under
and, unlike `serving`, a gate reads it. It belongs to the **weights**: every
case over one checkpoint takes the same `generation_config` from
`MODEL_CONFIGS` and gates against that checkpoint's single golden, so the two
DeepSeek-V4-Flash cases measure one number one way. A case changes its serve
argv, never its sampling parameters.

The values follow `reference_generation_config.json` in `LLM_validator` for the
weight family — `temperature` 1.0 for nearly every weight, `top_p` 1.0 or 0.95,
plus `top_k` / `min_p` / the penalties where the family sets them. `max_tokens`
is ours and uniform at 10240: below the 16384 served window, so a model that
starts repeating still stops inside one sample instead of burning the whole
window.

Changing any of these invalidates the goldens recorded under the old contract.
The nightly then reports `fail_config` naming the drifted knob rather than
`fail_accuracy`, and the fix is to re-record with `--update-golden`, not to
widen the band.

## Comparison rules

1. **Identity gate**: `chip` must match; `checkpoint_tail` (when present) must
   match — prevents cross-chip or cross-checkpoint comparison. The accuracy
   lane serves by Hugging Face id, so the tail is the whole `<org>/<name>`;
   when a case serves a path the helper keeps its last two segments, which
   survives a different mount root.
2. **Golden gates** (when a formal golden exists), in order:
   a. **Completeness**: `num == dataset.num_expected`.
   b. **Generation contract**: `generation` must equal the sampling parameters
      this run used, key by key. A drift, or a golden without the block, is
      reported as `fail_config` — a score from another contract cannot
      arbitrate this run, and blaming the weights for it would be wrong.
   c. **Band**: `golden.score - abs <= score <= golden.score + abs`, i.e.
      `golden ± 0.02` by default. Two points absorbs run-to-run noise on PPU
      and is still narrow enough that a real regression cannot hide; the band
      is symmetric so an abnormally high golden is caught as well.
3. **Threshold fallback** (only when no golden exists): `score >= 0.9`, so a
   first run cannot be reported as acceptable while scoring far below it.
   Configurable via `--threshold`.

Records still carrying the legacy `mode: lower_bound` tolerance keep their
one-sided bound and hard floor, so an older baseline continues to gate exactly
as it did.

## Nightly CI

`ppu-accuracy-nightly.yaml` runs every accuracy **case** once a night against
the wheels of the newest successful PPU CI build. The checkout inside the
runtime container is read-only, so `scripts/ci/ppu_e2e_accuracy.sh` copies
every `accuracy_*.json` from this directory into the writable results tree
before evaluating (`GOLDEN_SEED_DIR`, default
`/source/tests/e2e/models/_golden`). Consequences:

* A checkpoint with a committed golden here **is** gated on the band.
* A checkpoint without one reports `created`, uploads
  `accuracy_<model>_<chip>.pending-review.json` inside the run's
  `ppu-accuracy-<case_id>-*` artifact, and only enforces `score >= threshold`.

Several cases sharing one checkpoint therefore produce one golden candidate
per case but gate against the single baseline for that checkpoint; import the
first pending file you accept and the sibling cases will gate against it on
the next run.

Promotion stays a human step: review the pending artifact, commit it here as
`accuracy_<model>_<chip>.json`, and the next run gates against it.

`chip` is whatever device the nightly runner exposes, not the catalog `board`
field: the lane runs on the self-hosted `8-ppu-m890p` runner, so
`deepseek_v4_int8` is currently captured against the ZW-M890P int8 path.

## Capturing / refreshing

Run the eval script or test on the target PPU machine with `--update-golden`:

```bash
python scripts/eval/ppu_gsm8k_eval.py \
    --model-key deepseek_v4_int8 --update-golden
```

Add `--case-id <key> --server-cmd '[ ... ]'` to reproduce a specific catalog
case; both only affect the serve log and the argv, never the golden identity.

Or via pytest:

```bash
pytest tests/e2e/models/test_model_accuracy.py::test_deepseek_v4_int8_accuracy \
    --update-golden
```

## Pending-review → formal promotion

A `.pending-review.json` file never participates in comparison — an unreviewed
first run cannot silently become the baseline. Promote it by either:

1. Review the pending file, then copy it:
   `cp accuracy_<model>_<chip>.pending-review.json accuracy_<model>_<chip>.json`
2. Re-run with `--update-golden`, which writes the formal name directly and
   deletes the pending sibling.

## Model configurations

| Model key | Chip | Cards | Checkpoint | Baseline |
|-----------|------|----|-----------|----------|
| `deepseek_v4_int8` | 890p | 8 | `DeepSeek-V4-Flash-0731-W8A8-INT8` | 0.9689 seeded |
| `deepseek_v4_mxfp4` | 890p | 8 | `DeepSeek-V4-Flash-0731-MoE-Quant-W-MXFP4-A-MXFP4-Dense-Quant-W-FP8-PerChannel-A-FP8-PerToken` | 0.9689 seeded |
| `minimax-m3-fp8` | 890p | 8 | `MiniMax-M3-Quant-W-FP8-PerChannel-A-FP8-PerToken` | 0.9735 seeded |
| `glm_5_3_mxfp4` | 890p | 4 | `GLM-5.3-MoE-Quant-W-MXFP4-A-MXFP4-Dense-Quant-W-FP8-PerChannel-A-FP8-PerToken` | 0.9689 seeded |
| `glm_5_3_flash_mxfp4` | 890p | 2 | `GLM-5.3-Flash-MoE-Quant-W-MXFP4-A-MXFP4-Dense-Quant-W-FP8-PerChannel-A-FP8-PerToken` | none |
| `kimi_k2_6_mxfp4` | 890p | 8 | `Kimi-K2.6-MXFP4-FP8` | 0.9788 seeded |
| `qwen3.8-flash-next` | 890p | 4 | `Qwen3.8-Flash-Next-MoE-Quant-W-MXFP4-A-MXFP4-Dense-Quant-W-FP8-PerChannel-A-FP8-PerToken` | 0.9742 seeded |

`Cards` is the PPU count each catalog case asks the container for, not a
property of the weight or of the golden: a case may switch to DP+EP or
another TP without touching its baseline.

`deepseek_v4_int8` is listed as `890p` because the nightly lane runs on the
`8-ppu-m890p` runner even though its catalog `board` is `OAM-810E`. A run on a
real 810E host looks for `accuracy_deepseek_v4_int8_810e.json`, finds none,
and gates on the threshold instead of the wrong board's band.

`glm_5_3_flash_mxfp4` has no golden because the shared summary has no row for
that weight. It runs on the `score >= 0.9` fallback and uploads a pending-review
artifact; import the first one you accept. `qwen3.8-flash-next` is seeded from
the GSM8K number the checkpoint README publishes for this quantized export
(97.42), measured under the same sampling contract this lane runs.

`chip` here is the golden's identity, so a checkpoint evaluated on more than
one board needs one golden per board. Cases that only differ in parallel
layout (`tensor-parallel-size`) still share a golden: the band is the same
±2 points, which is what makes a TP8 baseline a meaningful reference for a
DP8+EP8 run of the same weights — if the two layouts really do diverge by more
than that, the case needs its own `model_key` rather than a wider band.

Sampling parameters per weight (`max_tokens` 10240 for every one of them):

| Model key | temperature | top_p | other | Reference row |
|-----------|-------------|-------|-------|---------------|
| `deepseek_v4_int8` | 1.0 | 1.0 | — | `deepseek_v4_flash` |
| `deepseek_v4_mxfp4` | 1.0 | 1.0 | — | `deepseek_v4_flash` |
| `minimax-m3-fp8` | 1.0 | 0.95 | — | `minimax_m3` |
| `glm_5_3_mxfp4` | 1.0 | 1.0 | — | `glm_5_3` |
| `glm_5_3_flash_mxfp4` | 1.0 | 1.0 | — | `glm_5_3` |
| `kimi_k2_6_mxfp4` | 1.0 | 0.95 | — | `kimi_k2_6` |
| `qwen3.8-flash-next` | 1.0 | 0.95 | `top_k` 20, `min_p` 0, `presence_penalty` 0, `repetition_penalty` 1.0 | every `qwen3_8` weight |
