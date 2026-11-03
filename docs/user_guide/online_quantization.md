# Loading-time online quantization

vLLM SAIL registers `fp8-to-int8`, `bf16-to-int8`, and `fp8-to-bf16` on PPU.
Conversion runs once after each layer loads; no separate quantized checkpoint or
complete intermediate BF16 model is required. INT8 execution reuses the existing
PPU Dense and MoE backends, with dynamic per-token activation quantization.

## Select a path

```bash
vllm serve /path/to/fp8-model --quantization fp8-to-int8 --dtype bfloat16
vllm serve /path/to/bf16-model --quantization bf16-to-int8 --dtype bfloat16
vllm serve /path/to/fp8-model --quantization fp8-to-bf16 --dtype bfloat16
```

The Python API accepts the same names through `LLM(quantization=..., dtype="bfloat16")`.
FP8 sources must declare `quant_method: fp8`, `weight_block_size: [128, 128]`,
E4M3 weights, and dynamic activations. BF16 sources must be unquantized checkpoints;
weight loaders reject other storage dtypes before implicit casting. Both paths
require BF16 activation dtype. FP8 decoding cannot recover precision lost during
original checkpoint quantization.

## Recipe selection

Source storage metadata and target precision are independent:

| Source | Matching model recipe | No matching model recipe |
| --- | --- | --- |
| FP8 | Apply recipe | Convert originally FP8 layers to INT8; preserve original BF16 layers |
| BF16 | Apply recipe | Apply bundled `_default` recipe |

FP8-to-BF16 is a standalone decoding path and does not apply an INT8 recipe.
An FP8 layer excluded by an INT8 recipe is still decoded using its scales before
BF16 execution. A BF16 layer selected by a recipe can be quantized to INT8.
Recipe rules never change the source dtype or replace source scale metadata.
Embeddings remain outside the Linear/MoE conversion interface.

Bundled policies are versioned JSON files in
`vllm_sail/model_executor/layers/quantization/online_conversion/recipes/`.
They adapt the qlean 0.2.0 layer rules carried by the original implementation;
the Qwen3.8-27B policy additionally records its offline INT8 reference and model
shape constraints. These rules are not hardware/model qualification claims.

Automatic selection matches the model identifier and local directory basename
against the bundled patterns, case-insensitively. Multiple matches are an error.
For renamed model directories, choose a bundled ID explicitly:

```bash
vllm serve /models/renamed --quantization bf16-to-int8 --dtype bfloat16 \
  --hf-overrides '{"sail_quantization_recipe":"qwen3_8_27b"}'
```

The legacy `ppu_quantization_recipe` override remains accepted; an explicit
`sail_quantization_recipe` takes precedence. Unknown IDs fail; paths and executable
recipes are not accepted. Recipe-provided
structural constraints apply to both automatic and explicit selection. Mixed-layer
indices must fit the model depth. `_default` can also be explicitly selected.
Logs report the source dtype and selected recipe ID, or `checkpoint` for the FP8
fallback. If a name matches a recipe but its constraints fail, loading fails rather
than silently falling back.

Schema version 1 supports symmetric per-output-channel W8 weights and dynamic
symmetric per-token A8 activations only. `ignored_layers` preserves floating
execution; `mix_layers` preserves the listed transformer blocks in BF16.
The default policy excludes embeddings, LM heads, visual towers, routers,
convolutions and selected hybrid-model projections. It is an explicit generic
policy, not an assertion that every unknown architecture has been accuracy-tested.

Checkpoint exclusion names are mapped independently from recipe names through the
model's weight mapper. Fused QKV/gate-up shards must agree on source storage and,
separately, target precision. Conflicting fused selections fail before loading.
Checkpoint `mix_layer` describes floating whole blocks; checkpoint `mix_layers`
retains the source fork's floating Linear-module selection. These source fields do
not override a selected target recipe. Without a matching recipe, original
floating modules remain floating. Recipe `mix_layers` independently selects target
floating blocks. Cached `scales_path` is unsupported.

`ignored_layers` takes precedence over `modules_to_not_convert` when nonempty,
matching the source loader. Explicit `fmt: null` is treated as unspecified E4M3.
The standalone `int8_quant_config.json` source metadata file is also accepted.

## Defaults and compatibility

Installing the plugin does not enable conversion automatically. Explicit quantization
arguments take precedence over environment policies; `--quantization fp8` retains
native FP8 behavior. BF16 conversion requires `--quantization bf16-to-int8`.

With no explicit quantization argument, `VLLM_SAIL_FP8_LOAD_POLICY` selects
`native`, `bf16`, or `int8`. Unset or empty retains native behavior. The corresponding
`VLLM_PPU_FP8_LOAD_POLICY` alias is accepted; SAIL takes precedence.

If no load policy is set, an explicitly set `VLLM_SAIL_ENABLE_FP8_TO_INT8`
(or legacy `VLLM_PPU_ENABLE_FP8_TO_INT8`) selects INT8 for `1` and BF16 for `0`.
Leaving it unset does not change the default. Composition with another upstream
online quantizer is rejected.

`--dtype auto` on PPU FP8 checkpoints resolves to BF16, including nested text
metadata. Explicit BF16 and FP16 activation dtypes remain accepted. Source decoding
and INT8 scale computation retain their fixed numerical precision regardless of
activation dtype.

## Numerical and parallel contract

FP8 decoding multiplies E4M3 values by 128-by-128 block scales in FP32, then rounds
to BF16. INT8 conversion uses FP32 row absmax divided by 127, clamped to FP32 epsilon;
weights are rounded and clipped to [-127, 127]. For the same source, recipe and
numerical implementation, online and offline INT8 weights and scales must match
exactly. Compare corresponding logical tensors after undoing layout transformations;
any mismatch must be investigated rather than accepted with a tolerance.

FP8 fused boundaries and TP shards must align to 128-element blocks. BF16 sources
do not inherit this FP8 alignment restriction. Independent MoE gate/up tail blocks
retain independent scale grids. Input-dimension shards share row maxima through
TP MAX reduction (and DP/PCP axes for flattened MoE shards), so scales represent
the full unsharded row.

MoE decoding and scale statistics exclude padding; target padding is zero-filled.
BF16 MoE execution inherits the unquantized backend's EPLB capability. INT8 MoE
inherits the target method's capability limits. Preprocessed source weights and
weight reload are not supported workflows. Restart with the source checkpoint to change conversion mode.
Backend-specific shape and execution restrictions still apply.

Conversion avoids a full intermediate BF16 model, but still temporarily holds the
source layer, converted weights and quantization scratch. MoE converts one expert
projection at a time. Peak memory and cold-start cost need measurement for the
actual checkpoint and TP/EP configuration.

## Verification

Dependency-free policy and registration tests:

```bash
python -m pytest tests/ut/test_online_conversion.py tests/ut/test_online_recipes.py -q
```

Torch numerical tests and a real two-rank Gloo scale-reduction check:

```bash
python -m pytest tests/tensor/test_online_conversion*.py -q
```

Real vLLM 0.31 CPU parameter/loader tests (device kernels mocked):

```bash
python -m pytest tests/integration/test_online_conversion_loading.py -q
```

Installed PPU environment, with qlean installed for the reference gate:

```bash
python -m pytest tests/e2e/test_online_conversion_device.py \
  tests/e2e/test_online_conversion_qlean.py -q --junitxml=online-conversion.xml
```

The qlean gate compares FP8 decoding, INT8 values and scales on the same device and
records dependency versions. Missing qlean or skipped device tests are not evidence
of equivalence. Full-model validation must cover Dense and MoE, TP1 and TP2+, recipe
and fallback selection, eager and graph execution. Compare weights/scales, prefill
and decode results, generated token IDs and stop reasons under fixed checkpoint
revisions, prompt IDs, sampling, SDK and backend settings. Run an A/A repeatability
check before attributing differences to conversion. Record loading time and peak
memory separately from steady-state inference performance.

These implementation paths have local tests; full PPU and model-level qualification
must be completed for each supported deployment before making accuracy claims.

### Model-run artifacts

`tools/validate_online_quantization.py` captures each run in a fresh process.
Provide an engine JSON containing ordinary `LLM` arguments (including model,
revision, quantization, TP size, `hf_overrides` and `enforce_eager`) and a prompts
JSON list with unique `id` and `prompt_token_ids` entries. Optional `expected_text`
is scored by stripped exact match; it is not a benchmark-specific answer grader.

```bash
python tools/validate_online_quantization.py run \
  --engine-config candidate-engine.json --prompts prompts.json --output candidate.json
python tools/validate_online_quantization.py run \
  --engine-config reference-engine.json --prompts prompts.json --output reference.json
python tools/validate_online_quantization.py compare \
  --reference reference.json --candidate candidate.json --output comparison.json
```

Use an equivalent qlean-generated checkpoint for the reference engine, with the
same tokenizer/prompt IDs and runtime settings. Repeat the reference run for A/A.
The comparison checks token IDs, stop reasons, prefill and decode top-logprobs
(default absolute tolerance 0.001). It reports per-rank runtime INT8 weight/scale
hash equality; `--require-weight-match` makes unequal hashes a failure when both
runs use the same packing, parameter names and TP configuration.

Artifacts record package versions, startup wall time, inference wall time and
per-worker peak allocated/reserved device memory. Startup includes model loading,
KV allocation and graph capture; its peak is not an isolated quantization-memory
measurement. Keep backend and SAIL SDK identity with the run artifacts.

Weight comparison is mandatory by default and uses exact hashes, including retained
floating weights and biases. Empty weight captures fail this check. Use
`--no-require-weight-match` only for generation diagnostics; that mode does not
qualify online/offline quantization equivalence. Logprob tolerance applies only to
inference outputs, never to captured weights or scales.
