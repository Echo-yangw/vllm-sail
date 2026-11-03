# Verification Guide

Validate the same package installation, SDK and hardware that will run the
workload. Keep the results of each check separate:

| Check | What it establishes |
| --- | --- |
| CPU unit tests | Python behavior, registration contracts and packaging invariants |
| Source regeneration checks | Generated upstream kernels match their declared inputs |
| HGGC build | Selected sources compile and link for the requested targets |
| Extension imports and strict preflight | Libraries load and representative operations are registered |
| Device correctness tests | Tested kernels match their numerical references on that device |
| Model run | The tested checkpoint and configuration complete inference |
| Benchmark | Performance under the recorded workload and environment |

Passing one check does not imply the later checks pass. A skipped test provides
no execution evidence for that feature.

## 1. Record the environment

After following [Installation](../getting_started/installation.md), run:

```bash
git rev-parse HEAD
python -c "from vllm_sail import collect_env; collect_env()"
python -c "import vllm_sail; print(vllm_sail.__file__)"
```

Record the PPU model and count, SDK version, SAIL PyTorch version, vLLM revision
and plugin revision. Confirm the printed package path points to the installation
being tested. A source checkout can shadow a separately installed native wheel.

Run the four-library import and strict preflight commands in
[Check the installation](../getting_started/installation.md#4-check-the-installation).
Resolve shared-library errors before running a model.

## 2. Run unit tests

From the plugin checkout, using its test environment:

```bash
python -m pytest tests/ut -q
```

The merge gate runs with only `requirements/dev.txt` installed, without torch,
vLLM or a device. Additional integration tests become available in a PPU
environment with their dependencies installed. Investigate failures and report
skip reasons; do not compare test counts from different environments as if they
cover the same paths.

Compatibility UT may intentionally import the plugin checkout beside installed
vLLM and torch. Record those import paths and set `VLLM_SOURCE_ROOT` to the same
exact upstream revision even when source-marked cases are deselected. This UT
mode does not verify the installed plugin wheel: separately compare its full
package contents with the wheel and run native imports, device tests and models
from that verified installation.

### Check upstream source compatibility

Tests live with the feature they cover, independently of the vLLM release that
introduced them. Tests marked `upstream_source` read a selected vLLM checkout
without importing vLLM, torch or the SDK:

```bash
VLLM_SOURCE_ROOT=/path/to/vllm python -m pytest tests/ut -q -rs \
  -m upstream_source --require-upstream-source
```

`VLLM_SOURCE_ROOT` names the checkout root containing `vllm/__init__.py`.
Without it, ordinary CPU runs skip source-dependent checks. The required-source
option makes a missing or invalid checkout an error, so CI cannot silently
skip these checks. The source tests cover selected upstream call sites, keyword
compatibility, quantization behavior and imported consumer inventories. They do
not establish that module imports, patch installation or device inference work.

### CI jobs

The [unit-test workflow](../../.github/workflows/ut.yaml) runs for pull requests,
pushes to `main` or `master`, and manual dispatch. Its jobs run independently:

| Job | Environment and checks | Failure handling |
| --- | --- | --- |
| CPU UT | Python 3.10–3.13; only `requirements/dev.txt`; UT and branch coverage | Test failure fails the workflow; coverage-upload failure does not |
| vLLM source (main-verified) | Python 3.12; checks out the exact verified main commit; runs `upstream_source` tests with a required source path | Checkout or test failure fails the workflow |

The source job reads the full SHA in
[verified main commit](../../.github/vllm-main-verified.commit) and logs the
resolved commit. Update this pin only after qualifying that exact revision.
Keep feature tests named after their behavior. The workflow does not install
vLLM or run PPU workloads; installed compatibility and device checks require
the prepared environment below. Repository branch protection determines which
job results are required for merging.

The separate [upstream canary](../../.github/workflows/upstream-canary.yaml) runs
weekly on Tuesday at 03:17 UTC and on manual dispatch. It installs upstream
`main` and checks patch drift. A drift failure opens or updates the tracking
issue and fails the job. Installation failure stops before that report step.
This scheduled check does not replace PR checks, and none of these public CPU
jobs runs PPU inference or `tests/e2e`.

## 3. Run device correctness tests

In the PPU environment, install `requirements/dev.txt`. Use an editable native
plugin build in the checkout, or ensure the test process imports the installed
wheel as described above. Start with activation and the affected feature:

```bash
pytest tests/e2e/test_native_activation.py -q -rs
pytest tests/e2e/test_deepseek_v4_native.py -q -rs
pytest tests/e2e/test_mxfp4_prepare.py -q -rs
pytest tests/e2e/tier_b/test_norm_quant.py -q -rs
```

The [`tests/e2e`](../../tests/e2e) directory also contains linear-attention,
indexer, cache and model-specific kernel tests. Select the relevant tests for
the change; run the full suite when qualifying an environment:

```bash
pytest tests/e2e -q -rs
```

These tests require a real PPU and the applicable SDK libraries. Verify that the
intended tests actually ran. Include dtype, shape, tolerance and device in a
correctness report, and test both architecture targets when claiming both.

Capability markers restrict tests to one chip: `cap80` runs only on compute
capability 8.0 (Zhenwu 810E) and `cap89` only on 8.9 (Zhenwu M890); other
devices skip them. `perf` marks performance benchmarks and `model_e2e` marks
offline model end-to-end tests. Select or exclude them with `-m`, for example
`pytest tests/e2e -q -rs -m "not perf"`.

### GDN fused MTP decode

Run the native GDN MTP accuracy matrix with a native plugin build:

```bash
python -m pytest tests/e2e/gdn/test_gdn_fused_mtp.py -q -rs \
  -o junit_family=xunit1 --junitxml=gdn-mtp-accuracy.xml
```

The tests call the native fused op directly and force the unfused reference onto
Triton, bypassing PLA. Selected cases also compare with an independent eager
PyTorch reference. On PPU, a missing native dispatch registration fails the
suite rather than skipping it.

Coverage includes all supported head ratios, BF16/FP32 recurrent state and norm
weights, FP16/BF16/FP32 gate bias, SiLU/sigmoid output gates, MTP lengths 1–8,
strided rows and state slots, padding and NULL slots, and repeated accepted-token
rewind with changing request lengths. Every round checks outputs, candidate
state writes, unused state storage and output-buffer guards. The suite also
tests numerical boundaries and the layer's post-convolution entry point.

The native-versus-Triton output keeps upstream's relative-L2 limit of `5e-4`;
pointwise output/state comparisons retain the upstream model-path tolerances
`atol=rtol=3e-2`. Each written slot additionally has a relative-L2 limit of `3e-2`.
Untouched slots and buffer guards must remain bitwise identical. The JUnit file
records relative-L2 and maximum absolute errors per round. These limits are
source-derived, not calibrated from a PPU run; investigate failures before
changing them.

The independent oracle has worked-example CPU tests that run when torch is
available:

```bash
python -m pytest tests/ut/test_gdn_mtp_reference.py -q
```

CPU oracle tests and device-test collection do not establish native accuracy.
Record actual execution counts and skips from the PPU command above.

## 4. Run a model

### Quick serving sanity

The standalone sanity runner launches one installed model, checks readiness
against a unique served-model identity, sends a deterministic arithmetic request,
checks its complete final answer and stops its owned server process group:

```bash
python scripts/eval/ppu_model_sanity.py --list
python scripts/eval/ppu_model_sanity.py \
  --profile deepseek-v4-community-tp8 \
  --checkpoint-root /path/to/datasets \
  --output /absolute/new-sanity-attempt \
  --with-stream
```

Profiles cover community DeepSeek V4 Flash 0731 TP8, its MXFP4/FP8-channel TP8
and DP8/EP8 low-latency variants, GLM 5.2 MXFP4/FP8 TP8, and MiniMax M3
FP8-channel TP8. Checkpoint suffixes and serving arguments are recorded in
[`model_sanity_profiles.json`](../../scripts/eval/model_sanity_profiles.json).
The dataset root defaults to `/ppusw/datasets`; set it to the actual mounted
root. Each attempt requires a new output directory. Run profiles sequentially
on free devices in the selected environment, preserving existing work.

The default sends one ordinary request; `--with-stream` adds a streamed request.
Both must return the final answer `42` and finish naturally. Truncated output,
reasoning without a final answer, a wrong answer, HTTP failure, serving or
shutdown ERROR logs, and failed cleanup fail the attempt. The process exits
nonzero on failure and retains its invocation, installed package metadata,
responses, metrics, serving log and result. Adjust readiness, request and token
budgets explicitly with `--ready-timeout`, `--request-timeout`, and `--max-tokens`.
Each request budget covers connection setup, response headers and the complete
body, including a streamed response that never finishes a line. Expiration
fails the attempt and enters service cleanup; cleanup has its own bounded wait.

This is a short inference check for the selected configuration. It does not
replace the accuracy evaluation below, kernel correctness checks, package-byte
verification or evidence that every configured optimization was active.

### Model verification

Follow the [quickstart](../getting_started/quickstart.md) with a checkpoint
appropriate for the available memory and supported precision. Confirm that:

- The PPU platform and intended compute backend are selected.
- Model loading and warm-up complete.
- The readiness endpoint succeeds and generation returns sensible output.
- Repeated generation and the intended context lengths complete successfully.

For MoE, quantized or multi-device workloads, repeat the check using the intended
expert backend, checkpoint layout and parallelism. A small dense-model smoke
test does not cover those paths.

## 5. Report correctness and performance

Include the exact command, checkpoint identifier and revision, dtype,
quantization, context length, device count and parallelism. For benchmarks, also
include the workload or dataset, input/output lengths, concurrency, warm-up,
throughput and latency measurements.

Useful tools include vLLM's `vllm bench --help` and the repository's
[`benchmarks/kernels`](../../benchmarks/kernels). Compare baseline and candidate
with the same software, hardware and workload. Explain the numerical reference
and tolerance used for kernel comparisons.

Report which checks passed, failed, skipped or were not run. Attach concise
logs and a minimal reproducer through the repository's issue or pull request
template; remove credentials and private model or host details before posting.
