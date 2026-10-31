# Short model feedback loop

Use `scripts/eval/ppu_model_sanity.py` from the repository root in the selected
runtime environment. It imports only the standard library; the child interpreter
runs installed vLLM with `-I` from the fresh external attempt directory. Neither
the source checkout nor `PYTHONPATH` should supply the model runtime.

```bash
python scripts/eval/ppu_model_sanity.py --list
python scripts/eval/ppu_model_sanity.py \
  --profile deepseek-v4-community-tp8 \
  --checkpoint-root /path/to/datasets \
  --output /absolute/new-attempt-directory \
  --with-stream
```

For checkpoint evidence, also pass `--execution-proof /absolute/installed-files.json`
from the reviewed installed-wheel byte verifier. It must declare status
`complete-package-byte-maps-match-wheels`, full `commits` and `wheel_sha256`
maps for `vllm` and `vllm-sail`, and a `packages` map containing each package's
absolute `root`, distribution `version` and complete relative `files` SHA256 map.
The runner rechecks the current interpreter's origins, versions and every package
file before launching, then records those execution identities and the receipt
digest in both invocation and result. This binds an already-reviewed receipt;
it does not independently establish the receipt's source-build provenance.

The default profile is the community DeepSeek V4 Flash 0731 TP8 baseline.
The other profiles cover its MXFP4/FP8-channel TP8 and DP8/EP8 low-latency
configurations, GLM 5.2 MXFP4/FP8 TP8, and MiniMax M3 FP8-channel TP8.
Checkpoint suffixes and structured serving arguments are maintained together in
`scripts/eval/model_sanity_profiles.json`. Select the agreed dataset mount with
`--checkpoint-root`; do not silently substitute a different model or precision.
The runner binds localhost and adds a unique served-model name. It preserves
profile settings, including DP versus TP, DeepEP, graphs and DSpark.
The DP8/EP8 profile expresses the requested 30-minute process-group timeout
using upstream's `--distributed-timeout-seconds 1800` and
`--cpu-distributed-timeout-seconds 1800` arguments.
The supplied `VLLM_SAIL_PROCESS_GROUP_TIMEOUT_M` environment variable is not
consumed by this plugin; retaining it would silently leave the default timeout.

The default request asks for 17 + 25 and requires the complete final answer
`42` with natural `stop`. Reasoning without a final answer, truncation, wrong
answers, stale model identities and broken SSE fail. The default 2048-token
generation budget and 3600-second readiness budget are bounded; these are
budgets, not measured performance. Freezing a different budget before execution
is allowed; reruns retain separate evidence. A quick probe sends one ordinary
request. Qualification also uses `--with-stream`, which sends one additional
request and requires `[DONE]` and a complete final answer.

Exact serving profiles can spend tens of minutes on a cold start. Record weight
loading, graph capture and JIT warmup separately from request latency. Keep
additive compiler/model caches separate from immutable package-byte evidence;
do not disable graphs or speculation to make an unchanged-profile claim faster.

Run profiles sequentially. Before launching, inspect existing device processes
and the named container's mounts, image and active work. Wait for busy devices
unless the user explicitly authorizes terminating identified services. An idle
utilization reading does not mean allocated device memory is available. Bind
any build/runtime environment preparation to the selected image; preserve its
SDK and Torch. Do not install candidate packages into a container with live
model services. If an isolated environment or container is needed, resolve that
choice within the user's authorization before changing shared state.

Retain each attempt's invocation, runner/profile hashes, installed metadata and
paths, readiness response, ordinary response, optional SSE bytes and assembled
response, metrics, full serving log, owned group identity and final result.
The runner records its actual PID/group and the owned server's PID/group during
execution, with Linux process-start records when available. It also binds the
metrics digest and actual server exit status in the raw result. Missing execution
records cannot be reconstructed as live observations after a service has exited.
The runner checks both serving and shutdown ERROR entries and verifies owned
process-group cleanup. Read the full log and answers; warnings remain evidence.
The runner's `cleanup.forced=false` only describes its own signals. The API
server can still log forced worker termination or resource-tracker warnings;
retain and report those rather than describing shutdown as warning-free.
Never overwrite an attempt directory or treat a failure's successful cleanup as
a model pass. Missing checkpoints are failures rather than skips.

This short gate does not independently establish wheel/source provenance,
numerical kernel correctness, model accuracy, tool calls, reasoning quality, performance,
or activation of every configured backend. In particular, DSpark argv alone
does not prove accepted speculative tokens. Collect those positive metrics and
the exact commit/wheel bindings separately under [acceptance.md](acceptance.md)
and [evidence-policy.md](evidence-policy.md). Do not feed this runner's smaller
result schema directly into `verify_checkpoint.py`, fabricate its missing
receipts or weaken the frozen policy to accept it.
