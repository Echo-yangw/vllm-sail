#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run a minimal PPU smoke test on the assembled image; do not run arbitrary supplied test commands.
set -Eeuo pipefail
: "${TEST_IMAGE:?TEST_IMAGE must be a candidate image reference}"
: "${RESULTS_DIR:?RESULTS_DIR must be a separate directory}"
: "${WHEELS_DIR:?Tested preview wheels must be mounted}"
CHECKS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WHEELS_DIR="$(realpath "$WHEELS_DIR")"
mkdir -p "$RESULTS_DIR"
RESULTS_DIR="$(realpath "$RESULTS_DIR")"
exec 9>"${PPU_CI_LOCK_FILE:-/tmp/vllm-sail-ppu-ci.lock}"
flock -w 3600 9
args=()
for dev in /dev/alixpu /dev/alixpu_ctl /dev/alixpu_sep /dev/alixpu_ppu0; do
  [[ -e "$dev" ]] || { printf 'Missing PPU device: %s\n' "$dev" >&2; exit 2; }
  args+=(--device="$dev" --group-add="$(stat -c %g "$dev")")
done
# The candidate image must not receive host Docker configuration, tokens or GitHub environment variables.
container="sail-smoke-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"
cleanup() {
  code=$?
  trap - EXIT
  if docker container inspect "$container" >/dev/null 2>&1; then
    docker stop --time=30 "$container" >/dev/null || echo 'Warning: failed to stop container' >&2
  fi
  exit "$code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
docker run --rm --name "$container" \
  --runtime=runc --ipc=host --network=none \
  --user "$(id -u):$(id -g)" "${args[@]}" \
  --env NVIDIA_VISIBLE_DEVICES=void --env CUDA_VISIBLE_DEVICES=0 \
  --env VLLM_PLUGINS=ppu --env VLLM_SAIL_USE_PLA=1 \
  --env VLLM_USE_RUST_FRONTEND=0 --env VLLM_USE_RUST_BENCH=0 \
  --env PYTHONPYCACHEPREFIX=/tmp/pycache --env HOME=/tmp \
  --mount "type=bind,source=$RESULTS_DIR,target=/results" \
  --mount "type=bind,source=$WHEELS_DIR,target=/wheels,readonly" \
  --mount "type=bind,source=$CHECKS_DIR,target=/checks,readonly" \
  --entrypoint bash "$TEST_IMAGE" -c '
    set -Eeuo pipefail
    unset PYTHONPATH PYTHONOPTIMIZE VLLM_VERSION VLLM_VERSION_OVERRIDE
    cd /tmp
    python /checks/verify_installed.py --wheels /wheels | tee /results/installed.json
    python -m pip check
    python - <<"PY"
import json
from importlib.metadata import version
import torch
import vllm
import vllm_sail
from vllm_sail.native.extensions import import_kernels
import_kernels(strict=True)
from vllm.platforms import current_platform
assert current_platform.is_ppu(), current_platform
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
x = torch.randn((7, 260), device="cuda", dtype=torch.float32)
y = torch.empty((7, 130), device="cuda", dtype=torch.float32)
reference = torch.nn.functional.silu(x[:, :130]) * x[:, 130:]
torch.ops._C.silu_and_mul(y, x)
torch.cuda.synchronize()
torch.testing.assert_close(y, reference, rtol=1e-5, atol=1e-5)
report = {"status": "passed", "vllm": version("vllm"), "vllm_sail": version("vllm-sail"),
          "capability": list(torch.cuda.get_device_capability(0))}
with open("/results/smoke.json", "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)
print(json.dumps(report))
PY
  ' 9>&- 2>&1 | tee "$RESULTS_DIR/smoke.log"
