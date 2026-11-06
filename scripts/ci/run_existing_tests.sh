#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run existing upstream vllm-sail tests in the same candidate image without installing or replacing the wheel under test.
set -Eeuo pipefail
: "${TEST_IMAGE:?A built image@sha256 reference must be configured}"
: "${SOURCE_DIR:?Must be an upstream checkout matching the wheel manifest}"
: "${RESULTS_DIR:?A results directory must be configured}"
: "${WHEELS_DIR:?Must be the verified original wheel directory}"
CHECKS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WHEELS_DIR="$(realpath "$WHEELS_DIR")"
PPU_BOARD="${PPU_BOARD:-ppu15}"
[[ "$PPU_BOARD" == ppu15 || "$PPU_BOARD" == ppu10 ]] || exit 2
[[ "$TEST_IMAGE" =~ @sha256:[0-9a-f]{64}$ ]] || { echo 'nightly must test an immutable digest' >&2; exit 2; }
SOURCE_DIR="$(realpath "$SOURCE_DIR")"
mkdir -p "$RESULTS_DIR"
RESULTS_DIR="$(realpath "$RESULTS_DIR")"
exec 9>"${PPU_CI_LOCK_FILE:-/tmp/vllm-sail-ppu-ci.lock}"
flock -w 3600 9
args=()
for dev in /dev/alixpu /dev/alixpu_ctl /dev/alixpu_sep /dev/alixpu_ppu0; do
  [[ -e "$dev" ]] || { printf 'Missing PPU device: %s\n' "$dev" >&2; exit 2; }
  args+=(--device="$dev" --group-add="$(stat -c %g "$dev")")
done
container="sail-existing-tests-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"
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
docker run --rm --name "$container" --runtime=runc --ipc=host --network=none \
  --user "$(id -u):$(id -g)" "${args[@]}" \
  --env NVIDIA_VISIBLE_DEVICES=void --env CUDA_VISIBLE_DEVICES=0 \
  --env VLLM_PLUGINS=ppu --env VLLM_SAIL_USE_PLA=1 \
  --env VLLM_USE_RUST_FRONTEND=0 --env VLLM_USE_RUST_BENCH=0 \
  --env PYTHONPYCACHEPREFIX=/tmp/pycache --env HOME=/tmp \
  --env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 --env PPU_BOARD="$PPU_BOARD" \
  --env OBSERVED_IMAGE="$TEST_IMAGE" \
  --mount "type=bind,source=$SOURCE_DIR,target=/source,readonly" \
  --mount "type=bind,source=$WHEELS_DIR,target=/wheels,readonly" \
  --mount "type=bind,source=$CHECKS_DIR,target=/checks,readonly" \
  --mount "type=bind,source=$RESULTS_DIR,target=/results" \
  --entrypoint bash "$TEST_IMAGE" -c '
    set -Eeuo pipefail
    unset PYTHONPATH PYTEST_ADDOPTS PYTHONOPTIMIZE VLLM_VERSION VLLM_VERSION_OVERRIDE
    cd /tmp
    python /checks/verify_installed.py --wheels /wheels | tee /results/installed.json
    cd /source
    python - <<"PY"
import shutil
from scripts.ci.ppu_ops_runner import copy_test_tree
copy_test_tree("/source", "/tmp/sail-existing-tests")
shutil.copytree("/source/tests/ut", "/tmp/sail-existing-tests/tests/ut")
PY
    cd /tmp/sail-existing-tests
    python -m pip check
    python -m pytest -c pyproject.toml tests/ut/test_patch_install.py \
      -k "patches_actually_land or install_is_idempotent" \
      -q -rs --junitxml=/results/ut.xml
    PPU_CI_CHECKOUT=/source PPU_CI_ISOLATED_ROOT=/tmp/sail-existing-tests \
      python -m pytest -c pyproject.toml -p tests.support.ppu_ci \
      tests/e2e/test_native_activation.py -q -rs \
      --ppu-ci-config /source/scripts/ci/ppu_ops_tests.json \
      --ppu-ci-board "$PPU_BOARD" --ppu-ci-report /results/nodes.json \
      --junitxml=/results/ops.xml
    cd /tmp
    python /checks/runtime_probe.py
  ' 9>&- 2>&1 | tee "$RESULTS_DIR/existing-tests.log"
# The caller then verifies the 2+9 actually executed results in XML and nodes.json; skips do not count as passes.
