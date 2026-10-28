#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# PPU performance-regression runner (kernel + model E2E lanes).
# Executed inside the PPU pod dispatched by t-head/ppu-scheduler-action, one
# lane per (chip, suite). It follows the PR49 runtime-container contract:
# same wheel install / platform self-check, but it drives tests/e2e/perf and
# turns the per-chip baseline comparison into a CI artifact + job summary.
#
# NO server is started; pytest drives vLLM's offline LLM API directly. Reviewed
# golden JSON is read from the source tree, while the run candidate and comparison
# report land in RESULTS_DIR so the outer runner can upload them for human review.
#
# Regression policy lives in _baseline_store.py: >10% relative regression WARNs
# (job summary only), >25% FAILs the pytest case (and therefore this lane). The
# first run on a chip writes a pending artifact candidate and never fails.
#
# Environment contract (injected via the action's extra_env):
#   SUITE              "kernel" or "model"
#   TEST_PATH          pytest target (e.g. tests/e2e/perf/test_kernel_perf.py)
#   PYTEST_MARKER      pytest -m expression (default: "perf")
#   WHEELS_DIR         directory holding the downloaded *.whl artifacts
#   RESULTS_DIR        absolute output dir for xml/logs/report/baseline/ppu-smi
#   HF_HOME            (model suite) HuggingFace cache root for offline model resolution
#   HF_HUB_OFFLINE     (model suite) "1" to prevent HF Hub network access
#   TRANSFORMERS_OFFLINE (model suite) "1" to prevent transformers network access
#   BASELINE_ROOT      read-only golden root (default: repo baselines dir)
#   BASELINE_OUTPUT_ROOT writable candidate root (default: RESULTS_DIR/baselines)
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

# Prevent image defaults from masking the installed version or changing frontend.
unset VLLM_VERSION
export VLLM_USE_RUST_FRONTEND=0 VLLM_USE_RUST_BENCH=0

SUITE="${SUITE:?SUITE is required (kernel|model)}"
TEST_PATH="${TEST_PATH:?TEST_PATH is required}"
PYTEST_MARKER="${PYTEST_MARKER:-perf}"
WHEELS_DIR="${WHEELS_DIR:?WHEELS_DIR is required}"
RESULTS_DIR="${RESULTS_DIR:?RESULTS_DIR is required}"
HF_HOME="${HF_HOME:-}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
BASELINE_ROOT="${BASELINE_ROOT:-${ROOT_DIR}/tests/e2e/perf/baselines}"
BASELINE_OUTPUT_ROOT="${BASELINE_OUTPUT_ROOT:-${RESULTS_DIR}/baselines}"

mkdir -p "${RESULTS_DIR}" "${BASELINE_OUTPUT_ROOT}"
exec > >(tee "${RESULTS_DIR}/perf-${SUITE}.log") 2>&1

# The internal HTTP(S) proxies do not route to NAS or loopback.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true

log() { printf '\n=== %s ===\n' "$1"; }

# ------------------------------------------------------------------
log "environment"
echo "SUITE=${SUITE} TEST_PATH=${TEST_PATH} PYTEST_MARKER=${PYTEST_MARKER:-<none>}"
echo "HF_HOME=${HF_HOME:-<none>} HF_HUB_OFFLINE=${HF_HUB_OFFLINE} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}"
echo "BASELINE_ROOT=${BASELINE_ROOT} BASELINE_OUTPUT_ROOT=${BASELINE_OUTPUT_ROOT} RESULTS_DIR=${RESULTS_DIR}"
python -c 'import sys; print("python", sys.version)'

# ------------------------------------------------------------------
log "install wheels"
WHEELS_DIR="$(cd -- "${WHEELS_DIR}" && pwd)"
verify_args=("${WHEELS_DIR}")
if [[ -n "${EXPECTED_VLLM_COMMIT:-}" ]]; then
    verify_args+=(--vllm-commit "${EXPECTED_VLLM_COMMIT}")
fi
if [[ -n "${EXPECTED_SAIL_COMMIT:-}" ]]; then
    verify_args+=(--sail-commit "${EXPECTED_SAIL_COMMIT}")
fi
python "${ROOT_DIR}/scripts/ci/ppu_wheel_manifest.py" verify "${verify_args[@]}"

mapfile -t vllm_wheels < <(
    find "${WHEELS_DIR}" -type f -name 'vllm-*.whl' -print | sort
)
mapfile -t sail_wheels < <(
    find "${WHEELS_DIR}" -type f -name 'vllm_sail-*.whl' -print | sort
)
((${#vllm_wheels[@]} == 1)) || { echo "expected one vLLM wheel under ${WHEELS_DIR}, got ${#vllm_wheels[@]}"; exit 1; }
((${#sail_wheels[@]} == 1)) || { echo "expected one vLLM SAIL wheel under ${WHEELS_DIR}, got ${#sail_wheels[@]}"; exit 1; }
echo "vllm wheel: ${vllm_wheels[0]}"
echo "sail wheel: ${sail_wheels[0]}"
# --no-deps keeps the SAIL-built torch/native ABI in the image untouched;
# --force-reinstall guarantees the freshly built wheels win over any preinstall.
python -m pip install --no-deps --force-reinstall "${vllm_wheels[0]}"
python -m pip install --no-deps --force-reinstall "${sail_wheels[0]}"

# Install test dependencies (pytest, etc.)
PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    python -m pip install -r "${ROOT_DIR}/requirements/dev.txt"

# ------------------------------------------------------------------
log "platform self-check"
cd /tmp
unset PYTHONPATH
python - <<'PY'
import torch, vllm, vllm_sail
from vllm.platforms import current_platform
from vllm_sail.compat import check_vllm_compatibility

check_vllm_compatibility(force=True)
assert current_platform.is_ppu(), f"expected PPU platform, got {current_platform}"
assert torch.cuda.is_available(), "perf pod requires a PPU device"
print("Device:", current_platform.get_device_name(0))
print("Capability:", current_platform.get_device_capability())
print("vLLM:", vllm.__version__)
print("vLLM SAIL:", vllm_sail.__version__)
PY

# ------------------------------------------------------------------
# For the model suite, export offline environment variables so that vLLM resolves
# HuggingFace model IDs from the local cache without network access. HF_HOME is
# inherited from the pod/runner environment (PR 88 contract); it is never
# hardcoded per-model.
if [[ "${SUITE}" == "model" ]]; then
    export HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
    if [[ -n "${HF_HOME}" ]]; then
        export HF_HOME
        test -d "${HF_HOME}" || { echo "HF_HOME directory missing: ${HF_HOME}"; exit 1; }
        log "model weights via HF_HOME=${HF_HOME} (offline=${HF_HUB_OFFLINE})"
    else
        log "model weights via default HF cache (offline=${HF_HUB_OFFLINE})"
    fi
fi

# ------------------------------------------------------------------
log "pytest perf ${SUITE}"
# Isolate the test run: copy tests/ and pyproject.toml into a clean directory
# that does NOT contain vllm_sail/.  pyproject.toml's pythonpath=["."] is now
# safe — "." only exposes tests/, so the installed wheel is never shadowed.
PERF_RUN_DIR="/tmp/ppu-perf-run"
rm -rf "${PERF_RUN_DIR}"
mkdir -p "${PERF_RUN_DIR}"
cp -r "${ROOT_DIR}/tests" "${PERF_RUN_DIR}/tests"
cp "${ROOT_DIR}/pyproject.toml" "${PERF_RUN_DIR}/pyproject.toml"
cd "${PERF_RUN_DIR}"

# The conftest perf_baseline fixture emits perf_report.{json,md} under this dir
# at session teardown; keep it inside RESULTS_DIR for artifact upload.
export VLLM_SAIL_PERF_REPORT_DIR="${RESULTS_DIR}/report"
mkdir -p "${VLLM_SAIL_PERF_REPORT_DIR}"

pytest_args=("${TEST_PATH}" -q -rs -m "${PYTEST_MARKER}"
             --junitxml="${RESULTS_DIR}/perf-${SUITE}.xml"
             --baseline-root="${BASELINE_ROOT}"
             --baseline-output-root="${BASELINE_OUTPUT_ROOT}")

set +e
python -m pytest "${pytest_args[@]}"
rc=$?
set -e

# ------------------------------------------------------------------
log "comparison table"
# Surface the per-metric current/baseline/delta table on stdout and, when the
# outer runner exposes it, in the GitHub job summary.
report_md="$(find "${VLLM_SAIL_PERF_REPORT_DIR}" -type f -name 'perf_report.md' | head -n 1 || true)"
if [[ -n "${report_md}" && -f "${report_md}" ]]; then
    cat "${report_md}"
    if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
        {
            echo "## perf regression - ${SUITE}"
            echo
            cat "${report_md}"
        } >> "${GITHUB_STEP_SUMMARY}"
    fi
else
    echo "no perf_report.md emitted (no perf case ran?)"
fi

# ------------------------------------------------------------------
log "collect evidence"
# The fixture writes the complete run candidate under BASELINE_OUTPUT_ROOT.
# Reviewed golden files remain untouched in BASELINE_ROOT.
find "${BASELINE_OUTPUT_ROOT}" -type f -name '*.json' -print || true

# ppu-smi occupancy evidence (best-effort; the tool may not exist in all images).
ppu-smi > "${RESULTS_DIR}/ppu-smi.log" 2>&1 || true

# ------------------------------------------------------------------
if (( rc != 0 )); then
    log "PERF ${SUITE} FAILED (pytest exit ${rc})"
    exit "${rc}"
fi
log "PERF ${SUITE} PASS"
