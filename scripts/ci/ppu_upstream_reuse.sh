#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Consume validated CI wheels; never rebuild or replace the preloaded SDK/Torch.
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${WHEELS_DIR:?WHEELS_DIR is required}"
: "${REUSE_RESULTS_DIR:?REUSE_RESULTS_DIR is required}"
: "${VLLM_SOURCE_ROOT:?VLLM_SOURCE_ROOT is required}"
: "${EXPECTED_VLLM_COMMIT:?EXPECTED_VLLM_COMMIT is required}"
: "${EXPECTED_SAIL_COMMIT:?EXPECTED_SAIL_COMMIT is required}"
: "${TESTED_SAIL_COMMIT:?TESTED_SAIL_COMMIT is required}"
for commit in "${EXPECTED_VLLM_COMMIT}" "${EXPECTED_SAIL_COMMIT}" "${TESTED_SAIL_COMMIT}"; do
    [[ "${commit}" =~ ^[0-9a-f]{40}$ ]] || { echo "Reuse requires immutable source SHAs" >&2; exit 2; }
done
mkdir -p "${REUSE_RESULTS_DIR}"
exec > >(tee "${REUSE_RESULTS_DIR}/environment.log") 2>&1
printf '810E skipped: runner unavailable\n'
printf 'tested_sha=%s\nwheel_source_sha=%s\nupstream_sha=%s\n' \
    "${TESTED_SAIL_COMMIT}" "${EXPECTED_SAIL_COMMIT}" "${EXPECTED_VLLM_COMMIT}"
unset VLLM_VERSION PYTHONPATH GH_TOKEN GITHUB_TOKEN
export VLLM_USE_RUST_FRONTEND=0 VLLM_USE_RUST_BENCH=0
export VLLM_PLUGINS=ppu
export EXPECTED_VLLM_COMMIT EXPECTED_SAIL_COMMIT TESTED_SAIL_COMMIT
# All invocations are fail-closed, including manual consumers of donor wheels.
python "${ROOT_DIR}/scripts/ci/ppu_wheel_manifest.py" verify "${WHEELS_DIR}" \
    --vllm-commit "${EXPECTED_VLLM_COMMIT}" --sail-commit "${EXPECTED_SAIL_COMMIT}"

# A tests-only working directory prevents checkouts from shadowing installed wheels.
WORK_DIR="$(mktemp -d /tmp/ppu-upstream-reuse.XXXXXX)"
export PIP_CONSTRAINT="${REUSE_RESULTS_DIR}/image-constraints.txt"
cd "${WORK_DIR}"
python "${ROOT_DIR}/scripts/ci/ppu_image_stack.py" capture \
    --sdk /usr/local/PPU_SDK --output "${REUSE_RESULTS_DIR}/image-stack.json" \
    --constraints "${PIP_CONSTRAINT}"
cleanup() {
    code=$?
    trap - EXIT
    if ! python "${ROOT_DIR}/scripts/ci/ppu_image_stack.py" verify \
        --sdk /usr/local/PPU_SDK --output "${REUSE_RESULTS_DIR}/image-stack.json"; then
        code=1
    fi
    cd /tmp
    rm -rf "${WORK_DIR}"
    if [[ "${code}" == "0" ]]; then
        echo "PPU UPSTREAM REUSE PASS (M890)"
    fi
    exit "${code}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
python -m pip install --no-index --no-deps --force-reinstall \
    "${WHEELS_DIR}"/vllm/*.whl "${WHEELS_DIR}"/vllm-sail/*.whl
PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    python -m pip install --no-cache-dir -r "${ROOT_DIR}/requirements/upstream-reuse.txt"
python "${ROOT_DIR}/scripts/ci/ppu_image_stack.py" verify \
    --sdk /usr/local/PPU_SDK --output "${REUSE_RESULTS_DIR}/image-stack.json"

python - <<'PY'
import json
import os
from pathlib import Path

import numba
import tblib
import torch
import vllm
import vllm_sail
from vllm.platforms import current_platform
from vllm_sail.compat import check_vllm_compatibility

check_vllm_compatibility(force=True)
assert current_platform.is_ppu(), current_platform
assert torch.cuda.is_available(), "reuse requires a real PPU driver/device"
count = torch.cuda.device_count()
assert count == 1, f"M890 single-card acceptance requires one visible card, got {count}"
name = torch.cuda.get_device_name(0)
capability = tuple(torch.cuda.get_device_capability(0))
platform_capability = current_platform.get_device_capability(0)
assert capability == (8, 9), f"non-M890 device: {name}, capability={capability}"
assert tuple(platform_capability) == (8, 9), platform_capability
assert "890" in name.upper(), f"non-M890 device name: {name}"
paths = {"torch": torch.__file__, "vllm": vllm.__file__, "vllm_sail": vllm_sail.__file__}
for package, path in paths.items():
    resolved = Path(path).resolve()
    assert not any(resolved.is_relative_to(root) for root in (Path("/source"), Path("/upstream"))), (package, path)
identity = {
    "tested_sha": os.environ["TESTED_SAIL_COMMIT"],
    "wheel_source_sha": os.environ["EXPECTED_SAIL_COMMIT"],
    "upstream_sha": os.environ["EXPECTED_VLLM_COMMIT"],
    "device_name": name,
    "device_count": count,
    "capability": capability,
    "platform": type(current_platform).__name__,
    "is_ppu": current_platform.is_ppu(),
    "cuda_available": torch.cuda.is_available(),
    "torch_version": str(torch.__version__),
    "numba_version": str(numba.__version__),
    "tblib_version": str(tblib.__version__),
    "imports": paths,
    "810e": "skipped: runner unavailable",
}
print(json.dumps(identity, sort_keys=True))
(Path(os.environ["REUSE_RESULTS_DIR"]) / "runtime-identity.json").write_text(
    json.dumps(identity, sort_keys=True, indent=2) + "\n"
)
PY

cp -a "${ROOT_DIR}/tests" "${WORK_DIR}/tests"
cp "${ROOT_DIR}/pyproject.toml" "${WORK_DIR}/pyproject.toml"
python -m pytest tests/upstream_reuse -q -rs -m upstream_source \
    --require-upstream-source --strict-upstream-reuse \
    --upstream-reuse-target=890p --upstream-reuse-report-dir=/artifacts/reuse \
    2>&1 | tee "${REUSE_RESULTS_DIR}/pytest.log"
