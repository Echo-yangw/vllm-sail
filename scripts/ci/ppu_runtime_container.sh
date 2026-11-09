#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run wheel validation in an isolated PPU runtime container.
set -Eeuo pipefail

mode=${1:?Usage: ppu_runtime_container.sh pin-ut|model-test|model-accuracy|kernel-perf|upstream-reuse}
case "${mode}" in
    pin-ut|model-test|model-accuracy|kernel-perf|upstream-reuse) ;;
    *) printf 'Unknown runtime mode: %s\n' "${mode}" >&2; exit 2 ;;
esac

: "${PPU_CI_SOURCE_DIR:?PPU_CI_SOURCE_DIR is required}"
: "${PPU_CI_SOURCE_SHA:?PPU_CI_SOURCE_SHA is required}"
: "${PPU_CI_WHEELS_DIR:?PPU_CI_WHEELS_DIR is required}"
: "${PPU_CI_ARTIFACT_DIR:?PPU_CI_ARTIFACT_DIR is required}"
: "${PPU_CI_DEVICE_COUNT:?PPU_CI_DEVICE_COUNT is required}"
: "${PPU_CI_REQUIRE_PR_ARTIFACT:?PPU_CI_REQUIRE_PR_ARTIFACT is required}"

source_dir=$(realpath "${PPU_CI_SOURCE_DIR}")
wheels_dir=$(realpath "${PPU_CI_WHEELS_DIR}")
mkdir -p "${PPU_CI_ARTIFACT_DIR}"
artifact_dir=$(realpath "${PPU_CI_ARTIFACT_DIR}")
git -C "${source_dir}" rev-parse --is-inside-work-tree >/dev/null
[[ "$(git -C "${source_dir}" rev-parse HEAD)" == "${PPU_CI_SOURCE_SHA}" ]] || {
    echo "PPU CI source checkout does not match PPU_CI_SOURCE_SHA" >&2
    exit 2
}
[[ -d "${wheels_dir}" ]] || { echo "PPU CI wheel directory is missing" >&2; exit 2; }
[[ "${PPU_CI_REQUIRE_PR_ARTIFACT}" =~ ^(true|false)$ ]] || {
    echo "PPU_CI_REQUIRE_PR_ARTIFACT must be true or false" >&2
    exit 2
}
expected_vllm_commit=$(tr -d '[:space:]' < "${source_dir}/.github/vllm-main-verified.commit")
expected_sail_commit="${PPU_CI_WHEEL_SOURCE_SHA:-${PPU_CI_SOURCE_SHA}}"
for commit in "${expected_vllm_commit}" "${expected_sail_commit}"; do
    [[ "${commit}" =~ ^[0-9a-f]{40}$ ]] || {
        echo "PPU CI expected commits must be immutable SHA-1 values" >&2
        exit 2
    }
done
[[ "${PPU_CI_DEVICE_COUNT}" =~ ^[1-8]$ ]] || {
    echo "PPU_CI_DEVICE_COUNT must be between 1 and 8" >&2
    exit 2
}

# transformers builds the cache for remote model code under HF_MODULES_CACHE,
# which defaults to ${HF_HOME}/modules. HF_HOME is mounted read-only below, so
# point it at a writable path; a weight that ships its own tokenizer stops at
# engine start with EROFS instead of loading.
hf_modules_cache=/tmp/hf-modules

exec 9>/tmp/vllm-sail-ppu-ci.lock
flock -n 9 || { echo "Another vLLM SAIL PPU job is active on this runner" >&2; exit 2; }

uid=$(id -u)
gid=$(id -g)
prefix="vllm-sail-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}-${mode}"
container=""
image_id=""
cleanup() {
    code=$?
    trap - EXIT
    set +e
    if [[ -n "${container}" ]]; then
        docker stop --time=2 "${container}" 9>&- >/dev/null 2>&1 || true
        docker rm --force "${container}" 9>&- >/dev/null 2>&1 || true
    fi
    if [[ -n "${image_id}" ]]; then
        docker run --rm --runtime=runc --network=none \
            --env NVIDIA_VISIBLE_DEVICES=void --user 0:0 \
            --mount "type=bind,source=${artifact_dir},target=/artifacts" \
            --entrypoint chown "${image_id}" -hR -- "${uid}:${gid}" /artifacts \
            9>&- >/dev/null 2>&1 || true
    fi
    exec 9>&-
    exit "${code}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

image_id=$(bash "${source_dir}/scripts/ci/ppu_runtime_image.sh" 9>&-)
printf '%s\n' "${image_id}" >"${artifact_dir}/runtime-image-id.txt"

container="${prefix}"
device_args=(
    --device=/dev/alixpu
    --device=/dev/alixpu_ctl
    --device=/dev/alixpu_sep
)
visible=()
for ((index = 0; index < PPU_CI_DEVICE_COUNT; index++)); do
    device_args+=(--device="/dev/alixpu_ppu${index}")
    visible+=("${index}")
done
visible_csv=$(IFS=,; echo "${visible[*]}")

create=(
    docker create --name "${container}" --runtime=runc
    --ipc=host --network=host
    --mount "type=bind,source=${source_dir},target=/source,readonly"
    --mount "type=bind,source=${wheels_dir},target=/wheels,readonly"
    --mount "type=bind,source=${artifact_dir},target=/artifacts"
    --env NVIDIA_VISIBLE_DEVICES=void
    --env CUDA_VISIBLE_DEVICES="${visible_csv}"
    --env PYTHONPYCACHEPREFIX=/tmp/pycache
    --env WHEELS_DIR=/wheels
    --env REQUIRE_PR_ARTIFACT="${PPU_CI_REQUIRE_PR_ARTIFACT}"
    --env EXPECTED_VLLM_COMMIT="${expected_vllm_commit}"
    --env EXPECTED_SAIL_COMMIT="${expected_sail_commit}"
)
create+=("${device_args[@]}")

# Resolve a catalog model id to the concrete snapshot directory of the offline
# HF cache and echo that absolute path. The cache is a view tree: snapshots/
# <revision> usually symlinks into a read-only checkpoint export that lives
# outside HF_HOME, so the caller has to mount the resolved target as well or
# the link dangles inside the container.
resolve_snapshot() {
    local model_id=$1
    local model_cache="${hf_home}/hub/models--${model_id//\//--}"
    [[ -f "${model_cache}/refs/main" ]] || {
        echo "Offline cache is missing ${model_id} refs/main" >&2
        exit 2
    }
    local model_revision
    model_revision=$(<"${model_cache}/refs/main")
    local model_snapshot
    model_snapshot=$(realpath "${model_cache}/snapshots/${model_revision}")
    [[ -d "${model_snapshot}" ]] || {
        echo "Offline cache snapshot is missing for ${model_id}" >&2
        exit 2
    }
    printf '%s\n' "${model_snapshot}"
}

# Mount the resolved snapshot when the view tree points outside HF_HOME, which
# is the normal layout on these runners.
mount_snapshot_if_external() {
    if [[ "$1" != "${hf_home}"/* ]]; then
        create+=(--mount "type=bind,source=$1,target=$1,readonly")
    fi
}

case "${mode}" in
    upstream-reuse)
        : "${PPU_CI_UPSTREAM_DIR:?PPU_CI_UPSTREAM_DIR is required for upstream-reuse}"
        [[ "${PPU_CI_DEVICE_COUNT}" == "1" ]] || {
            echo "upstream-reuse requires exactly one M890 device" >&2
            exit 2
        }
        upstream_dir=$(realpath "${PPU_CI_UPSTREAM_DIR}")
        [[ -f "${upstream_dir}/vllm/__init__.py" ]] &&
            [[ "$(git -c safe.directory="${upstream_dir}" -C "${upstream_dir}" rev-parse HEAD)" == "${expected_vllm_commit}" ]] || {
            echo "PPU CI upstream checkout does not match the tested pin" >&2
            exit 2
        }
        create+=(
            --shm-size=8g
            --mount "type=bind,source=${upstream_dir},target=/upstream,readonly"
            --env VLLM_SOURCE_ROOT=/upstream
            --env REUSE_RESULTS_DIR=/artifacts/reuse
            --env TESTED_SAIL_COMMIT="${PPU_CI_SOURCE_SHA}"
            --env VLLM_USE_RUST_FRONTEND=0
            --env VLLM_USE_RUST_BENCH=0
            --entrypoint bash "${image_id}"
            /source/scripts/ci/ppu_upstream_reuse.sh
        )
        ;;
    pin-ut)
        create+=(
            --shm-size=8g
            --env UT_RESULTS_DIR=/artifacts
            --env VLLM_USE_RUST_FRONTEND=0
            --env VLLM_USE_RUST_BENCH=0
            --entrypoint bash "${image_id}"
            /source/scripts/ci/ppu_pin_ut.sh
        )
        ;;
    model-test)
        MODEL_ID=${MODEL_ID:-${MODEL_PATH:-}}
        : "${MODEL_ID:?MODEL_ID or legacy MODEL_PATH is required for model-test}"
        : "${HF_HOME:?HF_HOME is required for model-test}"
        : "${SERVED_NAME:?SERVED_NAME is required for model-test}"
        : "${TP_SIZE:?TP_SIZE is required for model-test}"
        hf_home=$(realpath "${HF_HOME}")
        [[ -d "${hf_home}" ]] || { echo "HF_HOME is not a directory" >&2; exit 2; }
        [[ "${MODEL_ID}" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]] || {
            echo "MODEL_ID must be an owner/name Hugging Face ID" >&2
            exit 2
        }
        model_snapshot=$(resolve_snapshot "${MODEL_ID}")
        create+=(
            --shm-size=32g
            --mount "type=bind,source=${hf_home},target=${hf_home},readonly"
            --env MODEL_ID="${MODEL_ID}"
            --env HF_HOME="${hf_home}"
            --env HF_MODULES_CACHE="${hf_modules_cache}"
            --env HF_HUB_OFFLINE=1
            --env TRANSFORMERS_OFFLINE=1
            --env SERVED_NAME="${SERVED_NAME}"
            --env TP_SIZE="${TP_SIZE}"
            --env EXTRA_SERVE_FLAGS="${EXTRA_SERVE_FLAGS:-}"
            --env VLLM_USE_RUST_FRONTEND=0
            --env VLLM_USE_RUST_BENCH=0
        )
        mount_snapshot_if_external "${model_snapshot}"
        create+=(
            --entrypoint bash "${image_id}"
            /source/scripts/ci/ppu_e2e_smoke.sh
        )
        ;;
    model-accuracy)
        : "${GSM8K_DATASET_PATH:?GSM8K_DATASET_PATH is required for model-accuracy}"
        : "${PPU_CI_HF_HOME:?PPU_CI_HF_HOME is required for model-accuracy}"
        : "${PPU_CI_DG_CACHE_DIR:?PPU_CI_DG_CACHE_DIR is required for model-accuracy}"
        : "${MODEL_KEY:?MODEL_KEY is required for model-accuracy}"
        : "${MODEL_ID:?MODEL_ID is required for model-accuracy}"
        # Several cases may share one checkpoint; only the label differs.
        case_id="${CASE_ID:-${MODEL_KEY}}"
        hf_home=$(realpath "${PPU_CI_HF_HOME}")
        gsm8k_dataset=$(realpath "${GSM8K_DATASET_PATH}")
        dg_cache_root=$(realpath "${PPU_CI_DG_CACHE_DIR}")
        [[ -d "${hf_home}" ]] || {
            echo "PPU_CI_HF_HOME is not a directory" >&2
            exit 2
        }
        [[ -d "${gsm8k_dataset}" ]] || {
            echo "GSM8K_DATASET_PATH is not a directory" >&2
            exit 2
        }
        [[ -d "${dg_cache_root}" ]] || {
            echo "PPU_CI_DG_CACHE_DIR is not a directory" >&2
            exit 2
        }
        # DeepGEMM compiles its kernels on first use and caches the binaries so
        # the next start can skip the JIT pass. Left unset it writes under
        # VLLM_CACHE_ROOT, i.e. inside the container's writable layer, which
        # docker rm discards -- so every case of this lane recompiled from
        # scratch. Point it at the shared host root instead.
        #
        # One root serves every case and every image: deep_gemm keys the cache
        # by the toolchain that produced the binaries itself, so a stale entry
        # cannot be mistaken for a hit.
        #
        # Two names because deep_gemm reads DG_JIT_CACHE_DIR while the PPU SDK
        # builds also honour the older DG_CACHE_DIR, exactly like the vendor
        # recipes which export both. Pin them at one directory.
        model_snapshot=$(resolve_snapshot "${MODEL_ID}")
        create+=(
            --shm-size=32g
            --mount "type=bind,source=${hf_home},target=${hf_home},readonly"
            --mount "type=bind,source=${gsm8k_dataset},target=${gsm8k_dataset},readonly"
            --mount "type=bind,source=${dg_cache_root},target=${dg_cache_root}"
            --env HF_HOME="${hf_home}"
            --env HF_MODULES_CACHE="${hf_modules_cache}"
            --env HF_HUB_OFFLINE=1
            --env TRANSFORMERS_OFFLINE=1
            --env DG_JIT_CACHE_DIR="${dg_cache_root}"
            --env DG_CACHE_DIR="${dg_cache_root}"
            --env MODEL_ID="${MODEL_ID}"
            --env GSM8K_DATASET_PATH="${gsm8k_dataset}"
            --env MODEL_KEY="${MODEL_KEY}"
            --env CASE_ID="${case_id}"
            --env SERVER_CMD="${SERVER_CMD:-}"
            --env SERVER_ENV="${SERVER_ENV:-}"
            --env RESULTS_DIR=/artifacts
            --env UPDATE_GOLDEN="${UPDATE_GOLDEN:-0}"
            --env THRESHOLD="${THRESHOLD:-0.9}"
            --env VLLM_PORT="${VLLM_PORT:-9985}"
            --env VLLM_USE_RUST_FRONTEND=0
            --env VLLM_USE_RUST_BENCH=0
        )
        mount_snapshot_if_external "${model_snapshot}"
        create+=(
            --entrypoint bash "${image_id}"
            /source/scripts/ci/ppu_e2e_accuracy.sh
        )
        ;;
    kernel-perf)
        : "${SUITE:?SUITE is required for kernel-perf}"
        create+=(
            --shm-size=8g
            --env SUITE="${SUITE}"
            --env TEST_PATH=tests/e2e/perf/test_kernel_perf.py
            --env PYTEST_MARKER=perf
            --env WHEELS_DIR=/wheels
            --env RESULTS_DIR=/artifacts
            --env BASELINE_ROOT=/source/tests/e2e/perf/baselines
            --env BASELINE_OUTPUT_ROOT=/artifacts/baselines
            --env VLLM_USE_RUST_FRONTEND=0
            --env VLLM_USE_RUST_BENCH=0
            --entrypoint bash "${image_id}"
            /source/scripts/ci/perf_regression.sh
        )
        ;;
esac

"${create[@]}" 9>&- >"${artifact_dir}/container-id.txt"
docker start --attach "${container}" 9>&- 2>&1 | tee "${artifact_dir}/runtime.log"
