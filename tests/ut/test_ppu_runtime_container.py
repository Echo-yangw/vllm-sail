# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/ci/ppu_runtime_container.sh"


def test_runtime_wrapper_uses_preloaded_image_and_explicit_ppu_devices():
    text = SCRIPT.read_text()
    assert 'ppu_runtime_image.sh" 9>&-' in text
    resolver = (ROOT / "scripts/ci/ppu_runtime_image.sh").read_text()
    assert "Runner must configure PPU_CI_RUNTIME_IMAGE" in resolver
    assert 'docker image inspect "${PPU_CI_RUNTIME_IMAGE}"' in resolver
    assert 'docker pull "${PPU_CI_RUNTIME_IMAGE}" >"${pull_log}" 2>&1' in resolver
    assert "docker image ls" not in resolver
    assert "@sha256:" not in resolver
    assert "docker_release/llm" not in resolver
    assert "known_image_id" not in resolver
    assert "runtime-image.txt" not in text
    assert "--device=/dev/alixpu" in text
    assert "--device=/dev/alixpu_ctl" in text
    assert "--device=/dev/alixpu_sep" in text
    assert "/dev/alixpu_ppu${index}" in text
    assert "--env NVIDIA_VISIBLE_DEVICES=void" in text
    assert '--env CUDA_VISIBLE_DEVICES="${visible_csv}"' in text
    assert "--env PYTHONPYCACHEPREFIX=/tmp/pycache" in text
    assert "--privileged" not in text


def test_runtime_wrapper_mounts_inputs_read_only_and_artifacts_writable():
    text = SCRIPT.read_text()
    assert "target=/source,readonly" in text
    assert "target=/wheels,readonly" in text
    assert "source=${hf_home},target=${hf_home},readonly" in text
    # The view tree usually points outside HF_HOME, so both device lanes have to
    # mount the resolved snapshot explicitly.
    assert "mount_snapshot_if_external" in text
    assert "source=$1,target=$1,readonly" in text
    assert r"models--${model_id//\//--}" in text
    assert "--env HF_HUB_OFFLINE=1" in text
    assert "--env TRANSFORMERS_OFFLINE=1" in text
    assert 'target=/artifacts"' in text
    assert "/source/scripts/ci/ppu_pin_ut.sh" in text
    assert "/source/scripts/ci/ppu_e2e_smoke.sh" in text


def test_runtime_wrapper_keeps_dynamic_module_cache_writable():
    text = SCRIPT.read_text()
    # HF_HOME is mounted read-only, so transformers cannot materialise remote
    # model code underneath it; both device lanes have to redirect the cache.
    assert "hf_modules_cache=/tmp/hf-modules" in text
    assert text.count('--env HF_MODULES_CACHE="${hf_modules_cache}"') == 2


def test_runtime_wrapper_serializes_jobs_and_verifies_source_identity():
    text = SCRIPT.read_text()
    assert "flock -n 9" in text
    assert "rev-parse HEAD" in text
    assert "PPU_CI_SOURCE_SHA" in text
    assert ".github/vllm-main-verified.commit" in text
    assert '--env EXPECTED_VLLM_COMMIT="${expected_vllm_commit}"' in text
    assert '--env EXPECTED_SAIL_COMMIT="${expected_sail_commit}"' in text
    assert "docker stop --time=2" in text
    assert "docker rm --force" in text
