# SPDX-License-Identifier: Apache-2.0
"""Shared checkpoint / LLM / identity utilities for the PPU test suite.

Provides:

* **Checkpoint resolution** — mirrored NAS catalogue roots, availability
  guards and capacity guards (:func:`resolve_checkpoint`,
  :func:`checkpoint_skip_reason`).
* **Offline LLM construction** — :func:`build_llm` / :func:`release_llm`
  with the fixed parameter table (``max_model_len=4096``, ``dtype=bf16``,
  ``gpu_memory_utilization=0.90``, ``enforce_eager=True``, ``seed=0``).
* **Golden identity binding** — :func:`_checkpoint_tail`,
  :func:`_assert_chip`, :func:`_assert_checkpoint_identity`.
* **Chip → TP mapping** — :func:`chip_key_for_capability`,
  :func:`tensor_parallel_for_chip`.

Importing this module must **never** require ``torch`` or ``vllm``: both
are imported lazily inside the functions that need them, so collection
stays CPU-only.
"""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --- Fixed parameters (shared parameter table). ---
DEFAULT_MAX_MODEL_LEN = 4096
DEFAULT_DTYPE = "bfloat16"
DEFAULT_GPU_MEMORY_UTILIZATION = 0.90
DEFAULT_ENFORCE_EAGER = True
DEFAULT_SEED = 0
DEFAULT_TRUST_REMOTE_CODE = True
DEFAULT_TEMPERATURE = 0.0

#: Environment variable that may override the fixed checkpoint path.
MODEL_ROOT_ENV = "VLLM_SAIL_MODEL_ROOT"

#: File whose presence makes a directory a loadable checkpoint.
CHECKPOINT_MARKER = "config.json"

#: Mirrored NAS catalogue roots (same NFS export, two mount points).
CHECKPOINT_ROOT_MIRRORS: tuple[tuple[str, str], ...] = (
    ("/nas_aisw/datasets", "/ppusw/datasets"),
    ("/ppusw/datasets", "/nas_aisw/datasets"),
)

# Capability -> chip key.
CAPABILITY_TO_CHIP: dict[tuple[int, int], str] = {
    (8, 0): "810e",
    (8, 9): "890p",
}
DEFAULT_CHIP = "810e"

# Chip -> tensor_parallel_size.
CHIP_TO_TENSOR_PARALLEL_SIZE: dict[str, int] = {
    "810e": 8,
    "890p": 8,
}

#: Suffix marking an unreviewed first-run golden.
PENDING_REVIEW_SUFFIX = ".pending-review.json"

#: Root for per-run reports (git-ignored).
REPORTS_ROOT = Path(__file__).resolve().parent / "_reports"

#: File extensions counted when measuring a checkpoint's weight volume.
WEIGHT_FILE_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")


# ===================================================================
# Per-model fixed configuration
# ===================================================================


@dataclass(frozen=True)
class ModelConfig:
    """Per-model fixed configuration for E2E flows.

    Defaults reproduce the shared parameter table. ``env`` is applied
    to ``os.environ`` before the ``LLM`` is built; ``extra_llm_kwargs``
    is merged last into the ``LLM(...)`` call.
    """

    key: str
    checkpoint: str
    max_model_len: int = DEFAULT_MAX_MODEL_LEN
    dtype: str = DEFAULT_DTYPE
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION
    enforce_eager: bool = DEFAULT_ENFORCE_EAGER
    seed: int = DEFAULT_SEED
    trust_remote_code: bool = DEFAULT_TRUST_REMOTE_CODE
    env: Mapping[str, str] = field(default_factory=dict)
    extra_llm_kwargs: Mapping[str, Any] = field(default_factory=dict)


# ===================================================================
# Checkpoint resolution
# ===================================================================


def _mirror_roots(path: str) -> list[str]:
    """``path`` plus equivalents under every mirrored root."""
    candidates = [path]
    for source, target in CHECKPOINT_ROOT_MIRRORS:
        if path.startswith(source + "/"):
            candidates.append(target + path[len(source) :])
    return candidates


def _is_loadable(path: str) -> bool:
    """True when ``path`` holds a ``config.json``."""
    return os.path.isfile(os.path.join(path, CHECKPOINT_MARKER))


def resolve_checkpoint(default: str) -> str:
    """Return the runtime weight path.

    ``VLLM_SAIL_MODEL_ROOT`` wins outright.  Otherwise the recorded
    default is used when loadable, and the first mirrored root that
    *is* loadable when it is not.
    """
    override = os.environ.get(MODEL_ROOT_ENV)
    if override:
        return override
    for candidate in _mirror_roots(default):
        if _is_loadable(candidate):
            return candidate
    return default


def checkpoint_weight_bytes(path: str) -> int | None:
    """Total size in bytes of weight files under ``path``."""
    try:
        names = os.listdir(path)
    except OSError:
        return None
    total = 0
    found = False
    for name in names:
        if not name.endswith(WEIGHT_FILE_SUFFIXES):
            continue
        try:
            total += os.path.getsize(os.path.join(path, name))
        except OSError:
            return None
        found = True
    return total if found else None


def checkpoint_skip_reason(
    path: str,
    *,
    tensor_parallel_size: int | None = None,
    device_total_bytes: int | None = None,
) -> str | None:
    """Skip reason for an unusable checkpoint, or ``None``."""
    if not _is_loadable(path):
        return f"weights unavailable: {path}"
    if not tensor_parallel_size or not device_total_bytes:
        return None
    total = checkpoint_weight_bytes(path)
    if total is None:
        return None
    per_rank = total / int(tensor_parallel_size)
    if per_rank <= float(device_total_bytes):
        return None
    gib = float(1024**3)
    return (
        f"weights do not fit: {path} holds "
        f"{total / gib:.1f} GiB of weight files, "
        f"i.e. {per_rank / gib:.1f} GiB per rank at "
        f"tensor_parallel_size={int(tensor_parallel_size)}, "
        f"but a device here has only "
        f"{float(device_total_bytes) / gib:.1f} GiB in total"
    )


def _same_checkpoint(left: str, right: str) -> bool:
    """True when two paths name the same weight tree."""
    real_left = {os.path.realpath(p) for p in _mirror_roots(left)}
    real_right = {os.path.realpath(p) for p in _mirror_roots(right)}
    return bool(real_left & real_right)


def variant_checkpoint(default: str) -> str | None:
    """Resolve a variant, or ``None`` if deselected by override.

    When ``VLLM_SAIL_MODEL_ROOT`` is set it selects one variant;
    others skip so CI dispatch stays honest.
    """
    override = os.environ.get(MODEL_ROOT_ENV)
    if not override:
        return resolve_checkpoint(default)
    if _same_checkpoint(override, default):
        return override
    return None


# ===================================================================
# Chip / TP helpers
# ===================================================================


def chip_key_for_capability(capability: Any) -> str:
    """Map device capability onto ``810e`` / ``890p``."""
    if capability is None:
        return DEFAULT_CHIP
    key = (int(capability[0]), int(capability[1]))
    return CAPABILITY_TO_CHIP.get(key, DEFAULT_CHIP)


def tensor_parallel_for_chip(chip: str) -> int:
    """Fixed TP size of a chip key."""
    return CHIP_TO_TENSOR_PARALLEL_SIZE.get(
        chip, CHIP_TO_TENSOR_PARALLEL_SIZE[DEFAULT_CHIP]
    )


# ===================================================================
# Offline LLM construction / teardown
# ===================================================================


def build_llm(
    model_path: str,
    chip_profile: Any,
    config: ModelConfig | None = None,
):
    """Construct ``vllm.LLM`` with every parameter fixed.

    ``tensor_parallel_size`` comes from ``CHIP_TO_TENSOR_PARALLEL_SIZE``
    via the chip profile's capability; all other
    knobs come from ``config`` or the module defaults.  ``vllm`` is
    imported lazily.
    """
    if config is not None and config.env:
        for name, value in config.env.items():
            os.environ[name] = value
    tp = tensor_parallel_for_chip(
        chip_key_for_capability(getattr(chip_profile, "capability", None))
    )
    kwargs: dict[str, Any] = {
        "max_model_len": (config.max_model_len if config else DEFAULT_MAX_MODEL_LEN),
        "dtype": config.dtype if config else DEFAULT_DTYPE,
        "gpu_memory_utilization": (
            config.gpu_memory_utilization if config else DEFAULT_GPU_MEMORY_UTILIZATION
        ),
        "enforce_eager": (config.enforce_eager if config else DEFAULT_ENFORCE_EAGER),
        "seed": config.seed if config else DEFAULT_SEED,
        "trust_remote_code": (
            config.trust_remote_code if config else DEFAULT_TRUST_REMOTE_CODE
        ),
    }
    if config is not None and config.extra_llm_kwargs:
        kwargs.update(config.extra_llm_kwargs)
    from vllm import LLM

    return LLM(
        model=model_path,
        tensor_parallel_size=tp,
        **kwargs,
    )


def release_llm(llm: Any) -> None:
    """Best-effort teardown of an offline engine.

    Shuts the engine core process down so the KV-cache reservation
    is released and the next ``LLM(...)`` can allocate.  Never raises.
    """
    engine = getattr(llm, "llm_engine", None)
    core = getattr(engine, "engine_core", None)
    shutdown = getattr(core, "shutdown", None)
    if callable(shutdown):
        try:
            shutdown()
        except Exception as exc:
            warnings.warn(
                f"engine core shutdown failed: {exc!r}",
                stacklevel=2,
            )
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass


# ===================================================================
# Golden identity helpers
# ===================================================================


def _checkpoint_tail(checkpoint: Any) -> str | None:
    """Last two path segments (mount-independent identity).

    The accuracy lane passes the Hugging Face id rather than a path, so the
    tail of ``T-HEAD/<name>`` is the whole id; for a path it is the last two
    segments, which survive a different mount root.
    """
    if checkpoint is None:
        return None
    path = Path(str(checkpoint))
    parts = [part for part in path.parts if part not in {"/", "\\"}]
    return "/".join(parts[-2:]) if parts else str(path)


def _quantization_info(llm: Any) -> dict[str, Any] | None:
    """Best-effort quantization metadata from a built ``LLM``."""
    if llm is None:
        return None
    info: dict[str, Any] = {
        "hf_quantization_config": None,
        "quant_method": None,
    }
    try:
        model_config = llm.llm_engine.model_config
        hf_config = getattr(model_config, "hf_config", None) or getattr(
            model_config, "hf_text_config", None
        )
        quant_config = getattr(hf_config, "quantization_config", None)
        if isinstance(quant_config, Mapping):
            info["hf_quantization_config"] = dict(quant_config)
        quantization = getattr(model_config, "quantization", None)
        if isinstance(quantization, str):
            info["quant_method"] = quantization
        elif quantization is not None:
            info["quant_method"] = getattr(quantization, "name", str(quantization))
    except Exception:  # noqa: BLE001
        return None
    if all(v is None for v in info.values()):
        return None
    return info


def _checkpoint_identity(checkpoint: Any, llm: Any = None) -> dict[str, Any] | None:
    """Bundle identity fields for a golden record."""
    if checkpoint is None and llm is None:
        return None
    return {
        "checkpoint": (str(checkpoint) if checkpoint is not None else None),
        "checkpoint_tail": _checkpoint_tail(checkpoint),
        "quantization": _quantization_info(llm),
    }


# ===================================================================
# Golden file path helpers
# ===================================================================


def _model_from_path(golden_path: Path, chip: str) -> str:
    """Derive model key from ``<model>_<chip>.json``."""
    stem = golden_path.stem
    suffix = f"_{chip}"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def _pending_path(golden_path: Path) -> Path:
    """Pending-review sibling of a formal golden."""
    return golden_path.with_name(golden_path.stem + PENDING_REVIEW_SUFFIX)


def _write_golden(record: Mapping[str, Any], golden_path: Path) -> None:
    """Write a golden JSON record atomically."""
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    golden_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")


# ===================================================================
# Identity assertions
# ===================================================================


def _assert_chip(
    expected: Mapping[str, Any],
    chip: str,
    golden_path: Path,
) -> None:
    """Fail when the golden's chip does not match the device."""
    golden_chip = expected.get("chip")
    if golden_chip != chip:
        raise AssertionError(
            f"golden chip mismatch: {golden_path} records "
            f"chip={golden_chip!r} but the current device is "
            f"{chip!r}; per-chip goldens must never be "
            "cross-compared."
        )


def _assert_checkpoint_identity(
    expected: Mapping[str, Any],
    identity: Mapping[str, Any] | None,
    golden_path: Path,
) -> None:
    """Fail when the golden was captured from a different ckpt."""
    golden_tail = expected.get("checkpoint_tail")
    if golden_tail is None:
        warnings.warn(
            f"golden {golden_path} carries no checkpoint "
            "identity; it will be compared as-is.",
            stacklevel=3,
        )
        return
    actual_tail = (identity or {}).get("checkpoint_tail")
    if actual_tail is None:
        warnings.warn(
            "this run provides no checkpoint identity; "
            f"golden binding ({golden_tail!r}) not verified.",
            stacklevel=3,
        )
        return
    if golden_tail != actual_tail:
        raise AssertionError(
            f"golden checkpoint mismatch: golden {golden_path} "
            f"来自 {golden_tail!r}，本次运行 {actual_tail!r}，"
            "基准与 checkpoint 不匹配；如属有意换基线请先删除"
            "旧 golden 或用 --update-golden 重采。"
        )
