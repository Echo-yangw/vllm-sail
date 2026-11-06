# SPDX-License-Identifier: Apache-2.0
"""Let ``KernelConfig`` validation accept the PPU MoE backend names.

Upstream ``vllm/config/kernel.py`` defines ``MoEBackend = Literal[...]`` and
``KernelConfig.moe_backend`` is validated against it by pydantic. The fork adds
``"ppu_deep_gemm"`` and ``"ppu_acext"`` to the Literal (+4 lines including the
docstring). Without widening, ``--kernel-config moe_backend=ppu_deep_gemm`` and
``--moe-backend ppu_deep_gemm`` fail config validation before the patched MoE
oracles in ``vllm_sail.registry.moe_backends`` ever run.

Runtime CLI-name mapping is already covered: the oracles map ``ppu_deep_gemm`` /
``ppu_acext`` onto the injected enum members, and ``VLLM_PPU_MOE_BACKEND`` (the
primary, env-var-first knob) bypasses config validation entirely. The Literal
widening below is only for CLI parity, per
``docs/developer_guide/architecture.md``.

Mechanism (minimal): widen the ``MoEBackend`` module alias *and* the already
compiled pydantic dataclass schema — the alias alone cannot work because pydantic
compiles the ``Literal`` validator at class-creation time. So the field
annotation is replaced on the class, the dataclass field and the pydantic
``FieldInfo``, then ``rebuild_dataclass`` recompiles the validator. Also updates
``KernelConfig.__annotations__`` so ``get_type_hints``-driven argparse
``choices`` (``EngineArgs``) pick the widened set up.

Deliberately best-effort: this is the most pydantic-version-fragile line in the
plugin (§1.5), so failure logs a warning and degrades to the env-var knob
instead of crashing engine startup.
"""

from __future__ import annotations

from typing import Literal, get_args

from vllm_sail.patch.utils import PATCH_REGISTRY, PatchRecord

_MODULE = "vllm.config.kernel"
_TARGET = "vllm.config.kernel.MoEBackend"
PPU_MOE_BACKENDS = ("ppu_deep_gemm", "ppu_acext", "ppu_deep_gemm_w4a16", "ppu_deep_gemm_w4a8")
REASON = (
    "KernelConfig validates moe_backend against the MoEBackend Literal, which "
    "upstream does not extend. The PPU backend names must pass validation for "
    "--moe-backend / --kernel-config CLI parity; the env-var knob "
    "VLLM_PPU_MOE_BACKEND bypasses validation and remains the primary knob."
)
AFFECTED_VERSIONS = ">=0.32.0,<0.33.0"
REMOVE_WHEN = (
    "upstream gains a register_moe_backend() extension point mirroring "
    "register_linear_kernel (the plugin's #1 upstream RFC), or accepts the "
    "ppu_* members into the MoEBackend Literal."
)

#: (target, reason, affected versions, remove_when) for every patch installed
#: by this module. Read by the metadata-shape unit tests.
METADATA = ((_TARGET, REASON, AFFECTED_VERSIONS, REMOVE_WHEN),)

_installed = False


def _record() -> None:
    for record in PATCH_REGISTRY:
        if record.target == _TARGET and record.reason == REASON:
            return
    PATCH_REGISTRY.append(
        PatchRecord(
            target=_TARGET,
            reason=REASON,
            affected_versions=AFFECTED_VERSIONS,
            remove_when=REMOVE_WHEN,
            kind="value",
            was_missing=False,
            original_source=None,
        )
    )


def install() -> None:
    """Widen the Literal. Requires vLLM to be importable; idempotent."""
    global _installed
    if _installed:
        return

    import dataclasses

    from vllm.config import kernel as kernel_module
    from vllm.logger import init_logger

    logger = init_logger(__name__)
    kernel_config_cls = kernel_module.KernelConfig
    existing = get_args(kernel_module.MoEBackend)
    if all(name in existing for name in PPU_MOE_BACKENDS):
        _installed = True
        return

    try:
        from pydantic.dataclasses import rebuild_dataclass

        extended = Literal.__getitem__(tuple(existing) + PPU_MOE_BACKENDS)
        # Module-level type alias replacement: keeps later consumers of
        # vllm.config.kernel.MoEBackend (annotations, get_type_hints) seeing
        # the widened set.
        kernel_module.MoEBackend = extended
        # The already-compiled pydantic schema reads the annotation from all
        # three places; update each, then rebuild.
        kernel_config_cls.__annotations__["moe_backend"] = extended
        for field in dataclasses.fields(kernel_config_cls):
            if field.name == "moe_backend":
                field.type = extended
        kernel_config_cls.__pydantic_fields__["moe_backend"].annotation = extended
        rebuild_dataclass(kernel_config_cls, force=True)
    except Exception as exc:  # noqa: BLE001 - deliberate best-effort catch
        logger.warning(
            "Could not widen KernelConfig.moe_backend with %s (%s). "
            "VLLM_PPU_MOE_BACKEND remains the supported way to select PPU MoE "
            "backends; --moe-backend ppu_* will fail validation.",
            ", ".join(PPU_MOE_BACKENDS),
            exc,
        )
        return

    _installed = True
    _record()
    logger.debug(
        "Widened KernelConfig.moe_backend with %s", ", ".join(PPU_MOE_BACKENDS)
    )
