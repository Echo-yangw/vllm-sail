# SPDX-License-Identifier: Apache-2.0
"""The ``@patch`` decorator used to install PPU runtime patches into vLLM.

Adapted from vLLM-MetaX's ``vllm_metax/patch/utils.py``, with three additions:

1. **Mandatory metadata.** ``reason``, ``affected_versions`` and ``remove_when``
   are required keyword arguments, validated at import time. The on-demand
   ``tools/patch_manifest.py`` report displays the installed records' metadata.
2. **A registry.** Every installed patch is recorded in :data:`PATCH_REGISTRY`
   so tooling can enumerate patches, and so the drift harness
   (``tools/check_patch_drift.py``) can hash the upstream source each patch
   replaced.
3. **Explicit source capture** of the replaced original, for drift detection.

Two independent installation paths, chosen by whether ``target_attribute_name``
contains a ``.``:

* **module attribute** — installs the replacement object *unchanged*. This is
  what makes Triton objects patchable, but it means ``@patch`` must be the
  **outermost** decorator, outside ``@triton.jit`` / ``@triton.autotune``.
* **class attribute** (``"TargetClass.method"``) — resolves the class internally
  and preserves ``staticmethod`` / ``classmethod`` / ``property`` semantics.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar, cast

PatchTarget = TypeVar("PatchTarget")

#: Marker attribute on a replacement, mapping each target name it was installed
#: at to the original attribute that was replaced there.
PATCH_MARKER = "__vllm_sail_patch__"

_MISSING = object()


@dataclass(frozen=True)
class PatchRecord:
    """Bookkeeping for one installed patch."""

    target: str
    """Fully-qualified target, e.g. ``vllm.model_executor.custom_op.CustomOp.forward_ppu``."""
    reason: str
    affected_versions: str
    remove_when: str
    kind: str
    """``"module"`` or ``"class"``."""
    was_missing: bool
    """True when the attribute did not previously exist (``allow_missing``)."""
    original_source: str | None = field(default=None, repr=False)
    """Source of the replaced original, when retrievable. Used for drift detection."""


#: Every patch installed in this process, in installation order.
PATCH_REGISTRY: list[PatchRecord] = []


def _validate_metadata(
    target: str, reason: str, affected_versions: str, remove_when: str
) -> None:
    for name, value in (
        ("reason", reason),
        ("affected_versions", affected_versions),
        ("remove_when", remove_when),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Patch for {target} is missing required metadata {name!r}. "
                "Every patch must document why it exists, which vLLM versions "
                "it applies to, and the condition under which it can be "
                "deleted. See vllm_sail/patch/README.md."
            )


def _record_original_attribute(
    replacement_implementation: Any,
    full_target_name: str,
    original_attribute: Any,
) -> None:
    """Add one replaced attribute to a replacement's single patch marker."""
    original_attributes = getattr(replacement_implementation, PATCH_MARKER, None)
    if original_attributes is None:
        original_attributes = {}
    elif not isinstance(original_attributes, dict):
        raise TypeError(f"Invalid patch metadata on {replacement_implementation!r}")

    original_attributes[full_target_name] = original_attribute
    try:
        setattr(replacement_implementation, PATCH_MARKER, original_attributes)
    except (AttributeError, TypeError) as exc:
        raise TypeError(
            f"Replacement for {full_target_name} cannot store patch metadata"
        ) from exc


def _read_original_attribute(
    owner: Any,
    attribute_name: str,
    full_target_name: str,
    allow_missing: bool,
) -> Any:
    """Read an attribute without invoking descriptors or user lookup hooks."""
    try:
        return inspect.getattr_static(owner, attribute_name)
    except AttributeError as exc:
        if allow_missing:
            return _MISSING
        raise AttributeError(
            f"{full_target_name!r} does not exist. If you are intentionally "
            "adding a new attribute, pass allow_missing=True; otherwise the "
            "upstream target has moved or been renamed."
        ) from exc


def _raise_if_already_patched(
    original_implementation: Any,
    replacement_implementation: Any,
    full_target_name: str,
) -> None:
    """Reject both reapplying one object and installing a second patch."""
    if (
        original_implementation is replacement_implementation
        or getattr(original_implementation, PATCH_MARKER, None) is not None
    ):
        raise RuntimeError(
            f"{full_target_name} is already patched. Patch modules are applied "
            "once per process; if you are seeing this from a plugin hook, the "
            "idempotency guard in vllm_sail/__init__.py is not doing its job."
        )


def _capture_source(obj: Any) -> str | None:
    """Best-effort source text of ``obj``, for upstream-drift hashing."""
    if obj is _MISSING or obj is None:
        return None
    try:
        return inspect.getsource(_unwrap_class_descriptor(obj))
    except (OSError, TypeError):
        return None


def _patch_module_attribute(
    target_module_path: str,
    target_attribute_name: str | None,
    *,
    allow_missing: bool,
    reason: str,
    affected_versions: str,
    remove_when: str,
) -> Callable[[PatchTarget], PatchTarget]:
    """Replace a module-level function, class, Triton kernel, or other object.

    This path intentionally has no class-descriptor handling: the object returned
    by the inner decorators is installed in the module unchanged. That is why
    ``@patch`` must sit outside ``@triton.jit`` and ``@triton.autotune``.
    """
    if not target_module_path:
        raise ValueError("target_module_path must not be empty")

    def install_module_patch(replacement_attribute: PatchTarget) -> PatchTarget:
        resolved_attribute_name = target_attribute_name or getattr(
            replacement_attribute, "__name__", ""
        )
        if not resolved_attribute_name:
            raise ValueError(
                "target_attribute_name is required for unnamed replacement objects"
            )

        target_module = importlib.import_module(target_module_path)
        full_target_name = f"{target_module_path}.{resolved_attribute_name}"
        _validate_metadata(full_target_name, reason, affected_versions, remove_when)

        original_attribute = _read_original_attribute(
            target_module, resolved_attribute_name, full_target_name, allow_missing
        )

        if original_attribute is not _MISSING:
            _raise_if_already_patched(
                original_attribute, replacement_attribute, full_target_name
            )

        original_source = _capture_source(original_attribute)
        _record_original_attribute(
            replacement_attribute, full_target_name, original_attribute
        )
        setattr(target_module, resolved_attribute_name, replacement_attribute)

        PATCH_REGISTRY.append(
            PatchRecord(
                target=full_target_name,
                reason=reason,
                affected_versions=affected_versions,
                remove_when=remove_when,
                kind="module",
                was_missing=original_attribute is _MISSING,
                original_source=original_source,
            )
        )
        return replacement_attribute

    return install_module_patch


def _unwrap_class_descriptor(attribute: Any) -> Any:
    """Return the implementation stored inside a supported class descriptor."""
    if isinstance(attribute, staticmethod | classmethod):
        return attribute.__func__
    if isinstance(attribute, property):
        return attribute.fget
    return attribute


def _build_class_attribute(
    original_attribute: Any,
    replacement_attribute: Any,
    replacement_implementation: Any,
    full_target_name: str,
) -> Any:
    """Build a class attribute with the original descriptor semantics."""
    if original_attribute is _MISSING:
        return replacement_attribute
    if isinstance(original_attribute, staticmethod):
        return staticmethod(replacement_implementation)
    if isinstance(original_attribute, classmethod):
        return classmethod(replacement_implementation)
    if isinstance(original_attribute, property):
        return property(
            replacement_implementation,
            original_attribute.fset,
            original_attribute.fdel,
            original_attribute.__doc__,
        )
    if isinstance(replacement_attribute, staticmethod | classmethod | property):
        raise TypeError(
            f"Replacement descriptor does not match {full_target_name}: upstream "
            "is a plain function but the replacement is a descriptor."
        )
    return replacement_implementation


def _resolve_target_class(target_module: Any, target_class_path: str) -> type[Any]:
    """Resolve a dotted class path without importing it in the patch module."""
    target_class = target_module
    for class_name in target_class_path.split("."):
        try:
            target_class = getattr(target_class, class_name)
        except AttributeError as exc:
            raise AttributeError(
                f"Cannot resolve target class {target_class_path!r}"
            ) from exc
    if not isinstance(target_class, type):
        raise TypeError(f"{target_class_path!r} does not resolve to a class")
    return target_class


def _patch_class_method(
    target_module_path: str,
    target_attribute_path: str,
    *,
    allow_missing: bool,
    reason: str,
    affected_versions: str,
    remove_when: str,
) -> Callable[[PatchTarget], PatchTarget]:
    """Replace a class method or property through a dotted attribute path."""
    target_class_path, separator, target_attribute_name = (
        target_attribute_path.rpartition(".")
    )
    if not separator or not target_class_path or not target_attribute_name:
        raise ValueError("Class patches require a path such as 'TargetClass.method'")

    def install_class_patch(replacement_attribute: PatchTarget) -> PatchTarget:
        replacement_implementation = _unwrap_class_descriptor(replacement_attribute)
        target_module = importlib.import_module(target_module_path)
        target_class = _resolve_target_class(target_module, target_class_path)
        full_target_name = f"{target_module_path}.{target_attribute_path}"
        _validate_metadata(full_target_name, reason, affected_versions, remove_when)

        original_attribute = _read_original_attribute(
            target_class, target_attribute_name, full_target_name, allow_missing
        )

        # An additive patch installs into the target class's own namespace.
        # When the name exists only on a base class — e.g. SparseAttnIndexer
        # inheriting the patched CustomOp.forward_ppu — the attribute is
        # missing *on this class*: shadow it instead of tripping the
        # double-patch guard over the base's (already patched) attribute.
        if (
            allow_missing
            and original_attribute is not _MISSING
            and target_attribute_name not in vars(target_class)
        ):
            original_attribute = _MISSING

        if original_attribute is not _MISSING:
            _raise_if_already_patched(
                _unwrap_class_descriptor(original_attribute),
                replacement_implementation,
                full_target_name,
            )

        original_source = _capture_source(original_attribute)
        replacement_class_attribute = _build_class_attribute(
            original_attribute,
            replacement_attribute,
            replacement_implementation,
            full_target_name,
        )
        _record_original_attribute(
            replacement_implementation, full_target_name, original_attribute
        )
        setattr(target_class, target_attribute_name, replacement_class_attribute)

        PATCH_REGISTRY.append(
            PatchRecord(
                target=full_target_name,
                reason=reason,
                affected_versions=affected_versions,
                remove_when=remove_when,
                kind="class",
                was_missing=original_attribute is _MISSING,
                original_source=original_source,
            )
        )
        return cast(PatchTarget, replacement_class_attribute)

    return install_class_patch


def patch(
    target_module_path: str,
    target_attribute_name: str | None = None,
    *,
    reason: str,
    affected_versions: str,
    remove_when: str,
    allow_missing: bool = False,
) -> Callable[[PatchTarget], PatchTarget]:
    """Create a patch decorator for a module attribute or class attribute.

    Args:
        target_module_path: Import path of the module to patch.
        target_attribute_name: Attribute to replace. A bare or omitted name uses
            the module-attribute path (the replacement is installed unchanged,
            so this supports Triton JIT/autotune objects). A dotted path such as
            ``"TargetClass.method"`` uses the class-attribute path, which
            resolves the class internally and preserves method descriptors.
        reason: Why this patch exists. Required.
        affected_versions: vLLM version or range the patch applies to, e.g.
            ``">=0.32.0,<0.33.0"``. Required.
        remove_when: A verifiable condition under which this patch can be
            deleted, e.g. an upstream PR or capability. Required.
        allow_missing: Permit the target not to exist yet. Use only when
            deliberately *adding* a compatibility attribute (``is_ppu``,
            ``forward_ppu``); normal patches require the target to exist so a
            renamed upstream symbol fails loudly.

    Raises:
        RuntimeError: The target is already patched.
        AttributeError: The target does not exist and ``allow_missing`` is False.
        ValueError: Required metadata is missing.
    """
    if not isinstance(target_module_path, str) or not target_module_path:
        raise TypeError("target_module_path must be a non-empty module import path")

    if target_attribute_name is not None and "." in target_attribute_name:
        return _patch_class_method(
            target_module_path,
            target_attribute_name,
            allow_missing=allow_missing,
            reason=reason,
            affected_versions=affected_versions,
            remove_when=remove_when,
        )
    return _patch_module_attribute(
        target_module_path,
        target_attribute_name,
        allow_missing=allow_missing,
        reason=reason,
        affected_versions=affected_versions,
        remove_when=remove_when,
    )


def patch_value(
    target_module_path: str,
    target_attribute_name: str,
    value: Any,
    *,
    reason: str,
    affected_versions: str,
    remove_when: str,
    allow_missing: bool = False,
) -> Any:
    """Install a *data* value as a module attribute, with patch bookkeeping.

    :func:`patch` is a decorator and therefore only usable on callables and
    classes. Some upstream targets are plain data — module-level constants,
    dataclass instances, lookup tables — and those go through this imperative
    form so they still land in :data:`PATCH_REGISTRY` and still get the
    already-patched guard.

    Unlike :func:`patch`, an existing attribute that is *equal* to ``value`` is
    accepted as a no-op: data constants are frequently defined identically by
    upstream later, and that should be a clean removal signal rather than a
    crash. An existing attribute with a *different* value raises.
    """
    target_module = importlib.import_module(target_module_path)
    full_target_name = f"{target_module_path}.{target_attribute_name}"
    _validate_metadata(full_target_name, reason, affected_versions, remove_when)

    original_attribute = _read_original_attribute(
        target_module, target_attribute_name, full_target_name, allow_missing
    )

    if original_attribute is not _MISSING:
        if original_attribute == value:
            return original_attribute
        raise RuntimeError(
            f"{full_target_name} already exists with a different value "
            f"({original_attribute!r} != {value!r}). Upstream now defines this "
            "constant; delete the patch that installs it."
        )

    setattr(target_module, target_attribute_name, value)
    PATCH_REGISTRY.append(
        PatchRecord(
            target=full_target_name,
            reason=reason,
            affected_versions=affected_versions,
            remove_when=remove_when,
            kind="value",
            was_missing=original_attribute is _MISSING,
            original_source=None,
        )
    )
    return value


def original_of(replacement: Any, full_target_name: str) -> Any:
    """Return the attribute a patch replaced at ``full_target_name``.

    Useful inside a patch that wants to delegate to upstream behaviour, and in
    tests that assert a patch recorded what it replaced.
    """
    markers = getattr(replacement, PATCH_MARKER, None)
    if not isinstance(markers, dict) or full_target_name not in markers:
        raise KeyError(f"{replacement!r} did not patch {full_target_name}")
    original = markers[full_target_name]
    if original is _MISSING:
        raise KeyError(f"{full_target_name} did not previously exist")
    return original
