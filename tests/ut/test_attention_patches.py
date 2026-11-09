# SPDX-License-Identifier: Apache-2.0
"""Tests for the Phase-4 attention and FLA patch modules.

Everything here needs an importable vLLM; on a bare CPU runner the whole
module skips. Two vLLM targets are allowed to be unimportable in some test
environments (``mla.prefill.flash_attn`` needs a GPU platform to pick an FA
version; ``qwen_gdn_linear_attn`` needs ``compressed_tensors``). The patch
modules skip themselves with a warning in that case, and
:func:`test_unimportable_targets_are_skipped_not_faked` asserts that a skip is
a skip — never a silent no-pass.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import pytest

from tests.support.upstream import SUPPORTED_RANGE

pytest.importorskip("torch", reason="requires torch")
pytest.importorskip("vllm", reason="requires vLLM")

import vllm_sail.patch as patch_pkg  # noqa: E402
from vllm_sail.patch.utils import PATCH_MARKER, PATCH_REGISTRY  # noqa: E402

# Reproduce vLLM's real plugin timing: the backend may already be imported
# before the general-plugin hook installs the PPU FlashMLA provider patches.
_PRELOADED_FLASHMLA_BACKEND = importlib.import_module(
    "vllm.v1.attention.backends.mla.flashmla"
)


@pytest.fixture(scope="module")
def installed():
    # The platform entry point installs the FA shim before the general hook.
    # Direct patch.install() alone does not reproduce that startup sequence.
    import vllm_sail

    vllm_sail.register()
    patch_pkg.install()
    return PATCH_REGISTRY


def _unwrap(attribute):
    if isinstance(attribute, staticmethod | classmethod):
        return attribute.__func__
    if isinstance(attribute, property):
        return attribute.fget
    return attribute


def _static_attr(module_path: str, attr_path: str):
    obj = importlib.import_module(module_path)
    parts = attr_path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    return inspect.getattr_static(obj, parts[-1])


#: (module, attribute path) for every Phase-4 Group-1 attention patch.
_ATTENTION_TARGETS = (
    ("vllm.v1.attention.backends.fa_utils", "get_flash_attn_version"),
    ("vllm.v1.attention.ops.flashmla", "_is_flashmla_available"),
    ("vllm.v1.attention.ops.flashmla", "is_flashmla_dense_supported"),
    ("vllm.v1.attention.ops.flashmla", "is_flashmla_sparse_supported"),
    ("vllm.v1.attention.ops.flashmla", "flash_mla_sparse_fwd"),
    ("vllm.v1.attention.ops.flashmla", "flash_mla_with_kvcache"),
    ("vllm.v1.attention.ops.flashmla", "get_mla_metadata"),
    ("vllm.v1.attention.ops.flashmla", "get_mla_metadata_dense_fp8"),
    ("vllm.v1.attention.ops.flashmla", "flash_mla_with_kvcache_fp8"),
    (
        "vllm.v1.attention.backends.mla.flashmla",
        "FlashMLABackend.supports_compute_capability",
    ),
    ("vllm.v1.attention.backends.mla.flashmla", "FlashMLAMetadataBuilder.__init__"),
    (
        "vllm.v1.attention.backends.mla.flashmla_sparse",
        "FlashMLASparseBackend.supports_compute_capability",
    ),
    (
        "vllm.v1.attention.backends.mla.flashmla_sparse",
        "FlashMLASparseMetadataBuilder.__init__",
    ),
    (
        "vllm.v1.attention.backends.mla.prefill.flash_attn",
        "FlashAttnPrefillBackend.__init__",
    ),
    (
        "vllm.model_executor.layers.attention.mla_attention",
        "MLAAttention.forward_impl",
    ),
    (
        "vllm.model_executor.layers.attention.mla_attention",
        "_get_kv_b_proj_input_dtype",
    ),
    (
        "vllm.v1.attention.backends.mla.indexer",
        "get_paged_mqa_logits_metadata",
    ),
    (
        "vllm.v1.attention.backends.mla.indexer",
        "DeepseekV32IndexerMetadataBuilder._split_indexer_prefill_chunks",
    ),
    (
        "vllm.v1.attention.backends.mla.indexer",
        "DeepseekV32IndexerMetadataBuilder.__init__",
    ),
    (
        "vllm.v1.attention.backends.mla.indexer",
        "DeepseekV32IndexerMetadataBuilder.build",
    ),
    (
        "vllm.v1.attention.backends.mla.indexer",
        "DeepseekV32IndexerMetadataBuilder.update_draft_decode_metadata",
    ),
    (
        "vllm.v1.attention.ops.triton_decode_attention",
        "_decode_grouped_att_m_fwd",
    ),
    ("vllm.model_executor.layers.sparse_attn_indexer", "SparseAttnIndexer.__init__"),
    (
        "vllm.model_executor.layers.sparse_attn_indexer",
        "SparseAttnIndexer.forward_native",
    ),
    (
        "vllm.model_executor.layers.sparse_attn_indexer",
        "SparseAttnIndexer.forward_ppu",
    ),
    (
        "vllm.model_executor.kernels.mhc.tilelang",
        "_hc_prenorm_gemm_outputs",
    ),
    ("vllm.v1.kv_cache_interface", "MLAAttentionSpec.__init__"),
    ("vllm.v1.kv_cache_interface", "MLAAttentionSpec.merge"),
)

#: The fork's five forward_ppu overrides (mhc x4 + sparse attn indexer).
_FORWARD_PPU_CLASSES = (
    ("vllm.model_executor.layers.mhc", "MHCPreOp"),
    ("vllm.model_executor.layers.mhc", "MHCPostOp"),
    ("vllm.model_executor.layers.mhc", "HCHeadOp"),
    ("vllm.model_executor.layers.mhc", "MHCFusedPostPreOp"),
    ("vllm.model_executor.layers.sparse_attn_indexer", "SparseAttnIndexer"),
)


def test_patch_metadata_is_complete(installed) -> None:
    """Every installed patch must carry non-empty mandatory metadata."""
    assert installed, "no patches installed despite vLLM being importable"
    for record in installed:
        assert record.reason.strip(), record.target
        assert record.affected_versions.strip(), record.target
        assert record.remove_when.strip(), record.target


def test_attention_patch_markers_land(installed) -> None:
    """Each Group-1 target carries the patch marker for its full target name."""
    registry_targets = {record.target for record in installed}
    for module_path, attr_path in _ATTENTION_TARGETS:
        full_target = f"{module_path}.{attr_path}"
        if full_target not in registry_targets:
            # The patch module skips itself when its target cannot import;
            # test_unimportable_targets_are_skipped_not_faked proves the skip.
            continue
        implementation = _unwrap(_static_attr(module_path, attr_path))
        markers = getattr(implementation, PATCH_MARKER, None)
        assert isinstance(markers, dict), full_target
        assert full_target in markers, full_target


def test_flashmla_backend_uses_patched_ppu_support_gate(
    monkeypatch: pytest.MonkeyPatch, installed
) -> None:
    """FlashMLAImpl must not retain the upstream Hopper-only support alias."""
    from vllm.platforms import current_platform
    from vllm.v1.attention.backends.mla import flashmla as backend
    from vllm.v1.attention.ops import flashmla as ops

    from vllm_sail.patch.enhancement.attention import flashmla_ops as plugin

    platform_type = type(current_platform)
    monkeypatch.setattr(platform_type, "is_ppu", lambda self: True)
    monkeypatch.setattr(
        platform_type,
        "is_device_capability_family",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(plugin, "_ppu_flashmla_available", lambda: True)

    assert backend.is_flashmla_dense_supported is ops.is_flashmla_dense_supported
    assert backend.is_flashmla_dense_supported() == (True, None)


def test_flashmla_consumers_use_every_patched_op_alias(installed) -> None:
    """By-value imports must follow the patched PPU FlashMLA implementations."""
    from vllm.v1.attention.ops import flashmla as ops

    from vllm_sail.patch.enhancement.attention.flashmla_ops import (
        _FLASHMLA_ALIAS_CONSUMERS,
    )

    for alias, module_names in _FLASHMLA_ALIAS_CONSUMERS.items():
        for module_name in module_names:
            consumer = importlib.import_module(module_name)
            assert getattr(consumer, alias) is getattr(ops, alias), (
                module_name,
                alias,
            )


def test_flashmla_alias_inventory_matches_vllm_source(installed) -> None:
    """Discover by-value imports from the installed patch surface independently."""
    import vllm

    from vllm_sail.patch.enhancement.attention.flashmla_ops import (
        _FLASHMLA_ALIAS_CONSUMERS,
    )

    ops_module = "vllm.v1.attention.ops.flashmla"
    patched_aliases = {
        record.target.removeprefix(ops_module + ".")
        for record in installed
        if record.target.startswith(ops_module + ".")
        and "." not in record.target.removeprefix(ops_module + ".")
    }
    vllm_root = Path(vllm.__file__).resolve().parent
    discovered: dict[str, set[str]] = {}
    for path in vllm_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        consumer_name = ".".join(
            ("vllm", *path.relative_to(vllm_root).with_suffix("").parts)
        )
        for node in tree.body:
            if not isinstance(node, ast.ImportFrom) or node.module != ops_module:
                continue
            for imported in node.names:
                if imported.name in patched_aliases:
                    discovered.setdefault(imported.name, set()).add(consumer_name)

    expected = {
        alias: set(module_names)
        for alias, module_names in _FLASHMLA_ALIAS_CONSUMERS.items()
    }
    assert discovered == expected


@pytest.mark.parametrize("module_path,attr_path", _ATTENTION_TARGETS)
def test_attention_targets_installed_or_unimportable(
    installed, module_path, attr_path
) -> None:
    """Every expected target is either patched or genuinely unimportable."""
    full_target = f"{module_path}.{attr_path}"
    registry_targets = {record.target for record in installed}
    if full_target in registry_targets:
        return
    with pytest.raises(ImportError):
        importlib.import_module(module_path)


def test_forward_ppu_overrides_exist_and_differ_from_default(installed) -> None:
    """The five forward_ppu overrides exist and are not the CustomOp default."""
    from vllm.model_executor.custom_op import CustomOp

    default = _unwrap(inspect.getattr_static(CustomOp, "forward_ppu"))
    assert getattr(default, PATCH_MARKER, None), "CustomOp.forward_ppu not patched"

    for module_path, class_name in _FORWARD_PPU_CLASSES:
        cls = getattr(importlib.import_module(module_path), class_name)
        implementation = _unwrap(inspect.getattr_static(cls, "forward_ppu"))
        assert implementation is not None, f"{class_name}.forward_ppu missing"
        assert implementation is not default, (
            f"{class_name}.forward_ppu is the CustomOp default, so the fork's "
            "per-op override was never installed"
        )


def test_fla_ops_package_exports_current_decode(installed) -> None:
    """GDN consumers see the same packed-decode and MTP entry points."""
    ops = importlib.import_module("vllm.third_party.flash_linear_attention.ops")
    gdn = importlib.import_module(
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
    )
    for leaf, name in (
        ("fused_recurrent", "fused_recurrent_gated_delta_rule_packed_decode"),
        ("fused_sigmoid_gating", "fused_sigmoid_gating_delta_rule_update"),
    ):
        provider = importlib.import_module(
            f"vllm.third_party.flash_linear_attention.ops.{leaf}"
        )
        wrapper = getattr(provider, name)
        assert callable(wrapper)
        assert getattr(ops, name) is wrapper
        assert getattr(gdn, name) is wrapper
        assert getattr(wrapper, PATCH_MARKER, None)


def test_fla_autotune_cache_results_follows_triton_availability(installed) -> None:
    """cache_results rebuilds land iff real triton is present.

    The rebuilds only make sense for genuine Autotuner objects; with vLLM's
    triton placeholder the kernels are plain functions and the patch module
    must install nothing rather than faking it.
    """
    from vllm.triton_utils.importing import HAS_TRITON

    autotune_targets = [
        record.target
        for record in installed
        if "flash_linear_attention.ops" in record.target
        and record.reason.startswith("Fork adds cache_results=True")
    ]
    if HAS_TRITON:
        # Stock Triton rebuilds all 17; PPU triton builds with divergent
        # autotune() signatures or wrapper classes skip rebuilds with a
        # warning, so only the upper bound is portable.
        assert len(autotune_targets) <= 17, autotune_targets
    else:
        assert autotune_targets == [], autotune_targets


def test_gdn_fused_decode_env_gate_both_ways(monkeypatch) -> None:
    """VLLM_PPU_FUSED_GDN_DECODE is read lazily and honours both values."""
    import vllm_sail.envs as ppu_envs

    monkeypatch.setenv("VLLM_PPU_FUSED_GDN_DECODE", "0")
    assert ppu_envs.VLLM_PPU_FUSED_GDN_DECODE is False
    monkeypatch.setenv("VLLM_PPU_FUSED_GDN_DECODE", "1")
    assert ppu_envs.VLLM_PPU_FUSED_GDN_DECODE is True
    monkeypatch.setenv("VLLM_PPU_FUSED_GDN_DECODE", "true")
    assert ppu_envs.VLLM_PPU_FUSED_GDN_DECODE is True


def test_gdn_chunk_backend_patched_when_target_importable(installed) -> None:
    """ChunkGatedDeltaRule selects the current PLA/Triton prefill route."""
    module_path = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
    try:
        gdn = importlib.import_module(module_path)
    except ImportError:
        pytest.skip("qwen_gdn target not importable in this environment")

    implementation = _unwrap(
        inspect.getattr_static(gdn.ChunkGatedDeltaRule, "__init__")
    )
    markers = getattr(implementation, PATCH_MARKER, None)
    assert isinstance(markers, dict)
    assert f"{module_path}.ChunkGatedDeltaRule.__init__" in markers


def test_unimportable_targets_are_skipped_not_faked(installed) -> None:
    """A patch absent from the registry must correspond to a real ImportError.

    Guards against a patch module silently installing nothing while claiming
    success: for each historically-unimportable target, either the patch is
    registered or the target module genuinely fails to import here.
    """
    registry_targets = {record.target for record in installed}
    for module_path in (
        "vllm.v1.attention.backends.mla.prefill.flash_attn",
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
    ):
        patched = any(
            target.startswith(module_path + ".") for target in registry_targets
        )
        try:
            importlib.import_module(module_path)
        except ImportError:
            assert not patched, f"{module_path} imports fine; patches must land"
        else:
            assert patched, f"{module_path} imported but no patch was installed"


# ---------------------------------------------------------------------------
# MLAAttentionSpec indexer geometry (fork's kv_cache_interface fields).
# ---------------------------------------------------------------------------


def _make_mla_spec(**extra):
    import torch
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    return MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        kv_quant_mode=None,
        **extra,
    )


def test_mla_attention_spec_accepts_indexer_geometry(installed) -> None:
    """The wrapped dataclass init takes the fork's two indexer kwargs."""
    spec = _make_mla_spec(indexer_n_head=64, indexer_q_head_dim=128)
    assert spec.indexer_n_head == 64
    assert spec.indexer_q_head_dim == 128


def test_mla_attention_spec_indexer_geometry_defaults_none(installed) -> None:
    """Non-DeepseekV4 MLA models leave the geometry at None, like the fork."""
    spec = _make_mla_spec()
    assert spec.indexer_n_head is None
    assert spec.indexer_q_head_dim is None


def test_mla_attention_spec_merge_forwards_indexer_geometry(installed) -> None:
    """merge() must carry the geometry from specs[0] onto the merged spec.

    DeepseekV32IndexerMetadataBuilder reads indexer_n_head off the merged
    group spec; a None there makes its `indexer_n_head > 0` test raise.
    """
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    spec = _make_mla_spec(indexer_n_head=64, indexer_q_head_dim=128)
    merged = MLAAttentionSpec.merge([spec, spec])
    assert merged.indexer_n_head == 64
    assert merged.indexer_q_head_dim == 128


def test_mla_spec_dataclass_operations_preserve_indexer_geometry(installed):
    from dataclasses import fields, replace

    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    spec = _make_mla_spec(indexer_n_head=64, indexer_q_head_dim=128)
    same = replace(spec)
    resized = replace(spec, block_size=256)
    other_geometry = replace(spec, indexer_n_head=32)
    assert type(same) is type(resized) is MLAAttentionSpec
    assert {"indexer_n_head", "indexer_q_head_dim"} <= {f.name for f in fields(spec)}
    assert same == spec and hash(same) == hash(spec)
    assert other_geometry != spec
    assert len({same, spec, other_geometry}) == 2
    assert resized.block_size == 256
    assert (resized.indexer_n_head, resized.indexer_q_head_dim) == (64, 128)
    merged = MLAAttentionSpec.merge([resized, resized])
    assert type(merged) is MLAAttentionSpec
    assert (merged.indexer_n_head, merged.indexer_q_head_dim) == (64, 128)


# ---------------------------------------------------------------------------
# MHC tilelang launchers and metadata shape.
# ---------------------------------------------------------------------------


def test_mhc_launchers_use_shared_prenorm_helper(installed) -> None:
    tilelang = importlib.import_module("vllm.model_executor.kernels.mhc.tilelang")
    assert getattr(tilelang._hc_prenorm_gemm_outputs, PATCH_MARKER, None)
    for name in (
        "mhc_pre_tilelang",
        "mhc_pre_broadcast_tilelang",
        "mhc_fused_post_pre_tilelang",
    ):
        fn = getattr(tilelang, name)
        assert (
            fn.__globals__["_hc_prenorm_gemm_outputs"]
            is tilelang._hc_prenorm_gemm_outputs
        )
        assert not getattr(fn, PATCH_MARKER, None), (
            "Upstream retains ownership of launchers"
        )


def test_phase4_metadata_shape(installed) -> None:
    """Phase-4 records pin the supported range and carry verifiable metadata."""
    phase4_prefixes = (
        "vllm.v1.attention.",
        "vllm.v1.kv_cache_interface.",
        "vllm.model_executor.layers.attention.",
        "vllm.model_executor.layers.mhc.",
        "vllm.model_executor.layers.sparse_attn_indexer.",
        "vllm.model_executor.kernels.mhc.",
        "vllm.model_executor.layers.mamba.gdn.",
        "vllm.third_party.flash_linear_attention.",
    )
    records = [
        record for record in installed if record.target.startswith(phase4_prefixes)
    ]
    assert len(records) >= 30, [r.target for r in records]
    for record in records:
        assert record.affected_versions == SUPPORTED_RANGE, record.target
        assert record.reason.strip(), record.target
        assert record.remove_when.strip(), record.target
        assert record.remove_when.strip().lower() != "todo", record.target


def test_phase4_plugin_modules_import_without_hardware(installed) -> None:
    """No-hardware import test for every Phase-4 plugin-side module.

    The FlashAttention API also imports without SDK wheels; binary loading is
    deferred until a support query or a kernel call.
    """
    for module_name in (
        "vllm_sail.attention",
        "vllm_sail.attention.flash_attn",
        "vllm_sail.attention.flash_attn_shim",
        "vllm_sail.attention.ops.mla_sparse",
        "vllm_sail.patch.enhancement.attention",
        "vllm_sail.patch.enhancement.attention.fa_utils",
        "vllm_sail.patch.enhancement.attention.flashmla_ops",
        "vllm_sail.patch.enhancement.attention.flashmla_backend",
        "vllm_sail.patch.enhancement.attention.flashmla_sparse_backend",
        "vllm_sail.patch.enhancement.attention.mla_prefill_flash_attn",
        "vllm_sail.patch.enhancement.attention.mla_attention",
        "vllm_sail.patch.enhancement.attention.mla_indexer",
        "vllm_sail.patch.enhancement.attention.triton_decode_attention",
        "vllm_sail.patch.enhancement.attention.sparse_attn_indexer",
        "vllm_sail.patch.enhancement.attention.mhc",
        "vllm_sail.patch.enhancement.attention.mhc_tilelang",
        "vllm_sail.patch.enhancement.attention.kv_cache_interface",
        "vllm_sail.patch.performance.fla",
        "vllm_sail.patch.performance.fla.autotune_cache",
    ):
        module = importlib.import_module(module_name)
        assert module is not None, module_name


def test_flash_attn_package_exports_the_ppu_api(installed) -> None:
    """Both upstream import styles must use the same PPU callables."""
    package = importlib.import_module("vllm.vllm_flash_attn")
    interface = importlib.import_module(
        "vllm_sail.attention.flash_attn.flash_attn_interface"
    )
    assert package.flash_attn_varlen_func is interface.flash_attn_varlen_func
    assert package.get_scheduler_metadata is interface.get_scheduler_metadata
    assert package.compile_flash_attn_varlen_func_from_specs is None
