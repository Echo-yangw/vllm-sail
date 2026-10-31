# SPDX-License-Identifier: Apache-2.0
"""Tests for the Phase-5 model patch modules.

Two tiers, following the suite's established split:

* Bare CPU (no vLLM, no torch required): every module in
  ``vllm_sail/patch/enhancement/models/`` is parsed with ``ast`` and must
  carry well-formed mandatory metadata (``reason`` / ``affected_versions``
  / ``remove_when``) on every ``@patch`` call. The patch modules themselves
  import their vLLM targets at import time — that is the framework's
  convention (see ``custom_op_dispatch.py``) — so actually importing them
  is gated on vLLM being installed, exactly like ``test_patch_install.py``.
* With vLLM: importing the subpackage must apply every patch, land each
  replacement on its target with the marker recording the original, and
  record complete metadata in ``PATCH_REGISTRY``.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import pathlib
import py_compile
import sys

import pytest

from tests.support.source import assert_accepts_upstream_keywords

MODELS_PATCH_DIR = (
    pathlib.Path(__file__).parents[2] / "vllm_sail" / "patch" / "enhancement" / "models"
)

EXPECTED_MODULES = {
    "deepseek_mtp",
    "deepseek_v2",
    "deepseek_v4",
    "deepseek_v4_cache",
    "deepseek_v4_metadata",
    "deepseek_v4_compressor",
    "kimi_k3",
    "llama4",
    "minimax_m3",
    "moe_marlin_gate",
    "qwen3_fused_quant",
    "qwen3_moe",
    "qwen3_next",
    "step3p5_mtp",
}

REQUIRED_METADATA = ("reason", "affected_versions", "remove_when")


def _patch_module_files() -> list[pathlib.Path]:
    return sorted(
        path for path in MODELS_PATCH_DIR.glob("*.py") if path.name != "__init__.py"
    )


def _patch_calls(tree: ast.AST) -> list[ast.Call]:
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "patch":
            calls.append(node)
        elif isinstance(func, ast.Attribute) and func.attr == "patch":
            calls.append(node)
    return calls


def _string_value(node: ast.expr, constants: dict[str, str]) -> str:
    """Constant string, a module-level constant name, or a concatenation."""
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else ""
    if isinstance(node, ast.Name):
        return constants.get(node.id, "")
    if isinstance(node, ast.JoinedStr):
        return "<f-string>"
    parts = [_string_value(part, constants) for part in getattr(node, "values", [])]
    return "".join(parts)


def _module_constants(tree: ast.Module) -> dict[str, str]:
    constants: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, str):
                    constants[target.id] = node.value.value
    return constants


def test_all_expected_patch_modules_exist() -> None:
    on_disk = {path.stem for path in _patch_module_files()}
    assert on_disk == EXPECTED_MODULES, (
        "Model patch modules are the audited port of the fork's modified model "
        "files, the Marlin gate and the Kimi K3 latent-MoE tail; a missing or "
        "extra module means the port drifted from that inventory."
    )


def test_patch_modules_compile_on_bare_cpu() -> None:
    for path in _patch_module_files():
        py_compile.compile(str(path), doraise=True)
    py_compile.compile(str(MODELS_PATCH_DIR / "__init__.py"), doraise=True)


def test_every_patch_has_well_formed_metadata() -> None:
    """AST-level metadata check: runs with no vLLM and no torch."""
    for path in _patch_module_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        constants = _module_constants(tree)
        calls = _patch_calls(tree)
        assert calls, f"{path.name} contains no @patch call"
        for call in calls:
            keywords = {kw.arg: kw.value for kw in call.keywords if kw.arg}
            for field in REQUIRED_METADATA:
                assert field in keywords, f"{path.name}: patch missing {field}"
                value = _string_value(keywords[field], constants)
                assert value.strip(), f"{path.name}: patch {field} is empty"
            remove_when = _string_value(keywords["remove_when"], constants)
            assert remove_when.strip().lower() != "todo", (
                f"{path.name}: remove_when must be a verifiable condition"
            )


def test_subpackage_wiring_is_present() -> None:
    """The parent enhancement package imports models/.

    Guard against a silent behaviour change: the one-line wiring in
    ``vllm_sail/patch/enhancement/__init__.py`` is what makes
    ``vllm_sail.patch.install()`` apply the model patches, so deleting it must
    fail loudly here, not pass unnoticed.
    """
    parent = MODELS_PATCH_DIR.parent / "__init__.py"
    text = parent.read_text()
    assert "enhancement.models" in text or "import models" in text, (
        "models/ patches are no longer wired into the parent package; "
        "restore `from vllm_sail.patch.enhancement import models`."
    )


def test_full_package_chain_on_clean_upstream() -> None:
    """The full enhancement chain installs on a clean upstream vLLM.

    Regression guard for a former known issue: on clean upstream,
    ``SparseAttnIndexer`` subclasses ``CustomOp`` and inherits the
    ``forward_ppu`` attribute installed by the Phase-2 ``custom_op_dispatch``
    patch, which used to trip the double-patch guard of the Phase-4 additive
    ``SparseAttnIndexer.forward_ppu`` patch. The patch framework now shadows
    additive patches into the subclass namespace instead (see
    ``test_patch_framework.py``), so the chain must import cleanly.
    """
    pytest.importorskip("torch", reason="requires torch")
    pytest.importorskip("vllm", reason="requires vLLM")

    module = importlib.import_module("vllm_sail.patch.enhancement.models")
    assert module is not None


# ---------------------------------------------------------------------------
# Everything below needs a real vLLM to patch against.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def installed_model_patches():
    pytest.importorskip("torch", reason="requires torch")
    pytest.importorskip("vllm", reason="requires vLLM")

    # Load each models/ patch module in isolation instead of importing the
    # package: the parent enhancement/__init__ chain installs many more
    # patches, and loading in isolation keeps this fixture independent of
    # unrelated patch failures. (The former clean-upstream forward_ppu
    # collision that also motivated this is fixed in the patch framework.)
    import uuid

    from vllm_sail.patch.utils import PATCH_REGISTRY

    # If the full enhancement chain already ran in this process (fork mode:
    # test_full_package_chain_on_clean_upstream imported it successfully),
    # the targets are already patched; reuse those records instead of
    # re-applying, which the double-patch guard would reject.
    existing = [r for r in PATCH_REGISTRY if r.target in _EXPECTED_TARGETS]
    if len(existing) >= len(_EXPECTED_TARGETS):
        return existing

    registry_before = len(PATCH_REGISTRY)
    loaded = []
    for path in _patch_module_files():
        name = f"vllm_sail_model_patch_test_{path.stem}_{uuid.uuid4().hex[:8]}"
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except ImportError as exc:
            # A missing third-party dep of a vLLM model module (e.g.
            # compressed_tensors on a bare source checkout) is an
            # environment property; a broken plugin import is not.
            if (exc.name or "").startswith(("vllm_sail", "vllm")):
                raise
            pytest.skip(f"vLLM model modules need an unavailable dependency: {exc}")
        loaded.append(module)

    records = [
        r for r in PATCH_REGISTRY[registry_before:] if r.target in _EXPECTED_TARGETS
    ]
    assert records, "no model patch records were installed"
    return records


_EXPECTED_TARGETS = {
    "vllm.model_executor.models.deepseek_v2.Indexer.__init__",
    "vllm.model_executor.models.deepseek_v2.Indexer.forward",
    "vllm.model_executor.models.deepseek_v2.DeepseekV2Model.__init__",
    "vllm.model_executor.models.deepseek_v2.DeepseekV2Model.load_weights",
    "vllm.model_executor.models.deepseek_mtp.DeepSeekMTP.__init__",
    "vllm.model_executor.models.deepseek_mtp.DeepSeekMTP.load_weights",
    "vllm.models.deepseek_v4.nvidia.model._select_dsv4_attn_cls",
    (
        "vllm.models.kimi_k3.nvidia.latent_moe_runner.LatentMoERunner."
        "_overlap_allreduce_tail"
    ),
    ("vllm.models.kimi_k3.nvidia.latent_moe_runner.LatentMoERunner._select_tail_tier"),
    "vllm.models.kimi_k3.nvidia.latent_moe_runner.LatentMoERunner.__init__",
    "vllm.models.kimi_k3.nvidia.model.KimiRoutedOutputTransform.forward",
    "vllm.model_executor.models.llama4.Llama4MoE.custom_routing_function",
    "vllm.models.minimax_m3.nvidia.model.MiniMaxM3Model._init_fused_moe_finalize",
    "vllm.model_executor.models.qwen3_moe.Qwen3MoeSparseMoeBlock.__init__",
    "vllm.model_executor.models.qwen3_moe.Qwen3MoeDecoderLayer.__init__",
    "vllm.model_executor.models.qwen3_moe.Qwen3MoeModel.__init__",
    "vllm.model_executor.models.qwen3_next.th_nvtx_range_push",
    "vllm.model_executor.models.qwen3_next.th_nvtx_range_pop",
    "vllm.model_executor.models.step3p5_mtp.Step3p5MTP.load_weights",
    (
        "vllm.model_executor.layers.quantization.utils.marlin_utils."
        "check_moe_marlin_supports_config"
    ),
}


def test_all_model_patches_land(installed_model_patches) -> None:
    from vllm_sail.patch.utils import PATCH_MARKER

    targets = {record.target for record in installed_model_patches}
    assert _EXPECTED_TARGETS <= targets, _EXPECTED_TARGETS - targets

    for record in installed_model_patches:
        module_path, _, attribute_path = record.target.rpartition(".")
        if record.kind == "class":
            module_name, _, class_path = module_path.rpartition(".")
            module = importlib.import_module(module_name)
            owner = module
            for class_name in class_path.split("."):
                owner = getattr(owner, class_name)
            installed = getattr(owner, attribute_path)
        else:
            module = importlib.import_module(module_path)
            installed = getattr(module, attribute_path)
        marker_source = getattr(installed, PATCH_MARKER, None) or getattr(
            getattr(installed, "__wrapped__", None), PATCH_MARKER, None
        )
        assert marker_source is not None or record.was_missing, record.target


def test_kimi_k3_quantized_up_proj_forces_tier_one(
    installed_model_patches,
) -> None:
    """A quantized routed up-projection must never reach a raw-weight tier.

    Tier selection short-circuits before reading the batch, so ``None`` tensors
    are safe here.
    """
    from types import SimpleNamespace

    from vllm.models.kimi_k3.nvidia.latent_moe_runner import (
        LatentMoERunner,
        LatentTailTier,
    )

    runner = SimpleNamespace(_up_proj_is_quantized=True)
    assert (
        LatentMoERunner._select_tail_tier(runner, None, None)
        is LatentTailTier.ALLREDUCE_OVERLAP
    )


def test_deepseek_v4_selector_uses_ppu_flashmla(installed_model_patches) -> None:
    from types import SimpleNamespace
    from unittest import mock

    from vllm.models.deepseek_v4.nvidia import model as nvidia_model
    from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
        DeepseekV4FlashInferMLAAttention,
    )
    from vllm.models.deepseek_v4.nvidia.flashmla import (
        DeepseekV4FlashMLAAttention as NvidiaDeepseekV4FlashMLAAttention,
    )
    from vllm.platforms import current_platform
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    from vllm_sail.models.deepseek_v4.flashmla import (
        DeepseekV4FlashMLAAttention as PPUDeepseekV4FlashMLAAttention,
    )

    config = SimpleNamespace(
        attention_config=SimpleNamespace(
            backend=AttentionBackendEnum.FLASHMLA_SPARSE,
        )
    )
    platform_type = type(current_platform)
    with (
        mock.patch.object(platform_type, "get_device_capability", return_value=None),
        mock.patch.object(platform_type, "is_ppu", return_value=True),
    ):
        assert (
            nvidia_model._select_dsv4_attn_cls(config) is PPUDeepseekV4FlashMLAAttention
        )

    config.attention_config.backend = AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV4
    with (
        mock.patch.object(platform_type, "get_device_capability", return_value=None),
        mock.patch.object(platform_type, "is_ppu", return_value=True),
    ):
        assert (
            nvidia_model._select_dsv4_attn_cls(config)
            is DeepseekV4FlashInferMLAAttention
        )

    config.attention_config.backend = AttentionBackendEnum.FLASHMLA_SPARSE
    with (
        mock.patch.object(platform_type, "get_device_capability", return_value=None),
        mock.patch.object(platform_type, "is_ppu", return_value=False),
    ):
        assert (
            nvidia_model._select_dsv4_attn_cls(config)
            is NvidiaDeepseekV4FlashMLAAttention
        )


def test_ppu_flashmla_inherits_upstream_attention_flow(
    installed_model_patches,
) -> None:
    """Only the PPU output projection and head-padding policy may diverge."""
    from vllm.models.deepseek_v4.nvidia.flashmla import (
        DeepseekV4FlashMLAAttention as NvidiaDeepseekV4FlashMLAAttention,
    )

    from vllm_sail.models.deepseek_v4.flashmla import (
        DeepseekV4FlashMLAAttention as PPUDeepseekV4FlashMLAAttention,
    )

    assert issubclass(
        PPUDeepseekV4FlashMLAAttention,
        NvidiaDeepseekV4FlashMLAAttention,
    )
    assert (
        PPUDeepseekV4FlashMLAAttention.forward_mqa
        is NvidiaDeepseekV4FlashMLAAttention.forward_mqa
    )
    assert (
        PPUDeepseekV4FlashMLAAttention._forward_prefill
        is NvidiaDeepseekV4FlashMLAAttention._forward_prefill
    )


def test_deepseek_v4_mapper_remains_unpatched(installed_model_patches) -> None:
    """Installing the PPU model selector must not alter CUDA weight mapping."""
    from vllm.models.deepseek_v4.nvidia import model as nvidia_model

    from vllm_sail.patch.utils import PATCH_MARKER

    assert not hasattr(nvidia_model._make_deepseek_v4_weights_mapper, PATCH_MARKER)


def test_marlin_gate_defaults_off_on_ppu(installed_model_patches) -> None:
    """Decision D2(c): MoE Marlin stays disabled on PPU unless opted in.

    The gate short-circuits before touching its config argument, so a
    sentinel object is safe here.
    """
    import os
    from unittest import mock

    from vllm.model_executor.layers.quantization.utils import marlin_utils
    from vllm.platforms import current_platform

    os.environ.pop("VLLM_PPU_ENABLE_MOE_MARLIN", None)
    with mock.patch.object(type(current_platform), "is_ppu", return_value=True):
        assert marlin_utils.check_moe_marlin_supports_config(object()) is False


def test_reapplication_raises(installed_model_patches) -> None:
    """The double-patch guard must still fire for these targets."""
    from vllm_sail.patch.utils import patch

    with pytest.raises(RuntimeError, match="already patched"):

        @patch(
            "vllm.model_executor.models.llama4",
            "Llama4MoE.custom_routing_function",
            reason="test",
            affected_versions=">=0.27.0,<0.28.0",
            remove_when="never; test-only",
        )
        def custom_routing_function(*args, **kwargs):
            raise AssertionError("must not install")


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "local,local_name,upstream,upstream_name",
    [
        (
            "models/deepseek_v4_compressor",
            "__init__",
            "models/deepseek_v4/compressor",
            "DeepseekCompressor.__init__",
        ),
        (
            "models/deepseek_v4_cache",
            "dequantize_and_gather_k_cache",
            "models/deepseek_v4/common/ops/cache_utils",
            "dequantize_and_gather_k_cache",
        ),
    ],
)
def test_replacements_accept_upstream_keywords(
    upstream_source_root, local, local_name, upstream, upstream_name
):
    assert_accepts_upstream_keywords(
        local, local_name, upstream_source_root, upstream, upstream_name
    )
