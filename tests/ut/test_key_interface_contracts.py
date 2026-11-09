# SPDX-License-Identifier: Apache-2.0
"""Load-bearing upstream dependencies, expressed as members and real API calls."""

from __future__ import annotations

import ast
from types import SimpleNamespace

import pytest

from tests.support.source import ROOT, function
from tests.support.upstream import dataclass_method, resolve_symbol


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "owner,members",
    [
        ("vllm.platforms.interface.PlatformEnum", ("CUDA", "ROCM", "XPU")),
        (
            "vllm.platforms.interface.Platform",
            ("is_cuda", "is_cuda_alike", "is_out_of_tree", "is_sleep_mode_available"),
        ),
        (
            "vllm.platforms.cuda.NvmlCudaPlatform",
            ("get_device_capability", "get_device_name"),
        ),
        (
            "vllm.model_executor.custom_op.CustomOp",
            ("register", "dispatch_forward", "forward_native", "forward_cuda"),
        ),
        (
            "vllm.v1.attention.backend.AttentionBackend",
            (
                "get_impl_cls",
                "get_builder_cls",
                "validate_configuration",
                "supports_kv_cache_dtype",
            ),
        ),
        (
            "vllm.v1.attention.backend.AttentionMetadataBuilder",
            ("build", "build_for_cudagraph_capture", "update_block_table"),
        ),
        (
            "vllm.v1.attention.backends.registry.AttentionBackendEnum",
            ("get_class",),
        ),
    ],
)
def test_required_upstream_members(upstream_source_root, owner, members):
    assert resolve_symbol(upstream_source_root, owner).present, owner
    for member in members:
        target = f"{owner}.{member}"
        assert resolve_symbol(upstream_source_root, target).present, target


def attention_backend_problems(upstream_root):
    """Check the enum members the production platform actually references."""
    platform = ROOT / "vllm_sail/platform.py"
    members = {
        node.attr
        for node in ast.walk(ast.parse(platform.read_text()))
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "AttentionBackendEnum"
    }
    if not members:
        return ["no platform attention backend references found"]
    owner = "vllm.v1.attention.backends.registry.AttentionBackendEnum"
    return [
        f"missing platform attention backend: {owner}.{member}"
        for member in sorted(members)
        if not resolve_symbol(upstream_root, f"{owner}.{member}").present
    ]


@pytest.mark.upstream_source
def test_platform_attention_dependencies_exist(upstream_source_root):
    assert not attention_backend_problems(upstream_source_root)


@pytest.mark.upstream_source
@pytest.mark.parametrize("member", ["TRITON_ATTN", "FLEX_ATTENTION", "TORCH_SDPA"])
def test_platform_contract_rejects_removed_attention_members(
    upstream_source_root, tmp_path, member
):
    relative = "vllm/v1/attention/backends/registry.py"
    tree = ast.parse((upstream_source_root / relative).read_text())
    enum = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AttentionBackendEnum"
    )
    removed = [
        node
        for node in enum.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == member
            for target in node.targets
        )
    ]
    assert len(removed) == 1, f"fault setup requires the real {member} member"
    enum.body.remove(removed[0])
    mutated = tmp_path / relative
    mutated.parent.mkdir(parents=True)
    mutated.write_text(ast.unparse(tree))
    problems = attention_backend_problems(tmp_path)
    assert problems == [
        "missing platform attention backend: "
        f"vllm.v1.attention.backends.registry.AttentionBackendEnum.{member}"
    ]


@pytest.mark.upstream_source
def test_attention_registration_preserves_backend_identity(upstream_source_root):
    overrides, mamba = {}, {}
    register = function(
        upstream_source_root / "vllm/v1/attention/backends/registry.py",
        "register_backend",
        {"_ATTN_OVERRIDES": overrides, "_MAMBA_ATTN_OVERRIDES": mamba},
    )
    backend, other = object(), object()
    register(backend, "plugin.Backend")
    assert overrides == {backend: "plugin.Backend"} and not mamba

    class CustomBackend:
        pass

    assert register(other)(CustomBackend) is CustomBackend
    assert (
        overrides[other] == f"{CustomBackend.__module__}.{CustomBackend.__qualname__}"
    )
    register(backend, "plugin.Mamba", is_mamba=True)
    assert mamba == {backend: "plugin.Mamba"}
    assert overrides[backend] == "plugin.Backend"


@pytest.mark.upstream_source
def test_linear_registration_keeps_platform_buckets_separate(upstream_source_root):
    namespace = {
        name: {}
        for name in (
            "_POSSIBLE_KERNELS",
            "_POSSIBLE_INT8_KERNELS",
            "_POSSIBLE_FP8_KERNELS",
            "_POSSIBLE_MXFP8_KERNELS",
            "_POSSIBLE_NVFP4_KERNELS",
            "_POSSIBLE_MXFP4_KERNELS",
            "_POSSIBLE_FP8_BLOCK_KERNELS",
        )
    }
    register = function(
        upstream_source_root / "vllm/model_executor/kernels/linear/__init__.py",
        "register_linear_kernel",
        namespace,
    )
    platform, other, first, second = (
        object(),
        object(),
        type("First", (), {}),
        type("Second", (), {}),
    )
    register(first, platform, "fp8")
    register(second, platform, kernel_type="fp8")
    register(second, other, kernel_type="int8")
    assert namespace["_POSSIBLE_FP8_KERNELS"] == {platform: [first, second]}
    assert namespace["_POSSIBLE_INT8_KERNELS"] == {other: [second]}
    with pytest.raises(ValueError):
        register(first, platform, kernel_type="invalid")


@pytest.mark.upstream_source
def test_custom_op_registration_forwards_schema_dispatch_and_fake(upstream_source_root):
    calls = []

    def operation(x):
        return x

    def fake(x):
        return x

    def infer_schema(actual, *, mutates_args):
        assert actual is operation and mutates_args == ["x"]
        return "(Tensor x) -> Tensor"

    library = SimpleNamespace(
        define=lambda *args, **kwargs: calls.append(("define", args, kwargs)),
        impl=lambda *args, **kwargs: calls.append(("impl", args, kwargs)),
        _register_fake=lambda *args: calls.append(("fake", args)),
    )
    register = function(
        upstream_source_root / "vllm/utils/torch_utils.py",
        "direct_register_custom_op",
        {"infer_schema": infer_schema},
    )
    register(
        "ppu_op",
        operation,
        mutates_args=["x"],
        fake_impl=fake,
        target_lib=library,
        dispatch_key="CUDA",
    )
    assert calls == [
        ("define", ("ppu_op(Tensor x) -> Tensor",), {"tags": ()}),
        ("impl", ("ppu_op", operation), {"dispatch_key": "CUDA"}),
        ("fake", ("ppu_op", fake)),
    ]


@pytest.mark.upstream_source
def test_custom_op_registration_rejects_duplicates(upstream_source_root):
    registry = {}
    register = function(
        upstream_source_root / "vllm/model_executor/custom_op.py",
        "CustomOp.register",
        {"op_registry": registry},
    )

    class CustomOp:
        pass

    dimensions = {"x": 0}
    assert register(object, "ppu_op", dynamic_arg_dims=dimensions)(CustomOp) is CustomOp
    assert registry == {"ppu_op": CustomOp}
    assert CustomOp.name == "ppu_op" and CustomOp._dynamic_arg_dims is dimensions
    with pytest.raises(AssertionError, match="Duplicate op name"):
        register(object, "ppu_op")(type("Another", (), {}))


@pytest.mark.upstream_source
def test_indexer_subclass_generated_methods_remain_available(upstream_source_root):
    """Complement the real installer/replace/merge behavior test with source ABI."""
    import ast

    source = upstream_source_root / "vllm/v1/kv_cache_interface.py"
    classes = {
        node.name: node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef)
    }
    base = classes["MLAAttentionSpec"]
    assert {"MLAAttentionSpec", "HiddenStateCacheSpec"} <= classes.keys()
    for node in [
        base,
        *(
            node
            for node in classes.values()
            if any(
                isinstance(parent, ast.Name) and parent.id == "MLAAttentionSpec"
                for parent in node.bases
            )
        ),
    ]:
        for method in ("__init__", "__eq__", "__hash__"):
            assert dataclass_method(node, method), f"{node.name}.{method}"


@pytest.mark.upstream_source
def test_imperative_feature_dependencies_remain_available(upstream_source_root):
    import ast

    alias = resolve_symbol(upstream_source_root, "vllm.config.kernel.MoEBackend")
    field = resolve_symbol(
        upstream_source_root, "vllm.config.kernel.KernelConfig.moe_backend"
    )
    assert isinstance(alias.node, ast.Assign | ast.AnnAssign)
    assert isinstance(alias.node.value, ast.Subscript)
    assert (
        isinstance(alias.node.value.value, ast.Name)
        and alias.node.value.value.id == "Literal"
    )
    assert isinstance(field.node, ast.AnnAssign)
    assert (
        isinstance(field.node.annotation, ast.Name)
        and field.node.annotation.id == "MoEBackend"
    )
    handle = resolve_symbol(
        upstream_source_root,
        "vllm.model_executor.layers.minimax_rms_norm.rms_norm_tp._MINIMAX_FUSED_AR_RMS_QK",
    )
    assert handle.present
