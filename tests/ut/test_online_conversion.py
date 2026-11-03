# SPDX-License-Identifier: Apache-2.0
"""Public conversion policy and source contracts without runtime dependencies."""

import ast
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.support.source import definition, function
from vllm_sail import envs
from vllm_sail.model_executor.layers.quantization.online_conversion.config import (
    select_conversion,
    validate_fp8_config,
)

ROOT = Path(__file__).parents[2]
QUANT = "vllm_sail/model_executor/layers/quantization/"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in dir(envs):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("policy", [None, "native", "bf16", "int8"])
@pytest.mark.parametrize("user", [None, "fp8", "fp8-to-int8", "fp8-to-bf16", "other"])
def test_explicit_selection_wins(monkeypatch, policy, user):
    if policy is not None:
        monkeypatch.setenv("VLLM_SAIL_FP8_LOAD_POLICY", policy)
    expected = user if user in ("fp8-to-int8", "fp8-to-bf16") else None
    if user is None:
        expected = {"int8": "fp8-to-int8", "bf16": "fp8-to-bf16"}.get(policy)
    assert select_conversion("fp8", user, is_ppu=True) == expected
    assert select_conversion("fp8", user, is_ppu=False) is None
    assert select_conversion("compressed-tensors", user, is_ppu=True) is None


def test_legacy_alias_and_policy_precedence(monkeypatch):
    assert select_conversion("fp8", None, is_ppu=True) is None
    monkeypatch.setenv("VLLM_PPU_ENABLE_FP8_TO_INT8", "0")
    assert select_conversion("fp8", None, is_ppu=True) == "fp8-to-bf16"
    monkeypatch.setenv("VLLM_SAIL_ENABLE_FP8_TO_INT8", "1")
    assert select_conversion("fp8", None, is_ppu=True) == "fp8-to-int8"
    monkeypatch.setenv("VLLM_SAIL_FP8_LOAD_POLICY", "native")
    assert select_conversion("fp8", None, is_ppu=True) is None
    monkeypatch.setenv("VLLM_SAIL_FP8_LOAD_POLICY", "")
    assert select_conversion("fp8", None, is_ppu=True) is None
    monkeypatch.setenv("VLLM_SAIL_FP8_LOAD_POLICY", "invalid")
    with pytest.raises(ValueError, match="Expected one of"):
        select_conversion("fp8", None, is_ppu=True)


@pytest.mark.parametrize("is_ppu", [True, False])
def test_public_registry_registers_once_only_on_ppu(monkeypatch, is_ppu):
    import importlib.util
    import sys
    import types

    names = []
    registered = {}

    def register(name):
        def apply(cls):
            names.append(name)
            registered[name] = cls

        return apply

    modules = {
        "vllm.logger": {
            "init_logger": lambda _: SimpleNamespace(
                warning=lambda *a: None, debug=lambda *a: None
            )
        },
        "vllm.platforms": {"current_platform": SimpleNamespace(is_ppu=lambda: is_ppu)},
        "vllm.model_executor.layers.quantization": {
            "QUANTIZATION_METHODS": names,
            "register_quantization_config": register,
        },
    }
    for file, cls, name in (
        ("mixed_precision_w4", "MixedPrecisionW4Config", "mixed_precision_w4"),
        ("fp8_to_int8", "Fp8ToInt8Config", "fp8-to-int8"),
        ("fp8_to_bf16", "Fp8ToBf16Config", "fp8-to-bf16"),
        ("bf16_to_int8", "Bf16ToInt8Config", "bf16-to-int8"),
    ):
        modules[f"vllm_sail.model_executor.layers.quantization.{file}"] = {
            cls: type(cls, (), {"get_name": classmethod(lambda cls, n=name: n)})
        }
    modules["vllm_sail.models.deepseek_v4.quant_config"] = {
        "DeepseekV4FP8Config": object
    }
    for name, attrs in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        "_test_conversion_registry",
        ROOT / "vllm_sail/registry/quant_config/__init__.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert registered == {}
    module.register()
    module.register()
    expected = ["mixed_precision_w4"]
    if is_ppu:
        expected += ["fp8-to-int8", "fp8-to-bf16", "bf16-to-int8", "deepseek_v4_fp8"]
    assert names == expected


@pytest.mark.parametrize(
    "changes",
    [
        {"quant_method": None},
        {"weight_block_size": None},
        {"weight_block_size": [64, 128]},
        {"fmt": "e5m2"},
        {"activation_scheme": "static"},
        {"scales_path": "cache.safetensors"},
        {"mix_layers": [-1]},
        {"mix_layer": [True]},
        {"ignored_layers": "not-a-list"},
    ],
)
def test_invalid_checkpoint_metadata(changes):
    config = {"quant_method": "fp8", "weight_block_size": [128, 128], **changes}
    with pytest.raises(ValueError):
        validate_fp8_config(config)


def test_config_validation_does_not_mutate_input():
    config = {"quant_method": "fp8", "weight_block_size": [128, 128]}
    before = repr(config)
    validate_fp8_config(config)
    assert repr(config) == before


def test_mapper_is_idempotent_and_keeps_runtime_patterns():
    mapper = SimpleNamespace(apply_list=lambda names: ["mapped." + n for n in names])
    config = SimpleNamespace(
        _source_ignored=["layer", "re:^visual"],
        _target_ignored=["layer", "re:^visual", "re:.*in_proj_ba$"],
    )
    apply = function(QUANT + "fp8_to_int8.py", "Fp8ToInt8Config.apply_vllm_mapper", {})
    apply(config, mapper)
    expected = list(config.ignored_layers)
    apply(config, mapper)
    assert (
        config.ignored_layers
        == expected
        == [
            "mapped.layer",
            "re:^visual",
            "re:.*in_proj_ba$",
        ]
    )


@pytest.mark.upstream_source
def test_checkpoint_fused_selection(upstream_source_root, monkeypatch):
    import sys
    import types

    matcher = function(
        upstream_source_root
        / "vllm/model_executor/layers/quantization/utils/quant_utils.py",
        "is_layer_skipped",
        {"MappingProxyType": dict},
    )
    module = types.ModuleType(
        "vllm.model_executor.layers.quantization.utils.quant_utils"
    )
    module.is_layer_skipped = matcher
    monkeypatch.setitem(sys.modules, module.__name__, module)
    from vllm_sail.model_executor.layers.quantization.online_conversion.selection import (
        is_ignored,
    )

    prefix = "model.layers.0.mlp.gate_up_proj"
    mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    assert not is_ignored(prefix, ["model.layers.0.mlp.gate"], mapping)
    assert is_ignored(prefix, [prefix], mapping)
    with pytest.raises(ValueError, match="some but not all"):
        is_ignored(prefix, ["model.layers.0.mlp.gate_proj"], mapping)
    with pytest.raises(ValueError, match="some but not all"):
        is_ignored(prefix, [r"re:gate_proj$"], mapping)
    assert is_ignored(prefix, [r"re:(gate|up)_proj$"], mapping)


@pytest.mark.upstream_source
def test_moe_apply_matches_pinned_interface(upstream_source_root):
    local = definition(
        ROOT / (QUANT + "online_conversion/moe.py"), "Fp8ToInt8MoEMethod.apply"
    )
    upstream = definition(
        upstream_source_root / "vllm/model_executor/layers/quantization/fp8.py",
        "Fp8MoEMethod.apply",
    )
    assert [a.arg for a in local.args.args] == [a.arg for a in upstream.args.args]


def test_no_early_bound_oracle_or_fork_imports():
    for path in (ROOT / QUANT / "online_conversion").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "oracle" not in node.module
        assert "vllm.utils.ppu_deep_gemm" not in path.read_text()


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "user,policy,expected",
    [
        (None, "int8", "fp8-to-int8"),
        (None, "bf16", "fp8-to-bf16"),
        (None, "native", "fp8"),
        ("fp8", "int8", "fp8"),
        ("fp8-to-int8", "bf16", "fp8-to-int8"),
        ("fp8-to-bf16", "int8", "fp8-to-bf16"),
    ],
)
def test_upstream_model_config_resolution(
    upstream_source_root, monkeypatch, user, policy, expected
):
    from typing import Literal, cast, get_args

    monkeypatch.setenv("VLLM_SAIL_FP8_LOAD_POLICY", policy)

    def get_config(name):
        if name in ("fp8-to-int8", "fp8-to-bf16"):
            return SimpleNamespace(
                override_quantization_method=lambda cfg, user, **kw: select_conversion(
                    cfg["quant_method"], user, is_ppu=True
                )
            )
        return SimpleNamespace(override_quantization_method=lambda *a, **kw: None)

    registry = SimpleNamespace(
        QUANTIZATION_METHODS=["fp8", "fp8-to-int8", "fp8-to-bf16"],
        QuantizationMethods=Literal["fp8"],
        DEPRECATED_QUANTIZATION_METHODS=[],
        get_quantization_config=get_config,
    )
    verified = []
    verify = function(
        upstream_source_root / "vllm/config/model.py",
        "ModelConfig._verify_quantization",
        {
            "me_quant": registry,
            "cast": cast,
            "get_args": get_args,
            "current_platform": SimpleNamespace(verify_quantization=verified.append),
        },
    )
    model = SimpleNamespace(
        quantization=user,
        hf_config=None,
        model_arch_config=SimpleNamespace(quantization_config={"quant_method": "fp8"}),
    )
    verify(model)
    assert model.quantization == expected
    assert verified == [expected]


def test_floating_safety_patterns_do_not_match_fused_gate_up():
    from vllm_sail.model_executor.layers.quantization.online_conversion.recipe import (
        select_recipe,
    )

    patterns = select_recipe("unknown", None, "bf16")["ignored_layers"]
    assert not any(
        re.search(p[3:], "model.layers.0.mlp.gate_up_proj") for p in patterns
    )
    assert any(
        re.search(p[3:], "model.layers.0.linear_attn.in_proj_ba") for p in patterns
    )


@pytest.mark.parametrize("key", ["mix_layer", "mix_layers"])
def test_source_mixed_metadata_is_valid(key):
    config = {
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
        key: [0, 10],
        "fmt": None,
    }
    validate_fp8_config(config)
    from vllm_sail.model_executor.layers.quantization.online_conversion.selection import (
        is_mixed_layer,
    )

    assert is_mixed_layer("model.layers.0.mlp.down_proj", config[key])
    assert is_mixed_layer("layers.10.mlp.experts", config[key])
    assert not is_mixed_layer("model.layers.1.mlp.down_proj", config[key])


def test_legacy_recipe_selector_and_new_precedence():
    from vllm_sail.model_executor.layers.quantization.online_conversion.recipe import (
        select_recipe,
    )

    config = SimpleNamespace(ppu_quantization_recipe="_default")
    assert select_recipe("renamed", config, "fp8")["id"] == "_default"
    config.sail_quantization_recipe = "unknown"
    with pytest.raises(ValueError, match="Unknown"):
        select_recipe("renamed", config, "bf16")


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("is_ppu", [False, True])
@pytest.mark.parametrize("dtype", ["auto", "float16"])
def test_auto_dtype_preserves_explicit_choice(nested, is_ppu, dtype):
    resolve = function(
        "vllm_sail/patch/enhancement/online_quantization_dtype.py",
        "get_and_verify_dtype",
        {
            "current_platform": SimpleNamespace(is_ppu=lambda: is_ppu),
            "_upstream_get_dtype": lambda model_id, config, dtype, **kw: dtype,
        },
    )
    config = SimpleNamespace(quantization_config={"quant_method": "fp8"})
    if nested:
        config = SimpleNamespace(text_config=config)
    expected = "bfloat16" if is_ppu and dtype == "auto" else dtype
    assert resolve("model", config, dtype, is_pooling_model=False) == expected


@pytest.mark.parametrize("conversion", [False, True])
def test_qwen_mixed_constructor_keeps_conversion_config(conversion):
    # Exercise the actual model-constructor replacement, including its factory.
    # A subclass inheriting this initializer takes the same route.
    config = SimpleNamespace(
        quant_config=SimpleNamespace(handles_source_mixed_layers=conversion),
        model_config=SimpleNamespace(hf_text_config=object()),
    )
    calls = []

    def decoder(**kwargs):
        calls.append(kwargs)

    def upstream(self, *, vllm_config, prefix, decoder_layer_type):
        for index in (0, 1):
            decoder_layer_type(vllm_config=vllm_config, prefix=f"model.layers.{index}")

    init = function(
        "vllm_sail/patch/enhancement/models/qwen3_moe.py",
        "model_init",
        {
            "_mix_layers": lambda _: [0],
            "_upstream_model_init": upstream,
            "extract_layer_index": lambda prefix: int(prefix.rsplit(".", 1)[1]),
            "Qwen3MoeDecoderLayer": decoder,
            "VllmConfig": object,
        },
    )
    init(object(), vllm_config=config, decoder_layer_type=decoder)
    assert len(calls) == 2
    if conversion:
        assert all("quant_config" not in call for call in calls)
        assert all(
            call["vllm_config"].quant_config is config.quant_config for call in calls
        )
    else:
        assert calls[0]["quant_config"] is None
        assert calls[1]["quant_config"] is config.quant_config
