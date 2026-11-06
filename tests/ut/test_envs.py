# SPDX-License-Identifier: Apache-2.0
"""Tests for lazy SAIL environment parsing and PPU compatibility."""

from __future__ import annotations

import ast
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_sail import envs

BOOL_DEFAULTS = {
    "FUSED_RMSNORM_QUANT": False,
    "USE_OPT_TOKEN_GROUP_QUANT": False,
    "DENSE_BF16_DEEPGEMM": False,
    "USE_PLA": True,
    "DEEPGEMM_MOE_TP_FUSED": False,
    "DEEPGEMM_MOE_FUSED_ACT_QUANT": False,
    "FUSED_GDN_DECODE": True,
    "DISABLE_MOE_WNA16_CUDA": False,
    "FORCE_MOE_WNA16_CUDA": False,
    "ENABLE_MOE_MARLIN": False,
    "USE_TRITON_INT8_QUANT": True,
    "ENABLE_FP8_TO_INT8": False,
    "NVTX_PROFILE": False,
    "NVTX_DUMP_TOPK": False,
    "NVTX_VFA_DUMP_SEQLEN": False,
}
BACKENDS = ("MOE_BACKEND", "DENSE_BACKEND")
# String env vars with a non-None default and no legacy VLLM_PPU_* alias.
STRING_DEFAULTS = {"KPOOL_DECODE_LOGITS": "non_paged", "QSA_KERNEL": "auto"}
INT_DEFAULTS = {"KPOOL_MAX_DENSE_POOLS": 1 << 21}


@pytest.fixture(autouse=True)
def clean_plugin_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in dir(envs):
        monkeypatch.delenv(name, raising=False)


def test_defaults_and_canonical_snapshot() -> None:
    expected = {f"VLLM_SAIL_{key}": value for key, value in BOOL_DEFAULTS.items()}
    expected.update({f"VLLM_SAIL_{key}": None for key in BACKENDS})
    expected.update(
        {f"VLLM_SAIL_{key}": value for key, value in STRING_DEFAULTS.items()}
    )
    expected.update({f"VLLM_SAIL_{key}": value for key, value in INT_DEFAULTS.items()})
    expected["VLLM_DEEPEPLL_RECV_HOOK"] = True
    expected["PPU_SDK"] = "/usr/local/PPU_SDK"
    expected["VLLM_GDN_DECODE_KERNEL"] = "auto"
    expected["VLLM_SAIL_FP8_LOAD_POLICY"] = None
    assert envs.snapshot() == expected
    assert set(envs.environment_variables) == set(expected)
    for suffix in (*BOOL_DEFAULTS, *BACKENDS):
        assert getattr(envs, f"VLLM_PPU_{suffix}") == expected[f"VLLM_SAIL_{suffix}"]


@pytest.mark.parametrize("suffix", BOOL_DEFAULTS)
def test_boolean_aliases_are_lazy_and_sail_takes_precedence(
    monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    sail, ppu = f"VLLM_SAIL_{suffix}", f"VLLM_PPU_{suffix}"
    for raw, expected in (("1", True), ("0", False)):
        monkeypatch.setenv(ppu, raw)
        assert getattr(envs, sail) is expected
        assert getattr(envs, ppu) is expected
    monkeypatch.setenv(ppu, "1")
    for raw in ("0", "false", ""):
        monkeypatch.setenv(sail, raw)
        assert getattr(envs, sail) is False
        assert getattr(envs, ppu) is False
        assert envs.snapshot()[sail] is False
    monkeypatch.setenv(ppu, "0")
    monkeypatch.setenv(sail, "1")
    assert getattr(envs, sail) is True
    assert getattr(envs, ppu) is True
    monkeypatch.delenv(sail)
    assert getattr(envs, sail) is False


@pytest.mark.parametrize("prefix", ["VLLM_SAIL", "VLLM_PPU"])
@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on", " On "])
def test_truthy_parsing(monkeypatch: pytest.MonkeyPatch, prefix: str, raw: str) -> None:
    monkeypatch.setenv(f"{prefix}_NVTX_PROFILE", raw)
    assert envs.VLLM_SAIL_NVTX_PROFILE is True


@pytest.mark.parametrize("suffix", BACKENDS)
def test_backend_aliases_and_precedence(
    monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    sail, ppu = f"VLLM_SAIL_{suffix}", f"VLLM_PPU_{suffix}"
    for name in (ppu, sail):
        for raw, expected in (
            (" DeepGEMM ", "deepgemm"),
            ("ACEXT", "acext"),
            ("triton", "triton"),
        ):
            monkeypatch.setenv(name, raw)
            assert getattr(envs, sail) == expected
            assert getattr(envs, ppu) == expected
    monkeypatch.setenv(ppu, "deepgemm")
    for raw in ("", " "):
        monkeypatch.setenv(sail, raw)
        assert getattr(envs, sail) is None
        assert getattr(envs, ppu) is None
    monkeypatch.setenv(sail, "invalid")
    with pytest.raises(ValueError, match=sail):
        getattr(envs, ppu)
    monkeypatch.delenv(sail)
    monkeypatch.setenv(ppu, "invalid")
    with pytest.raises(ValueError, match=ppu):
        getattr(envs, sail)
    monkeypatch.setenv(sail, "acext")
    assert getattr(envs, sail) == "acext"


def test_oldest_nvtx_alias_has_lowest_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAIL_NVTX_PROFILE", "1")
    assert envs.VLLM_SAIL_NVTX_PROFILE is True
    assert envs.is_set("VLLM_SAIL_NVTX_PROFILE") is True
    monkeypatch.setenv("VLLM_PPU_NVTX_PROFILE", "0")
    assert envs.VLLM_SAIL_NVTX_PROFILE is False
    monkeypatch.setenv("VLLM_SAIL_NVTX_PROFILE", "1")
    assert envs.VLLM_PPU_NVTX_PROFILE is True


@pytest.mark.parametrize("suffix", ["USE_PLA", "MOE_BACKEND"])
@pytest.mark.parametrize("prefix", ["VLLM_SAIL", "VLLM_PPU"])
def test_is_set_recognizes_either_name_even_when_empty(
    monkeypatch: pytest.MonkeyPatch, suffix: str, prefix: str
) -> None:
    sail, ppu = f"VLLM_SAIL_{suffix}", f"VLLM_PPU_{suffix}"
    assert envs.is_set(sail) is False
    assert envs.is_set(ppu) is False
    monkeypatch.setenv(f"{prefix}_{suffix}", "")
    assert envs.is_set(sail) is True
    assert envs.is_set(ppu) is True


def test_sdk_path_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    assert envs.PPU_SDK == "/usr/local/PPU_SDK"
    monkeypatch.setenv("PPU_SDK", "/opt/sail")
    assert envs.PPU_SDK == "/opt/sail"


def test_shared_upstream_variable_keeps_its_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_DEEPEPLL_RECV_HOOK", "0")
    assert envs.VLLM_DEEPEPLL_RECV_HOOK is False
    assert envs.is_set("VLLM_DEEPEPLL_RECV_HOOK") is True


def test_choice_helper_supports_unregistered_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    getter = envs._choice("TEST_SAIL_CHOICE", None, ("alpha", "beta"))
    monkeypatch.setenv("TEST_SAIL_CHOICE", "BeTa")
    assert getter() == "beta"
    monkeypatch.setenv("TEST_SAIL_CHOICE", "gamma")
    with pytest.raises(ValueError, match="TEST_SAIL_CHOICE"):
        getter()


def test_dir_lists_resolvable_names_and_unknown_attributes_raise() -> None:
    assert set(envs.environment_variables).issubset(dir(envs))
    assert "VLLM_PPU_NVTX_PROFILE" in dir(envs)
    for name in dir(envs):
        getattr(envs, name)
    with pytest.raises(AttributeError, match="VLLM_SAIL_UNKNOWN"):
        _ = envs.VLLM_SAIL_UNKNOWN


def _load_gdn_functions(relative_path, names, namespace):
    """Execute only the selected original function bodies, without device imports or bypassing device-test gates."""
    path = Path(__file__).parents[2] / relative_path
    tree = ast.parse(path.read_text(), filename=str(path))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in functions} == set(names)
    for node in functions:
        node.decorator_list = []
    future = ast.parse("from __future__ import annotations").body
    module = ast.Module(body=[*future, *functions], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)


@pytest.mark.parametrize("kind", ["decode", "prefill"])
@pytest.mark.parametrize(
    "raw,expected",
    [(None, True), ("1", True), ("0", False), ("true", True), ("false", False)],
)
def test_gdn_e2e_env_cases_ignore_inherited_ci_environment(
    monkeypatch, kind, raw, expected
):
    """Reuse E2E environment test functions so stale interfaces are caught before device CI."""
    name = (
        "test_ppu_pla_cuda_env_resolution"
        if kind == "decode"
        else "test_ppu_pla_env_resolution"
    )
    namespace = {"envs": envs}
    _load_gdn_functions(f"tests/e2e/gdn/test_gdn_pla_{kind}.py", [name], namespace)
    monkeypatch.setenv("VLLM_SAIL_USE_PLA", "1")
    monkeypatch.setenv("VLLM_PPU_USE_PLA", "0")
    namespace[name](monkeypatch, raw, expected)


@pytest.mark.parametrize(
    "name",
    [
        "test_resolver_disabled_by_env",
        "test_resolver_skips_non_ppu",
        "test_resolver_enabled_on_ppu",
        "test_resolver_tolerates_missing_pla",
        "test_resolver_honors_env_after_resolution",
    ],
)
def test_gdn_prefill_e2e_resolver_stubs_match_production(monkeypatch, name):
    """Execute real resolver bodies and E2E stubs without simulating numerical kernels."""
    resolver = types.ModuleType("_gdn_prefill_env_test")
    resolver.__dict__.update(
        envs=envs,
        current_platform=SimpleNamespace(is_ppu=lambda: True),
        _lock=threading.Lock(),
        _resolved=False,
        _chunk_fwd_fn=None,
        _supported_head_configs=frozenset(),
        logger=SimpleNamespace(warning_once=lambda *a: None, info_once=lambda *a: None),
    )
    _load_gdn_functions(
        "vllm_sail/attention/pla_prefill.py",
        [
            "_resolve_sail_cuda_pla_prefill",
            "get_sail_cuda_pla_prefill_fwd",
            "get_sail_cuda_pla_prefill_head_configs",
        ],
        resolver.__dict__,
    )
    namespace = dict(
        sail_cuda_pla_prefill=resolver,
        SimpleNamespace=SimpleNamespace,
        types=types,
        sys=sys,
        HV=8,
        HK=4,
    )
    _load_gdn_functions(
        "tests/e2e/gdn/test_gdn_pla_prefill.py",
        ["_install_fake_pla", "_stub_resolver_env", name],
        namespace,
    )
    if name == "test_resolver_tolerates_missing_pla":
        # Devices may have cached PLA submodules; blocking only the parent cannot simulate a missing package.
        namespace["_install_fake_pla"](monkeypatch)
    namespace[name](monkeypatch)


@pytest.mark.parametrize(
    "name", ["VLLM_SAIL_KPOOL_DECODE_LOGITS", "VLLM_PPU_KPOOL_DECODE_LOGITS"]
)
def test_kpool_rejects_incomplete_paged_logits(monkeypatch, name):
    monkeypatch.delenv("VLLM_SAIL_KPOOL_DECODE_LOGITS", raising=False)
    monkeypatch.delenv("VLLM_PPU_KPOOL_DECODE_LOGITS", raising=False)
    monkeypatch.setenv(name, "paged")
    with pytest.raises(ValueError, match="Expected one of non_paged"):
        _ = envs.VLLM_SAIL_KPOOL_DECODE_LOGITS


@pytest.mark.parametrize("name", ["VLLM_SAIL_QSA_KERNEL", "VLLM_PPU_QSA_KERNEL"])
def test_qsa_kernel_rejects_unknown_value(monkeypatch, name):
    """A typo must fail loudly, not silently select the Triton kernel."""
    monkeypatch.delenv("VLLM_SAIL_QSA_KERNEL", raising=False)
    monkeypatch.delenv("VLLM_PPU_QSA_KERNEL", raising=False)
    monkeypatch.setenv(name, "fa3")
    with pytest.raises(ValueError, match="Expected one of auto, cuda, triton"):
        _ = envs.VLLM_SAIL_QSA_KERNEL


@pytest.mark.parametrize(
    "raw,expected", [("CUDA", "cuda"), (" triton ", "triton"), ("Auto", "auto")]
)
def test_qsa_kernel_normalizes_case_and_space(monkeypatch, raw, expected):
    monkeypatch.setenv("VLLM_SAIL_QSA_KERNEL", raw)
    assert envs.VLLM_SAIL_QSA_KERNEL == expected


def test_qsa_kernel_sail_name_wins_over_ppu_alias(monkeypatch):
    monkeypatch.setenv("VLLM_SAIL_QSA_KERNEL", "cuda")
    monkeypatch.setenv("VLLM_PPU_QSA_KERNEL", "triton")
    assert envs.VLLM_SAIL_QSA_KERNEL == "cuda"
    assert envs.VLLM_PPU_QSA_KERNEL == "cuda"
