# SPDX-License-Identifier: Apache-2.0
"""Exercise the real PLA loader under explicit and automatic GDN choices."""

import sys
from types import SimpleNamespace as NS

import pytest

from tests.support.modules import load_patch
from vllm_sail import envs
from vllm_sail.attention import gdn_decode


@pytest.fixture
def pla_loader(modules, monkeypatch):
    monkeypatch.delenv("VLLM_GDN_DECODE_KERNEL", raising=False)
    monkeypatch.setenv("VLLM_SAIL_USE_PLA", "1")
    modules("vllm_sail.envs", **envs.__dict__)
    modules("vllm_sail.attention.gdn_decode", **gdn_decode.__dict__)
    modules("vllm.platforms", current_platform=NS(is_ppu=lambda: True))
    modules(
        "vllm.logger",
        init_logger=lambda _: NS(
            info_once=lambda *a: None, warning_once=lambda *a: None
        ),
    )
    return load_patch("vllm_sail/attention/pla_decode.py")


def test_explicit_choices_control_real_loader_and_cached_handles(
    pla_loader, modules, monkeypatch
):
    def decoder():
        return "decode"

    def packed():
        return "packed"

    modules(
        "pla.decode",
        fused_sigmoid_gating_delta_rule_forward_k_last=decoder,
        fused_sigmoid_gating_delta_rule_forward_k_last_packed=packed,
    )
    monkeypatch.setenv("VLLM_SAIL_USE_PLA", "0")
    assert pla_loader.get_sail_cuda_pla_k_last() is None
    assert not pla_loader._resolved
    monkeypatch.setenv("VLLM_GDN_DECODE_KERNEL", "pla")
    assert pla_loader.get_sail_cuda_pla_k_last() is decoder
    assert pla_loader.get_sail_cuda_pla_k_last_packed() is packed
    monkeypatch.setenv("VLLM_SAIL_USE_PLA", "1")
    for choice in ("cuda", "triton"):
        monkeypatch.setenv("VLLM_GDN_DECODE_KERNEL", choice)
        assert pla_loader.get_sail_cuda_pla_k_last() is None
        assert pla_loader.get_sail_cuda_pla_k_last_packed() is None
    # The legacy switch itself remains available for prefill/KDA callers.
    assert envs.VLLM_SAIL_USE_PLA is True


def test_missing_pla_falls_back_only_for_auto(pla_loader, monkeypatch):
    monkeypatch.setitem(sys.modules, "pla", None)
    assert pla_loader.get_sail_cuda_pla_k_last() is None
    monkeypatch.setenv("VLLM_GDN_DECODE_KERNEL", "pla")
    with pytest.raises(ImportError, match="VLLM_GDN_DECODE_KERNEL=pla"):
        pla_loader.get_sail_cuda_pla_k_last()


def test_shared_library_loader_failure_falls_back_only_for_auto(
    pla_loader, modules, monkeypatch
):
    def unavailable(name):
        raise OSError("PLA shared library ABI mismatch")

    modules("pla.decode", __getattr__=unavailable)
    assert pla_loader.get_sail_cuda_pla_k_last() is None
    monkeypatch.setenv("VLLM_GDN_DECODE_KERNEL", "pla")
    with pytest.raises(ImportError, match="ABI mismatch") as error:
        pla_loader.get_sail_cuda_pla_k_last()
    assert isinstance(error.value.__cause__, OSError)
