# SPDX-License-Identifier: Apache-2.0
"""The installed DSA indexer patch resolves PPU auto dtype through its provider."""

from types import SimpleNamespace as NS

import pytest

from tests.ut.test_ppu_kernel_capabilities import load_patch
from tests.ut.test_ppu_kernel_capabilities import modules as _modules

modules = _modules


class AttentionConfig:
    def __init__(self, value):
        self.indexer_kv_dtype = value

    def resolve_indexer_kv_dtype(self, default):
        return default if self.indexer_kv_dtype == "auto" else self.indexer_kv_dtype


def config(value):
    return NS(attention_config=AttentionConfig(value))


@pytest.fixture
def installed_indexer(modules, patch_utils_module):
    def load(capability, *, ppu=True):
        calls = []
        platform = NS(
            is_ppu=lambda: ppu, is_device_capability=lambda cap: capability == cap
        )
        modules("torch", float32="fp32")
        modules("vllm.platforms", current_platform=platform)
        modules(
            "vllm.v1.attention.backend",
            CommonAttentionMetadata=type("CommonAttentionMetadata", (), {}),
        )
        modules("vllm_sail.patch.utils", patch_value=patch_utils_module.patch_value)
        modules("vllm_sail.utils.deep_gemm")

        class Builder:
            def __init__(self, *args):
                raise AssertionError("not constructing metadata in dtype test")

            @staticmethod
            def _split_indexer_prefill_chunks(*args, **kwargs):
                raise AssertionError("not splitting prefills in dtype test")

            def build(self, *args):
                raise AssertionError("not building a step in dtype test")

            def update_draft_decode_metadata(self, metadata):
                raise AssertionError("not refreshing draft metadata in dtype test")

        def upstream(value):
            calls.append(value)
            return "upstream"

        provider = modules(
            "vllm.v1.attention.backends.mla.indexer",
            DeepseekV32IndexerMetadataBuilder=Builder,
            DeepseekV32IndexerMetadata=type("DeepseekV32IndexerMetadata", (), {}),
            get_paged_mqa_logits_metadata=lambda *args, **kwargs: None,
            dsa_indexer_uses_fp4=upstream,
            _supports_flattened_device_query_lens=lambda: False,
            DSA_INDEXER_KV_DTYPES=("fp8", "mxfp4"),
        )
        patch = load_patch("vllm_sail/patch/enhancement/attention/mla_indexer.py")
        assert provider.dsa_indexer_uses_fp4 is patch.dsa_indexer_uses_fp4
        return provider.dsa_indexer_uses_fp4, calls

    return load


@pytest.mark.parametrize(
    "capability,configured,expected",
    [
        (89, "auto", True),
        (80, "auto", False),
        (89, "fp8", False),
        (80, "fp8", False),
        (89, "mxfp4", True),
    ],
)
def test_indexer_cache_dtype_resolution(
    installed_indexer, capability, configured, expected
):
    uses_fp4, calls = installed_indexer(capability)
    assert uses_fp4(config(configured)) is expected
    assert calls == []


def test_mxfp4_stays_rejected_below_sm_89(installed_indexer):
    uses_fp4, _ = installed_indexer(80)
    with pytest.raises(ValueError, match="requires capability sm_89"):
        uses_fp4(config("mxfp4"))


def test_non_ppu_delegates_to_upstream(installed_indexer):
    uses_fp4, calls = installed_indexer(100, ppu=False)
    value = config("auto")
    assert uses_fp4(value) == "upstream"
    assert calls == [value]
