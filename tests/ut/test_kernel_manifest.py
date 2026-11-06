# SPDX-License-Identifier: Apache-2.0
"""Integrity of the shipped kernel corpus manifest, and the reader's refusals.

``csrc/upstream/manifest.toml`` is the single source of truth behind the port
tool, the CMake source lists, the runtime stubs and the docs table, so a
malformed or drifted manifest is a build-and-runtime bug rather than a docs bug.
These tests are stdlib-only: the reader never imports torch or vLLM, and neither
does the merge gate.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pytest

from vllm_sail.native import manifest as m

# The upstream ops the plugin's own Python already resolves through the shared
# `_C` / `_moe_C` namespaces, from docs/developer_guide/kernels.md
# section 2.3. `scaled_int8_quant` is deliberately absent: it is a Python wrapper
# in vllm/_custom_ops.py over the two real `*_scaled_int8_quant` ops, not an op.
BORROWED_OPS = (
    "cp_gather_and_upconvert_fp8_kv_cache",
    "cp_gather_cache",
    "cp_gather_indexer_k_quant_cache",
    "cutlass_scaled_mm",
    "cutlass_scaled_mm_azp",
    "dynamic_scaled_int8_quant",
    "gather_and_maybe_dequant_cache",
    "indexer_k_quant_and_cache",
    "per_token_group_fp8_quant",
    "per_token_group_quant_int8",
    "persistent_masked_m_silu_mul_quant",
    "persistent_topk",
    "scaled_fp4_quant",
    "static_scaled_int8_quant",
    "top_k_per_row_decode",
    "top_k_per_row_prefill",
)

# The accepted phase-7 regressions: sailify carries no CUTLASS mappings and this
# build wires no CUTLASS, so these lose their implementation and gain a stub
# (section 2.3, tier x). The two GEMM ops are CUTLASS to the core;
# `scaled_fp4_quant` is not -- only its 175-line entry TU is, so it returns once
# that entry is rewritten CUTLASS-free.
ACCEPTED_REGRESSIONS = frozenset(
    {"cutlass_scaled_mm", "cutlass_scaled_mm_azp", "scaled_fp4_quant"}
)

# Plugin-original, registered by vllm_sail._moe_C into `_ppu_moe_C`, therefore not
# part of the ported upstream corpus.
PLUGIN_OWNED_OPS = ("ep_scatter_2_cuda", "top_k_per_row_prefill_bf16")


@pytest.fixture(scope="module")
def shipped() -> m.Manifest:
    return m.load()


def test_upstream_ref_matches_the_verified_vllm_commit(shipped: m.Manifest) -> None:
    """Runtime compatibility and CI must select the same upstream source."""
    pin = (
        (Path(__file__).parents[2] / ".github/vllm-main-verified.commit")
        .read_text()
        .strip()
    )
    assert re.fullmatch(r"[0-9a-f]{40}", shipped.upstream_ref)
    assert shipped.upstream_ref == pin


def test_every_tier_and_hazard_is_declared(shipped: m.Manifest) -> None:
    assert shipped.tiers
    assert shipped.hazards
    for entry in shipped:
        assert entry.tier in shipped.tiers, entry.path
        for hazard in entry.hazards:
            assert hazard in shipped.hazards, entry.path


def test_paths_are_unique(shipped: m.Manifest) -> None:
    paths = [e.path for e in shipped]
    assert sorted(paths) == sorted(set(paths))


def test_no_op_is_claimed_twice(shipped: m.Manifest) -> None:
    claims = [op for e in shipped for op in e.ops]
    assert sorted(claims) == sorted(set(claims))


def test_no_cutlass_hazard_in_a_compiled_tier(shipped: m.Manifest) -> None:
    assert [e.path for e in shipped.compiled() if "cutlass" in e.hazards] == []


def test_compiled_tiers_are_known_and_never_include_x(shipped: m.Manifest) -> None:
    assert shipped.compiled_tiers
    assert set(shipped.compiled_tiers) <= set(shipped.tiers)
    assert "x" not in shipped.compiled_tiers


def test_prune_bindings_name_real_bindings_entries(shipped: m.Manifest) -> None:
    by_path = {e.path: e for e in shipped}
    assert shipped.prune_bindings
    for path in shipped.prune_bindings:
        assert path in by_path
        assert by_path[path].is_bindings


def test_rename_extension_retargets_upstreams_own_module_names(
    shipped: m.Manifest,
) -> None:
    """Sources must be upstream's names, targets must be ours.

    ``REGISTER_EXTENSION(NAME)`` defines ``PyInit_<NAME>``, so the rename is what
    stops a verbatim copy from exporting upstream's symbol.
    """
    assert shipped.rename_extension
    for upstream, ours in shipped.rename_extension.items():
        assert upstream.endswith("_stable_libtorch"), upstream
        assert ours.startswith("_upstream"), ours
        assert upstream != ours


def test_rename_extension_targets_match_the_cmake_and_python_module_names(
    shipped: m.Manifest,
) -> None:
    """The three-way agreement that only breaks on a PPU host if it drifts.

    A rename target names the ``PyInit_`` symbol the ported TU will export; CMake
    decides the ``.so`` filename; ``extensions.py`` decides what gets imported. If
    any pair disagrees the extension builds and then fails to import, which is
    unobservable without hardware -- so it is gated here.
    """
    # Inspect the checkout under test, even when Python resolves ``vllm_sail``
    # from an installed wheel that intentionally does not ship build sources.
    root = Path(__file__).resolve().parents[2]
    cmake = (root / "CMakeLists.txt").read_text(encoding="utf-8")
    loader = (root / "vllm_sail" / "native" / "extensions.py").read_text(
        encoding="utf-8"
    )

    targets = set(shipped.rename_extension.values())
    cmake_targets = set(re.findall(r"^ppu_add_extension\((_\w+)", cmake, re.MULTILINE))

    # A subset, not equal: CMake also builds the plugin-original `_C` / `_moe_C`,
    # which are ours already and so have nothing to rename.
    assert targets <= cmake_targets
    assert cmake_targets - targets == {"_C", "_moe_C"}
    for target in targets:
        assert f'"vllm_sail.{target}"' in loader


@pytest.mark.parametrize(
    ("path", "vendored"),
    [
        (
            "csrc/libtorch_stable/cache_kernels.cu",
            "csrc/upstream/libtorch_stable/cache_kernels.hg",
        ),
        (
            "csrc/libtorch_stable/moe/moe_wna16.cu",
            "csrc/upstream/libtorch_stable/moe/moe_wna16.hg",
        ),
        (
            "csrc/libtorch_stable/torch_bindings.cpp",
            "csrc/upstream/libtorch_stable/torch_bindings.cpp",
        ),
    ],
)
def test_vendored_path_mirrors_upstream_layout(path: str, vendored: str) -> None:
    assert m.Entry(path=path, tier="a").vendored == vendored


def test_is_moe_is_exactly_the_moe_subtree(shipped: m.Manifest) -> None:
    for entry in shipped:
        assert entry.is_moe == entry.path.startswith("csrc/moe/")


def test_compiled_sources_partition_the_compiled_entries(shipped: m.Manifest) -> None:
    sources, moe_sources = shipped.compiled_sources()
    compiled = shipped.compiled()

    assert sources == tuple(e.vendored for e in compiled if not e.is_moe)
    assert moe_sources == tuple(e.vendored for e in compiled if e.is_moe)
    assert set(sources) | set(moe_sources) == {e.vendored for e in compiled}
    assert set(sources) & set(moe_sources) == set()
    assert sources and moe_sources


def test_every_vendored_hg_source_is_compiled(shipped: m.Manifest) -> None:
    """A normal native build must not silently omit a vendored device TU."""
    root = Path(__file__).parents[2]
    sources, moe_sources = shipped.compiled_sources()
    configured = {path for path in (*sources, *moe_sources) if path.endswith(".hg")}
    vendored = {
        path.relative_to(root).as_posix()
        for path in (root / "csrc" / "upstream").rglob("*.hg")
    }

    assert configured == vendored


def test_spec_section_2_3_lists_sixteen_borrowed_ops() -> None:
    assert len(BORROWED_OPS) == 16
    assert sorted(BORROWED_OPS) == list(BORROWED_OPS)


def test_tier_a_registers_every_borrowed_op(shipped: m.Manifest) -> None:
    """Tier a must cover everything the plugin already calls.

    Anything here other than the two accepted CUTLASS regressions is a silent
    breakage of an existing PPU code path: the op resolves to nothing and the
    call site fails at runtime.
    """
    tier_a = set(shipped.ops(shipped.of_tier("a")))
    missing = sorted(set(BORROWED_OPS) - ACCEPTED_REGRESSIONS - tier_a)
    assert missing == []


def test_accepted_regressions_are_excluded_and_stubbable(shipped: m.Manifest) -> None:
    assert ACCEPTED_REGRESSIONS <= set(shipped.excluded_ops())


@pytest.mark.parametrize(
    "op",
    [
        "fused_deepseek_v4_kv_rope_insert",
        "fused_minimax_m3_qknorm_rope_kv_insert",
        "fused_kimi_k3_mla_decode_q_concat_ds_mla_insert",
        "fused_kimi_k3_mla_decode_q_concat_kv_cache_fp8_insert",
        "fused_kimi_k3_mla_decode_q_concat_kv_cache_insert",
        "fused_kimi_k3_mla_key_concat_ds_mla_insert",
        "fused_kimi_k3_mla_key_concat_kv_cache_insert",
        "fused_kimi_k3_mla_qkv_quant_kv_cache_fp8_insert",
        "concat_and_cache_mla_rope_fused",
        "selective_scan_fwd",
        "rms_norm_static_fp8_quant",
        "fused_add_rms_norm_static_fp8_quant",
        "rms_norm_dynamic_per_token_quant",
        "rms_norm_per_block_quant",
        "silu_and_mul_per_block_quant",
        "fused_qk_norm_rope",
    ],
)
def test_model_kernel_dependencies_are_compiled_and_bound(shipped, op):
    """Check the generated registration as well as the requested build set."""
    assert op in shipped.ops(shipped.compiled())
    root = Path(__file__).parents[2]
    bindings = (root / "csrc/upstream/torch_bindings.cpp").read_text()
    assert f'"{op}(' in bindings or f'"{op}( ' in bindings
    namespace = "_C_cache_ops" if op == "concat_and_cache_mla_rope_fused" else "_C"
    cuda_block = bindings.split(
        f"STABLE_TORCH_LIBRARY_IMPL({namespace}, CUDA, ops) {{"
    )[1].split("\n}")[0]
    assert f"TORCH_BOX(&{op})" in cuda_block
    from vllm_sail.native.stubs import load_schemas

    assert op not in {entry.name for entry in load_schemas()}


@pytest.mark.parametrize(
    "op",
    [
        "topk_softplus_sqrt",
        "fused_deepseek_v4_kv_rope_insert",
        "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
        "fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert",
        "fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert",
    ],
)
def test_deepseek_v4_routing_and_cache_ops_are_compiled(shipped: m.Manifest, op: str):
    # These are reached by DeepSeek V4 profiling and the first real forward.
    assert op in shipped.ops(shipped.compiled())
    assert op not in shipped.excluded_ops()


@pytest.mark.parametrize("op", PLUGIN_OWNED_OPS)
def test_plugin_original_ops_are_not_in_the_upstream_corpus(
    shipped: m.Manifest, op: str
) -> None:
    assert op not in shipped.ops()


def test_python_wrapper_is_not_mistaken_for_an_op(shipped: m.Manifest) -> None:
    """`scaled_int8_quant` is vllm/_custom_ops.py glue, so no TU may claim it."""
    assert "scaled_int8_quant" not in shipped.ops()


_VALID = """\
schema = 1
upstream_ref = "v0.27.1"
compiled_tiers = ["a"]

[tiers]
a = "compiled"
x = "excluded"

[hazards]
cutlass = "CUTLASS / CuTe in the include closure"

[[entry]]
path = "csrc/libtorch_stable/layernorm_kernels.cu"
tier = "a"
ops = ["rms_norm"]
"""


def _write(tmp_path: Path, body: str) -> Path:
    src = tmp_path / "manifest.toml"
    src.write_text(textwrap.dedent(body), encoding="utf-8")
    return src


def test_the_reference_manifest_used_by_the_refusal_cases_is_valid(
    tmp_path: Path,
) -> None:
    """Each refusal case below mutates this one document, so it must load."""
    loaded = m.load(_write(tmp_path, _VALID))
    assert loaded.upstream_ref == "v0.27.1"
    assert loaded.compiled_tiers == ("a",)
    assert [e.path for e in loaded.compiled()] == [
        "csrc/libtorch_stable/layernorm_kernels.cu"
    ]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        pytest.param(
            """\
            schema = 2
            upstream_ref = "v0.27.1"
            compiled_tiers = ["a"]
            [tiers]
            a = "compiled"
            [[entry]]
            path = "a.cu"
            tier = "a"
            """,
            "this reader speaks 1",
            id="wrong-schema",
        ),
        pytest.param(
            """\
            schema = 1
            compiled_tiers = ["a"]
            [tiers]
            a = "compiled"
            [[entry]]
            path = "a.cu"
            tier = "a"
            """,
            "upstream_ref must be a non-empty",
            id="missing-upstream-ref",
        ),
        pytest.param(
            """\
            schema = 1
            upstream_ref = "   "
            compiled_tiers = ["a"]
            [tiers]
            a = "compiled"
            [[entry]]
            path = "a.cu"
            tier = "a"
            """,
            "upstream_ref must be a non-empty",
            id="blank-upstream-ref",
        ),
        pytest.param(
            """\
            schema = 1
            upstream_ref = "v0.27.1"
            compiled_tiers = ["a"]
            [[entry]]
            path = "a.cu"
            tier = "a"
            """,
            "no tiers declared",
            id="no-tiers-table",
        ),
        pytest.param(
            """\
            schema = 1
            upstream_ref = "v0.27.1"
            [tiers]
            a = "compiled"
            [[entry]]
            path = "a.cu"
            tier = "a"
            """,
            "compiled_tiers is empty or missing",
            id="missing-compiled-tiers",
        ),
        pytest.param(
            """\
            schema = 1
            upstream_ref = "v0.27.1"
            compiled_tiers = []
            [tiers]
            a = "compiled"
            [[entry]]
            path = "a.cu"
            tier = "a"
            """,
            "compiled_tiers is empty or missing",
            id="empty-compiled-tiers",
        ),
        pytest.param(
            """\
            schema = 1
            upstream_ref = "v0.27.1"
            compiled_tiers = ["a", "z"]
            [tiers]
            a = "compiled"
            [[entry]]
            path = "a.cu"
            tier = "a"
            """,
            "compiled_tiers names unknown tier 'z'",
            id="compiled-tier-undeclared",
        ),
        pytest.param(
            """\
            schema = 1
            upstream_ref = "v0.27.1"
            compiled_tiers = ["a"]
            [tiers]
            a = "compiled"
            [[entry]]
            tier = "a"
            """,
            "an entry has no path",
            id="entry-without-path",
        ),
        pytest.param(
            """\
            schema = 1
            upstream_ref = "v0.27.1"
            compiled_tiers = ["a"]
            [tiers]
            a = "compiled"
            [[entry]]
            path = "a.cu"
            tier = "q"
            """,
            "a.cu: unknown tier 'q'",
            id="entry-tier-undeclared",
        ),
        pytest.param(
            """\
            schema = 1
            upstream_ref = "v0.27.1"
            compiled_tiers = ["a"]
            [tiers]
            a = "compiled"
            [hazards]
            cub = "CUB"
            [[entry]]
            path = "a.cu"
            tier = "a"
            hazards = ["cub", "tcgen05"]
            """,
            "a.cu: unknown hazard 'tcgen05'",
            id="entry-hazard-undeclared",
        ),
        pytest.param(
            """\
            schema = 1
            upstream_ref = "v0.27.1"
            compiled_tiers = ["a"]
            [tiers]
            a = "compiled"
            [[entry]]
            path = "a.cu"
            tier = "a"
            [[entry]]
            path = "a.cu"
            tier = "a"
            """,
            "duplicate entry 'a.cu'",
            id="duplicate-path",
        ),
        pytest.param(
            """\
            schema = 1
            upstream_ref = "v0.27.1"
            compiled_tiers = ["a"]
            [tiers]
            a = "compiled"
            [[entry]]
            path = "a.cu"
            tier = "a"
            ops = ["rms_norm"]
            [[entry]]
            path = "b.cu"
            tier = "a"
            ops = ["rms_norm"]
            """,
            "op 'rms_norm' is claimed by both a.cu and b.cu",
            id="op-claimed-twice",
        ),
        pytest.param(
            """\
            schema = 1
            upstream_ref = "v0.27.1"
            compiled_tiers = ["a"]
            [tiers]
            a = "compiled"
            [hazards]
            cutlass = "CUTLASS"
            [[entry]]
            path = "a.cu"
            tier = "a"
            hazards = ["cutlass"]
            """,
            "has a cutlass hazard but sits in compiled tier",
            id="cutlass-in-compiled-tier",
        ),
        pytest.param(
            """\
            schema = 1
            upstream_ref = "v0.27.1"
            compiled_tiers = ["a"]
            [tiers]
            a = "compiled"
            [overlay]
            prune_bindings = ["csrc/nope/torch_bindings.cpp"]
            [[entry]]
            path = "a.cu"
            tier = "a"
            """,
            "prune_bindings names unknown 'csrc/nope/torch_bindings.cpp'",
            id="prune-bindings-unknown",
        ),
        pytest.param(
            """\
            schema = 1
            upstream_ref = "v0.27.1"
            compiled_tiers = ["a"]
            [tiers]
            a = "compiled"
            """,
            "no entries",
            id="no-entries",
        ),
    ],
)
def test_load_refuses_malformed_manifest(
    tmp_path: Path, body: str, message: str
) -> None:
    with pytest.raises(m.ManifestError, match=re.escape(message)):
        m.load(_write(tmp_path, body))


def test_load_refuses_compiled_tiers_captured_by_the_tiers_table(
    tmp_path: Path,
) -> None:
    """The TOML footgun this manifest's own comment warns about.

    A bare key written *after* a table header belongs to that table, so
    ``compiled_tiers`` written below ``[tiers]`` becomes ``tiers.compiled_tiers``
    and the top-level key vanishes. That must fail loudly rather than quietly
    compile nothing.
    """
    body = """\
    schema = 1
    upstream_ref = "v0.27.1"

    [tiers]
    a = "compiled"
    compiled_tiers = ["a"]

    [[entry]]
    path = "a.cu"
    tier = "a"
    """
    with pytest.raises(m.ManifestError, match="compiled_tiers is empty or missing"):
        m.load(_write(tmp_path, body))


def test_load_refuses_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(m.ManifestError, match="manifest not found"):
        m.load(tmp_path / "absent" / "manifest.toml")
