# SPDX-License-Identifier: Apache-2.0
"""The Qwen4Exp QSA indexer's MXFP4 compressed cache route.

Three tiers, following the suite's split:

* Bare CPU: ``ops/qsa_prepare_fp4.py`` is a copied upstream body, and
  ``_qsa_prepare_kernel`` is not a patch target, so ``tools/check_patch_drift.py``
  never hashes it. Collapsing its ``PPU MODIFICATION`` regions must reproduce
  upstream's text exactly; that is the only drift guard the kernel has.
* With torch: the MXFP4 page geometry helpers are pure tensor math and run on
  CPU, as does the DeepGEMM wrapper's call shape against a recorder.
* With torch and vLLM: the patch module's platform delegation and the cache
  spec it publishes.

Device tests cover the Triton packing kernel and the DeepGEMM launches.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
from types import SimpleNamespace
from unittest import mock

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
PREPARE_FP4 = ROOT / "vllm_sail" / "models" / "qwen4_exp" / "ops" / "qsa_prepare_fp4.py"
UPSTREAM_PREPARE = "vllm/models/qwen4_exp/nvidia/ops/qsa_prepare.py"

_PATCH_MODULE = "vllm_sail.patch.enhancement.models.qwen4_exp_qsa_fp4"


def _definition_lines(path: pathlib.Path, name: str) -> list[str]:
    source = path.read_text(encoding="utf-8").splitlines()
    tree = ast.parse("\n".join(source))
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1, f"missing or ambiguous {name!r} in {path}"
    (node,) = matches
    return source[node.lineno - 1 : node.end_lineno]


def _upstream_view(lines: list[str]) -> list[str]:
    """Collapse ``PPU MODIFICATION`` regions back to the text they replaced.

    A region either only adds, or it wraps upstream's statement in the ``else``
    of a PPU-only branch; the ``else`` body, dedented one level, is then exactly
    upstream's text.
    """
    collapsed: list[str] = []
    index = 0
    while index < len(lines):
        if "PPU MODIFICATION: begin" not in lines[index]:
            collapsed.append(lines[index])
            index += 1
            continue
        end = index
        while "PPU MODIFICATION: end" not in lines[end]:
            end += 1
        region = lines[index + 1 : end]
        else_at = next(
            (i for i in reversed(range(len(region))) if region[i].strip() == "else:"),
            None,
        )
        if else_at is not None:
            body = region[else_at + 1 :]
            kept = min(len(line) - len(line.lstrip()) for line in body if line.strip())
            drop = kept - (len(region[else_at]) - len(region[else_at].lstrip()))
            collapsed.extend(line[drop:] if line.strip() else line for line in body)
        index = end + 1
    return collapsed


@pytest.mark.upstream_source
@pytest.mark.parametrize("name", ["_qsa_prepare_kernel", "qsa_prepare"])
def test_copied_prepare_reduces_to_upstream(upstream_source_root, name):
    copied = _upstream_view(_definition_lines(PREPARE_FP4, name))
    assert copied == _definition_lines(upstream_source_root / UPSTREAM_PREPARE, name)


@pytest.fixture(scope="module")
def mxfp4():
    pytest.importorskip("torch", reason="requires torch")
    pytest.importorskip("vllm", reason="requires vLLM")
    return importlib.import_module("vllm_sail.models.qwen4_exp.ops.mxfp4")


def test_mxfp4_row_geometry(mxfp4):
    assert mxfp4.MXFP4_LOGICAL_HEAD_DIM == 128
    assert mxfp4.MXFP4_PACKED_HEAD_DIM == 64
    # Four 32-dim block scales fit in the int32 the kernels take as ``k_sf``.
    assert mxfp4.MXFP4_SCALE_BYTES == 4
    assert mxfp4.MXFP4_ROW_BYTES == 68


@pytest.mark.parametrize(
    "shape, match",
    [
        ((1, 64, 2, 68), "one kv head"),
        ((1, 64, 1, 128), "68 bytes wide"),
        ((1, 100, 1, 68), "must divide"),
    ],
)
def test_as_qsa_fp4_pages_rejects_a_layout_it_cannot_address(mxfp4, shape, match):
    import torch

    cache = torch.zeros(shape, dtype=torch.uint8)
    with pytest.raises(ValueError, match=match):
        mxfp4.as_qsa_fp4_pages(cache)


def test_as_qsa_fp4_pages_splits_a_block_into_pages(mxfp4):
    import torch

    cache = torch.zeros((3, 128, 1, 68), dtype=torch.uint8)
    assert mxfp4.qsa_fp4_pages_per_block(128) == 2
    assert mxfp4.as_qsa_fp4_pages(cache).shape == (6, 64, 1, 68)


def test_expand_qsa_fp4_block_table(mxfp4):
    import torch

    table = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    assert mxfp4.expand_qsa_fp4_block_table(table, 1) is table
    assert mxfp4.expand_qsa_fp4_block_table(table, 2).tolist() == [
        [2, 3, 4, 5],
        [6, 7, 8, 9],
    ]


def test_gather_qsa_fp4_keys_reads_the_planar_page(mxfp4):
    """Split a planar page back into the per-row values and scales.

    Capacity past ``cu_seqlens[-1]`` aliases the last request, which is safe
    because every row end stays inside its own request; assert it rather than
    leave it implicit.
    """
    import torch

    rows, packed = mxfp4.MXFP4_PAGE_ROWS, mxfp4.MXFP4_PACKED_HEAD_DIM
    num_pages, capacity = 8, 8
    logical = torch.arange(num_pages * rows, dtype=torch.int32)
    value_bytes = ((logical[:, None] + torch.arange(packed)[None, :]) & 0xFF).to(
        torch.uint8
    )
    scale_bytes = logical.view(torch.uint8).view(num_pages * rows, 4)
    cache = torch.cat([value_bytes, scale_bytes], dim=1).view(num_pages, rows, 1, -1)

    block_table = torch.tensor([[7, 2, 0], [5, 1, 3]], dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int64)
    k_values, k_sf = mxfp4.gather_qsa_fp4_keys(cache, block_table, cu_seqlens, capacity)

    assert k_values.shape == (capacity, packed) and k_values.dtype == torch.int8
    assert k_sf.shape == (capacity,) and k_sf.dtype == torch.int32
    # Rows 0-2 of request 0 come from page 7, rows 0-1 of request 1 from page 5,
    # and the three spare slots keep walking page 5.
    assert k_sf.tolist() == [7 * rows + r for r in (0, 1, 2)] + [
        5 * rows + r for r in (0, 1, 2, 3, 4)
    ]
    for slot, logical_row in enumerate(k_sf.tolist()):
        want = ((logical_row + torch.arange(packed)) & 0xFF).to(torch.uint8)
        assert torch.equal(k_values[slot].view(torch.uint8), want)


def test_fp4_avg_logits_call_shape(monkeypatch):
    """The prefill scorer's contract with DeepGEMM, without a device.

    ``clean_logits`` cannot be set with compression, ``max_seqlen_k`` must be
    the caller's real bound rather than DeepGEMM's 0 default, and the logits
    must not come back bf16, which is the FP4 entry points' own default.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm", reason="requires vLLM")
    from vllm_sail.utils import deep_gemm

    calls: list[tuple[tuple, dict]] = []

    def recorder(*args, **kwargs):
        calls.append((args, kwargs))
        return "logits"

    monkeypatch.setattr(deep_gemm, "_fp4_mqa_avg_logits_impl", recorder)
    assert deep_gemm.has_fp4_mqa_avg_logits()

    q, kv, ks, ke = object(), object(), object(), object()
    assert deep_gemm.fp4_mqa_avg_logits(q, kv, ks, ke, 96) == "logits"
    args, kwargs = calls[0]
    assert args == (q, kv, ks, ke)
    assert kwargs["clean_logits"] is False
    assert kwargs["max_seqlen_k"] == 96
    assert kwargs["logits_dtype"] is torch.float32
    assert "q_scale" not in kwargs

    calls.clear()
    deep_gemm.fp4_mqa_avg_logits(q, kv, ks, ke, 96, q_scale="scale")
    assert calls[0][1]["q_scale"] == "scale"


def test_fp4_avg_metadata_extra_matches_the_schedule_assert():
    pytest.importorskip("vllm", reason="requires vLLM")
    from vllm_sail.utils import deep_gemm

    assert deep_gemm.fp4_avg_metadata_extra(3) == (
        3,
        deep_gemm.FP4_AVG_NUM_HEADS,
        deep_gemm.FP4_AVG_HEAD_DIM,
        1,
    )


def test_require_fp4_mqa_avg_logits_needs_both_kernels(monkeypatch):
    pytest.importorskip("vllm", reason="requires vLLM")
    from vllm_sail.models.qwen4_exp.ops import fp4_select
    from vllm_sail.utils import deep_gemm

    monkeypatch.setattr(deep_gemm, "has_fp4_mqa_avg_logits", lambda: True)
    for paged in (False, True):
        monkeypatch.setattr(
            deep_gemm, "has_fp4_paged_mqa_avg_logits", lambda paged=paged: paged
        )
        if paged:
            fp4_select.require_fp4_mqa_avg_logits()
        else:
            with pytest.raises(RuntimeError, match="fp8_fp4_paged_mqa_avg_logits"):
                fp4_select.require_fp4_mqa_avg_logits()


@pytest.fixture(scope="module")
def qsa_fp4_patch():
    """The module object whose globals the installed wrappers actually read.

    ``test_model_patches`` executes every ``models/`` patch file under a
    synthetic module name, so the targets can already be patched while
    ``sys.modules`` has no entry for the real import path. Importing it then
    would trip the double-patch guard; follow the installed wrapper's
    ``__module__`` instead.
    """
    pytest.importorskip("torch", reason="requires torch")
    pytest.importorskip("vllm", reason="requires vLLM")
    import sys

    from vllm.models.qwen4_exp.nvidia.ops import qsa_prepare as prepare_ops

    from vllm_sail.patch.utils import PATCH_MARKER

    installed = prepare_ops.qsa_prepare
    if getattr(installed, PATCH_MARKER, None) is None:
        return importlib.import_module(_PATCH_MODULE)
    owner = sys.modules.get(getattr(installed, "__module__", ""))
    assert owner is not None, "the installed qsa_prepare wrapper lost its module"
    return owner


def _is_ppu(value: bool):
    from vllm.platforms import current_platform

    return mock.patch.object(type(current_platform), "is_ppu", return_value=value)


@pytest.mark.parametrize(
    "wrapper, upstream, fp4",
    [
        ("qsa_prepare", "_upstream_qsa_prepare", "_fp4_qsa_prepare"),
        (
            "qsa_select_paged_decode",
            "_upstream_select_paged_decode",
            "_fp4_select_paged_decode",
        ),
        (
            "qsa_select_paged_prefill",
            "_upstream_select_paged_prefill",
            "_fp4_select_paged_prefill",
        ),
        (
            "warmup_qsa_mqa_paged_decode",
            "_upstream_warmup_paged_decode",
            "_fp4_warmup_paged_decode",
        ),
    ],
)
def test_scoring_wrappers_delegate_on_platform(
    qsa_fp4_patch, wrapper, upstream, fp4, monkeypatch
):
    calls: list[str] = []
    for name, label in ((upstream, "upstream"), (fp4, "fp4")):
        monkeypatch.setattr(
            qsa_fp4_patch, name, lambda label=label: calls.append(label)
        )
    body = getattr(qsa_fp4_patch, wrapper)

    with _is_ppu(False):
        assert body() is None and calls == ["upstream"]
    calls.clear()
    with _is_ppu(True):
        assert body() is None and calls == ["fp4"]


def test_wrappers_are_installed_on_every_target(qsa_fp4_patch):
    from vllm.models.qwen4_exp.nvidia import qsa_indexer as indexer_alias
    from vllm.models.qwen4_exp.nvidia.ops import qsa_indexer as ops
    from vllm.models.qwen4_exp.nvidia.ops import qsa_prepare as prepare_ops

    assert prepare_ops.qsa_prepare is qsa_fp4_patch.qsa_prepare
    assert indexer_alias.qsa_prepare is qsa_fp4_patch.qsa_prepare
    assert ops.qsa_select_paged_decode is qsa_fp4_patch.qsa_select_paged_decode
    assert ops.qsa_select_paged_prefill is qsa_fp4_patch.qsa_select_paged_prefill
    assert ops.warmup_qsa_mqa_paged_decode is qsa_fp4_patch.warmup_qsa_mqa_paged_decode


def test_compressed_cache_init_publishes_an_mxfp4_row(qsa_fp4_patch, monkeypatch):
    import torch
    from vllm.models.qwen4_exp.common.qsa_indexer import QSACompressedKeyCache

    calls: list[dict] = []
    probes: list[bool] = []
    monkeypatch.setattr(
        qsa_fp4_patch,
        "_QSAIndexerCache",
        SimpleNamespace(__init__=lambda self, **kwargs: calls.append(kwargs)),
    )
    monkeypatch.setattr(
        qsa_fp4_patch,
        "require_fp4_mqa_avg_logits",
        lambda: probes.append(True),
    )
    init = QSACompressedKeyCache.__init__
    kwargs = {
        "head_size": 128,
        "dtype": torch.bfloat16,
        "cache_config": object(),
        "prefix": "model.layers.0.self_attn.indexer.compressed_k_cache",
        "vllm_config": object(),
    }

    with _is_ppu(False):
        init(object(), **kwargs)
    assert calls[-1]["head_size"] == 128
    assert calls[-1]["dtype"] is torch.bfloat16
    assert calls[-1]["compress_ratio"] == 1
    assert probes == []

    with _is_ppu(True):
        init(object(), **kwargs, compress_ratio=4)
    assert calls[-1]["head_size"] == qsa_fp4_patch.MXFP4_ROW_BYTES
    assert calls[-1]["dtype"] is torch.uint8
    assert calls[-1]["compress_ratio"] == 4
    assert probes == [True]

    # The FP4 cache has no fp8 fallback: --indexer-kv-dtype fp8 must fail at
    # model load rather than bind a cache no scorer can read.
    with _is_ppu(True), pytest.raises(NotImplementedError, match="indexer-kv-dtype"):
        init(object(), **kwargs, dtype=torch.float8_e4m3fn)


def test_kernel_block_sizes_follow_the_fp4_page(qsa_fp4_patch):
    import torch
    from vllm.models.qwen4_exp.common.qsa_indexer import QSAIndexerBackend
    from vllm.v1.attention.backend import MultipleOf
    from vllm.v1.kv_cache_interface import CircularBufferSpec, MLAAttentionSpec

    block_sizes = QSAIndexerBackend.get_supported_kernel_block_sizes
    compressed = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=68,
        dtype=torch.uint8,
        tokens_per_state=4,
    )
    raw = CircularBufferSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )

    with _is_ppu(True):
        # One framework block must hold whole 64-row FP4 pages, so a
        # compress_ratio of 4 pushes the block size to a multiple of 256.
        assert block_sizes(compressed) == [MultipleOf(64 * 4)]
        # The raw key ring is bf16 and never reaches an FP4 kernel.
        assert block_sizes(raw) == [MultipleOf(1)]
        assert block_sizes(None) == [MultipleOf(1)]
    with _is_ppu(False):
        assert block_sizes(compressed) == [MultipleOf(1)]
