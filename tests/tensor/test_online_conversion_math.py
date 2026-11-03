# SPDX-License-Identifier: Apache-2.0
"""Tensor-only conversion reference tests (torch required; no vLLM or PPU)."""

import pytest

torch = pytest.importorskip("torch")

from vllm_sail.model_executor.layers.quantization.online_conversion.quant_utils import (  # noqa: E402
    dequantize_fp8,
    quantize_int8,
)


@pytest.mark.parametrize("shape", [(1, 1), (129, 257), (256, 384)])
def test_dequantization_matches_full_grid_reference(shape):
    torch.manual_seed(4)
    weight = torch.randn(shape).to(torch.float8_e4m3fn)
    scales = torch.rand(((shape[0] + 127) // 128, (shape[1] + 127) // 128)) + 0.01
    expanded = scales.repeat_interleave(128, 0).repeat_interleave(128, 1)
    expected = (weight.float() * expanded[: shape[0], : shape[1]]).to(torch.bfloat16)
    assert torch.equal(dequantize_fp8(weight, scales), expected)


def test_independent_gate_up_tail_grids():
    gate = torch.ones((65, 128)).to(torch.float8_e4m3fn)
    up = torch.ones((65, 128)).to(torch.float8_e4m3fn)
    decoded = torch.cat(
        [
            dequantize_fp8(gate, torch.tensor([[2.0]])),
            dequantize_fp8(up, torch.tensor([[3.0]])),
        ]
    )
    assert torch.equal(decoded[:65], torch.full((65, 128), 2.0, dtype=torch.bfloat16))
    assert torch.equal(decoded[65:], torch.full((65, 128), 3.0, dtype=torch.bfloat16))


def test_int8_reference_zero_and_rounding():
    weight = torch.tensor(
        [[0, 0, 0, 0], [127, -127, 0.5, 1.5], [1e-10, -1e-10, 0, 0]],
        dtype=torch.bfloat16,
    )
    q, s = quantize_int8(weight)
    reference = (weight.float().abs().amax(1, keepdim=True) / 127).clamp_min(
        torch.finfo(torch.float32).eps
    )
    assert torch.equal(s, reference)
    assert torch.equal(
        q, (weight.float() / reference).round().clamp(-127, 127).to(torch.int8)
    )
    assert q[1].tolist() == [127, -127, 0, 2]


def test_input_shards_use_full_row_scale(monkeypatch):
    full = torch.tensor([[127.0, 2.0, 254.0, -3.0]], dtype=torch.bfloat16)
    expected_q, expected_s = quantize_int8(full)
    calls = []
    group = object()

    def reduce(maxima, op, group):
        calls.append((op, group))
        maxima.copy_(full.float().abs().amax(1, keepdim=True))

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    parts = [quantize_int8(w, tp_group=group) for w in full.chunk(2, 1)]
    assert torch.equal(torch.cat([q for q, _ in parts], 1), expected_q)
    assert all(torch.equal(s, expected_s) for _, s in parts)
    assert calls == [(torch.distributed.ReduceOp.MAX, group)] * 2


@pytest.mark.parametrize("scale", [float("nan"), float("inf"), 0.0, -1.0])
def test_rejects_missing_or_invalid_scales(scale):
    weight = torch.ones((128, 128)).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="finite and positive"):
        dequantize_fp8(weight, torch.tensor([[scale]]))


def test_rejects_bad_dtype_grid_and_nonfinite_weight():
    with pytest.raises(ValueError, match="E4M3"):
        dequantize_fp8(torch.ones((128, 128)), torch.ones((1, 1)))
    with pytest.raises(ValueError, match="grid"):
        dequantize_fp8(
            torch.ones((128, 128)).to(torch.float8_e4m3fn), torch.ones((2, 1))
        )
    with pytest.raises(ValueError, match="non-finite"):
        quantize_int8(torch.tensor([[float("nan")]], dtype=torch.bfloat16))


@pytest.mark.parametrize("to_int8", [False, True])
@pytest.mark.parametrize("gated", [False, True])
def test_moe_conversion_keeps_expert_and_gate_scale_grids(to_int8, gated):
    from types import SimpleNamespace

    from tests.support.source import function

    # Execute the production conversion body with real tensors; backend setup
    # is covered separately on PPU, not replaced by a numerical kernel mock.
    convert = function(
        "vllm_sail/model_executor/layers/quantization/online_conversion/moe.py",
        "Fp8SourceMoE._convert",
        {
            "torch": torch,
            "Parameter": torch.nn.Parameter,
            "dequantize_fp8": dequantize_fp8,
            "quantize_int8": quantize_int8,
            "replace_parameter": lambda layer, name, value: setattr(
                layer, name, torch.nn.Parameter(value, requires_grad=False)
            ),
        },
    )
    layer = torch.nn.Module()
    layer.orig_dtype = torch.bfloat16
    shards = 2 if gated else 1
    layer.w13_weight = torch.nn.Parameter(
        torch.ones((2, shards * 65, 128)).to(torch.float8_e4m3fn), requires_grad=False
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.ones((2, 128, 65)).to(torch.float8_e4m3fn), requires_grad=False
    )
    layer.w13_weight_scale_inv = torch.nn.Parameter(
        torch.arange(1, 2 * shards + 1).float().reshape(2, shards, 1),
        requires_grad=False,
    )
    layer.w2_weight_scale_inv = torch.nn.Parameter(
        torch.tensor([[[3.0]], [[5.0]]]), requires_grad=False
    )
    expected = {}
    for name in ("w13", "w2"):
        source = getattr(layer, name + "_weight")
        scales = getattr(layer, name + "_weight_scale_inv")
        count = shards if name == "w13" else 1
        expected[name] = torch.stack(
            [
                torch.cat(
                    [
                        dequantize_fp8(w, s)
                        for w, s in zip(
                            source[e].chunk(count, 0),
                            scales[e].chunk(count, 0),
                            strict=True,
                        )
                    ]
                )
                for e in range(2)
            ]
        )
    convert(
        SimpleNamespace(moe=SimpleNamespace(tp_size=1, is_act_and_mul=gated)),
        layer,
        to_int8=to_int8,
    )
    for name in ("w13", "w2"):
        if to_int8:
            for e in range(2):
                q, s = quantize_int8(expected[name][e])
                assert torch.equal(getattr(layer, name + "_weight")[e], q)
                assert torch.equal(getattr(layer, name + "_weight_scale")[e], s)
        else:
            assert torch.equal(getattr(layer, name + "_weight"), expected[name])
    assert layer.weight_block_size is None


def test_loader_checks_before_cast_and_forwards_moe_arguments():
    from functools import wraps

    from tests.support.source import function

    make_loader = function(
        "vllm_sail/model_executor/layers/quantization/online_conversion/loading.py",
        "checked_loader",
        {"wraps": wraps},
    )
    calls = []

    def load(param, weight, *args, **kwargs):
        calls.append((args, kwargs))
        param.copy_(weight)
        return True

    param = torch.empty((1, 1), dtype=torch.float8_e4m3fn)
    guarded = make_loader(load, torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="storage/selection mismatch"):
        guarded(param, torch.ones((1, 1), dtype=torch.bfloat16))
    assert calls == []
    assert guarded(
        param, torch.ones_like(param), "w1", expert_id=3, return_success=True
    )
    assert calls == [(("w1",), {"expert_id": 3, "return_success": True})]


@pytest.mark.parametrize("to_int8", [False, True])
@pytest.mark.parametrize("source_dtype", ["fp8", "bf16"])
def test_moe_padding_does_not_enter_decode_or_quantization(to_int8, source_dtype):
    from types import SimpleNamespace

    from tests.support.source import function

    convert = function(
        "vllm_sail/model_executor/layers/quantization/online_conversion/moe.py",
        "Fp8SourceMoE._convert",
        {
            "torch": torch,
            "Parameter": torch.nn.Parameter,
            "dequantize_fp8": dequantize_fp8,
            "quantize_int8": quantize_int8,
            "replace_parameter": lambda layer, name, value: setattr(layer, name, value),
        },
    )
    layer = torch.nn.Module()
    layer.orig_dtype = torch.bfloat16
    hidden, intermediate, padded = 129, 65, 256
    references = {}
    for name in ("w13", "w2"):
        count = 2 if name == "w13" else 1
        rows, cols = (intermediate, hidden) if name == "w13" else (hidden, intermediate)
        full = torch.full(
            (2, count * padded, padded), float("nan"), dtype=torch.bfloat16
        )
        scale = (
            torch.full((2, count * 2, 2), float("nan"))
            if source_dtype == "fp8"
            else None
        )
        references[name] = []
        for e in range(2):
            parts = []
            for shard in range(count):
                values = torch.full(
                    (rows, cols), float(e + shard + 1), dtype=torch.bfloat16
                )
                full[e, shard * padded : shard * padded + rows, :cols] = values
                if scale is not None:
                    scale[
                        e,
                        shard * 2 : shard * 2 + (rows + 127) // 128,
                        : (cols + 127) // 128,
                    ] = 2
                    values = values * 2
                parts.append(values)
            references[name].append(parts)
        if source_dtype == "fp8":
            full = full.to(torch.float8_e4m3fn)
        setattr(layer, name + "_weight", full)
        setattr(layer, name + "_weight_scale_inv", scale)
    convert(
        SimpleNamespace(
            moe=SimpleNamespace(tp_size=1, is_act_and_mul=True),
            source_hidden_size=hidden,
            source_intermediate_size=intermediate,
        ),
        layer,
        to_int8=to_int8,
    )
    for name in ("w13", "w2"):
        for expert, parts in enumerate(references[name]):
            for shard, expected in enumerate(parts):
                actual = getattr(layer, name + "_weight")[expert].chunk(len(parts), 0)[
                    shard
                ]
                rows, cols = expected.shape
                if to_int8:
                    expected, scales = quantize_int8(expected)
                    actual_scales = getattr(layer, name + "_weight_scale")[
                        expert
                    ].chunk(len(parts), 0)[shard]
                    assert torch.equal(actual_scales[:rows], scales)
                assert torch.equal(actual[:rows, :cols], expected)
                assert not torch.count_nonzero(actual[rows:])
                assert not torch.count_nonzero(actual[:, cols:])


def test_flattened_parallel_axes_reduce_the_same_full_row_maximum(monkeypatch):
    full = torch.tensor([[1, 2, 3, 127]], dtype=torch.bfloat16)
    expected_q, expected_scale = quantize_int8(full)
    groups = (object(), object())
    visited = []

    def reduce(maxima, op, group):
        visited.append(group)
        maxima.fill_(2 if group is groups[0] else 127)

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    actual_q, actual_scale = quantize_int8(full[:, :1], tp_group=groups)
    assert visited == list(groups)
    assert torch.equal(actual_q, expected_q[:, :1])
    assert torch.equal(actual_scale, expected_scale)
