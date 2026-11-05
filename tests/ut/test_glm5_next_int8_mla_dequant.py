# SPDX-License-Identifier: Apache-2.0
"""INT8 MLA projection dequantisation for GLM-5.3-Flash (no device required).

The patch installs a replacement of
``vllm.models.glm5next.common.model._try_load_fp8_attn_proj``.  These tests
exercise the replacement directly with fake parameters, so no GPU/PPU and no
model weights are needed.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm.models.glm5next.common import model as glm5next_model  # noqa: E402

from vllm_sail.patch import PATCH_REGISTRY  # noqa: E402
from vllm_sail.patch.bugfix import (  # noqa: E402
    glm5_next_int8_mla_dequant as int8_patch,
)

_PREFIX = "model.language_model.layers.3.self_attn"


class _FakeParam:
    """Minimal stand-in for a vLLM ``Parameter`` with the loader contract."""

    def __init__(self, dtype: torch.dtype):
        self.dtype = dtype
        self.loads: list[tuple[torch.Tensor, object]] = []

    def weight_loader(self, param, weight, shard_id=None):
        assert param is self
        self.loads.append((weight.clone(), shard_id))


def test_patch_is_installed_and_targeted():
    targets = {record.target for record in PATCH_REGISTRY}
    assert "vllm.models.glm5next.common.model._try_load_fp8_attn_proj" in targets
    assert glm5next_model._try_load_fp8_attn_proj is int8_patch._try_load_fp8_attn_proj


def test_int8_weight_and_scale_are_dequantised():
    weight = torch.randint(-127, 128, (8, 16), dtype=torch.int8)
    scale = torch.rand(8, 1) * 1e-3
    param = _FakeParam(torch.bfloat16)
    params_dict = {f"{_PREFIX}.o_proj.weight": param}
    buf, loaded = {}, set()

    assert glm5next_model._try_load_fp8_attn_proj(
        f"{_PREFIX}.o_proj.weight", weight, buf, params_dict, loaded, 0
    )
    assert loaded == set()  # waiting for the scale
    assert glm5next_model._try_load_fp8_attn_proj(
        f"{_PREFIX}.o_proj.weight_scale", scale, buf, params_dict, loaded, 0
    )
    assert loaded == {f"{_PREFIX}.o_proj.weight"}
    assert len(param.loads) == 1
    loaded_weight, shard_id = param.loads[0]
    assert shard_id is None
    assert loaded_weight.dtype == torch.bfloat16
    expected = (weight.float() * scale).to(torch.bfloat16)
    assert torch.equal(loaded_weight, expected)


def test_scale_may_arrive_before_weight():
    weight = torch.randint(-127, 128, (8, 16), dtype=torch.int8)
    scale = torch.rand(8, 1) * 1e-3
    param = _FakeParam(torch.bfloat16)
    params_dict = {f"{_PREFIX}.o_proj.weight": param}
    buf, loaded = {}, set()

    assert glm5next_model._try_load_fp8_attn_proj(
        f"{_PREFIX}.o_proj.weight_scale", scale, buf, params_dict, loaded, 0
    )
    assert glm5next_model._try_load_fp8_attn_proj(
        f"{_PREFIX}.o_proj.weight", weight, buf, params_dict, loaded, 0
    )
    assert len(param.loads) == 1
    assert torch.equal(param.loads[0][0], (weight.float() * scale).to(torch.bfloat16))


def test_quantised_target_is_left_to_the_regular_loader():
    weight = torch.randint(-127, 128, (8, 16), dtype=torch.int8)
    scale = torch.rand(8, 1) * 1e-3
    param = _FakeParam(torch.int8)  # module keeps the projection quantised
    params_dict = {f"{_PREFIX}.o_proj.weight": param}
    buf, loaded = {}, set()

    for name, tensor in (
        (f"{_PREFIX}.o_proj.weight", weight),
        (f"{_PREFIX}.o_proj.weight_scale", scale),
    ):
        assert not glm5next_model._try_load_fp8_attn_proj(
            name, tensor, buf, params_dict, loaded, 0
        )
    assert param.loads == []
    assert buf == {}


def test_fp8_block_dequant_path_is_unchanged():
    weight = torch.randn(256, 128).to(torch.float8_e4m3fn)
    scale_inv = torch.rand(2, 1) * 1e-2
    param = _FakeParam(torch.bfloat16)
    params_dict = {f"{_PREFIX}.o_proj.weight": param}
    buf, loaded = {}, set()

    assert glm5next_model._try_load_fp8_attn_proj(
        f"{_PREFIX}.o_proj.weight", weight, buf, params_dict, loaded, 0
    )
    assert glm5next_model._try_load_fp8_attn_proj(
        f"{_PREFIX}.o_proj.weight_scale_inv", scale_inv, buf, params_dict, loaded, 0
    )
    assert loaded == {f"{_PREFIX}.o_proj.weight"}
    loaded_weight = param.loads[0][0]
    assert loaded_weight.dtype == torch.bfloat16
    expected = glm5next_model._dequant_fp8_block(weight, scale_inv, 128)
    assert torch.equal(loaded_weight, expected)


def test_fused_q_a_and_kv_a_shards_are_forwarded():
    q_a = torch.randint(-127, 128, (6, 16), dtype=torch.int8)
    q_a_scale = torch.rand(6, 1) * 1e-3
    kv_a = torch.randint(-127, 128, (4, 16), dtype=torch.int8)
    kv_a_scale = torch.rand(4, 1) * 1e-3
    fused = _FakeParam(torch.bfloat16)
    params_dict = {f"{_PREFIX}.fused_qkv_a_proj.weight": fused}
    buf, loaded = {}, set()

    for name, tensor in (
        (f"{_PREFIX}.q_a_proj.weight", q_a),
        (f"{_PREFIX}.q_a_proj.weight_scale", q_a_scale),
        (f"{_PREFIX}.kv_a_proj_with_mqa.weight", kv_a),
        (f"{_PREFIX}.kv_a_proj_with_mqa.weight_scale", kv_a_scale),
    ):
        assert glm5next_model._try_load_fp8_attn_proj(
            name, tensor, buf, params_dict, loaded, 0
        )

    assert loaded == {f"{_PREFIX}.fused_qkv_a_proj.weight"}
    assert [shard for _, shard in fused.loads] == [0, 1]


def test_nope_kv_a_padding_is_still_appended():
    kv_a = torch.randint(-127, 128, (4, 16), dtype=torch.int8)
    kv_a_scale = torch.rand(4, 1) * 1e-3
    fused = _FakeParam(torch.bfloat16)
    params_dict = {f"{_PREFIX}.fused_qkv_a_proj.weight": fused}
    buf, loaded = {}, set()

    glm5next_model._try_load_fp8_attn_proj(
        f"{_PREFIX}.kv_a_proj_with_mqa.weight", kv_a, buf, params_dict, loaded, 2
    )
    glm5next_model._try_load_fp8_attn_proj(
        f"{_PREFIX}.kv_a_proj_with_mqa.weight_scale",
        kv_a_scale,
        buf,
        params_dict,
        loaded,
        2,
    )

    loaded_weight, shard_id = fused.loads[0]
    assert shard_id == 1
    assert loaded_weight.shape == (6, 16)
    assert torch.all(loaded_weight[4:] == 0)
    assert torch.equal(
        loaded_weight[:4], (kv_a.float() * kv_a_scale).to(torch.bfloat16)
    )


def test_unrelated_names_are_ignored():
    param = _FakeParam(torch.bfloat16)
    params_dict = {f"{_PREFIX}.o_proj.weight": param}
    buf, loaded = {}, set()

    assert not glm5next_model._try_load_fp8_attn_proj(
        f"{_PREFIX}.o_proj.bias", torch.zeros(8), buf, params_dict, loaded, 0
    )
    assert not glm5next_model._try_load_fp8_attn_proj(
        "model.language_model.layers.3.mlp.gate.weight",
        torch.zeros(8, dtype=torch.int8),
        buf,
        params_dict,
        loaded,
        0,
    )
    assert param.loads == []


def test_fp8_per_channel_weight_and_scale_are_dequantised():
    """FP8-PerChannel checkpoints use Quark ``weight_scale`` of shape [out, 1].

    The upstream block-FP8 map would expand it to an out*in tensor and fail
    (observed on the 890P MXFP4/FP8 checkpoint); the patch must dequantise it
    row-wise.
    """
    weight = torch.randn(8, 16).to(torch.float8_e4m3fn)
    scale = torch.rand(8, 1) * 1e-2
    param = _FakeParam(torch.bfloat16)
    params_dict = {f"{_PREFIX}.o_proj.weight": param}
    buf, loaded = {}, set()

    assert glm5next_model._try_load_fp8_attn_proj(
        f"{_PREFIX}.o_proj.weight", weight, buf, params_dict, loaded, 0
    )
    assert glm5next_model._try_load_fp8_attn_proj(
        f"{_PREFIX}.o_proj.weight_scale", scale, buf, params_dict, loaded, 0
    )
    assert loaded == {f"{_PREFIX}.o_proj.weight"}
    loaded_weight, shard_id = param.loads[0]
    assert shard_id is None
    assert loaded_weight.dtype == torch.bfloat16
    assert torch.equal(loaded_weight, (weight.float() * scale).to(torch.bfloat16))


def test_fp8_per_channel_one_dimensional_scale_is_supported():
    weight = torch.randn(4, 8).to(torch.float8_e4m3fn)
    scale = torch.rand(4) * 1e-2
    param = _FakeParam(torch.bfloat16)
    params_dict = {f"{_PREFIX}.o_proj.weight": param}
    buf, loaded = {}, set()

    for name, tensor in (
        (f"{_PREFIX}.o_proj.weight", weight),
        (f"{_PREFIX}.o_proj.weight_scale", scale),
    ):
        glm5next_model._try_load_fp8_attn_proj(
            name, tensor, buf, params_dict, loaded, 0
        )

    assert loaded == {f"{_PREFIX}.o_proj.weight"}
    assert torch.equal(
        param.loads[0][0], (weight.float() * scale.reshape(-1, 1)).to(torch.bfloat16)
    )


def test_fp8_per_tensor_scale_is_dequantised():
    weight = torch.randn(4, 8).to(torch.float8_e4m3fn)
    scale = torch.tensor(0.5)
    param = _FakeParam(torch.bfloat16)
    params_dict = {f"{_PREFIX}.o_proj.weight": param}
    buf, loaded = {}, set()

    for name, tensor in (
        (f"{_PREFIX}.o_proj.weight", weight),
        (f"{_PREFIX}.o_proj.weight_scale", scale),
    ):
        glm5next_model._try_load_fp8_attn_proj(
            name, tensor, buf, params_dict, loaded, 0
        )

    assert loaded == {f"{_PREFIX}.o_proj.weight"}
    loaded_weight, shard_id = param.loads[0]
    assert shard_id is None
    expected = (weight.float() * scale).to(torch.bfloat16)
    assert torch.equal(loaded_weight, expected)


def test_quantised_fp8_target_is_left_to_the_regular_loader():
    """A target that carries ``weight_scale`` is natively quantised."""
    weight = torch.randn(8, 16).to(torch.float8_e4m3fn)
    scale = torch.rand(8, 1) * 1e-2
    params_dict = {
        f"{_PREFIX}.o_proj.weight": _FakeParam(torch.float8_e4m3fn),
        f"{_PREFIX}.o_proj.weight_scale": _FakeParam(torch.float32),
    }
    buf, loaded = {}, set()

    for name, tensor in (
        (f"{_PREFIX}.o_proj.weight", weight),
        (f"{_PREFIX}.o_proj.weight_scale", scale),
    ):
        assert not glm5next_model._try_load_fp8_attn_proj(
            name, tensor, buf, params_dict, loaded, 0
        )
    assert buf == {}
