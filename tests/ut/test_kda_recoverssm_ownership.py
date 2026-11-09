# SPDX-License-Identifier: Apache-2.0
"""RecoverSSM layer ownership and speculative dispatch, using host doubles."""

import ast
import math
from dataclasses import dataclass
from types import MethodType
from types import SimpleNamespace as NS

import pytest

from tests.support.source import ROOT, definition, function

UPSTREAM = "vllm/models/kimi_k3/nvidia/kda.py"
KERNELS = "vllm/models/kimi_k3/nvidia/recoverssm/kernels.py"
RECORDS = "vllm/models/kimi_k3/nvidia/ops/recoverssm.py"


class Tensor:
    dtype = "bf16"
    device = "host-double"

    def __init__(self, shape, root=None):
        self.shape = tuple(shape)
        self.root = self if root is None else root
        self.copies = []

    def size(self, axis):
        return self.shape[axis]

    def numel(self):
        return math.prod(self.shape)

    def element_size(self):
        return 4 if self.dtype == "fp32" else 2

    def __getitem__(self, index):
        indices = index if isinstance(index, tuple) else (index,)
        shape = []
        for axis, length in enumerate(self.shape):
            item = indices[axis] if axis < len(indices) else slice(None)
            if isinstance(item, slice):
                start, stop, step = item.indices(length)
                assert step == 1
                shape.append(stop - start)
        return Tensor(shape, self.root)

    def view(self, *shape):
        assert math.prod(shape) == self.numel()
        return Tensor(shape, self.root)

    def split(self, width, dim):
        dim %= len(self.shape)
        assert self.shape[dim] % width == 0
        shape = (*self.shape[:dim], width, *self.shape[dim + 1 :])
        return tuple(Tensor(shape, self.root) for _ in range(self.shape[dim] // width))

    def transpose(self, left, right):
        shape = list(self.shape)
        shape[left], shape[right] = shape[right], shape[left]
        return Tensor(shape, self.root)

    def copy_(self, source):
        self.copies.append(source)
        return self


def constructor_records_block(path, name):
    """Execute just the constructor's records/backend ownership transaction."""
    node = definition(ROOT / path, name)
    starts = [
        i
        for i, statement in enumerate(node.body)
        if isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Attribute)
        and statement.target.attr == "recoverssm_backend"
    ]
    assert len(starts) == 1, "constructor must initialize layer-owned records"
    start = starts[0]
    end = next(
        i
        for i, statement in enumerate(node.body[start:], start)
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "decode_conv1d_weight"
            for t in statement.targets
        )
    )
    return ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *node.body[start:end],
        ],
        type_ignores=[],
    )


def check_constructor(path, name, recover, capability, source_root=None):
    events, allocated = [], []
    ppu = capability != 100

    def allocate(*args):
        allocated.append(args)
        return NS(correction=Tensor((2, 2, 4, 128)), key_gate=Tensor((2, 2, 4, 256)))

    class Cute:
        def __init__(self, **kwargs):
            assert not ppu
            events.append(("cute", kwargs))

        def compile(self):
            events.append(("compile",))

    namespace = {
        "__name__": __name__,
        "torch": NS(bfloat16="bf16", float32="fp32"),
        "current_platform": NS(
            is_cuda=lambda: True,
            is_device_capability_family=lambda cap: not ppu and cap == 100,
        ),
        "K": 128,
        "CONV_WIDTH": 4,
        "MAX_WINDOW": 8,
        "KDARecoverSSMRecords": NS(allocate=allocate),
        "CuteRecoverSSMVerify": Cute,
        "can_use_cute_recoverssm": lambda **kw: (
            not ppu,
            "host predicate" if ppu else None,
        ),
        "compile_state_commit": lambda *a, **kw: events.append(("commit", a, kw)),
        "is_conv_state_dim_first": lambda: True,
        "logger": NS(info_once=lambda *a: None),
        "vllm_config": NS(
            scheduler_config=NS(max_num_seqs=2), model_config=NS(dtype="bf16")
        ),
        "conv_state_dtype": "bf16",
        "recurrent_state_dtype": "bf16",
        "kda_config": {"kda_layers": [0, 1]},
    }
    if source_root is not None:
        function(source_root / KERNELS, "can_use_cute_recoverssm", namespace)
        tensor_class = namespace["torch"]

        def empty(shape, *, dtype, device):
            allocated.append((shape, dtype, device))
            tensor = Tensor(shape)
            tensor.dtype = dtype
            return tensor

        tensor_class.empty = empty
        records = function(source_root / RECORDS, "KDARecoverSSMRecords", namespace)
        namespace["KDARecoverSSMRecords"] = dataclass(frozen=True)(records)
    code = compile(
        ast.fix_missing_locations(constructor_records_block(path, name)),
        str(path),
        "exec",
    )
    layers = []
    for _ in range(2):
        layer = NS(
            use_recoverssm=recover,
            local_num_heads=2,
            spec_query_len=4,
            head_dim=128,
            conv_size=4,
            gate_lower_bound=-5.0,
            conv1d=NS(weight=NS(device="host-double", dtype="bf16"), bias=None),
            cache_config=NS(mamba_cache_mode="align"),
        )
        namespace["self"] = layer
        exec(code, namespace)
        layers.append(layer)
        assert (
            layer.recoverssm_backend == ("triton" if ppu else "cute")
            if recover
            else layer.recoverssm_backend is None
        )
        assert (layer.recoverssm_records is not None) is recover
        assert (layer.recoverssm_verify is not None) is (recover and not ppu)
    if recover:
        assert layers[0].recoverssm_records is not layers[1].recoverssm_records
        assert all(
            layer.recoverssm_records.correction.shape == (2, 2, 4, 128)
            for layer in layers
        )
        assert all(
            layer.recoverssm_records.key_gate.shape == (2, 2, 4, 256)
            for layer in layers
        )
        if source_root is None:
            assert allocated == [(2, 2, 4, 128, "bf16", "host-double")] * 2
        else:
            assert (
                allocated
                == [
                    ((2, 2, 4, 128), "fp32", "host-double"),
                    ((2, 2, 4, 256), "bf16", "host-double"),
                ]
                * 2
            )
    else:
        assert not allocated
    if recover and not ppu:
        assert [e[0] for e in events] == ["cute", "compile", "commit"] * 2
        assert events[0][1] == {
            "num_heads": 2,
            "window": 4,
            "conv_dim_first": True,
            "conv_weight_dtype": "bf16",
            "device": "host-double",
        }
        assert events[2] == ("commit", (2, 4, "host-double"), {"align": True})
    else:
        assert events == []


@pytest.mark.upstream_source
@pytest.mark.parametrize("recover", [False, True])
@pytest.mark.parametrize("capability", [80, 89, 100])
def test_actual_records_allocation_and_platform_predicate(
    upstream_source_root, recover, capability
):
    check_constructor(
        upstream_source_root / UPSTREAM,
        "KimiK3DeltaAttention.__init__",
        recover,
        capability,
        upstream_source_root,
    )


def check_forward(path, name, modules, recover, dim_first, real_verify_path=None):
    events = []
    mixed, g, beta, gate, output = (
        Tensor((3, 12)),
        Tensor((1, 3, 1, 4)),
        Tensor((1, 3, 1)),
        Tensor((3, 4)),
        Tensor((1, 3, 1, 4)),
    )
    conv = Tensor((1, 12, 3) if dim_first else (1, 3, 12))
    state, accepted, indices, starts = (
        Tensor((1, 1, 4, 4)),
        Tensor((1,)),
        Tensor((1, 3)),
        Tensor((2,)),
    )
    metadata = NS(
        has_initial_state=None,
        non_spec_query_start_loc=None,
        non_spec_state_indices_tensor=None,
        spec_token_indx=None,
        non_spec_token_indx=None,
        spec_token_start=None,
        non_spec_token_start=None,
        spec_state_indices_tensor=indices,
        spec_query_start_loc=starts,
        num_accepted_tokens=accepted,
        num_actual_tokens=3,
        checkpoint=None,
        num_spec_decodes=1,
        num_prefills=0,
        num_decodes=0,
    )

    def convolution(*args, **kwargs):
        events.append(("conv", args, kwargs))
        return kwargs["out"]

    def recurrent(**kwargs):
        events.append(("recurrent", kwargs))
        return kwargs["out"], None

    def verify(*args):
        events.append(("verify", args))
        return args[-1]

    def normalize(actual, actual_gate):
        assert actual is output and actual_gate is gate
        events.append(("norm",))
        return output

    modules("vllm_sail.envs", VLLM_SAIL_USE_PLA=False)
    modules(
        "vllm.models.kimi_k3.nvidia.ops.recoverssm",
        kda_recoverssm_verify=lambda **kw: pytest.fail(
            "old KV-owned records path reached"
        ),
    )
    modules(
        "vllm.models.kimi_k3.nvidia.ops.third_party.kda",
        chunk_kda_with_fused_gate=lambda **kw: pytest.fail("unexpected prefill"),
        fused_recurrent_kda=recurrent,
        fused_recurrent_kda_packed_decode=lambda **kw: pytest.fail("unexpected decode"),
    )
    layer = NS(
        prefix="kda",
        kv_cache=(conv, state, object()),
        kda_spec_decode_backend="triton",
        kda_decode_backend="triton",
        local_projection_size=4,
        head_dim=4,
        spec_query_len=8,
        use_recoverssm=recover,
        _recoverssm_verify=verify,
        conv1d=NS(weight=Tensor((12, 1, 4)), bias=None),
        A_log=object(),
        dt_bias=object(),
        gate_lower_bound=-5.0,
        o_norm=normalize,
    )
    namespace = {
        "get_forward_context": lambda: NS(attn_metadata={"kda": metadata}),
        "KimiK3KDAMetadata": NS,
        "is_conv_state_dim_first": lambda: dim_first,
        "torch": NS(empty_like=lambda x: Tensor(x.shape)),
        "causal_conv1d_update": convolution,
        "rearrange": lambda x, pattern, d: Tensor(
            (1, x.shape[0], x.shape[1] // d, d), x.root
        ),
    }
    if real_verify_path is not None:
        layer.recoverssm_records = object()
        layer.recoverssm_backend = "triton"

        def verify_records(**kwargs):
            events.append(("records", kwargs))
            return kwargs["out"]

        namespace["kda_recoverssm_verify"] = verify_records
        layer._recoverssm_verify = MethodType(
            function(
                real_verify_path, "KimiK3DeltaAttention._recoverssm_verify", namespace
            ),
            layer,
        )
    function(path, name, namespace)(layer, mixed, g, gate, beta, output)
    expected_events = (
        ["conv", "records", "norm"]
        if real_verify_path is not None
        else ["verify", "norm"]
        if recover
        else ["conv", "recurrent", "norm"]
    )
    assert [e[0] for e in events] == expected_events
    assert output.copies == [output]
    if real_verify_path is not None:
        convolution_args = events[0][2]
        assert convolution_args["max_query_len"] == 8
        assert convolution_args["num_accepted_tokens"] is accepted
        record_args = events[1][1]
        assert record_args["records"] is layer.recoverssm_records
        assert record_args["checkpoint_state"] is state
        assert record_args["state_indices"].root is indices
        assert record_args["query_start_loc"].root is starts
        assert record_args["spec_query_len"] == 8
        assert record_args["out"].root is output
    elif recover:
        (
            actual_meta,
            actual_mixed,
            actual_g,
            actual_beta,
            weights,
            actual_starts,
            actual_out,
        ) = events[0][1]
        assert actual_meta is metadata
        assert (
            actual_mixed.root is mixed
            and actual_g.root is g
            and actual_beta.root is beta
        )
        assert weights.shape == (12, 4) and actual_starts.root is starts
        assert actual_out.root is output and actual_out.shape == output.shape
    else:
        args, kwargs = events[0][1:]
        assert args[0].root is mixed and args[1].root is conv
        assert args[1].shape == (1, 12, 3)
        assert (
            kwargs["num_accepted_tokens"] is accepted
            and kwargs["query_start_loc"] is starts
        )
        assert kwargs["max_query_len"] == 3 and kwargs["validate_data"] is False
        recurrent_args = events[1][1]
        assert recurrent_args["initial_state"] is state
        assert recurrent_args["ssm_state_indices"] is indices
        assert recurrent_args["num_accepted_tokens"] is accepted
        assert recurrent_args["out"].root is output


@pytest.mark.upstream_source
@pytest.mark.parametrize("recover", [False, True])
@pytest.mark.parametrize("dim_first", [False, True])
def test_actual_speculative_forward_ownership(
    upstream_source_root, modules, recover, dim_first
):
    check_forward(
        upstream_source_root / UPSTREAM,
        "KimiK3DeltaAttention._forward",
        modules,
        recover,
        dim_first,
    )


@pytest.mark.upstream_source
@pytest.mark.parametrize("dim_first", [False, True])
def test_actual_verify_helper_consumes_layer_records(
    upstream_source_root, modules, dim_first
):
    check_forward(
        upstream_source_root / UPSTREAM,
        "KimiK3DeltaAttention._forward",
        modules,
        True,
        dim_first,
        upstream_source_root / UPSTREAM,
    )
