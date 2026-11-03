# SPDX-License-Identifier: Apache-2.0
"""Compressed draft caches preserve sampling for reordered request states."""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.ppu


@pytest.fixture(scope="module")
def sampler_runtime():
    import torch

    import vllm_sail
    from vllm_sail.native.extensions import import_kernels

    import_kernels(strict=True)
    vllm_sail.register_out_of_tree()
    from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample

    return SimpleNamespace(torch=torch, sample=rejection_sample, Cache=DraftLogitsCache)


def make_inputs(runtime, shards, mode):
    torch = runtime.torch
    generator = torch.Generator().manual_seed(7193)
    requests, steps, vocab = 4, 3, 129
    capacity = requests * shards
    rank = shards - 1
    states = rank + shards * torch.tensor([3, 0, 2, 1], dtype=torch.int32)
    temperature = torch.linspace(0.7, 1.3, capacity)
    full_draft = torch.randn(capacity, steps, vocab, generator=generator)
    target_rows, proposed = [], []
    for state in states.tolist():
        scaled = full_draft[state] / temperature[state]
        tokens = torch.multinomial(scaled.softmax(-1), 1, generator=generator).squeeze(
            1
        )
        target = torch.cat(
            [scaled.clone(), torch.randn(1, vocab, generator=generator)], dim=0
        )
        # Equal distributions accept the first two drafts. The last proposal
        # has negligible target mass, exercising recovery after that prefix.
        target[steps - 1, tokens[-1]] = -30
        target_rows.append(target)
        proposed.append(torch.cat([torch.zeros(1, dtype=torch.int64), tokens]))
    logits = requests * (steps + 1)
    full_cache = None
    compressed = None
    if mode != "no-draft":
        full_cache = runtime.Cache(
            capacity, steps, vocab, dtype=torch.float32, fill=0, device="cuda"
        )
        full_cache.values.copy_(full_draft)
        compressed = runtime.Cache(
            requests, steps, vocab, dtype=torch.float32, fill=0, device="cuda"
        )
        compressed.row_mapping = torch.full(
            (capacity,), -1, dtype=torch.int32, device="cuda"
        )
        mapped_rows = torch.tensor([2, 0, 3, 1], device="cuda")
        compressed.row_mapping[states.to("cuda")] = mapped_rows.to(torch.int32)
        compressed.values.index_copy_(0, mapped_rows, full_draft[states].to("cuda"))
    inputs = {
        "target_logits": torch.cat(target_rows).to("cuda"),
        "draft_logits_cache": full_cache,
        "draft_sampled": torch.cat(proposed).to("cuda"),
        "cu_num_logits": (
            torch.arange(requests + 1, dtype=torch.int32) * (steps + 1)
        ).to("cuda"),
        "pos": (torch.arange(logits, dtype=torch.int32) + 67).to("cuda"),
        "idx_mapping": states.to("cuda"),
        "expanded_idx_mapping": states.repeat_interleave(steps + 1).to("cuda"),
        "expanded_local_pos": torch.arange(steps + 1, dtype=torch.int32)
        .repeat(requests)
        .to("cuda"),
        "temperature": temperature.to("cuda"),
        "seed": (torch.arange(capacity, dtype=torch.int64) + 401).to("cuda"),
        "num_speculative_steps": steps,
    }
    if mode == "block":
        inputs["use_block_verification"] = True
    elif mode == "fly":
        inputs["fly_window_size"] = 2
    elif mode == "synthetic":
        inputs["synthetic_conditional_rates"] = torch.tensor(
            [1.0, 1.0, 0.0], device="cuda"
        )
    elif mode == "watermark":
        inputs.update(
            contexts=torch.arange(logits * 2, dtype=torch.int32, device="cuda").view(
                logits, 2
            ),
            watermarking=torch.ones(capacity, dtype=torch.bool, device="cuda"),
            watermarking_skip_mask=torch.zeros(
                logits * 2, dtype=torch.bool, device="cuda"
            )[::2],
            watermark_key=0x123456789ABCDEF,
        )
    return inputs, compressed


@pytest.mark.parametrize("shards", [2, 4])
@pytest.mark.parametrize(
    "mode", ["ordinary", "block", "watermark", "fly", "synthetic", "no-draft"]
)
def test_sharded_cache_matches_full_cache_and_graph_replay(
    sampler_runtime, shards, mode
):
    runtime = sampler_runtime
    torch = runtime.torch
    inputs, compressed = make_inputs(runtime, shards, mode)
    expected, expected_counts = runtime.sample(**inputs)
    sharded = dict(inputs, draft_logits_cache=compressed)
    actual, counts = runtime.sample(**sharded)
    torch.cuda.synchronize()

    def check(values, lengths):
        torch.testing.assert_close(lengths, expected_counts, rtol=0, atol=0)
        assert bool(((lengths >= 1) & (lengths <= 4)).all())
        valid = torch.arange(4, device="cuda")[None, :] < lengths[:, None]
        torch.testing.assert_close(values[valid], expected[valid], rtol=0, atol=0)
        assert bool(((values[valid] >= 0) & (values[valid] < 129)).all())

    check(actual, counts)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed, replayed_counts = runtime.sample(**sharded)
    if compressed is not None:
        # Change the public mapping and physical rows after capture while
        # preserving each request's distribution. Replay must read new rows.
        mapped_rows = torch.tensor([1, 3, 0, 2], device="cuda")
        states = inputs["idx_mapping"]
        compressed.row_mapping.fill_(-1)
        compressed.row_mapping[states] = mapped_rows.to(torch.int32)
        compressed.values.index_copy_(
            0, mapped_rows, inputs["draft_logits_cache"].values[states]
        )
    graph.replay()
    torch.cuda.synchronize()
    check(replayed, replayed_counts)
