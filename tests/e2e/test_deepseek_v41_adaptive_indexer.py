# SPDX-License-Identifier: Apache-2.0
"""Device-length flattening and paged-logits oracle for adaptive verification."""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.ppu


def _allocate_v41_indexer_cache(compress_ratio, dim):
    import torch
    from vllm.models.deepseek_v41.attention import DeepseekV4IndexerCache
    from vllm.v1.kv_cache_interface import (
        KVCacheLayout,
        KVCacheTensor,
        create_kv_cache_views,
    )
    from vllm.v1.worker.utils import select_common_block_size

    # No weights/config construction is needed to exercise the model's actual
    # cache-spec producer, backend selector and padded BLHNC allocation.
    layer = SimpleNamespace(
        head_dim=dim + 4,
        dtype=torch.uint8,
        cache_config=SimpleNamespace(block_size=128),
        compress_ratio=compress_ratio,
        sparse_logits=False,
        rocm_mxfp4=False,
    )
    config = SimpleNamespace(cache_config=SimpleNamespace(cache_dtype="fp8_ds_mla"))
    spec = DeepseekV4IndexerCache.get_kv_cache_spec(layer, config)
    backend = DeepseekV4IndexerCache.get_attn_backend(layer)
    kernel_block = select_common_block_size(spec.block_size, [backend])
    assert spec.block_size == kernel_block == 64 * compress_ratio
    # Leave space for another layer's page between consecutive indexer pages.
    stride = spec.page_size_bytes + 512
    raw = torch.zeros(8 * stride, dtype=torch.uint8, device="cuda")
    allocation = KVCacheTensor(
        size=raw.numel(),
        layers=["indexer"],
        layer_stride=spec.page_size_bytes,
        block_stride=stride,
    )
    views = create_kv_cache_views(
        raw, spec, 8, KVCacheLayout.BLHNC, allocation, kernel_block
    )
    DeepseekV4IndexerCache.bind_kv_cache(layer, views[0])
    assert layer.kv_cache.shape == (8, 64, dim + 4)
    assert layer.kv_cache.stride(0) == stride
    return layer.kv_cache.unsqueeze(-2)


def test_adaptive_indexer_backend_gate():
    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm.v1.attention.backends.mla.indexer import DeepseekV41IndexerBackend
    from vllm.v1.worker.gpu.attn_utils import (
        get_query_lens_mismatch_unsupported_backend,
    )

    group = SimpleNamespace(backend=DeepseekV41IndexerBackend, layer_names=["indexer"])
    assert get_query_lens_mismatch_unsupported_backend([[group]]) is None


@pytest.mark.parametrize("compress_ratio", [1, 2])
def test_device_lengths_flattened_logits_graph_replay(compress_ratio):
    import torch

    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadataBuilder

    from vllm_sail.utils.deep_gemm import (
        fp8_paged_mqa_logits,
        get_num_sms,
        get_paged_mqa_logits_metadata,
    )

    # Same total budget; CPU plans [3,3], device reallocates to [5,1] or [1,5].
    # Use the real V4.1 cache spec/backend/allocation path. Hard-coding a
    # 64-row tensor here hid the model's unsupported 128-row C1 pages.
    capacity, tokens, heads, dim = 8, 6, 64, 128
    cache = _allocate_v41_indexer_cache(compress_ratio, dim)
    block_size = cache.shape[1]
    assert block_size == 64
    builder = object.__new__(DeepseekV32IndexerMetadataBuilder)
    builder.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(enable_adaptive_verification=True)
    )
    builder.supports_varlen = False
    builder.decode_seq_lens_buffer = torch.zeros(
        capacity, dtype=torch.int32, device="cuda"
    )
    builder.decode_lens_buffer = torch.zeros(capacity, dtype=torch.int32, device="cuda")
    builder.expanded_block_table_buffer = torch.zeros(
        capacity, 4, dtype=torch.int32, device="cuda"
    )
    builder.arange_buffer = torch.arange(capacity, dtype=torch.int32, device="cuda")
    cpu_lens = torch.tensor([3, 3], dtype=torch.int32)
    lens = torch.tensor([5, 1], dtype=torch.int32, device="cuda")
    starts = torch.tensor([0, 5], dtype=torch.int32, device="cuda")
    context = torch.tensor([127, 185], dtype=torch.int32, device="cuda")
    seq_lens = context + lens
    table_cpu = torch.tensor([[4, 0, 6, 2], [5, 1, 7, 3]], dtype=torch.int32)
    table = table_cpu.cuda()
    generator = torch.Generator().manual_seed(83)
    q_cpu = (torch.randn(capacity, 1, heads, dim, generator=generator) / 8).to(
        torch.float8_e4m3fn
    )
    k_cpu = (torch.randn(8, block_size, dim, generator=generator) / 8).to(
        torch.float8_e4m3fn
    )
    scales_cpu = torch.rand(8, block_size, generator=generator) / 2 + 0.25
    weight_cpu = torch.rand(capacity, heads, generator=generator) / heads
    packed = torch.cat(
        (k_cpu.view(torch.uint8).flatten(1), scales_cpu.view(torch.uint8).flatten(1)),
        dim=1,
    )
    cache.copy_(packed.reshape(8, block_size, 1, dim + 4))
    q, weights = q_cpu.cuda(), weight_cpu.cuda()
    sms = get_num_sms()

    def step():
        ends, blocks, lengths, batch, padding = builder._prepare_decode_tensors(
            seq_lens,
            table,
            lens,
            cpu_lens,
            starts,
            num_decodes=2,
            num_decode_tokens=capacity,
            use_native=False,
            next_n=6,
            max_decode_len=3,
        )
        assert batch == capacity and not padding
        ends = (ends // compress_ratio).unsqueeze(-1)
        schedule = get_paged_mqa_logits_metadata(ends, block_size, sms, None)
        # Match the model: 2D context lengths require external top-k masking.
        # The logits kernel only promises values within each visible prefix.
        logits = fp8_paged_mqa_logits(
            q, cache, weights, ends, blocks, schedule, 256, False
        )
        return ends, blocks, lengths, logits

    # Compile first, then capture and replay with changed device-only boundaries.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = step()
    for device_lens in ([5, 1], [1, 5]):
        lens.copy_(torch.tensor(device_lens, dtype=torch.int32))
        starts.copy_(torch.tensor([0, device_lens[0]], dtype=torch.int32))
        seq_lens.copy_(context + lens)
        graph.replay()
        ends, blocks, lengths, logits = (tensor.cpu() for tensor in actual)
        expected_ends, request_ids = [], []
        for req, count in enumerate(device_lens):
            base = (127, 185)[req]
            expected_ends.extend((base + i + 1) // compress_ratio for i in range(count))
            request_ids.extend([req] * count)
        assert ends.flatten().tolist() == expected_ends + [0] * (capacity - tokens)
        assert lengths.tolist() == [1] * capacity
        torch.testing.assert_close(
            blocks[:tokens], table_cpu[request_ids], rtol=0, atol=0
        )
        for row, (req, end) in enumerate(zip(request_ids, expected_ends, strict=False)):
            positions = torch.arange(end)
            pages = table_cpu[req, positions // block_size].long()
            slots = positions % block_size
            keys = k_cpu.float()[pages, slots] * scales_cpu[pages, slots, None]
            reference = (
                (q_cpu[row, 0].float() @ keys.T).clamp_min(0) * weight_cpu[row, :, None]
            ).sum(0)
            torch.testing.assert_close(
                logits[row, :end], reference, rtol=0.01, atol=0.002
            )
