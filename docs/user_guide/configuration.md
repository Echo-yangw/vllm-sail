# Configuration

Set environment variables before starting vLLM so the server and worker processes
inherit the same configuration. Runtime defaults are defined in
[`vllm_sail/envs.py`](../../vllm_sail/envs.py).

Every `VLLM_SAIL_*` runtime variable accepts the corresponding `VLLM_PPU_*`
name for compatibility. The SAIL name takes precedence whenever it is set,
including `0` or an empty string. An empty boolean is false; an empty backend
selection uses its default. Boolean truthy values are `true`, `1`, `yes` and
`on`, case-insensitively. `is_set()` recognizes either name, and `snapshot()`
reports effective values under SAIL names. `VLLM_DEEPEPLL_RECV_HOOK` keeps its
shared upstream name.

## Plugin discovery

vLLM discovers the installed plugin through package entry points. If you use
vLLM's plugin allowlist, include `ppu`:

```bash
export VLLM_PLUGINS=ppu
```

The same name enables platform discovery and the general hook that installs
registrations and model integrations. Include any other plugins needed by your
application in the allowlist.

## Backend selection

| Variable | Default | Values / behavior |
| --- | --- | --- |
| `VLLM_SAIL_MOE_BACKEND` | Automatic | `deepgemm`, `acext`, `triton` |
| `VLLM_SAIL_DENSE_BACKEND` | Automatic | `deepgemm`, `acext`, `triton` |
| `VLLM_SAIL_DENSE_BF16_DEEPGEMM` | `0` | Opt into the PPU 1.5 BF16 dense DeepGEMM path |
| `VLLM_SAIL_DEEPGEMM_MOE_TP_FUSED` | `0` | Enable the supported fused tensor-parallel MoE path |
| `VLLM_SAIL_DEEPGEMM_MOE_FUSED_ACT_QUANT` | `0` | Fuse the activation and the second MoE GEMM's input quantization into DeepGEMM's SiLU/SiTU kernel for the FP8 and MXFP4 experts; the BF16 (w4a16 or unquantized) experts use its values-only kernel, which also replaces the standalone `vllm::silu_and_mul` / `vllm::situ_and_mul` that unquantized BF16 dense MLPs and shared experts run on |
| `VLLM_SAIL_USE_PLA` | `1` | Allow PLA linear-attention paths where available |
| `VLLM_GDN_DECODE_KERNEL` | `auto` on PPU | `auto`, `cuda`, `pla`, `triton`; selects GDN decode independently of prefill and KDA |
| `VLLM_SAIL_QSA_KERNEL` | `auto` on PPU | `auto`, `cuda`, `triton`; selects the Qwen4Exp QSA sparse-attention kernel |
| `VLLM_SAIL_FUSED_RMSNORM_QUANT` | `0` | Enable supported fused RMSNorm and quantization paths |
| `VLLM_SAIL_USE_OPT_TOKEN_GROUP_QUANT` | `0` | Enable the optimized token-group quantization path |
| `VLLM_SAIL_USE_TRITON_INT8_QUANT` | `1` | Use Triton dynamic per-token INT8 quantization |
| `VLLM_DEEPEPLL_RECV_HOOK` | `1` | Use the deferred receive hook; `0` selects the event-based fallback |

For example, to request the MoE DeepGEMM backend:

```bash
VLLM_SAIL_MOE_BACKEND=deepgemm vllm serve /path/to/model
```

Leave backend variables unset for normal capability-based selection. An explicit
choice still requires the corresponding library, device support and compatible
tensor layout. Avoid setting overlapping backend controls while diagnosing a
selection issue.

On PPU, GDN decode `auto` prefers PLA when the legacy `VLLM_SAIL_USE_PLA`
switch permits it and the model uses supported head dimensions and FP32 SSM
state. Otherwise it tries the registered native fused kernel, then Triton.
An explicit `pla`, `cuda` or `triton` choice overrides that legacy switch for
GDN decode only. Prefill and KDA retain their existing selection.

PLA decode requires `--mamba-ssm-cache-dtype float32` and K=V=128; the resolver
does not change an existing cache dtype. Missing dependencies or unsupported
static configurations fail early for explicit choices. Native `cuda` requires
a real CUDA dispatch implementation, not just an operation schema or an
unsupported-op stub. Its fused MTP kernel applies only to eligible batches;
other batches use the existing Triton path. The new HGGC port still requires
native build and device correctness verification before deployment.

On PPU, Qwen4Exp QSA (query-sparse attention) `auto` selects FA3's QSA kernel
when every gate passes and otherwise keeps the upstream Triton kernel, logging
the reason once per distinct decision. The gates are: a BF16 main KV cache
(FA3's QSA path rejects FP8 outright), head dimension 256, at most 32 query
heads per KV head, and a `flash_attn_3` build whose QSA entry point carries the
`qsa_allow_aiu` argument and passes a numerical self-test. `cuda` and `triton`
pin either side for A/B measurement; an explicit `cuda` that fails a gate raises
at layer construction rather than falling back silently. Selection is per layer,
so a model whose layers disagree on KV-cache dtype resolves each independently.

Note that FA3's hand-tuned QSA kernels require one KV head per rank under the
interleaved KV-cache layout this pin allocates. Zhenwu M890 reaches a tuned
kernel only at tensor-parallel size 2; other sizes, including tensor-parallel
size 1, run FA3's QSA-aware generic kernel. Correctness is unaffected. Use
`triton` to compare against the Triton kernel on your own shape.

## Compatibility controls

For FP8/BF16 conversion and recipe selection, see [Online quantization](online_quantization.md).
`VLLM_SAIL_FP8_LOAD_POLICY=native|bf16|int8` applies only when no explicit
quantization argument was supplied. Leaving it unset preserves native behavior;
the guide describes legacy opt-in variable precedence and source-format limits.

| Variable | Default | Behavior |
| --- | --- | --- |
| `VLLM_SAIL_DISABLE_MOE_WNA16_CUDA` | `0` | Disable the native WNA16 MoE path; takes precedence over `VLLM_SAIL_FORCE_MOE_WNA16_CUDA` |
| `VLLM_SAIL_FORCE_MOE_WNA16_CUDA` | `0` | Request the native WNA16 path; raises if the kernel is unavailable |
| `VLLM_SAIL_ENABLE_MOE_MARLIN` | `0` | Legacy selection gate; does not enable excluded native Marlin kernels |
| `VLLM_SAIL_FUSED_GDN_DECODE` | `1` | Legacy getter; dispatch now uses `VLLM_SAIL_USE_PLA` |

## Profiling

| Variable | Default | Behavior |
| --- | --- | --- |
| `VLLM_SAIL_NVTX_PROFILE` | `0` | Enable optional NVTX instrumentation |
| `VLLM_SAIL_NVTX_DUMP_TOPK` | `0` | Include top-k details in NVTX labels |
| `VLLM_SAIL_NVTX_VFA_DUMP_SEQLEN` | `0` | Include sequence lengths in FlashAttention NVTX labels |

`SAIL_NVTX_PROFILE` is also accepted, after `VLLM_PPU_NVTX_PROFILE` in precedence.
The profiling integration requires the optional `nvtx` package; install it with
`python -m pip install --no-build-isolation -e '.[profiling]'` from the repository
root. The SAIL SDK's optional `model_prof` package adds iteration hooks.

## Version metadata

Commit-pinned installations must report the full source SHA recorded in the
packaged kernel manifest. Leave `VLLM_VERSION` and `VLLM_VERSION_OVERRIDE` unset
so the plugin can verify that identity.

For release-tag metadata, `VLLM_VERSION` can state the actual release base when a
source build reports an ambiguous version. Use `VLLM_VERSION=0.32.0` only for a
checkout verified against that release. Commit-pinned installations still
require their full source SHA.

## Inspect the active environment

```bash
python -c "from vllm_sail import collect_env; collect_env()"
```

The report includes versions, device information, SDK package availability and
the PPU environment-variable snapshot. Package presence is a discovery check;
successful loading and execution require the
[verification steps](verification_guide.md).

`PPU_SDK` is also read at runtime and defaults to `/usr/local/PPU_SDK`. When
importing ACEXT fails on an unresolved cuBLAS symbol, the plugin preloads
`CUDA_SDK/lib64/libcublas.so` from that root and retries.

Build-time settings such as `PPU_SDK`, `HGCC`, `CMAKE_HGCC_DIR` and
`PYTORCH_SAIL_ARCH` are documented in [HGGC kernels](../developer_guide/kernels.md#build-settings).
