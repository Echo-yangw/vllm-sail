#include "ops.h"
#include "cuda_utils.h"
#include "core/registration.h"

#include <torch/csrc/stable/library.h>

// Register ops with STABLE_TORCH_LIBRARY for libtorch stable ABI compatibility.
// Note: We register under namespace "_C" so ops are accessible as
// torch.ops._C.<op_name> for compatibility with existing code.
STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
  // Compute per-token-group FP8 quantized tensor and scaling factor.
  // The dummy arguments are here so we can correctly fuse with RMSNorm.
  ops.def(
      "per_token_group_fp8_quant(Tensor input, Tensor! output_q, Tensor! "
      "output_s, "
      "int group_size, float eps, float fp8_min, float fp8_max, bool "
      "scale_ue8m0, bool dummy_is_scale_transposed, bool dummy_is_tma_aligned "
      ") -> ()");
  // Compute per-token-group 8-bit quantized tensor and UE8M0-packed,
  // TMA-aligned scales for DeepGEMM.
  ops.def(
      "per_token_group_fp8_quant_packed(Tensor input, Tensor! output_q, "
      "Tensor! output_s_packed, int group_size, float eps, float fp8_min, "
      "float fp8_max) -> ()");
  // Compute per-token-group INT8 quantized tensor and scaling factor.
  ops.def(
      "per_token_group_quant_int8(Tensor input, Tensor! output_q, Tensor! "
      "output_s, int group_size, float eps, float int8_min, float int8_max) -> "
      "()");
  ops.def("get_cuda_view_from_cpu_tensor(Tensor cpu_tensor) -> Tensor");

#ifndef USE_ROCM
  // conditionally compiled so impl registrations are in source file
#endif

#ifndef USE_ROCM
#endif

  // Apply Root Mean Square (RMS) Normalization to the input tensor.
  ops.def(
      "rms_norm(Tensor! result, Tensor input, Tensor? weight, float epsilon) "
      "-> "
      "()");

  // In-place fused Add and RMS Normalization.
  ops.def(
      "fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor? weight, "
      "float epsilon) -> ()");

  // Layernorm-quant
  // Apply Root Mean Square (RMS) Normalization to the input tensor.
  ops.def(
      "rms_norm_static_fp8_quant(Tensor! result, Tensor input, Tensor weight, "
      "Tensor scale, float epsilon) -> "
      "()");

  // In-place fused Add and RMS Normalization.
  ops.def(
      "fused_add_rms_norm_static_fp8_quant(Tensor! result, Tensor input, "
      "Tensor! residual, Tensor weight, "
      "Tensor scale, float epsilon) -> ()");

  // Fused Layernorm + Quant kernels
  ops.def(
      "rms_norm_dynamic_per_token_quant(Tensor! result, Tensor input, "
      "Tensor weight, Tensor! scale, float epsilon, "
      "Tensor? scale_ub, Tensor!? residual) -> ()");

  // Fused Layernorm + Block quant kernels
  ops.def(
      "rms_norm_per_block_quant(Tensor! result, Tensor input, "
      "Tensor weight, Tensor! scale, float epsilon, "
      "Tensor? scale_ub, Tensor!? residual, int group_size, "
      "bool is_scale_transposed) -> ()");

  // Fused SiLU+Mul + per-block quantization
  ops.def(
      "silu_and_mul_per_block_quant("
      "Tensor! out, "
      "Tensor input, "
      "Tensor! scales, "
      "int group_size, "
      "Tensor? scale_ub=None, "
      "bool is_scale_transposed=False) -> ()");

  // Rotary embedding
  // Apply GPT-NeoX or GPT-J style rotary embedding to query and key.
  ops.def(
      "rotary_embedding(Tensor positions, Tensor! query,"
      "                 Tensor!? key, int head_size,"
      "                 Tensor cos_sin_cache, bool is_neox, int "
      "rope_dim_offset=0, bool inverse=False) -> ()");

  // Function for fused QK Norm and RoPE
  ops.def(
      "fused_qk_norm_rope(Tensor! qkv, int num_heads_q, "
      "int num_heads_k, int num_heads_v, int head_dim, float eps, "
      "Tensor q_weight, Tensor k_weight, Tensor cos_sin_cache, "
      "bool is_neox, Tensor position_ids, "
      "int forced_token_heads_per_warp=-1) -> ()");
  ops.def(
      "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert("
      "Tensor q_in, Tensor kv, Tensor! k_cache, "
      "Tensor slot_mapping, Tensor position_ids, Tensor cos_sin_cache, "
      "int q_head_padded, float eps, int cache_block_size, "
      "bool apply_q_norm=True, bool kv_mxfp8=False, bool apply_q_rope=True, "
      "bool is_q_interleaved=False) -> Tensor");

  // FlashInfer V4 full-cache variants: write Q in place (bf16) or to a separate
  // FP8 tensor, and KV into a contiguous 512-wide token-strided cache.
  ops.def(
      "fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert("
      "Tensor! q, Tensor kv, Tensor! k_cache, Tensor slot_mapping, "
      "Tensor position_ids, Tensor cos_sin_cache, float eps, "
      "int cache_block_size, bool apply_q_norm=True) -> ()");
  ops.def(
      "fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert("
      "Tensor q, Tensor kv, Tensor! q_fp8, Tensor! k_cache, "
      "Tensor slot_mapping, Tensor position_ids, Tensor cos_sin_cache, "
      "Tensor fp8_scale, Tensor q_fp8_scale_inv, float eps, "
      "int cache_block_size, bool apply_q_norm=True) -> ()");

  // Kimi-K3 MLA epilogues: optional RoPE followed by concat/cache insertion.
  ops.def(
      "fused_kimi_k3_mla_key_concat_kv_cache_insert("
      "Tensor! q, Tensor k_nope, Tensor k_pe, Tensor kv_c_normed, "
      "Tensor! k_out, Tensor! k_cache, Tensor slot_mapping, "
      "int cache_block_size, Tensor? position_ids=None, "
      "Tensor? cos_sin_cache=None) -> ()");
  ops.def(
      "fused_kimi_k3_mla_key_concat_ds_mla_insert("
      "Tensor! q, Tensor k_nope, Tensor k_pe, Tensor kv_c_normed, "
      "Tensor! k_out, Tensor! k_cache, Tensor slot_mapping, "
      "int cache_block_size, Tensor? position_ids=None, "
      "Tensor? cos_sin_cache=None) -> ()");
  ops.def(
      "fused_kimi_k3_mla_kv_concat(Tensor k_nope, Tensor k_pe, Tensor! k_out) "
      "-> ()");
  ops.def(
      "fused_kimi_k3_mla_kv_concat_quant_fp8("
      "Tensor k_nope, Tensor k_pe, Tensor v, Tensor! k_fp8, Tensor! v_fp8) "
      "-> ()");
  ops.def(
      "fused_kimi_k3_mla_qkv_quant_kv_cache_fp8_insert("
      "Tensor q, Tensor k_nope, Tensor k_pe, Tensor kv_c_normed, Tensor v, "
      "Tensor! q_fp8, Tensor! k_fp8, Tensor! v_fp8, Tensor! k_cache, "
      "Tensor slot_mapping, Tensor q_scale_inv, Tensor k_scale_inv, "
      "Tensor v_scale_inv, Tensor cache_scale_inv, int cache_block_size, "
      "Tensor? position_ids=None, Tensor? cos_sin_cache=None) -> ()");

  // Kimi-K3 MLA decode epilogue: concat mqa_q = [ql_nope | q_pe] and insert the
  // latent [kv_c_normed | k_pe] into the paged cache (bf16 / fp8 / fp8_ds_mla).
  ops.def(
      "fused_kimi_k3_mla_decode_q_concat_kv_cache_insert("
      "Tensor ql_nope, Tensor q_pe, Tensor kv_c_normed, Tensor k_pe, "
      "Tensor! mqa_q, Tensor! k_cache, Tensor slot_mapping, "
      "int cache_block_size, Tensor? position_ids=None, "
      "Tensor? cos_sin_cache=None) -> ()");
  ops.def(
      "fused_kimi_k3_mla_decode_q_concat_kv_cache_fp8_insert("
      "Tensor ql_nope, Tensor q_pe, Tensor kv_c_normed, Tensor k_pe, "
      "Tensor! mqa_q, Tensor! k_cache, Tensor slot_mapping, "
      "Tensor q_scale_inv, Tensor cache_scale_inv, int cache_block_size, "
      "Tensor? position_ids=None, Tensor? cos_sin_cache=None) -> ()");
  ops.def(
      "fused_kimi_k3_mla_decode_q_concat_ds_mla_insert("
      "Tensor ql_nope, Tensor q_pe, Tensor kv_c_normed, Tensor k_pe, "
      "Tensor! mqa_q, Tensor! k_cache, Tensor slot_mapping, "
      "int cache_block_size, Tensor? position_ids=None, "
      "Tensor? cos_sin_cache=None) -> ()");

#ifndef USE_ROCM
#endif

#ifndef USE_ROCM
#endif

  // Horizontally-fused MiniMax-M3 QK-norm + partial NeoX RoPE + KV-insert.
  ops.def(
      "fused_minimax_m3_qknorm_rope_kv_insert("
      "Tensor! qkv, Tensor q_norm_weight, Tensor k_norm_weight, "
      "Tensor cos_sin_cache, Tensor positions, int num_heads, "
      "int num_kv_heads, int rotary_dim, float eps, "
      "Tensor? index_q_norm_weight, Tensor? index_k_norm_weight, "
      "int num_index_heads, "
      "Tensor? slot_mapping, Tensor? index_slot_mapping, "
      "Tensor!? kv_cache, Tensor!? index_cache, "
      "int block_size, Tensor!? q_out, Tensor!? index_q_out, "
      "str kv_cache_dtype, bool skip_index_branch=False, "
      "Tensor!? q_fp8_out=None, float q_fp8_scale=1.0, "
      "Tensor? kv_k_scale=None, Tensor? kv_v_scale=None, "
      "bool write_qkv=True) -> ()");

#ifdef VLLM_ENABLE_FUSED_KDA_DECODE
#endif

#ifdef VLLM_ENABLE_FUSED_GDN_DECODE
  ops.def(
      "fused_gdn_decode_post_conv_mtp("
      "Tensor mixed_qkv, Tensor a, Tensor b, Tensor A_log, Tensor dt_bias, "
      "Tensor state_indices, Tensor cu_seqlens, Tensor num_accepted_tokens, "
      "Tensor! state, Tensor output_gate, Tensor norm_weight, Tensor! out, "
      "float scale, float norm_eps=1e-5, "
      "str output_gate_activation='silu') -> ()");
#endif

#ifdef VLLM_ENABLE_FUSED_KDA_CHUNK
#endif

#ifdef VLLM_ENABLE_BEAM_KERNELS
#endif

#ifdef VLLM_ENABLE_KIMI_K3_ATTN_RES
#endif

  // Apply repetition penalties to logits in-place.
  ops.def(
      "apply_repetition_penalties_(Tensor! logits, Tensor prompt_mask, "
      "Tensor output_mask, Tensor repetition_penalties) -> ()");

  // Optimized top-k per row operations.
  ops.def(
      "top_k_per_row_prefill(Tensor logits, Tensor rowStarts, Tensor rowEnds, "
      "Tensor! indices, int numRows, int stride0, "
      "int stride1, int topK) -> ()");

  ops.def(
      "top_k_per_row_decode(Tensor logits, int next_n, "
      "Tensor seq_lens, Tensor! indices, "
      "int numRows, int stride0, int stride1, int topK) -> ()");

  ops.def(
      "persistent_topk(Tensor logits, Tensor lengths, Tensor! output, "
      "Tensor workspace, int k, int max_seq_len) -> ()");

#ifdef VLLM_ENABLE_COOPERATIVE_TOPK
#endif

  // Activation ops
  ops.def(
      "persistent_masked_m_silu_mul_quant(Tensor input, Tensor counts, Tensor! "
      "y_q, Tensor! y_s, bool use_ue8m0) -> ()");
  ops.def("weak_ref_tensor(Tensor input) -> Tensor");

  // Activation function used in SwiGLU.
  ops.def("silu_and_mul(Tensor! result, Tensor input) -> ()");

  ops.def("mul_and_silu(Tensor! out, Tensor input) -> ()");

  // SwiGLU activation with input clamping.
  // alpha scales the sigmoid (gate * sigmoid(alpha * gate)); beta is added to
  // the up half (up + beta). Defaults alpha=1.0, beta=0.0 give silu(gate)*up.
  ops.def(
      "silu_and_mul_with_clamp(Tensor! result, Tensor input, float limit, "
      "float alpha=1.0, float beta=0.0) -> ()");

  // SwiGLU activation with FP8 quantization.
  ops.def(
      "silu_and_mul_quant(Tensor! result, Tensor input, Tensor scale) -> ()");

  // Activation function used in GeGLU with `none` approximation.
  ops.def("gelu_and_mul(Tensor! out, Tensor input) -> ()");

  // Activation function used in GeGLU with `tanh` approximation.
  ops.def("gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()");

  // FATReLU implementation.
  ops.def("fatrelu_and_mul(Tensor! out, Tensor input, float threshold) -> ()");

  ops.def(
      "swigluoai_and_mul(Tensor! out, Tensor input, float alpha=1.702, float "
      "limit=7.0) "
      "-> ()");

  // SituGLU implementation used in Kimi models.
  ops.def(
      "situ_and_mul(Tensor! out, Tensor input, float beta=1.0, float "
      "linear_beta=-1.0) -> ()");
  // Fused SituGLU activation + dynamic FP8 quantization for the Humming w2 path
  // (writes the fp8 down input and its float32 scale). group_size=0 ->
  // per-token scale [.., 1]; group_size=128 -> k-major block-FP8 scale [..,
  // d/128].
  ops.def(
      "situ_and_mul_quant(Tensor! out, Tensor! scale, Tensor input, "
      "float beta=1.0, float linear_beta=-1.0, int group_size=0, "
      "Tensor? num_valid_tokens=None, int topk=1) -> ()");
  ops.def(
      "masked_moe_activation(Tensor! out, Tensor input, Tensor "
      "valid_token_counts, str activation, float clamp_limit=0.0, float "
      "alpha=1.0, float beta=0.0, float situ_beta=1.0, float "
      "situ_linear_beta=-1.0) -> ()");

  // GELU implementation used in GPT-2.
  ops.def("gelu_new(Tensor! out, Tensor input) -> ()");

  // Approximate GELU implementation.
  ops.def("gelu_fast(Tensor! out, Tensor input) -> ()");

  // Quick GELU implementation.
  ops.def("gelu_quick(Tensor! out, Tensor input) -> ()");

  // relu(x)^2 activation from https://arxiv.org/abs/2109.08668v2
  ops.def("relu_squared(Tensor! out, Tensor input) -> ()");

  // Compute int8 quantized tensor for given scaling factor.
  ops.def(
      "static_scaled_int8_quant(Tensor! result, Tensor input, Tensor scale,"
      "Tensor? azp) -> ()");

  // Compute int8 quantized tensor and scaling factor
  ops.def(
      "dynamic_scaled_int8_quant(Tensor! result, Tensor input, Tensor! scale, "
      "Tensor!? azp) -> ()");

  // Compute FP8 quantized tensor for given scaling factor.
  // Supports per-tensor, per-channel, per-token, and arbitrary 2D group
  // scaling. Optional group_m/group_n specify the group shape explicitly;
  // required for 1D scales to disambiguate per-channel vs per-token.
  ops.def(
      "static_scaled_fp8_quant(Tensor! result, Tensor input, Tensor scale, "
      "int[]? group_shape=None) -> ()");

  // Compute dynamic-per-tensor FP8 quantized tensor and scaling factor.
  ops.def(
      "dynamic_scaled_fp8_quant(Tensor! result, Tensor input, Tensor! scale) "
      "-> "
      "()");

  // Compute dynamic-per-token FP8 quantized tensor and scaling factor.
  ops.def(
      "dynamic_per_token_scaled_fp8_quant(Tensor! result, Tensor input, "
      "Tensor! scale, Tensor? scale_ub) -> "
      "()");

  // Mamba selective scan kernel
  ops.def(
      "selective_scan_fwd(Tensor! u, Tensor! delta,"
      "Tensor! A, Tensor! B, Tensor! C,"
      "Tensor? D_, Tensor!? z_, Tensor? delta_bias_,"
      "bool delta_softplus,"
      "Tensor? query_start_loc,"
      "Tensor? cache_indices,"
      "Tensor? has_initial_state,"
      "Tensor! ssm_states,"
      "int null_block_id,"
      "int block_size,"
      "Tensor? block_idx_first_scheduled_token,"
      "Tensor? block_idx_last_scheduled_token,"
      "Tensor? initial_state_idx,"
      "Tensor? cu_chunk_seqlen,"
      "Tensor? last_chunk_indices) -> ()");

  // LongCat n-gram embedding index kernel. All tensor args are marked mutable
  // to match the (non-const) stable-Tensor& C++ signature; only ne_token_table
  // and n_gram_ids are actually written in place.
  // Fused vocab-parallel embedding: gather the rows this rank owns and write
  // zeros for the rest, so the following all-reduce reconstructs the full
  // embedding.
  ops.def(
      "vocab_parallel_embedding(Tensor! out, Tensor input_ids, Tensor weight, "
      "int org_vocab_start_index, int org_vocab_end_index, "
      "int num_org_vocab_padding, int added_vocab_start_index, "
      "int added_vocab_end_index) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl("vocab_parallel_embedding", TORCH_BOX(&vocab_parallel_embedding));

  // Per-token group quantization
  ops.impl("per_token_group_fp8_quant", TORCH_BOX(&per_token_group_quant_fp8));
  ops.impl("per_token_group_fp8_quant_packed",
           TORCH_BOX(&per_token_group_quant_8bit_packed));
  ops.impl("per_token_group_quant_int8",
           TORCH_BOX(&per_token_group_quant_int8));

#ifndef USE_ROCM

  // DSV3 fused A GEMM: conditionally compiled so impl registration is in
  // source file (dsv3_fused_a_gemm.cu)
#endif

  // Layernorm kernels (shared CUDA/ROCm)
  ops.impl("rms_norm", TORCH_BOX(&rms_norm));
  ops.impl("fused_add_rms_norm", TORCH_BOX(&fused_add_rms_norm));

  // Layernorm-quant kernels (shared CUDA/ROCm)
  ops.impl("rms_norm_static_fp8_quant", TORCH_BOX(&rms_norm_static_fp8_quant));
  ops.impl("fused_add_rms_norm_static_fp8_quant",
           TORCH_BOX(&fused_add_rms_norm_static_fp8_quant));

  // Fused layernorm + dynamic per-token quant kernels (shared CUDA/ROCm)
  ops.impl("rms_norm_dynamic_per_token_quant",
           TORCH_BOX(&rms_norm_dynamic_per_token_quant));
  ops.impl("rms_norm_per_block_quant", TORCH_BOX(&rms_norm_per_block_quant));
  ops.impl("silu_and_mul_per_block_quant",
           TORCH_BOX(&silu_and_mul_per_block_quant));

  // Positional encoding kernels (shared CUDA/ROCm)
  ops.impl("rotary_embedding", TORCH_BOX(&rotary_embedding));
  ops.impl("fused_qk_norm_rope", TORCH_BOX(&fused_qk_norm_rope));
  ops.impl("fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
           TORCH_BOX(&fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert));
  ops.impl(
      "fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert",
      TORCH_BOX(&fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert));
  ops.impl(
      "fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert",
      TORCH_BOX(&fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert));
  ops.impl("fused_kimi_k3_mla_key_concat_kv_cache_insert",
           TORCH_BOX(&fused_kimi_k3_mla_key_concat_kv_cache_insert));
  ops.impl("fused_kimi_k3_mla_key_concat_ds_mla_insert",
           TORCH_BOX(&fused_kimi_k3_mla_key_concat_ds_mla_insert));
  ops.impl("fused_kimi_k3_mla_kv_concat",
           TORCH_BOX(&fused_kimi_k3_mla_kv_concat));
  ops.impl("fused_kimi_k3_mla_kv_concat_quant_fp8",
           TORCH_BOX(&fused_kimi_k3_mla_kv_concat_quant_fp8));
  ops.impl("fused_kimi_k3_mla_qkv_quant_kv_cache_fp8_insert",
           TORCH_BOX(&fused_kimi_k3_mla_qkv_quant_kv_cache_fp8_insert));
  ops.impl("fused_kimi_k3_mla_decode_q_concat_kv_cache_insert",
           TORCH_BOX(&fused_kimi_k3_mla_decode_q_concat_kv_cache_insert));
  ops.impl("fused_kimi_k3_mla_decode_q_concat_kv_cache_fp8_insert",
           TORCH_BOX(&fused_kimi_k3_mla_decode_q_concat_kv_cache_fp8_insert));
  ops.impl("fused_kimi_k3_mla_decode_q_concat_ds_mla_insert",
           TORCH_BOX(&fused_kimi_k3_mla_decode_q_concat_ds_mla_insert));
#ifndef USE_ROCM
#endif
  ops.impl("fused_minimax_m3_qknorm_rope_kv_insert",
           TORCH_BOX(&fused_minimax_m3_qknorm_rope_kv_insert));
#ifdef VLLM_ENABLE_FUSED_KDA_DECODE
#endif

#ifdef VLLM_ENABLE_FUSED_GDN_DECODE
  ops.impl("fused_gdn_decode_post_conv_mtp",
           TORCH_BOX(&fused_gdn_decode_post_conv_mtp));
#endif

#ifdef VLLM_ENABLE_FUSED_KDA_CHUNK
#endif

#ifdef VLLM_ENABLE_BEAM_KERNELS
#endif

#ifdef VLLM_ENABLE_KIMI_K3_ATTN_RES
#endif

  // Sampler kernels (shared CUDA/ROCm)
  ops.impl("apply_repetition_penalties_",
           TORCH_BOX(&apply_repetition_penalties_));
  ops.impl("top_k_per_row_prefill", TORCH_BOX(&top_k_per_row_prefill));
  ops.impl("top_k_per_row_decode", TORCH_BOX(&top_k_per_row_decode));
  ops.impl("persistent_topk", TORCH_BOX(&persistent_topk));
#ifdef VLLM_ENABLE_COOPERATIVE_TOPK
#endif

  // Activation kernels (shared CUDA/ROCm)
  ops.impl("persistent_masked_m_silu_mul_quant",
           TORCH_BOX(&persistent_masked_m_silu_mul_quant));
  ops.impl("weak_ref_tensor", TORCH_BOX(&weak_ref_tensor));
  ops.impl("silu_and_mul_quant", TORCH_BOX(&silu_and_mul_quant));
  ops.impl("silu_and_mul", TORCH_BOX(&silu_and_mul));
  ops.impl("mul_and_silu", TORCH_BOX(&mul_and_silu));
  ops.impl("gelu_and_mul", TORCH_BOX(&gelu_and_mul));
  ops.impl("gelu_tanh_and_mul", TORCH_BOX(&gelu_tanh_and_mul));
  ops.impl("fatrelu_and_mul", TORCH_BOX(&fatrelu_and_mul));
  ops.impl("swigluoai_and_mul", TORCH_BOX(&swigluoai_and_mul));
  ops.impl("situ_and_mul", TORCH_BOX(&situ_and_mul));
  ops.impl("situ_and_mul_quant", TORCH_BOX(&situ_and_mul_quant));
  ops.impl("masked_moe_activation", TORCH_BOX(&masked_moe_activation));
  ops.impl("gelu_new", TORCH_BOX(&gelu_new));
  ops.impl("gelu_fast", TORCH_BOX(&gelu_fast));
  ops.impl("gelu_quick", TORCH_BOX(&gelu_quick));
  ops.impl("relu_squared", TORCH_BOX(&relu_squared));
  ops.impl("silu_and_mul_with_clamp", TORCH_BOX(&silu_and_mul_clamp));

  // INT8 quantization kernels
  ops.impl("static_scaled_int8_quant", TORCH_BOX(&static_scaled_int8_quant));
  ops.impl("dynamic_scaled_int8_quant", TORCH_BOX(&dynamic_scaled_int8_quant));

  // FP8 quantization kernels
  ops.impl("static_scaled_fp8_quant", TORCH_BOX(&static_scaled_fp8_quant));
  ops.impl("dynamic_scaled_fp8_quant", TORCH_BOX(&dynamic_scaled_fp8_quant));
  ops.impl("dynamic_per_token_scaled_fp8_quant",
           TORCH_BOX(&dynamic_per_token_scaled_fp8_quant));

  // Mamba kernels
  ops.impl("selective_scan_fwd", TORCH_BOX(&selective_scan_fwd));
}

STABLE_TORCH_LIBRARY_IMPL(_C, CPU, ops) {
  ops.impl("get_cuda_view_from_cpu_tensor",
           TORCH_BOX(&get_cuda_view_from_cpu_tensor));
}

STABLE_TORCH_LIBRARY_FRAGMENT(_C_cuda_utils, cuda_utils) {
  cuda_utils.def("get_device_attribute(int attribute, int device_id) -> int");
  cuda_utils.def(
      "get_max_shared_memory_per_block_device_attribute(int device_id) -> int");
}

STABLE_TORCH_LIBRARY_IMPL(_C_cuda_utils, CompositeExplicitAutograd,
                          cuda_utils) {
  cuda_utils.impl("get_device_attribute", TORCH_BOX(&get_device_attribute));
  cuda_utils.impl("get_max_shared_memory_per_block_device_attribute",
                  TORCH_BOX(&get_max_shared_memory_per_block_device_attribute));
}

// These capability-check functions take only primitive args (no tensors), so
// there is no device to dispatch on. CompositeExplicitAutograd makes them
// available for all backends. This is the stable ABI equivalent of calling
// ops.impl("op_name", &func) without a dispatch key in the non-stable API.
STABLE_TORCH_LIBRARY_IMPL(_C, CompositeExplicitAutograd, ops) {
#ifndef USE_ROCM
#endif
}

// Cache ops
STABLE_TORCH_LIBRARY_FRAGMENT(_C_cache_ops, ops) {
  // Swap in (out) the cache blocks from src to dst.
  ops.def(
      "swap_blocks(Tensor src, Tensor! dst,"
      "            int block_size_in_bytes, Tensor block_mapping) -> ()");

  // Batch swap: submit all block copies in a single driver call.
  ops.def(
      "swap_blocks_batch(Tensor src_ptrs, Tensor dst_ptrs,"
      "                  Tensor sizes,"
      "                  bool is_src_access_order_any=False) -> ()");

  // Reshape the key and value tensors and cache them.
  ops.def(
      "reshape_and_cache(Tensor key, Tensor value,"
      "                  Tensor! key_cache, Tensor! value_cache,"
      "                  Tensor slot_mapping,"
      "                  str kv_cache_dtype,"
      "                  Tensor k_scale, Tensor v_scale,"
      "                  bool interleaved_v=False) -> ()");

  // Reshape the key and value tensors and cache them.
  ops.def(
      "reshape_and_cache_flash(Tensor key, Tensor value,"
      "                        Tensor! key_cache,"
      "                        Tensor! value_cache,"
      "                        Tensor slot_mapping,"
      "                        str kv_cache_dtype,"
      "                        Tensor k_scale, Tensor v_scale) -> ()");

  // Concat kv_c and k_pe and cache them.
  ops.def(
      "concat_and_cache_mla(Tensor kv_c, Tensor k_pe,"
      "                     Tensor! kv_cache,"
      "                     Tensor slot_mapping,"
      "                     str kv_cache_dtype,"
      "                     Tensor scale,"
      "                     bool skip_scale=False) -> ()");

  // Grouped concat_and_cache_mla across all layers. Each layer's cache base
  // pointer and optional plain-FP8 scale are read from device tensors.
  ops.def(
      "concat_and_cache_mla_grouped(Tensor kv_c, Tensor k_pe,"
      "                             Tensor kv_cache_ptrs,"
      "                             Tensor slot_mapping,"
      "                             int block_size, int block_stride,"
      "                             int entry_stride,"
      "                             Tensor? kv_scales=None,"
      "                             str kv_cache_dtype='auto') -> ()");

#ifndef USE_ROCM

#endif  // !USE_ROCM

  // Rotate Q and K, then write to kv cache for MLA
  ops.def(
      "concat_and_cache_mla_rope_fused("
      "                     Tensor positions,"
      "                     Tensor! q_pe,"
      "                     Tensor! k_pe,"
      "                     Tensor kv_c,"
      "                     Tensor cos_sin_cache,"
      "                     bool is_neox,"
      "                     Tensor slot_mapping,"
      "                     Tensor! kv_cache,"
      "                     str kv_cache_dtype,"
      "                     Tensor kv_cache_scale) -> ()");

  // Convert the key and value cache to fp8 data type.
  ops.def(
      "convert_fp8(Tensor! dst_cache, Tensor src_cache, float scale, "
      "str kv_cache_dtype) -> ()");

  // Gather cache blocks from src_cache to dst, dequantizing from
  // src_cache's dtype to dst's dtype if necessary.
  ops.def(
      "gather_and_maybe_dequant_cache(Tensor src_cache, Tensor! dst, "
      "                               Tensor block_table, Tensor cu_seq_lens, "
      "                               Tensor token_to_seq, "
      "                               int num_tokens, "
      "                               str kv_cache_dtype, "
      "                               Tensor scale, Tensor? seq_starts) -> ()");

  ops.def(
      "cp_gather_cache(Tensor src_cache, Tensor! dst, Tensor block_table, "
      "Tensor cu_seq_lens, int batch_size, Tensor? seq_starts) -> ()");

  ops.def(
      "cp_gather_and_upconvert_fp8_kv_cache(Tensor src_cache, Tensor! dst, "
      "Tensor block_table, Tensor workspace_starts, int batch_size, Tensor? "
      "seq_starts, Tensor? host_cache=None, Tensor? host_row_ids=None, Tensor? "
      "device_row_ids=None) -> ()");

  ops.def(
      "indexer_k_quant_and_cache(Tensor k, Tensor! kv_cache, Tensor "
      "slot_mapping, "
      "int quant_block_size, str kv_cache_dtype) -> ()");

  ops.def("concat_mla_q(Tensor ql_nope, Tensor q_pe, Tensor! q_out) -> ()");

  ops.def(
      "cp_gather_indexer_k_quant_cache(Tensor kv_cache, Tensor! dst_k, Tensor! "
      "dst_scale, Tensor block_table, Tensor cu_seq_lens) -> ()");
}

STABLE_TORCH_LIBRARY_FRAGMENT(_C_custom_ar, custom_ar) {
}

STABLE_TORCH_LIBRARY_IMPL(_C_custom_ar, CUDA, custom_ar) {
}

STABLE_TORCH_LIBRARY_IMPL(_C_custom_ar, CPU, custom_ar) {
}

STABLE_TORCH_LIBRARY_IMPL(_C_custom_ar, CompositeExplicitAutograd, custom_ar) {
}

STABLE_TORCH_LIBRARY_IMPL(_C_cache_ops, CPU, ops) {
  ops.impl("swap_blocks_batch", TORCH_BOX(&swap_blocks_batch));
}

STABLE_TORCH_LIBRARY_IMPL(_C_cache_ops, CUDA, ops) {
  ops.impl("swap_blocks", TORCH_BOX(&swap_blocks));
  ops.impl("reshape_and_cache", TORCH_BOX(&reshape_and_cache));
  ops.impl("reshape_and_cache_flash", TORCH_BOX(&reshape_and_cache_flash));
  ops.impl("concat_and_cache_mla", TORCH_BOX(&concat_and_cache_mla));
  ops.impl("concat_and_cache_mla_grouped",
           TORCH_BOX(&concat_and_cache_mla_grouped));

#ifndef USE_ROCM
#endif  // !USE_ROCM
  ops.impl("concat_and_cache_mla_rope_fused",
           TORCH_BOX(&concat_and_cache_mla_rope_fused));
  ops.impl("convert_fp8", TORCH_BOX(&convert_fp8));
  ops.impl("gather_and_maybe_dequant_cache",
           TORCH_BOX(&gather_and_maybe_dequant_cache));
  ops.impl("cp_gather_cache", TORCH_BOX(&cp_gather_cache));
  ops.impl("cp_gather_and_upconvert_fp8_kv_cache",
           TORCH_BOX(&cp_gather_and_upconvert_fp8_kv_cache));
  ops.impl("indexer_k_quant_and_cache", TORCH_BOX(&indexer_k_quant_and_cache));
  ops.impl("concat_mla_q", TORCH_BOX(&concat_mla_q));
  ops.impl("cp_gather_indexer_k_quant_cache",
           TORCH_BOX(&cp_gather_indexer_k_quant_cache));
}

REGISTER_EXTENSION(_upstream_C)
