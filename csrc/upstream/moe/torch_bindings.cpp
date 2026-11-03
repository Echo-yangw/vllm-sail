#include "core/registration.h"
#include "moe_ops.h"

#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY_FRAGMENT(_moe_C, m) {
  // Apply topk softmax to the gating outputs.
  m.def(
      "topk_softmax(Tensor! topk_weights, Tensor! topk_indices, Tensor! "
      "token_expert_indices, Tensor gating_output, bool renormalize, Tensor? "
      "bias, Tensor? is_padding) -> ()");

  // Apply topk sigmoid to the gating outputs.
  m.def(
      "topk_sigmoid(Tensor! topk_weights, Tensor! topk_indices, Tensor! "
      "token_expert_indices, Tensor gating_output, bool renormalize, "
      "Tensor? bias, float routed_scaling_factor, Tensor? is_padding) -> ()");

  m.def(
      "topk_softplus_sqrt(Tensor! topk_weights, Tensor! topk_indices, Tensor! "
      "token_expert_indices, Tensor gating_output, bool renormalize, float "
      "routed_scaling_factor, Tensor? "
      "bias, Tensor? input_ids, Tensor? tid2eid, Tensor? is_padding, "
      "Tensor? bias_vl=None, int image_sentinel_lo=0) -> ()");

#ifdef USE_CUDA
#endif

  // Calculate the result of moe by summing up the partial results
  // from all selected experts. topk_ids/expert_map are optional and, when
  // both given, enable pad-aware reduce that skips (token, expert)
  // slots that were never actually computed (unrouted, or routed to an
  // expert not owned by this rank under expert parallelism).
  m.def(
      "moe_sum(Tensor input, Tensor! output, Tensor? topk_ids=None, "
      "Tensor? expert_map=None) -> ()");

  // Aligning the number of tokens to be processed by each expert such
  // that it is divisible by the block size.
  m.def(
      "moe_align_block_size(Tensor topk_ids, int num_experts,"
      "                     int block_size, Tensor! sorted_token_ids,"
      "                     Tensor! experts_ids,"
      "                     Tensor! num_tokens_post_pad,"
      "                     Tensor? maybe_expert_map,"
      "                     Tensor(a!)? scatter_idx=None) -> ()");

#ifdef USE_CUDA
#endif

  // Aligning the number of tokens to be processed by each expert such
  // that it is divisible by the block size, but for the batched case.
  m.def(
      "batched_moe_align_block_size(int max_tokens_per_batch,"
      "                     int block_size, Tensor expert_num_tokens,"
      "                     Tensor! sorted_token_ids,"
      "                     Tensor! experts_ids,"
      "                     Tensor! num_tokens_post_pad) -> ()");

  // Aligning the number of tokens to be processed by each expert such
  // that it is divisible by the block size.
  m.def(
      "moe_lora_align_block_size(Tensor topk_ids,"
      "                     Tensor token_lora_mapping,"
      "                     int num_experts,"
      "                     int block_size, int max_loras, "
      "                     int max_num_tokens_padded, "
      "                     int max_num_m_blocks, "
      "                     Tensor !sorted_token_ids,"
      "                     Tensor !experts_ids,"
      "                     Tensor !num_tokens_post_pad,"
      "                     Tensor !adapter_enabled,"
      "                     Tensor !lora_ids,"
      "                     Tensor? maybe_expert_map) -> () ");
#ifndef USE_ROCM

  // Apply grouped topk routing to select experts.
  m.def(
      "grouped_topk(Tensor scores, int n_group, int "
      "topk_group, int topk, bool renormalize, float "
      "routed_scaling_factor, Tensor bias, int scoring_func) -> (Tensor, "
      "Tensor)");
#endif
}

STABLE_TORCH_LIBRARY_IMPL(_moe_C, CUDA, m) {
  m.impl("topk_softmax", TORCH_BOX(&topk_softmax));
  m.impl("topk_sigmoid", TORCH_BOX(&topk_sigmoid));
#ifdef USE_CUDA
#endif
  m.impl("topk_softplus_sqrt", TORCH_BOX(&topk_softplus_sqrt));
  m.impl("moe_sum", TORCH_BOX(&moe_sum));
  m.impl("moe_align_block_size", TORCH_BOX(&moe_align_block_size));
#ifdef USE_CUDA
#endif
  m.impl("batched_moe_align_block_size",
         TORCH_BOX(&batched_moe_align_block_size));
  m.impl("moe_lora_align_block_size", TORCH_BOX(&moe_lora_align_block_size));
#ifndef USE_ROCM
  m.impl("grouped_topk", TORCH_BOX(&grouped_topk));
#endif
}

#ifndef USE_ROCM
// Primitive-only ops have no tensor to dispatch on.
STABLE_TORCH_LIBRARY_IMPL(_moe_C, CompositeExplicitAutograd, m) {
}
#endif

REGISTER_EXTENSION(_upstream_moe_C)
