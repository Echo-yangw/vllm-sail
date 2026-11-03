# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: I001  (import order below is functional; see the docstring)
"""PPU compatibility patches.

These are not bug fixes and not optimisations — they are the small set of
additions that let a CUDA-compatible plugin reuse upstream vLLM's CUDA code
paths. Removing any of them breaks PPU.

Import order is significant for `platform_predicates` (must come first: every
other patch may call `current_platform.is_ppu()`) and for `attention` (must
come after `custom_op_dispatch`, which installs the `forward_ppu` dispatch
reroute that the attention `forward_ppu` additions rely on).
"""

from vllm_sail.patch.enhancement import platform_predicates  # noqa: F401  (must be first)
from vllm_sail.patch.enhancement import pynccl  # noqa: F401  (before communicators)
from vllm_sail.patch.enhancement import import_gates  # noqa: F401  (before kernel consumers)
from vllm_sail.patch.enhancement import custom_op_dispatch  # noqa: F401
from vllm_sail.patch.enhancement import attention  # noqa: F401  (needs custom_op_dispatch)
from vllm_sail.patch.enhancement import fused_act  # noqa: F401  (needs custom_op_dispatch)
from vllm_sail.patch.enhancement import compressed_tensors_ppu
from vllm_sail.patch.enhancement import deepep  # noqa: F401
from vllm_sail.patch.enhancement import deep_gemm_redirect  # noqa: F401
from vllm_sail.patch.enhancement import fused_moe_ppu
from vllm_sail.patch.enhancement import int8_quant  # noqa: F401
from vllm_sail.patch.enhancement import kernel_warmup  # noqa: F401
from vllm_sail.patch.enhancement import modular_kernel  # noqa: F401
from vllm_sail.patch.enhancement import online_quantization_dtype  # noqa: F401
from vllm_sail.patch.enhancement import quant_keys  # noqa: F401
from vllm_sail.patch.enhancement import tuned_config_lookup  # noqa: F401
from vllm_sail.patch.enhancement import models  # noqa: F401
from vllm_sail.patch.enhancement import mxfp4
from vllm_sail.patch.enhancement import residual  # noqa: F401

# Leaf-style modules (like residual/): define their replacements without vLLM
# imports at module scope, so installation is explicit.
compressed_tensors_ppu.install()
fused_moe_ppu.install()
mxfp4.install()

# After generic quantization adapters, before runtime calls consume tuned configs.
from vllm_sail.patch.enhancement import triton_moe  # noqa: F401

from vllm_sail.patch.enhancement import dense  # noqa: F401

from vllm_sail.patch.enhancement import fp8_quant  # noqa: F401
from vllm_sail.patch.enhancement.models import qwen3_fused_quant  # noqa: F401

from vllm_sail.patch.enhancement import runtime  # noqa: F401
