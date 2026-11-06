# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Route Qwen4Exp QSA through PPU FA3's CUDA kernel when it can run the shape.

``Qwen4ExpQSAImpl`` hard-wires the Triton sparse-attention kernel. PPU's
``flash_attn_3`` wheel also instantiates QSA, and on Zhenwu M890 it measured
faster than Triton on both prefill (-6.7% to -11.6% TTFT from 4k to 100k) and
decode (+5.5% output tokens/s), so this dispatches to it per layer.

The dispatch is thin. The decision's pure half -- the shape and KV-dtype gates
and the ``auto``/``cuda``/``triton`` resolution -- lives in
:mod:`vllm_sail.models.qwen4_exp.qsa_fa3` so ``tests/ut`` covers it on a bare
interpreter; the kernel sequence and its scratch workspace live in
:mod:`vllm_sail.models.qwen4_exp.qsa_fa3_forward` so the device suite exercises
the same code the patched forward runs. What is left here is the two probes that
need a live platform and wheel, and the two patched methods.
"""

from __future__ import annotations

import torch
from vllm.logger import init_logger

# Upstream has an import cycle: ``qsa`` does ``from . import model`` while
# ``model`` does ``from .qsa import Qwen4ExpQSAAttention``, so importing ``qsa``
# first dies on a partially initialised module. vLLM only ever reaches ``qsa``
# through ``model``, but this patch module is imported at plugin registration --
# before any model is loaded -- so it has to establish that order itself.
from vllm.models.qwen4_exp.nvidia import model  # noqa: F401
from vllm.models.qwen4_exp.nvidia import qsa
from vllm.platforms import current_platform

from vllm_sail import envs
from vllm_sail.models.qwen4_exp.qsa_fa3 import (
    QSAFA3Kernel,
    qsa_fa3_kv_dtype_reason,
    qsa_fa3_shape_reason,
    resolve_qsa_fa3_kernel,
)
from vllm_sail.models.qwen4_exp.qsa_fa3_forward import (
    qsa_fa3_sparse_attention,
    qsa_fa3_workspace,
    reserve_qsa_fa3_workspace,
)
from vllm_sail.patch.utils import patch

# Named under ``vllm``, not ``__name__``: vLLM raises only its own ``vllm`` logger
# to INFO and leaves the root logger at WARNING, so every ``info_once`` on a
# ``vllm_sail.*`` logger is silently dropped in a real server. The kernel-selection
# line below is the only operator-visible evidence of which QSA kernel is running;
# invisible, it makes VLLM_SAIL_QSA_KERNEL=cuda indistinguishable from a quiet
# fallback. Same pattern as vllm_sail.attention.flash_attn_shim.
logger = init_logger("vllm.ppu.qwen4_exp_qsa")

_MODULE = "vllm.models.qwen4_exp.nvidia.qsa"
_REASON = (
    "PPU FA3's QSA kernel is faster than the Triton one on Zhenwu M890 for the "
    "shapes it instantiates, and upstream hard-wires the Triton kernel."
)
_AFFECTED = ">=0.32.0,<0.33.0"
_REMOVE_WHEN = (
    "Qwen4ExpQSAImpl grows a backend-selection hook that a plugin can register "
    "an FA3 QSA implementation with."
)


def qsa_fa3_unsupported_reason(
    *,
    head_size: int,
    num_heads: int,
    num_kv_heads: int,
    kv_cache_dtype: str,
    device: int | None = None,
) -> str | None:
    """Why FA3's QSA cannot run this config, or ``None`` if it can.

    Ordered cheapest check first; the wheel probe is last because it issues real
    kernel launches.

    Args:
        head_size: main attention head dim.
        num_heads: / num_kv_heads: per-rank query / KV head counts.
        kv_cache_dtype: the layer's ``--kv-cache-dtype`` string.
        device: device to probe; defaults to the current one.
    """
    reason = qsa_fa3_shape_reason(head_size, num_heads, num_kv_heads)
    if reason is not None:
        return reason
    reason = qsa_fa3_kv_dtype_reason(kv_cache_dtype)
    if reason is not None:
        return reason
    if not current_platform.is_ppu():
        return "FA3 QSA ships only in PPU's flash_attn_3 wheel"
    # Imported here, not at module scope: the patch package loads on every
    # platform, and the probe module needs the PPU wheel's symbols to exist.
    from vllm_sail.attention.flash_attn._qsa import (
        fa3_qsa_unsupported_reason,
        is_fa3_qsa_supported,
    )

    if not is_fa3_qsa_supported(device):
        return (
            "the installed FA3 wheel cannot run QSA "
            f"({fa3_qsa_unsupported_reason(device)})"
        )
    return None


_SELECTION_CACHE: dict[tuple[int, int, int, str], QSAFA3Kernel] = {}


def select_qsa_fa3_kernel(
    *,
    head_size: int,
    num_heads: int,
    num_kv_heads: int,
    kv_cache_dtype: str,
) -> QSAFA3Kernel:
    """Resolve the kernel for one layer shape and reserve the scratch FA3 needs.

    Called once per layer, so cached per shape and logged once per distinct
    outcome. A shape that wants FA3 but cannot get a workspace is downgraded to
    Triton -- or raised on, when FA3 was pinned -- so ``forward`` dispatches on
    ``use_fa3`` alone.
    """
    requested = envs.VLLM_SAIL_QSA_KERNEL
    key = (head_size, num_heads, num_kv_heads, kv_cache_dtype)
    cached = _SELECTION_CACHE.get(key)
    if cached is not None and cached.requested == requested:
        return cached

    selection = resolve_qsa_fa3_kernel(
        requested,
        cuda_reason=(
            None
            if requested == "triton"
            else qsa_fa3_unsupported_reason(
                head_size=head_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                kv_cache_dtype=kv_cache_dtype,
            )
        ),
    )
    if selection.use_fa3:
        workspace, reason = reserve_qsa_fa3_workspace()
        if workspace is None:
            if requested == "cuda":
                raise ValueError(
                    f"VLLM_SAIL_QSA_KERNEL=cuda is not supported: {reason}"
                )
            selection = QSAFA3Kernel(requested, "triton", reason)

    _SELECTION_CACHE[key] = selection
    if selection.use_fa3:
        logger.info_once("Qwen4Exp QSA is on PPU FA3's CUDA kernel.")
    elif requested == "auto":
        logger.warning_once(
            "Qwen4Exp QSA falls back to the Triton kernel because %s. Set "
            "VLLM_SAIL_QSA_KERNEL=cuda to make this fatal, or =triton to silence it.",
            selection.reason,
        )
    else:
        logger.info_once("Qwen4Exp QSA is on the Triton kernel (pinned by env).")
    return selection


_impl_init = qsa.Qwen4ExpQSAImpl.__init__
_impl_forward = qsa.Qwen4ExpQSAImpl.forward


@patch(
    _MODULE,
    "Qwen4ExpQSAImpl.__init__",
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def impl_init(
    self,
    num_heads: int,
    head_size: int,
    scale: float,
    num_kv_heads: int,
    kv_cache_dtype: str,
) -> None:
    _impl_init(self, num_heads, head_size, scale, num_kv_heads, kv_cache_dtype)
    # Resolved here, not per step: the answer depends only on the shape, the
    # environment and the wheel, and reserving the workspace here keeps the
    # allocation out of the CUDA graph capture the builder allows.
    self._sail_qsa_fa3 = select_qsa_fa3_kernel(
        head_size=head_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        kv_cache_dtype=kv_cache_dtype,
    )
    if self._sail_qsa_fa3.use_fa3:
        # select_qsa_fa3_kernel reserved it, and downgraded to Triton if it could
        # not, so this is never None on the FA3 branch.
        self._sail_qsa_fa3_workspace = qsa_fa3_workspace()


@patch(
    _MODULE,
    "Qwen4ExpQSAImpl.forward",
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def impl_forward(
    self,
    layer: torch.nn.Module,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: qsa.Qwen4ExpQSAMetadata,
    output: torch.Tensor,
    *,
    token_to_req: torch.Tensor,
    use_prefill_config: bool,
    output_gate: torch.Tensor,
) -> torch.Tensor:
    if not self._sail_qsa_fa3.use_fa3:
        return _impl_forward(
            self,
            layer,
            query,
            kv_cache,
            attn_metadata,
            output,
            token_to_req=token_to_req,
            use_prefill_config=use_prefill_config,
            output_gate=output_gate,
        )

    # The full tensor, not just the live rows: the owner allocates it with
    # new_empty, so padding rows hold garbage that o_proj would otherwise read.
    output.zero_()
    num_tokens = attn_metadata.num_actual_tokens
    if num_tokens == 0:
        return output

    qsa_fa3_sparse_attention(
        query[:num_tokens],
        kv_cache,
        packed_indices=layer.topk_indices_buffer[:num_tokens],
        block_table=attn_metadata.block_table,
        token_to_req=token_to_req[:num_tokens],
        output=output[:num_tokens],
        output_gate=output_gate[:num_tokens],
        head_size=self.head_size,
        scale=self.scale,
        workspace=self._sail_qsa_fa3_workspace,
    )
    return output
