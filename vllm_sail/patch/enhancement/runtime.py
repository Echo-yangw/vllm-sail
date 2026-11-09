# SPDX-License-Identifier: Apache-2.0
"""PPU input-batch invalidation and race-free speculative sample insertion."""

from __future__ import annotations

import functools
import sys

from vllm.platforms import current_platform
from vllm.v1.worker import gpu_model_runner as _runner
from vllm.v1.worker.gpu.spec_decode import rejection_sampler_utils as _sampler

from vllm_sail.patch._upstream import runtime as _bodies
from vllm_sail.patch.bodies import bind_body
from vllm_sail.patch.utils import patch


def _install(module, owner, attribute, body):
    original = getattr(owner, attribute)
    replacement = bind_body(body, module)

    @functools.wraps(original)
    def dispatch(*args, **kwargs):
        if current_platform.is_ppu():
            return replacement(*args, **kwargs)
        return original(*args, **kwargs)

    target = attribute if owner is module else owner.__name__ + "." + attribute
    metadata = dict(
        reason="PPU must reinitialize batches on context-length changes and insert speculative tokens within one warp.",
        affected_versions=">=0.32.0,<0.33.0",
        remove_when="Upstream includes the context-length invalidation and one-warp insertion fixes.",
    )
    patch(module.__name__, target, **metadata)(dispatch)
    if owner is module:
        for name in _CONSUMERS:
            consumer = sys.modules.get(name)
            if consumer is not None and getattr(consumer, attribute, None) is original:
                patch(name, attribute, **metadata)(dispatch)


_CONSUMERS = [
    "vllm.v1.worker.gpu.spec_decode.rejection_sampler",
    "vllm.v1.watermarking.spec_decode",
]
_install(
    _runner,
    _runner.GPUModelRunner,
    "may_reinitialize_input_batch",
    _bodies.may_reinitialize_input_batch,
)
_install(_sampler, _sampler, "rejection_sample", _bodies.rejection_sample)
