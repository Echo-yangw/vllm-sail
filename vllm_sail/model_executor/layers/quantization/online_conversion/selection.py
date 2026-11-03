# SPDX-License-Identifier: Apache-2.0
"""Checkpoint exclusions use exact names, fused expansion and explicit regex."""

import re


def is_ignored(prefix, entries, fused_mapping):
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        is_layer_skipped,
    )

    regex = [entry[3:] for entry in entries if entry.startswith("re:")]
    plain = [entry for entry in entries if not entry.startswith("re:")]

    def matches(name):
        return any(re.search(pattern, name) for pattern in regex) or (
            bool(plain)
            and is_layer_skipped(prefix=name, ignored_layers=plain, fused_mapping={})
        )

    if matches(prefix):
        return True
    parent, separator, module = prefix.rpartition(".")
    if module in fused_mapping:
        flags = [matches(parent + separator + shard) for shard in fused_mapping[module]]
        if any(flags) and not all(flags):
            raise ValueError(
                f"Detected some but not all shards of {prefix} are quantized"
            )
        return bool(flags) and all(flags)
    return False


def is_mixed_layer(prefix, indices):
    """Match a transformer block without confusing layer 1 with layer 10."""
    match = re.search(r"(?:^|\.)layers\.(\d+)\.", prefix)
    return match is not None and int(match.group(1)) in indices
