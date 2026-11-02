# SPDX-License-Identifier: Apache-2.0
"""Exclusion coverage for the Qwen4Exp PLE n-gram embedding.

``Qwen4ExpPLEEmbeddingMethod.from_quant_config`` picks the storage format of the
PLE n-gram table. Upstream dispatches over an explicit whitelist — ``None``,
``ModelOptMixedPrecisionConfig``, ``ModelOptQuantConfigBase`` and ``Fp8Config``
— and rejects everything else:

.. code-block:: python

    if not isinstance(quant_config, Fp8Config):
        raise NotImplementedError(
            "Qwen4Exp PLE embedding does not support quantization config "
            f"{type(quant_config).__name__}"
        )

The PPU quantization pipeline emits two families the whitelist misses. Its FP8
per-channel weight / per-token dynamic activation checkpoints load as
``CompressedTensorsConfig``; its MXFP4-expert + FP8-channelwise-dense
checkpoints load as ``Mxfp4Config`` (whose ``from_config``
``patch/enhancement/mxfp4.py`` already teaches to read ``ignore``). Loading a
Qwen4Exp model from either kills every worker inside ``load_model`` before the
API server can bind a port:

.. code-block:: text

    NotImplementedError: Qwen4Exp PLE embedding does not support quantization
    config CompressedTensorsConfig

    NotImplementedError: Qwen4Exp PLE embedding does not support quantization
    config Mxfp4Config

Those checkpoints do not quantize the PLE table at all. Every PLE weight is
listed in ``quantization_config.ignore`` — the ``ple.ple_embedding.ngram_embedding.shard_*``
tensors (one per ``split_ngram_parts`` entry) plus ``ple.conv1d``,
``ple.key_proj`` and the PLE norms — so the table is stored in ``params_dtype``
and must be loaded that way. Each family spells that list differently:
``CompressedTensorsConfig.ignore`` versus ``Mxfp4Config.ignored_layers``. The
``ModelOptQuantConfigBase`` branch (``is_layer_excluded``) and the
``Fp8Config`` branch (``is_layer_skipped`` plus a ``shard_`` prefix scan)
already honour exactly this exclusion; these two families never reach a check.

Both lists reach the dispatcher already rewritten from checkpoint naming
(``model.language_model.*``) into vLLM module naming
(``language_model.model.*``), which is what makes them comparable to
``prefix``: ``CompressedTensorsConfig.apply_vllm_mapper`` does that upstream,
and ``patch/enhancement/mxfp4.py`` does it for ``Mxfp4Config``, whose upstream
``apply_vllm_mapper`` is the no-op base.

This patch adds that missing branch and delegates everything else, so a
checkpoint that genuinely *does* quantize the PLE table still fails loudly
upstream instead of silently loading unquantized weights over quantized ones.
An ``Mxfp4Config`` carrying no exclusions at all is such a checkpoint: the
qwen4-max config quantizes the table to FP8 and lists nothing in ``ignore``,
so it keeps raising.

Leaf-style module (like ``compressed_tensors_ppu.py``): importable on a bare
CPU runner, patches install when ``install()`` runs.
"""

from __future__ import annotations

from vllm_sail.patch.utils import patch

_MODULE = "vllm.models.qwen4_exp.common.ngram_embedding"
AFFECTED_VERSIONS = ">=0.31.0,<0.32.0"

REASON_PLE_IGNORE = (
    "Upstream's Qwen4ExpPLEEmbeddingMethod.from_quant_config whitelists None, "
    "ModelOpt and Fp8Config and raises NotImplementedError for anything else, "
    "so a CompressedTensorsConfig or Mxfp4Config checkpoint that lists every "
    "PLE weight in quantization_config.ignore still fails to load. The ModelOpt "
    "and Fp8 branches already honour that exclusion; these two families carry "
    "it in different attributes (ignore / ignored_layers) and never reach a "
    "check."
)
REMOVE_WHEN_PLE_IGNORE = (
    "upstream's Qwen4ExpPLEEmbeddingMethod.from_quant_config grows "
    "CompressedTensorsConfig and Mxfp4Config branches that return the "
    "unquantized method when the PLE table is excluded by "
    "quantization_config.ignore."
)

METADATA = (
    (
        f"{_MODULE}.Qwen4ExpPLEEmbeddingMethod.from_quant_config",
        REASON_PLE_IGNORE,
        AFFECTED_VERSIONS,
        REMOVE_WHEN_PLE_IGNORE,
    ),
)

_installed = False


def install() -> None:
    """Apply the patch. Requires vLLM to be importable; idempotent."""
    global _installed
    if _installed:
        return

    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
        CompressedTensorsConfig,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
        should_ignore_layer,
    )
    from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        is_layer_skipped,
    )
    from vllm.models.qwen4_exp.common import ngram_embedding as _ng

    # Captured before patching so the replacement can delegate to upstream.
    # from_quant_config is a staticmethod, so class access yields the plain
    # function and it is called without a leading self/cls.
    _upstream_from_quant_config = _ng.Qwen4ExpPLEEmbeddingMethod.from_quant_config

    def _ple_ignore_list(quant_config):
        """The exclusion list of a family this patch rescues, else ``None``.

        The two families spell the attribute differently, and an
        ``Mxfp4Config`` may carry no list at all: a checkpoint that really does
        quantize the PLE table has nothing to exclude.
        """
        if isinstance(quant_config, CompressedTensorsConfig):
            return quant_config.ignore
        if isinstance(quant_config, Mxfp4Config):
            return quant_config.ignored_layers
        return None

    def _ple_table_is_ignored(quant_config, prefix: str) -> bool:
        """Whether the checkpoint stores this PLE table unquantized."""
        ignore = _ple_ignore_list(quant_config)
        if not ignore:
            return False
        fused_mapping = quant_config.packed_modules_mapping
        # Each family's own helper, so its list is matched the way the rest of
        # vLLM matches that family.
        if isinstance(quant_config, Mxfp4Config):
            named_directly = is_layer_skipped(
                prefix=prefix, ignored_layers=ignore, fused_mapping=fused_mapping
            )
        else:
            named_directly = should_ignore_layer(
                prefix, ignore=ignore, fused_mapping=fused_mapping
            )
        if named_directly:
            return True
        # The table is one runtime parameter assembled from shard_* checkpoint
        # tensors, so an ignore list naming the shards excludes the whole table
        # without ever naming its parent. Mirrors the upstream Fp8Config branch.
        shard_prefix = f"{prefix}.shard_"
        return any(name.startswith(shard_prefix) for name in ignore)

    @patch(
        _MODULE,
        "Qwen4ExpPLEEmbeddingMethod.from_quant_config",
        reason=REASON_PLE_IGNORE,
        affected_versions=AFFECTED_VERSIONS,
        remove_when=REMOVE_WHEN_PLE_IGNORE,
    )
    def _from_quant_config(quant_config, prefix, embedding_dtype=None):
        # Upstream resolves an explicitly declared FP8 PLE dtype before it ever
        # looks at the quantization config, so delegating first keeps that
        # precedence (and every config family upstream does handle) intact.
        # Only the blanket rejection of the two PPU families is rescued, and
        # only when the checkpoint's ignore list says the table is unquantized;
        # a checkpoint that really does quantize the PLE table re-raises
        # upstream's error instead of loading it wrong.
        try:
            return _upstream_from_quant_config(quant_config, prefix, embedding_dtype)
        except NotImplementedError:
            if _ple_table_is_ignored(quant_config, prefix):
                return _ng.Qwen4ExpPLEUnquantizedEmbeddingMethod()
            raise

    _installed = True
