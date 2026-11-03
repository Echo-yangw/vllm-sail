# SPDX-License-Identifier: Apache-2.0
"""Refuse the fused QKV+indexer GEMM when a checkpoint splits their precision.

Failure record (vllm-sail main, vLLM pin ``5f63ea461ccd``, PPU 890p, tp4)::

    ValueError: Detected some but not all shards of
    language_model.model.layers.3.self_attn.qkv_proj are quantized.
    All shards of fused layers to have the same precision.

    linear.py:293 LinearBase.__init__
      -> base_config.py:312 resolve_quant_method
      -> vllm_sail/patch/enhancement/mxfp4.py:237 _get_quant_method
      -> quant_utils.py:679 is_layer_skipped

The fused layer is ``Qwen4ExpQSAAttention.qkv_proj``. vLLM 0.32 added
``fuse_indexer_projection`` (``qsa.py:508``), which rebuilds it as
``Qwen4ExpQSAQKVIndexerLinear`` — a ``MergedColumnParallelLinear`` whose fourth
shard is the QSA indexer's ``index_qk_proj`` — and listed that shard in
``Qwen4ExpForCausalLM.packed_modules_mapping`` (``model.py:697``). 0.31.0 listed
only ``q_proj``/``k_proj``/``v_proj`` (``model.py:647``), which is why the same
checkpoint served there.

That mapping doubles as the fused->shard expansion table ``is_layer_skipped``
uses to assert one precision per fused layer, and
``SupportsQuant._maybe_apply_model_mapping`` injects it into
``quant_config.packed_modules_mapping`` from ``__new__`` — before any layer is
built, and ahead of the LoRA override in ``__init__`` that would trim it.

The two formats that disagree, per ``Qwen3.8-Flash-Next-MoE-Quant-W-MXFP4-A-MXFP4
-Dense-Quant-W-FP8-PerChannel-A-FP8-PerToken`` (12 full-attention layers, and
the MTP layer is excluded wholesale so it stays consistent):

===========================================  ======================  ==========
tensor                                       dtype / scale           declared in
===========================================  ======================  ==========
``self_attn.{q,k,v,o}_proj.weight``          F8_E4M3 + F32 [N, 1]    fp8_channelwise_layers
``self_attn.indexer.index_qk_proj.weight``   BF16, no scale          ignore
===========================================  ======================  ==========

So shards 0-2 are FP8 per-channel and shard 3 is BF16. A merged linear carries
one quant method, so no method can hold both; silencing the check would leave
``qkv_proj`` on upstream's ``UnquantizedLinearMethod`` (every ``LinearBase``
branch of ``Mxfp4Config.get_quant_method`` returns it), which cannot hold F8_E4M3
weights at all. The fusion has to be refused instead, which restores the 0.31.0
structure exactly: ``qkv_proj`` = 3 FP8 shards, ``indexer.index_qk_proj`` =
standalone BF16 ``ReplicatedLinear`` matched by ``ignore``. Cost is one extra
``[T, 2560] x [2560, 640]`` GEMM per full-attention layer (12 of 48).

Only ``quant_config.packed_modules_mapping`` needs trimming, not the model
instance's: ``Qwen4ExpForCausalLM.load_weights`` uses ``AutoWeightsLoader`` and
never expands ``packed_modules_mapping``, and the attention module's own
``load_weights`` (``qsa.py:538``) installs the ``indexer.index_qk_proj`` ->
``qkv_proj`` shard-3 mapper only when it fuses. With the gate off the indexer
weight therefore lands on the surviving standalone ``ReplicatedLinear``, and
``forward`` (``qsa.py:639``) takes the matching branch. ``index_qk_size`` is
read only inside the fused branch, so nothing dangles.

SGLang needs no equivalent guard because it never fuses this projection
(``qsa_indexer.py:76`` keeps ``index_qk_proj`` standalone and
``qwen3_5.py:1651`` declares three shards), yet enforces the same invariant
(``quark/utils.py should_ignore_layer``). Its
``shared_experts_fusion_disable_reason`` -> ``quant_blocks_shared_experts_fusion``
-> ``quant_config.can_fuse_shared_expert()`` chain is the shape followed here:
a precision mismatch vetoes a fusion, decided before any layer is built, with a
human-readable reason.

Sibling checkpoints are homogeneous and keep the fusion: the official
``Qwen3.8-Flash-Next-FP8`` skips q/k/v/o *and* the indexer, and
``Qwen3.8-Flash-Next-Quant-W-FP8-PerChannel-A-FP8-PerToken`` quantizes all four
(F8_E4M3 + F32 [640, 1] for the indexer).

Leaf-style module (like ``mxfp4``): importable on a bare CPU runner, patches
install when ``install()`` runs. ``install()`` must enter the upstream package
through ``model`` rather than ``qsa``: the two import each other, and the plugin
hook runs before vLLM loads any model, so importing ``qsa`` first is the one
order that raises.

That import carries one unavoidable side effect, visible as nine extra rows in
``patch/manifest.md``. ``model.py`` reaches ``quantization.quark.quark`` via
``ple_layer`` -> ``ngram_embedding`` -> ``common/ngram_embedding``, so
``quark_moe`` is already in ``sys.modules`` when ``registry.moe_backends``
runs after ``patch.install()``, and that registry patches every oracle alias a
*loaded* consumer still holds. A consumer imported later captures the PPU oracle
directly and needs no patch, so the end state is unchanged -- only reached
eagerly. qwen4_exp is the first model package this plugin imports whose upstream
chain reaches quark; the coupling belongs to that chain, not to this patch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm_sail.patch.utils import patch

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.quantization import QuantizationConfig

_QSA_MODULE = "vllm.models.qwen4_exp.nvidia.qsa"
AFFECTED_VERSIONS = ">=0.32.0,<0.33.0"

_FUSED_QKV = "qkv_proj"
_INDEXER_SHARD = "indexer.index_qk_proj"
_INDEX_QK_SUFFIX = ".indexer.index_qk_proj"

# Every quantization config names its skip list differently; the first
# populated one is authoritative for that checkpoint.
_SKIP_LIST_ATTRS = ("ignored_layers", "ignore", "modules_to_not_convert")

REASON_INIT = (
    "vLLM 0.32 lists indexer.index_qk_proj as a fourth qkv_proj shard in "
    "Qwen4ExpForCausalLM.packed_modules_mapping, which SupportsQuant injects "
    "into quant_config.packed_modules_mapping and is_layer_skipped uses as the "
    "fused->shard expansion table. Checkpoints that quantize q/k/v but leave "
    "the indexer unquantized make the four shards disagree, so building "
    "qkv_proj raises 'Detected some but not all shards ... are quantized'. "
    "Delegates to upstream after dropping the indexer shard from the injected "
    "mapping, the same override upstream applies for lora_config."
)
REASON_GATE = (
    "fuse_indexer_projection must follow the trimmed packed_modules_mapping: a "
    "MergedColumnParallelLinear carries one quant method, so an unquantized "
    "indexer shard cannot join FP8 q/k/v shards. Installed as a property "
    "because upstream assigns the flag and reads it back inside __init__ "
    "before building the merged projection, which is the only point where the "
    "fusion can still be refused without rebuilding qkv_proj."
)
REMOVE_WHEN = (
    "upstream gates the QSA indexer projection fusion on per-shard "
    "quantization the way it already gates on lora_config, or "
    "Qwen4ExpForCausalLM.packed_modules_mapping stops listing "
    "indexer.index_qk_proj as a qkv_proj shard."
)

_logged_reasons: set[str] = set()
_installed = False


def _skipped_layer_names(quant_config) -> frozenset[str]:
    for attr in _SKIP_LIST_ATTRS:
        names = getattr(quant_config, attr, None)
        if names:
            return frozenset(names)
    return frozenset()


def indexer_fusion_block_reason(quant_config) -> str | None:
    """Why the indexer projection cannot join the fused QKV GEMM, else None.

    A shard is only a mismatch when its own attention block quantizes the
    neighbouring q/k/v: the MTP layer of these checkpoints keeps the whole
    block unquantized, so its indexer entry is consistent and must not veto.
    """
    if quant_config is None:
        return None
    skipped = _skipped_layer_names(quant_config)
    if not skipped:
        return None
    for entry in sorted(skipped):
        if not entry.endswith(_INDEX_QK_SUFFIX):
            continue
        attention_prefix = entry[: -len(_INDEX_QK_SUFFIX)]
        if f"{attention_prefix}.q_proj" in skipped:
            continue
        return (
            f"e.g. {attention_prefix}.q_proj/k_proj/v_proj are quantized but "
            f"{entry} is not, and a merged QKV+indexer GEMM carries a single "
            "quant method"
        )
    return None


def trim_indexer_shard(quant_config) -> bool:
    """Drop the indexer shard from the injected fused->shard expansion table."""
    mapping = getattr(quant_config, "packed_modules_mapping", None)
    if not mapping:
        return False
    shards = mapping.get(_FUSED_QKV)
    if not shards or _INDEXER_SHARD not in shards:
        return False
    mapping[_FUSED_QKV] = [shard for shard in shards if shard != _INDEXER_SHARD]
    return True


def _announce(logger, reason: str) -> None:
    if reason in _logged_reasons:
        return
    _logged_reasons.add(reason)
    # WARNING, not INFO: vllm.logger.init_logger leaves loggers outside the
    # "vllm" namespace at effective level WARNING, so an INFO here is dropped
    # and the disabled fusion would be invisible in the serve log.
    logger.warning(
        "Disabling the fused QKV+indexer projection for the QSA attention: %s. "
        "The indexer projection runs as a separate GEMM instead.",
        reason,
    )


def install() -> None:
    """Apply the patches. Requires vLLM to be importable; idempotent."""
    global _installed
    if _installed:
        return

    from vllm.logger import init_logger

    logger = init_logger(__name__)

    try:
        # qsa.py:63 imports model.py and model.py:90 imports Qwen4ExpQSAAttention
        # back out of qsa.py, so entering through qsa leaves it partially
        # initialized and raises. model.py first breaks the cycle.
        import vllm.models.qwen4_exp.nvidia.model  # noqa: F401
        from vllm.models.qwen4_exp.nvidia import qsa as _qsa
    except ImportError as exc:
        logger.warning(
            "Skipping the QSA indexer projection fusion patch: the upstream "
            "qwen4_exp model cannot be imported (%s). It needs "
            "transformers>=5.16.1,<5.20.0.",
            exc,
        )
        return

    _upstream_init = _qsa.Qwen4ExpQSAAttention.__init__

    @patch(
        _QSA_MODULE,
        "Qwen4ExpQSAAttention.__init__",
        reason=REASON_INIT,
        affected_versions=AFFECTED_VERSIONS,
        remove_when=REMOVE_WHEN,
    )
    def _init(
        self,
        *,
        vllm_config: VllmConfig,
        config,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        # PPU MODIFICATION: begin
        # Runs before upstream builds qkv_proj, which is where the shard
        # consistency check raises; the flag is read back by the
        # fuse_indexer_projection property patched below.
        reason = indexer_fusion_block_reason(quant_config)
        self._ppu_indexer_fusion_blocked = reason is not None
        if reason is not None:
            trim_indexer_shard(quant_config)
            _announce(logger, reason)
        # PPU MODIFICATION: end
        _upstream_init(
            self,
            vllm_config=vllm_config,
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=prefix,
        )

    def _fuse_indexer_projection_get(self) -> bool:
        return bool(
            self.__dict__.get("_ppu_fuse_indexer_projection", False)
            and not self.__dict__.get("_ppu_indexer_fusion_blocked", False)
        )

    def _fuse_indexer_projection_set(self, value: bool) -> None:
        self._ppu_fuse_indexer_projection = bool(value)

    patch(
        _QSA_MODULE,
        "Qwen4ExpQSAAttention.fuse_indexer_projection",
        allow_missing=True,
        reason=REASON_GATE,
        affected_versions=AFFECTED_VERSIONS,
        remove_when=REMOVE_WHEN,
    )(property(_fuse_indexer_projection_get, _fuse_indexer_projection_set))

    _installed = True
