# ruff: noqa: E731, W291, W293, UP037
# SPDX-License-Identifier: Apache-2.0
"""MXFP4 quantization support for PPU on stock upstream.

The fork modifies ``vllm/model_executor/layers/quantization/mxfp4.py`` (+96)
and ``vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`` (+88) to run
MXFP4 checkpoints on PPU. Backend enum members, kernel-class resolution,
priority lists and activation keys are already covered by
``vllm_sail/registry/moe_backends/mxfp4.py``; this module covers the rest:

* ``Mxfp4Config`` gains ``fp8_channelwise_layers`` (PPU dense layers quantized
  FP8 channelwise via a CompressedTensors scheme), reads ``ignore`` /
  ``fp8_channelwise_layers`` in ``from_config``, routes ignored ``RoutedExperts``
  to ``UnquantizedFusedMoEMethod`` (mixed-precision MTP blocks), remaps the
  channelwise list in ``apply_vllm_mapper`` and matches a dense Linear in the
  checkpoint, mapped-runtime and text-only MTP-draft naming spaces.
* ``GptOssMxfp4MoEMethod`` / ``Mxfp4MoEMethod`` select W4A4
  (``kMxfp4Dynamic`` activations) on PPU sm90+, falling back to W4A16 on sm80.
* ``Mxfp4MoEMethod.get_fused_moe_quant_config`` threads ``swiglu_alpha`` /
  ``swiglu_beta`` into the quant config for the PPU backends.
* The oracle's ``make_mxfp4_moe_quant_config``, both weight-conversion
  functions (``deep_gemm.preprocess_mxfp4_scales``) and the size round-up gain
  PPU-backend branches.

Leaf-style module (like ``patch/enhancement/residual/``): importable on a bare
CPU runner, patches install when ``install()`` runs.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from vllm_sail.patch.utils import patch

if TYPE_CHECKING:
    import torch

_QMODULE = "vllm.model_executor.layers.quantization.mxfp4"
_OMODULE = "vllm.model_executor.layers.fused_moe.oracle.mxfp4"
AFFECTED_VERSIONS = ">=0.32.0,<0.33.0"

# Explicit per-kind member lists, keyed by the kernel path each member takes:
# "w4a16" reuses MARLIN's 16-bit activation packing, "w4a16mma" is DeepGEMM's
# own W4A16 MMA kernel, "w4a8"/"w4a4" are the FP8/MXFP4 activation paths.
# Suffix/substring sniffing silently misfiles a member -- an "_FP8" W4A8 entry
# used to fall into the "w4a4" group because it carries neither "_BF16" nor
# "W4A8".
_PPU_BACKEND_GROUPS = {
    "w4a4": ("PPU_DEEPGEMM_MXFP4", "BATCHED_PPU_DEEPGEMM_MXFP4"),
    "w4a16": ("PPU_DEEPGEMM_MXFP4_BF16", "BATCHED_PPU_DEEPGEMM_MXFP4_BF16"),
    "w4a16mma": (
        "PPU_DEEPGEMM_MXFP4_BF16_MMA",
        "BATCHED_PPU_DEEPGEMM_MXFP4_BF16_MMA",
    ),
    "w4a8": ("PPU_DEEPGEMM_MXFP4_FP8", "BATCHED_PPU_DEEPGEMM_MXFP4_FP8"),
}
_PPU_BACKEND_NAMES = tuple(
    name for names in _PPU_BACKEND_GROUPS.values() for name in names
)

REASON_CONFIG = (
    "The fork adds fp8_channelwise_layers to Mxfp4Config (PPU dense layers "
    "quantized FP8 channelwise), reads 'ignore'/'fp8_channelwise_layers' in "
    "from_config, routes ignored RoutedExperts to UnquantizedFusedMoEMethod "
    "for mixed-precision checkpoints, and remaps the list in "
    "apply_vllm_mapper; upstream has none of this."
)
REASON_METHOD_SELECT = (
    "PPU runs MXFP4 as W4A4 (mxfp4 activations) on sm90+ and W4A16 on sm80; "
    "upstream selects without the kMxfp4Dynamic activation key, so the PPU "
    "DeepGEMM MXFP4 experts would never be picked."
)
REASON_QUANT_CONFIG = (
    "The fork threads swiglu_alpha/swiglu_beta into the PPU MXFP4 quant "
    "config; upstream's get_fused_moe_quant_config does not pass them."
)
REASON_ORACLE = (
    "The PPU MXFP4 backends need an ocp_mx quant config (block_shape [1,16]), "
    "deep_gemm.preprocess_mxfp4_scales weight-scale preprocessing, and "
    "32-alignment of hidden/intermediate sizes; upstream's oracle has no "
    "branch for them."
)
REMOVE_WHEN = (
    "upstream grows first-class PPU MXFP4 support (fp8_channelwise_layers "
    "config plumbing, W4A4 selection, and PPU branches in the MXFP4 oracle), "
    "or the PPU MXFP4 flow is dropped."
)

METADATA = (
    (f"{_QMODULE}.Mxfp4Config.__init__", REASON_CONFIG, AFFECTED_VERSIONS, REMOVE_WHEN),
    (
        f"{_QMODULE}.Mxfp4Config.from_config",
        REASON_CONFIG,
        AFFECTED_VERSIONS,
        REMOVE_WHEN,
    ),
    (
        f"{_QMODULE}.Mxfp4Config.get_quant_method",
        REASON_CONFIG,
        AFFECTED_VERSIONS,
        REMOVE_WHEN,
    ),
    (
        f"{_QMODULE}.Mxfp4Config.apply_vllm_mapper",
        REASON_CONFIG,
        AFFECTED_VERSIONS,
        REMOVE_WHEN,
    ),
    (
        f"{_QMODULE}.GptOssMxfp4MoEMethod.__init__",
        REASON_METHOD_SELECT,
        AFFECTED_VERSIONS,
        REMOVE_WHEN,
    ),
    (
        f"{_QMODULE}.Mxfp4MoEMethod.get_fused_moe_quant_config",
        REASON_QUANT_CONFIG,
        AFFECTED_VERSIONS,
        REMOVE_WHEN,
    ),
    (
        f"{_OMODULE}.make_mxfp4_moe_quant_config",
        REASON_ORACLE,
        AFFECTED_VERSIONS,
        REMOVE_WHEN,
    ),
    (
        f"{_OMODULE}.convert_weight_to_mxfp4_moe_kernel_format",
        REASON_ORACLE,
        AFFECTED_VERSIONS,
        REMOVE_WHEN,
    ),
    (
        f"{_OMODULE}.convert_gpt_oss_weight_to_mxfp4_moe_kernel_format",
        REASON_ORACLE,
        AFFECTED_VERSIONS,
        REMOVE_WHEN,
    ),
    (
        f"{_OMODULE}.mxfp4_round_up_hidden_size_and_intermediate_size",
        REASON_ORACLE,
        AFFECTED_VERSIONS,
        REMOVE_WHEN,
    ),
)

_installed = False

#: A mixed MXFP4 + FP8-channelwise export names the same projection in three
#: spaces: the checkpoint quant lists (``model.language_model.layers.N.``), the
#: runtime text tower after the weight mapper (``language_model.model.layers.N.``)
#: and the text-only MTP draft (``model.layers.N.``, whose parameters live under
#: an extra ``mtp_block`` attribute). Strip the tower namespace and the MTP
#: attribute so one entry can match its own layer in all three spaces, while the
#: layer index stays in the path and keeps any other layer from matching.
_TOWER_PREFIX = re.compile(
    r"^(?:model\.language_model\.|language_model\.model\.|language_model\.|model\.)"
)


def normalize_layer_path(path: str) -> str:
    """Tower-free layer path, with the layer index preserved."""
    return _TOWER_PREFIX.sub("", path.replace(".mtp_block.", "."))


def channelwise_paths(layers: list[str]) -> list[str]:
    """Channelwise entries in their tower-free form."""
    return [normalize_layer_path(name) for name in layers]


def _ppu_backends(kind: str = "all") -> tuple[Any, ...]:
    """The PPU enum members, or () if the registry has not extended the enum."""
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend

    names = _PPU_BACKEND_NAMES if kind == "all" else _PPU_BACKEND_GROUPS.get(kind, ())
    return tuple(
        member
        for name in names
        if (member := getattr(Mxfp4MoeBackend, name, None)) is not None
    )


def install() -> None:
    """Apply the patches. Requires vLLM to be importable; idempotent."""
    global _installed
    if _installed:
        return

    import torch
    from vllm.model_executor.layers.fused_moe import (
        RoutedExperts,
        UnquantizedFusedMoEMethod,
    )
    from vllm.model_executor.layers.fused_moe.oracle import mxfp4 as oracle
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.quantization import mxfp4 as _quant
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        is_layer_skipped,
    )

    _upstream_config_init = _quant.Mxfp4Config.__init__
    _upstream_get_quant_method = _quant.Mxfp4Config.get_quant_method
    _upstream_apply_mapper = _quant.Mxfp4Config.apply_vllm_mapper
    _upstream_gptoss_init = _quant.GptOssMxfp4MoEMethod.__init__
    _upstream_get_fm_quant_config = _quant.Mxfp4MoEMethod.get_fused_moe_quant_config
    _upstream_make_quant_config = oracle.make_mxfp4_moe_quant_config
    _upstream_convert_weight = oracle.convert_weight_to_mxfp4_moe_kernel_format
    _upstream_convert_gpt_oss = oracle.convert_gpt_oss_weight_to_mxfp4_moe_kernel_format
    _upstream_round_up = oracle.mxfp4_round_up_hidden_size_and_intermediate_size

    @patch(
        _QMODULE,
        "Mxfp4Config.__init__",
        reason=REASON_CONFIG,
        affected_versions=AFFECTED_VERSIONS,
        remove_when=REMOVE_WHEN,
    )
    def _config_init(
        self,
        ignored_layers: list[str] | None = None,
        fp8_channelwise_layers: list[str] | None = None,
    ) -> None:
        _upstream_config_init(self, ignored_layers)
        self.fp8_channelwise_layers: list[str] = fp8_channelwise_layers or []

    @patch(
        _QMODULE,
        "Mxfp4Config.from_config",
        reason=REASON_CONFIG,
        affected_versions=AFFECTED_VERSIONS,
        remove_when=REMOVE_WHEN,
    )
    def _from_config(cls, config):
        from vllm.platforms import current_platform

        ignored_layers = config.get("ignore") if isinstance(config, dict) else None
        fp8_channelwise_layers = None
        if current_platform.is_ppu() and isinstance(config, dict):
            fp8_channelwise_layers = config.get("fp8_channelwise_layers")
        return cls(
            ignored_layers=ignored_layers,
            fp8_channelwise_layers=fp8_channelwise_layers,
        )

    @patch(
        _QMODULE,
        "Mxfp4Config.get_quant_method",
        reason=REASON_CONFIG,
        affected_versions=AFFECTED_VERSIONS,
        remove_when=REMOVE_WHEN,
    )
    def _get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from vllm.platforms import current_platform

        if isinstance(layer, RoutedExperts):
            if self.ignored_layers and is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedFusedMoEMethod(layer.moe_config)
        elif isinstance(layer, LinearBase) and current_platform.is_ppu():
            # PPU MODIFICATION: begin
            # Match the list in the naming space it was mapped into first, then
            # retry tower-free so the text-only MTP draft -- whose prefixes are
            # ``model.layers.N.*`` and whose parameters sit under ``mtp_block``
            # -- finds the scheme its own layer was exported with. Without the
            # retry the draft's shared-expert Linears build unquantized, carry
            # no ``weight_scale`` parameter, and weight loading aborts on a
            # KeyError for the scale the checkpoint does provide.
            matched = bool(self.fp8_channelwise_layers) and (
                is_layer_skipped(
                    prefix=prefix,
                    ignored_layers=self.fp8_channelwise_layers,
                    fused_mapping=self.packed_modules_mapping,
                    match_mode="substring",
                )
                or is_layer_skipped(
                    prefix=normalize_layer_path(prefix),
                    ignored_layers=channelwise_paths(self.fp8_channelwise_layers),
                    fused_mapping=self.packed_modules_mapping,
                    match_mode="substring",
                )
            )
            # PPU MODIFICATION: end
            if matched:
                from compressed_tensors.quantization import (
                    QuantizationArgs,
                    QuantizationStrategy,
                    QuantizationType,
                )
                from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
                    CompressedTensorsLinearMethod,
                )
                from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (  # noqa: E501
                    CompressedTensorsW8A8Fp8,
                )

                channelwise_args = QuantizationArgs(
                    strategy=QuantizationStrategy.CHANNEL,
                    type=QuantizationType.FLOAT,
                    num_bits=8,
                    symmetric=True,
                )
                scheme = CompressedTensorsW8A8Fp8(
                    weight_quant=channelwise_args,
                    is_static_input_scheme=False,
                )
                layer.scheme = scheme
                return CompressedTensorsLinearMethod(self)
        return _upstream_get_quant_method(self, layer, prefix)

    @patch(
        _QMODULE,
        "Mxfp4Config.apply_vllm_mapper",
        reason=REASON_CONFIG,
        affected_versions=AFFECTED_VERSIONS,
        remove_when=REMOVE_WHEN,
    )
    def _apply_vllm_mapper(self, hf_to_vllm_mapper) -> None:
        _upstream_apply_mapper(self, hf_to_vllm_mapper)
        from vllm.platforms import current_platform

        if current_platform.is_ppu() and self.fp8_channelwise_layers:
            self.fp8_channelwise_layers = hf_to_vllm_mapper.apply_list(
                self.fp8_channelwise_layers
            )

    def _ppu_select(moe):
        """The fork's PPU backend selection: W4A4 on sm90+, W4A16 on sm80."""
        from vllm.platforms import current_platform

        if not current_platform.is_device_capability(
            (8, 0)
        ) and moe.moe_backend not in ("marlin", "ppu_deep_gemm_w4a16"):
            if moe.moe_backend == "ppu_deep_gemm_w4a8":
                # W4A8: per-token dynamic FP8 activations.
                from vllm.model_executor.layers.quantization.utils.quant_utils import (
                    kFp8DynamicTokenSym,
                )

                return oracle.select_mxfp4_moe_backend(
                    moe, activation_key=kFp8DynamicTokenSym
                )
            return oracle.select_mxfp4_moe_backend(
                moe, activation_key=oracle.kMxfp4Dynamic
            )
        return oracle.select_mxfp4_moe_backend(moe)

    @patch(
        _QMODULE,
        "GptOssMxfp4MoEMethod.__init__",
        reason=REASON_METHOD_SELECT,
        affected_versions=AFFECTED_VERSIONS,
        remove_when=REMOVE_WHEN,
    )
    def _gptoss_init(self, moe) -> None:
        _upstream_gptoss_init(self, moe)
        from vllm.platforms import current_platform

        if current_platform.is_ppu():
            self.mxfp4_backend, self.experts_cls = _ppu_select(moe)

    @patch(
        _QMODULE,
        "Mxfp4MoEMethod.get_fused_moe_quant_config",
        reason=REASON_QUANT_CONFIG,
        affected_versions=AFFECTED_VERSIONS,
        remove_when=REMOVE_WHEN,
    )
    def _get_fused_moe_quant_config(self, layer):
        if self.mxfp4_backend in _ppu_backends():
            return oracle.make_mxfp4_moe_quant_config(
                mxfp4_backend=self.mxfp4_backend,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                w1_bias=getattr(layer, "w13_bias", None),
                w2_bias=getattr(layer, "w2_bias", None),
                gemm1_alpha=getattr(layer, "swiglu_alpha", None),
                gemm1_beta=getattr(layer, "swiglu_beta", None),
                swiglu_limit=getattr(layer, "swiglu_limit", None),
                layer=layer,
            )
        return _upstream_get_fm_quant_config(self, layer)

    @patch(
        _OMODULE,
        "make_mxfp4_moe_quant_config",
        reason=REASON_ORACLE,
        affected_versions=AFFECTED_VERSIONS,
        remove_when=REMOVE_WHEN,
    )
    def _make_quant_config(
        mxfp4_backend,
        w1_scale,
        w2_scale,
        gemm1_alpha: float | None = None,
        gemm1_beta: float | None = None,
        swiglu_limit: float | None = None,
        w1_bias: torch.Tensor | None = None,
        w2_bias: torch.Tensor | None = None,
        a1_scale: torch.Tensor | None = None,
        a2_scale: torch.Tensor | None = None,
        layer=None,
    ):
        if mxfp4_backend in _ppu_backends("w4a16") + _ppu_backends("w4a16mma"):
            mxfp4_backend = oracle.Mxfp4MoeBackend.MARLIN
        if mxfp4_backend in _ppu_backends("w4a8"):
            from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
            from vllm.platforms import current_platform

            quant_config = FusedMoEQuantConfig.make(
                quant_dtype=current_platform.fp8_dtype(),
                per_act_token_quant=True,
                weight_dtype="mxfp4",
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                w1_bias=w1_bias,
                w2_bias=w2_bias,
                gemm1_clamp_limit=swiglu_limit,
            )
            # Attached by _convert_ppu_w4a8_weights during weight conversion,
            # which runs before _build_moe_kernel -> this function.
            quant_config.w13_inv_scale = float(getattr(layer, "w13_inv_scale", 1.0))
            quant_config.w2_inv_scale = float(getattr(layer, "w2_inv_scale", 1.0))
            return quant_config
        if mxfp4_backend in _ppu_backends("w4a4"):
            from vllm.model_executor.layers.fused_moe.config import (
                ocp_mx_moe_quant_config,
            )

            return ocp_mx_moe_quant_config(
                quant_dtype="mxfp4",
                weight_dtype="mxfp4",
                w1_bias=w1_bias,
                w2_bias=w2_bias,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                block_shape=[1, 16],  # block_shape[1]: mxfp4 block size // 2
                gemm1_alpha=gemm1_alpha,
                gemm1_beta=gemm1_beta,
                gemm1_clamp_limit=swiglu_limit,
            )
        return _upstream_make_quant_config(
            mxfp4_backend=mxfp4_backend,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            gemm1_alpha=gemm1_alpha,
            gemm1_beta=gemm1_beta,
            swiglu_limit=swiglu_limit,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            layer=layer,
        )

    def _convert_ppu_weights(
        w13_weight,
        w2_weight,
        w13_weight_scale,
        w2_weight_scale,
        w13_bias,
        w2_bias,
    ):
        from deep_gemm import preprocess_mxfp4_scales

        if w13_bias is not None:
            w13_bias = w13_bias.to(torch.float32)
        if w2_bias is not None:
            w2_bias = w2_bias.to(torch.float32)
        w13_weight_scale = torch.nn.Parameter(
            preprocess_mxfp4_scales(w13_weight_scale), requires_grad=False
        )
        w2_weight_scale = torch.nn.Parameter(
            preprocess_mxfp4_scales(w2_weight_scale), requires_grad=False
        )
        return (
            w13_weight,
            w2_weight,
            w13_weight_scale,
            w2_weight_scale,
            w13_bias,
            w2_bias,
        )

    def _convert_ppu_w4a8_weights(
        layer,
        w13_weight,
        w2_weight,
        w13_weight_scale,
        w2_weight_scale,
        w13_bias,
        w2_bias,
    ):
        """Preprocess W4A8-MXFP4 weights via DeepGEMM.

        Runs during ``process_weights_after_loading`` before
        ``_build_moe_kernel`` so the inverse scales can be attached to the
        layer and threaded into the quant config. w2's packed K is padded to
        a 128-element boundary with zero codes / E8M0-unity scales.
        """
        import torch.nn.functional as F

        from vllm_sail.utils.deep_gemm import (
            get_preprocess_mode,
            preprocess_weights_and_scales,
        )

        if w13_bias is not None or w2_bias is not None:
            raise ValueError("PPU MXFP4 W4A8 experts do not support biases.")

        mode = get_preprocess_mode().W4A8_MXFP4_MMA

        # GEMM2 contracts over the packed w2 K; align it to 128 so the
        # activation can be zero-padded (zero fp8 codes contribute nothing).
        pad = (-w2_weight.size(2) * 2) % 128
        if pad:
            w2_weight = F.pad(w2_weight, (0, pad // 2))
            # E8M0 scale bytes: 127 encodes 2**0 (unity).
            w2_weight_scale = F.pad(w2_weight_scale, (0, pad // 32), value=127)

        w13_weight, w13_weight_scale, w13_inv_scale = preprocess_weights_and_scales(
            mode, False, w13_weight, w13_weight_scale
        )
        w2_weight, w2_weight_scale, w2_inv_scale = preprocess_weights_and_scales(
            mode, False, w2_weight, w2_weight_scale
        )
        layer.w13_inv_scale = float(w13_inv_scale)
        layer.w2_inv_scale = float(w2_inv_scale)
        return (
            w13_weight,
            w2_weight,
            torch.nn.Parameter(w13_weight_scale, requires_grad=False),
            torch.nn.Parameter(w2_weight_scale, requires_grad=False),
            w13_bias,
            w2_bias,
        )

    @patch(
        _OMODULE,
        "convert_weight_to_mxfp4_moe_kernel_format",
        reason=REASON_ORACLE,
        affected_versions=AFFECTED_VERSIONS,
        remove_when=REMOVE_WHEN,
    )
    def _convert_weight(
        mxfp4_backend,
        layer: torch.nn.Module,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        w13_weight_scale: torch.Tensor,
        w2_weight_scale: torch.Tensor,
        w13_bias: torch.Tensor | None = None,
        w2_bias: torch.Tensor | None = None,
        _cache_permute_indices: dict | None = None,
        activation=None,
        use_separated_a4w4: bool = False,
        activation_key=None,
    ):
        if mxfp4_backend in _ppu_backends("w4a16"):
            import vllm.envs as envs

            if (envs.VLLM_MARLIN_INPUT_DTYPE or "").lower() in ("int8", "fp8"):
                raise ValueError(
                    "PPU MXFP4 BF16 experts require 16-bit activation weight packing; unset VLLM_MARLIN_INPUT_DTYPE."
                )
            mxfp4_backend = oracle.Mxfp4MoeBackend.MARLIN
        if mxfp4_backend in _ppu_backends("w4a16mma"):
            from vllm_sail.model_executor.layers.fused_moe.deep_gemm_utils import (
                preprocess_mxfp4_w4a16_scales,
            )

            return (
                w13_weight,
                w2_weight,
                torch.nn.Parameter(
                    preprocess_mxfp4_w4a16_scales(w13_weight_scale), requires_grad=False
                ),
                torch.nn.Parameter(
                    preprocess_mxfp4_w4a16_scales(w2_weight_scale), requires_grad=False
                ),
                w13_bias,
                w2_bias,
            )
        if mxfp4_backend in _ppu_backends("w4a4"):
            return _convert_ppu_weights(
                w13_weight,
                w2_weight,
                w13_weight_scale,
                w2_weight_scale,
                w13_bias,
                w2_bias,
            )
        if mxfp4_backend in _ppu_backends("w4a8"):
            return _convert_ppu_w4a8_weights(
                layer,
                w13_weight,
                w2_weight,
                w13_weight_scale,
                w2_weight_scale,
                w13_bias,
                w2_bias,
            )
        return _upstream_convert_weight(
            mxfp4_backend,
            layer,
            w13_weight,
            w2_weight,
            w13_weight_scale,
            w2_weight_scale,
            w13_bias,
            w2_bias,
            _cache_permute_indices,
            activation=activation,
            use_separated_a4w4=use_separated_a4w4,
            activation_key=activation_key,
        )

    @patch(
        _OMODULE,
        "convert_gpt_oss_weight_to_mxfp4_moe_kernel_format",
        reason=REASON_ORACLE,
        affected_versions=AFFECTED_VERSIONS,
        remove_when=REMOVE_WHEN,
    )
    def _convert_gpt_oss_weight(
        mxfp4_backend,
        layer: torch.nn.Module,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        w13_weight_scale: torch.Tensor,
        w2_weight_scale: torch.Tensor,
        w13_bias: torch.Tensor | None = None,
        w2_bias: torch.Tensor | None = None,
        w13_input_scale: torch.Tensor | None = None,
        w2_input_scale: torch.Tensor | None = None,
        _cache_permute_indices: dict | None = None,
    ):
        if mxfp4_backend in _ppu_backends("w4a16"):
            import vllm.envs as envs

            if (envs.VLLM_MARLIN_INPUT_DTYPE or "").lower() in ("int8", "fp8"):
                raise ValueError(
                    "PPU MXFP4 BF16 experts require 16-bit activation weight packing; unset VLLM_MARLIN_INPUT_DTYPE."
                )
            mxfp4_backend = oracle.Mxfp4MoeBackend.MARLIN
        if mxfp4_backend in _ppu_backends("w4a16mma"):
            from vllm_sail.model_executor.layers.fused_moe.deep_gemm_utils import (
                preprocess_mxfp4_w4a16_scales,
            )

            return (
                w13_weight,
                w2_weight,
                torch.nn.Parameter(
                    preprocess_mxfp4_w4a16_scales(w13_weight_scale), requires_grad=False
                ),
                torch.nn.Parameter(
                    preprocess_mxfp4_w4a16_scales(w2_weight_scale), requires_grad=False
                ),
                w13_bias,
                w2_bias,
            )
        if mxfp4_backend in _ppu_backends("w4a4"):
            return _convert_ppu_weights(
                w13_weight,
                w2_weight,
                w13_weight_scale,
                w2_weight_scale,
                w13_bias,
                w2_bias,
            )
        if mxfp4_backend in _ppu_backends("w4a8"):
            return _convert_ppu_w4a8_weights(
                layer,
                w13_weight,
                w2_weight,
                w13_weight_scale,
                w2_weight_scale,
                w13_bias,
                w2_bias,
            )
        return _upstream_convert_gpt_oss(
            mxfp4_backend,
            layer,
            w13_weight,
            w2_weight,
            w13_weight_scale,
            w2_weight_scale,
            w13_bias,
            w2_bias,
            w13_input_scale,
            w2_input_scale,
            _cache_permute_indices,
        )

    @patch(
        _OMODULE,
        "mxfp4_round_up_hidden_size_and_intermediate_size",
        reason=REASON_ORACLE,
        affected_versions=AFFECTED_VERSIONS,
        remove_when=REMOVE_WHEN,
    )
    def _round_up_sizes(
        backend, hidden_size: int, intermediate_size: int, activation=None
    ):
        if backend in _ppu_backends():
            from vllm.utils.math_utils import round_up

            if backend in _ppu_backends("w4a16"):
                return _upstream_round_up(
                    oracle.Mxfp4MoeBackend.MARLIN,
                    hidden_size,
                    intermediate_size,
                    activation=activation,
                )
            alignment = (
                64
                if backend in _ppu_backends("w4a16mma") + _ppu_backends("w4a8")
                else 32
            )
            intermediate_size = round_up(intermediate_size, alignment)
            hidden_size = round_up(hidden_size, alignment)
            return hidden_size, intermediate_size
        return _upstream_round_up(
            backend, hidden_size, intermediate_size, activation=activation
        )

    _installed = True
