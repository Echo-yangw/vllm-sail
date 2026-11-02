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
  to ``UnquantizedFusedMoEMethod`` (mixed-precision MTP blocks), and remaps the
  ignored and channelwise lists in ``apply_vllm_mapper``.
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

from typing import TYPE_CHECKING, Any

from vllm_sail.patch.utils import patch

if TYPE_CHECKING:
    import torch

_QMODULE = "vllm.model_executor.layers.quantization.mxfp4"
_OMODULE = "vllm.model_executor.layers.fused_moe.oracle.mxfp4"
AFFECTED_VERSIONS = ">=0.31.0,<0.32.0"

_PPU_BACKEND_NAMES = (
    "PPU_DEEPGEMM_MXFP4",
    "BATCHED_PPU_DEEPGEMM_MXFP4",
    "PPU_DEEPGEMM_MXFP4_BF16",
    "BATCHED_PPU_DEEPGEMM_MXFP4_BF16",
    "PPU_DEEPGEMM_MXFP4_BF16_MMA",
    "BATCHED_PPU_DEEPGEMM_MXFP4_BF16_MMA",
)

REASON_CONFIG = (
    "The fork adds fp8_channelwise_layers to Mxfp4Config (PPU dense layers "
    "quantized FP8 channelwise), reads 'ignore'/'fp8_channelwise_layers' in "
    "from_config, routes ignored RoutedExperts to UnquantizedFusedMoEMethod "
    "for mixed-precision checkpoints, and remaps both lists in "
    "apply_vllm_mapper -- upstream inherits the no-op base, so ignored_layers "
    "keeps checkpoint naming and never matches a vLLM prefix."
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
    "config plumbing, W4A4 selection, PPU branches in the MXFP4 oracle, and an "
    "apply_vllm_mapper override that rewrites ignored_layers), or the PPU "
    "MXFP4 flow is dropped."
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


def _ppu_backends(kind: str = "all") -> tuple[Any, ...]:
    """The PPU enum members, or () if the registry has not extended the enum."""
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend

    return tuple(
        member
        for name in _PPU_BACKEND_NAMES
        if kind == "all"
        or (kind == "w4a4" and "_BF16" not in name)
        or (kind == "bf16" and name.endswith("_BF16"))
        or (kind == "mma" and name.endswith("_MMA"))
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
        elif (
            isinstance(layer, LinearBase)
            and current_platform.is_ppu()
            and self.fp8_channelwise_layers
            and is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.fp8_channelwise_layers,
                fused_mapping=self.packed_modules_mapping,
                match_mode="substring",
            )
        ):
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
        # Upstream's Mxfp4Config inherits QuantizationConfig's no-op
        # apply_vllm_mapper, so its exclusion list keeps checkpoint naming
        # (model.language_model.*) while the module tree uses vLLM naming
        # (language_model.model.*). Every other quantization family rewrites
        # its list here; without that, is_layer_skipped never matches and the
        # list is dead weight.
        if self.ignored_layers:
            self.ignored_layers = hf_to_vllm_mapper.apply_list(self.ignored_layers)

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
        if mxfp4_backend in _ppu_backends("bf16") + _ppu_backends("mma"):
            mxfp4_backend = oracle.Mxfp4MoeBackend.MARLIN
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
    ):
        if mxfp4_backend in _ppu_backends("bf16"):
            import vllm.envs as envs

            if (envs.VLLM_MARLIN_INPUT_DTYPE or "").lower() in ("int8", "fp8"):
                raise ValueError(
                    "PPU MXFP4 BF16 experts require 16-bit activation weight packing; unset VLLM_MARLIN_INPUT_DTYPE."
                )
            mxfp4_backend = oracle.Mxfp4MoeBackend.MARLIN
        if mxfp4_backend in _ppu_backends("mma"):
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
        if mxfp4_backend in _ppu_backends("bf16"):
            import vllm.envs as envs

            if (envs.VLLM_MARLIN_INPUT_DTYPE or "").lower() in ("int8", "fp8"):
                raise ValueError(
                    "PPU MXFP4 BF16 experts require 16-bit activation weight packing; unset VLLM_MARLIN_INPUT_DTYPE."
                )
            mxfp4_backend = oracle.Mxfp4MoeBackend.MARLIN
        if mxfp4_backend in _ppu_backends("mma"):
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

            if backend in _ppu_backends("bf16"):
                return _upstream_round_up(
                    oracle.Mxfp4MoeBackend.MARLIN,
                    hidden_size,
                    intermediate_size,
                    activation=activation,
                )
            alignment = 64 if backend in _ppu_backends("mma") else 32
            intermediate_size = round_up(intermediate_size, alignment)
            hidden_size = round_up(hidden_size, alignment)
            return hidden_size, intermediate_size
        return _upstream_round_up(
            backend, hidden_size, intermediate_size, activation=activation
        )

    _installed = True
