# SPDX-License-Identifier: Apache-2.0
"""Let GLM-5.3-Flash checkpoints declare ``deepseek_sparse_attention`` layers.

Upstream extends Transformers' strict ``ALLOWED_LAYER_TYPES`` set per model type
through ``vllm.transformers_utils.config._PATCH_HF_ALLOWED_LAYER_TYPES``. The
table carries GLM-5.2's ``glm_moe_dsa`` entry but not GLM-5.3-Flash's
``glm5_next``.

On the 0.31 line the checkpoints are parsed by vLLM's own bundled
``Glm5NextConfig`` (``vllm/transformers_utils/configs/glm5_next.py``), so a
Transformers release without a ``glm5_next`` entry still loads. The table entry
is a defensive guard for the paths that do go through Transformers' per-model-type
allow-list (configs parsed by ``AutoConfig``, or releases that reject unknown
layer types before any weight or kernel is touched) and a no-op wherever the
bundled config is used.

The table is data, so ``patch_value`` would be the natural hook, but it can only
*add* a constant: an existing attribute with a different value raises. Replace
the module attribute with a copy carrying the extra entry instead. Consumers
read the table from the module namespace at call time, so the copy is what they
see.

``_AllowedLayerTypes`` is a ``dict`` subclass only because the patch framework
records the replaced object *on the replacement*, and a plain ``dict`` rejects
attribute assignment. It behaves exactly like the upstream table.
"""

from __future__ import annotations

from vllm.transformers_utils import config as _config_module

from vllm_sail.patch.utils import patch

_AFFECTED = ">=0.31.0,<0.32.0"
_MODULE = "vllm.transformers_utils.config"

#: GLM-5.3-Flash's sparse layers; upstream transformers has no such entry yet.
_GLM5_NEXT_LAYER_TYPES = ("deepseek_sparse_attention",)


class _AllowedLayerTypes(dict):
    """Copy of upstream's table that can carry the patch marker."""


_upstream_allowed_layer_types = _config_module._PATCH_HF_ALLOWED_LAYER_TYPES
_extended_allowed_layer_types = _AllowedLayerTypes(_upstream_allowed_layer_types)
# setdefault keeps a future upstream entry authoritative instead of duplicating.
_extended_allowed_layer_types.setdefault("glm5_next", _GLM5_NEXT_LAYER_TYPES)

patch(
    _MODULE,
    "_PATCH_HF_ALLOWED_LAYER_TYPES",
    reason=(
        "GLM-5.3-Flash (model_type 'glm5_next') declares "
        "'deepseek_sparse_attention' layer_types and upstream only lists "
        "'glm_moe_dsa' (GLM-5.2) in the per-model-type allow-list. The 0.31 "
        "line parses these checkpoints with vLLM's bundled Glm5NextConfig, so "
        "this entry is a defensive guard for Transformers-parsed configs "
        "(AutoConfig, or releases without a glm5_next registry entry) and a "
        "no-op where the bundled config is used."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream lists 'glm5_next' in _PATCH_HF_ALLOWED_LAYER_TYPES, at which "
        "point this patch installs an identical table and can be deleted."
    ),
)(_extended_allowed_layer_types)
