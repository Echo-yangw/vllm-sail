# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PPU kernels and selection glue for the Qwen4Exp QSA owner.

Importing this package pulls in nothing: the leaf modules import ``torch``,
``triton`` and ``vllm`` at module top level and are reached only from the
patched ``Qwen4ExpQSAImpl``, which itself only exists once
``vllm.models.qwen4_exp.nvidia.qsa`` has been imported.
"""
