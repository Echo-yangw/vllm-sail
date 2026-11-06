# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp QSA Triton kernels.

This ``__init__`` deliberately imports nothing, matching
``vllm_sail/models/deepseek_v4/ops/__init__.py``: the leaf modules import
``torch`` and ``vllm.triton_utils`` at module top level. Import them directly,
under the QSA selection gate.
"""
