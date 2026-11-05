# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp QSA indexer kernels for PPU.

These modules import ``torch`` and DeepGEMM at module top level, so they must
not be imported eagerly. This ``__init__`` deliberately imports nothing;
import the leaf modules directly from the patches that need them.
"""
