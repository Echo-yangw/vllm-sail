# SPDX-License-Identifier: Apache-2.0
"""Strict tensor/launch doubles shared by speculative-sampling contract tests."""


class Tensor:
    def __init__(self, *shape, dtype="float32", contiguous=True):
        self.shape, self.dtype = shape, dtype
        self.ndim = len(shape)
        self._contiguous = contiguous

    def stride(self, axis):
        return 1 if self._contiguous else 2

    def size(self, axis):
        return self.shape[axis]

    def new_empty(self, *shape, dtype):
        return Tensor(*shape, dtype=dtype)

    def is_contiguous(self):
        return self._contiguous

    def contiguous(self):
        return Tensor(*self.shape, dtype=self.dtype)

    def view(self, dtype):
        return Tensor(*self.shape, dtype=dtype)


class Kernel:
    def __init__(self, call=lambda *a, **kw: None):
        self.call = call

    def __getitem__(self, grid):
        return self.call
