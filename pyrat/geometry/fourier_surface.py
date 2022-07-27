from __future__ import annotations
import numpy as np


class FourierSurface:
    def __init__(self, c, a0, a, b):
        assert c.shape == a0.shape
        assert a.shape == b.shape
        assert a0.shape == a.shape[:-1]
        self._c = c
        self._a0 = a0
        self._a = a
        self._b = b

    def shape(self):
        return self._c.shape

    def __add__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

    def __matmul__(self, other):
        raise NotImplementedError
