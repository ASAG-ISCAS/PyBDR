from __future__ import annotations

import numbers

import numpy as np

from .geometry import Geometry


class Interval(Geometry):
    def __init__(self, bd: np.ndarray):
        assert bd.shape[0] == 2 and bd.ndim >= 2
        assert np.all(bd[0] <= bd[1])
        self._bd = bd
        self._is_empty = False

    # =============================================== property
    @property
    def bd(self) -> np.ndarray:
        return self._bd

    @property
    def inf(self) -> np.ndarray:
        return self._bd[0]

    @property
    def sup(self) -> np.ndarray:
        return self._bd[1]

    @property
    def dim(self) -> int:
        return self._bd.shape[1:]

    @property
    def is_empty(self) -> bool:
        return self._is_empty

    @property
    def vertices(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def center(self) -> np.ndarray:
        return self._bd.sum(axis=0) / 2

    # =============================================== operator
    def __contains__(self, item):
        raise NotImplementedError

    def __add__(self, other):
        if isinstance(other, Interval):
            return Interval(
                np.stack(
                    [np.minimum(self.inf, other.inf), np.maximum(self.sup, other.sup)]
                )
            )
        elif isinstance(other, numbers.Real):
            return Interval(self._bd + other)
        raise NotImplementedError

    def __sub__(self, other):
        return self + (-other)

    def __pos__(self):
        return self

    def __neg__(self):
        return self * -1

    def __str__(self):
        return str(self._bd.T)

    def __matmul__(self, other):
        raise NotImplementedError(
            "Unsupported operation for now,"
            "For doing multiplication with real number, please use '*' instead."
        )

    def __mul__(self, other):
        if isinstance(other, numbers.Real):
            return Interval(self._bd * other)
        raise NotImplementedError

    # =============================================== static method
    @staticmethod
    def empty(dim: int):
        i = Interval(np.zeros(2, dim, dtype=float))
        i._is_empty = True
        return i

    @staticmethod
    def rand(dim: int):
        bd = np.sort(np.random.rand(2, dim), axis=0)
        return Interval(bd)
