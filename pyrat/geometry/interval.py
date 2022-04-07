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
        return self._bd.sum(axis=0) * 0.5

    # =============================================== class method
    @classmethod
    def ops(cls):
        return {
            "__add__": cls.__add__,
            "__radd__": cls.__radd__,
            "__iadd__": cls.__iadd__,
            "__sub__": cls.__sub__,
            "__rsub__": cls.__rsub__,
            "__isub__": cls.__isub__,
            "__mul__": cls.__mul__,
            "__matmul__": cls.__matmul__,
            "__truediv__": cls.__truediv__,
            "__rtruediv__": cls.__rtruediv__,
            "__pow__": cls.__pow__,
            "__getitem__": cls.__getitem__,
            "__abs__": cls.__abs__,
        }

    # =============================================== operator
    def __abs__(self):
        bd = np.sort(abs(self._bd), axis=0)
        return Interval(bd)

    def __getitem__(self, item):
        assert isinstance(item, int) and item >= 0
        return Interval(self._bd[:, item].reshape((-1, 1)))

    def __truediv__(self, other):
        if isinstance(other, numbers.Real):
            return self * (1 / other)
        else:
            raise NotImplementedError

    def __rtruediv__(self, other):
        if isinstance(other, numbers.Real):
            bd = other / self._bd
            if other >= 0:
                return Interval(np.flip(bd, axis=0))
            return Interval(bd)
        else:
            raise NotImplementedError

    def __pow__(self, power, modulo=None):
        if isinstance(power, numbers.Real):
            if power >= 0:
                return Interval(self.bd**power)
            return Interval(np.flip(self._bd**power))
        else:
            raise NotImplementedError

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
        elif isinstance(other, np.ndarray):
            return Interval(self._bd + other)
        raise NotImplementedError

    def __radd__(self, other):
        return self + other

    def __iadd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __isub__(self, other):
        return self - other

    def __rsub__(self, other):
        return other - self

    def __pos__(self):
        return self

    def __neg__(self):
        return self * -1

    def __str__(self):
        return str(self._bd[0]) + " \n" + str(self._bd[1])

    def __matmul__(self, other):
        raise NotImplementedError(
            "Unsupported operation for now"
            "For doing multiplication with real number, please use '*' instead."
        )

    def __mul__(self, other):
        if isinstance(other, numbers.Real):
            return Interval(self._bd * other)
        elif isinstance(other, Interval):
            aa = self._bd[0] * other._bd[0]
            ab = self._bd[0] * other._bd[1]
            ba = self._bd[1] * other._bd[0]
            bb = self._bd[1] * other._bd[1]
            print(aa.shape)
            exit(False)

            raise NotImplementedError  # TODO
        return other * self
        # raise NotImplementedError

    def __rmul__(self, other):
        return self * other

    def __or__(self, other):
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
