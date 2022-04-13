from __future__ import annotations

import math
import numbers

import numpy as np

from .geometry import Geometry


class IntervalOld(Geometry):
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
    def c(self) -> np.ndarray:
        return self._bd.sum(axis=0) * 0.5

    @property
    def info(self):
        info = "\n ------------- interval info ------------- \n"
        info += str(self.dim) + "\n"
        info += str(self._bd) + "\n"
        return info

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
            "sin": cls.sin,
            "cos": cls.cos,
            "asin": cls.asin,
            "acos": cls.acos,
            "asinh": cls.asinh,
            "acosh": cls.acosh,
            "tan": cls.tan,
            "tanh": cls.tanh,
            "sinh": cls.sinh,
            "cosh": cls.cosh,
            "atan": cls.atan,
            "atan2": cls.atan2,
            "exp": cls.exp,
        }

    # =============================================== operator

    def __abs__(self):
        bd = np.sort(abs(self._bd), axis=0)
        return IntervalOld(bd)

    def __getitem__(self, item):
        assert isinstance(item, int) and item >= 0
        return IntervalOld(self._bd[:, item].reshape((-1, 1)))

    def __truediv__(self, other):
        if isinstance(other, numbers.Real):
            return self * (1 / other)
        else:
            raise NotImplementedError

    def __rtruediv__(self, other):
        if isinstance(other, numbers.Real):
            bd = np.sort(other / self._bd, axis=0)
            return IntervalOld(bd)
        else:
            raise NotImplementedError

    def __pow__(self, power, modulo=None):
        if isinstance(power, numbers.Real):
            if power >= 0:
                bd = abs(self._bd) ** power
                return IntervalOld(bd * np.sign(self._bd))
            return IntervalOld(np.flip(self._bd**power))
        else:
            raise NotImplementedError

    def __contains__(self, item):
        raise NotImplementedError

    def __add__(self, other):
        if isinstance(other, IntervalOld):
            return IntervalOld(
                np.stack(
                    [np.minimum(self.inf, other.inf), np.maximum(self.sup, other.sup)]
                )
            )
        elif isinstance(other, numbers.Real):
            return IntervalOld(self._bd + other)
        elif isinstance(other, np.ndarray):
            return IntervalOld(self._bd + other)
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
        raise NotImplementedError

    def __str__(self):
        return str(self._bd[0]) + " \n" + str(self._bd[1])

    def __matmul__(self, other):
        raise NotImplementedError(
            "Unsupported operation for now"
            "For doing multiplication with real number, please use '*' instead."
        )

    def __mul__(self, other):
        if isinstance(other, numbers.Real):
            bd = np.sort(self._bd * other, axis=0)
            return IntervalOld(bd)
        elif isinstance(other, IntervalOld):
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
        i = IntervalOld(np.zeros(2, dim, dtype=float))
        i._is_empty = True
        return i

    @staticmethod
    def rand(dim: int):
        bd = np.sort(np.random.rand(2, dim), axis=0)
        return IntervalOld(bd)

    @staticmethod
    def sin(x: IntervalOld):
        raise NotImplementedError

    @staticmethod
    def cos(x: IntervalOld):
        raise NotImplementedError

    @staticmethod
    def asin(x: IntervalOld):
        raise NotImplementedError

    @staticmethod
    def acos(x: IntervalOld):
        raise NotImplementedError

    @staticmethod
    def asinh(x: IntervalOld):
        raise NotImplementedError

    @staticmethod
    def acosh(x: IntervalOld):
        raise NotImplementedError

    @staticmethod
    def tan(x: IntervalOld):
        raise NotImplementedError

    @staticmethod
    def tanh(x: IntervalOld):
        raise NotImplementedError

    @staticmethod
    def sinh(x: IntervalOld):
        raise NotImplementedError

    @staticmethod
    def cosh(x: IntervalOld):
        raise NotImplementedError

    @staticmethod
    def atan(x: IntervalOld):
        raise NotImplementedError

    @staticmethod
    def atan2(a: IntervalOld, b: IntervalOld):
        raise NotImplementedError

    @staticmethod
    def exp(x: IntervalOld):
        bd = np.hstack([np.exp(x.inf), np.exp(x.sup)])
        return IntervalOld(bd)

    # =============================================== public method
    def reduce(self, method: str, order: int):
        raise NotImplementedError

    def proj(self, dims):
        raise NotImplementedError
