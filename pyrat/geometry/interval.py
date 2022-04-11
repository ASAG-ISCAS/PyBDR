from __future__ import annotations

import numbers

import numpy as np
from numpy.typing import ArrayLike

import pyrat.util.functional.auxiliary as aux
from .geometry import Geometry


class Interval(Geometry):
    def __init__(self, inf: ArrayLike, sup: ArrayLike):
        inf, sup = np.array(inf), np.array(sup)
        assert inf.shape == sup.shape
        assert inf.ndim <= 2
        assert np.all(inf <= sup)
        self._inf = inf
        self._sup = sup

    # =============================================== property
    @property
    def inf(self) -> np.ndarray:
        return self._inf

    @property
    def sup(self) -> np.ndarray:
        return self._sup

    @property
    def dim(self) -> int:
        return -1 if self.is_empty else self.inf.shape

    @property
    def is_empty(self) -> bool:
        return aux.is_empty(self.inf) or aux.is_empty(self.sup)

    @property
    def c(self) -> np.ndarray:
        assert not self.is_empty
        return (self._inf + self._sup) * 0.5

    @property
    def vertices(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def info(self):
        raise NotImplementedError

    # =============================================== class method
    @classmethod
    def functions(cls):
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

    # =============================================== operators
    def __contains__(self, item):
        raise NotImplementedError

    def __str__(self):
        return (
            "VVVVVVVVVVVVVVVVVVVVVVVVVVV Interval Info VVVVVVVVVVVVVVVVVVVVVVVVVVV\n"
            + str(self.inf)
            + "\n"
            + str(self.sup)
        )

    def __abs__(self):
        return Interval(abs(self.inf), abs(self.sup))

    def __len__(self):
        return self.dim  # override the len() functional operator

    def __add__(self, other):
        if isinstance(other, Interval):
            assert self.dim == other.dim
            return Interval(self.inf + other.inf, self.sup + other.sup)
        elif isinstance(other, (numbers.Real, np.ndarray)):
            return Interval(self.inf + other, self.sup + other)
        else:
            raise NotImplementedError

    def __getitem__(self, item):
        assert isinstance(item, (int, ArrayLike))
        return Interval(self.inf[item], self.sup[item])

    def __setitem__(self, key, value):
        assert isinstance(key, (int, ArrayLike))
        assert isinstance(value, (Interval, ArrayLike))
        if isinstance(value, Interval):
            self._inf[key] = value.inf
            self._sup[key] = value.sup
        elif isinstance(value, ArrayLike):
            self._inf[key] = value
            self._sup[key] = value
        else:
            raise NotImplementedError

    def __matmul__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

    def __pos__(self):
        raise NotImplementedError

    def __neg__(self):
        raise NotImplementedError

    def __or__(self, other):
        raise NotImplementedError

    def __sub__(self, other):
        raise NotImplementedError

    # =============================================== public method
    @staticmethod
    def empty(dim: int):
        raise NotImplementedError

    @staticmethod
    def proj(self, dims):
        raise NotImplementedError

    @staticmethod
    def rand(dim: int):
        raise NotImplementedError

    @staticmethod
    def reduce(self, method: str, order: int):
        raise NotImplementedError
