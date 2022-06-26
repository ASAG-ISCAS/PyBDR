from __future__ import annotations

import numpy as np
from numbers import Real
from numpy.typing import ArrayLike
import pyrat.util.functional.auxiliary as aux
from pyrat.geometry.geometry import Geometry
from typing import TYPE_CHECKING


if TYPE_CHECKING:  # for type hint
    pass


class IntervalMatrix(Geometry.Base):
    def __init__(self, inf: ArrayLike, sup: ArrayLike):
        inf = inf if isinstance(inf, np.ndarray) else np.array(inf, dtype=float)
        sup = sup if isinstance(sup, np.ndarray) else np.array(sup, dtype=float)
        assert inf.shape == sup.shape
        assert inf.ndim == 2
        assert np.all(inf <= sup)
        self._inf = inf
        self._sup = sup
        self._type = Geometry.TYPE.INTERVAL_MATRIX

    # =============================================== property
    @property
    def inf(self):
        return self._inf

    @property
    def sup(self):
        return self._sup

    @property
    def c(self):
        return (self.inf + self.sup) * 0.5

    @property
    def rad(self):
        return (self.sup - self.inf) * 0.5

    @property
    def is_empty(self) -> bool:
        return aux.is_empty(self.inf) or aux.is_empty(self.sup)

    @property
    def is_zero(self) -> bool:
        return np.all(abs(self.sup - self.inf) <= np.finfo(float).eps) and np.all(
            abs(self.sup) <= np.finfo(float).eps
        )

    @property
    def vertices(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def info(self):
        info = "\n ------------- Interval BEGIN ------------- \n"
        info += ">>> dimension -- inf -- sup\n"
        info += str(self.dim) + "\n"
        info += str(self.inf) + "\n"
        info += str(self.sup) + "\n"
        info += "\n ------------- Interval END ------------- \n"
        return info

    @property
    def dim(self):
        return None if self.is_empty else self.inf.shape

    @property
    def type(self) -> Geometry.TYPE:
        return self._type

    # =============================================== class method
    @classmethod
    def functional(cls):
        raise NotImplementedError

    # =============================================== operator
    def __contains__(self, item):
        raise NotImplementedError

    def __str__(self):
        return self.info

    def __abs__(self):
        np_idx = (self.inf < 0) & (self.sup > 0)
        nn_idx = (self.inf < 0) & (self.sup <= 0)
        inf, sup = self.inf, self.sup
        sup[np_idx] = np.maximum(abs(self.inf[np_idx]), abs(self.sup[np_idx]))
        inf[np_idx] = 0
        sup[nn_idx] = abs(self.inf[nn_idx])
        sup[nn_idx] = abs(self.sup[nn_idx])
        return IntervalMatrix(inf, sup)

    def __add__(self, other):
        if isinstance(other, IntervalMatrix):
            assert self.dim == other.dim
            return IntervalMatrix(self.inf + other.inf, self.sup + other.sup)
        elif isinstance(other, (Real, np.ndarray)):
            return IntervalMatrix(self.inf + other, self.sup + other)
        else:
            raise NotImplementedError

    def __radd__(self, other):
        return self + other

    def __iadd__(self, other):
        return self + other

    def __getitem__(self, item):
        return IntervalMatrix(self.inf[item], self.sup[item])

    def __setitem__(self, key, value):
        if isinstance(value, IntervalMatrix):
            self._inf[key] = value.inf
            self._sup[key] = value.sup
        else:
            self._inf[key] = value
            self._sup[key] = value

    def __matmul__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        def __mul_real(rhs: Real):
            infm, supm = self.inf * rhs, self.sup * rhs
            inf, sup = np.minimum(infm, supm), np.maximum(infm, supm)
            return IntervalMatrix(inf, sup)

        def __mul_interval_matrix(rhs: IntervalMatrix):
            bd = np.stack(
                [
                    self.inf * rhs.inf,
                    self.inf * rhs.sup,
                    self.sup * rhs.inf,
                    self.sup * rhs.sup,
                ]
            )
            inf, sup = np.min(bd, axis=0), np.max(bd, axis=0)
            return IntervalMatrix(inf, sup)

        if isinstance(other, Real):
            return __mul_real(other)
        elif isinstance(other, Geometry.Base):
            if other.type == Geometry.TYPE.ZONOTOPE:
                return NotImplemented
            elif other.type == Geometry.TYPE.INTERVAL_MATRIX:
                return __mul_interval_matrix(other)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def __rmul__(self, other):
        if isinstance(other, Geometry.Base):
            if other.type == Geometry.TYPE.INTERVAL:
                return NotImplemented
        else:
            raise NotImplementedError

    def __neg__(self):
        raise NotImplementedError

    def __or__(self, other):
        raise NotImplementedError

    def __pos__(self):
        return self

    def __sub__(self, other):
        raise NotImplementedError

    # =============================================== public method
    def reduce(self, method: str, order: int):
        raise NotImplementedError

    def proj(self, dims):
        raise NotImplementedError

    def boundary(self, max_dist: float, element: Geometry.TYPE):
        raise NotImplementedError

    # =============================================== static method
    @staticmethod
    def empty(dim: int):
        raise NotImplementedError

    @staticmethod
    def rand(dim: int):
        raise NotImplementedError
