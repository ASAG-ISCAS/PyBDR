from __future__ import annotations

from numbers import Real

import numpy as np
from numpy.typing import ArrayLike

import pyrat.util.functional.auxiliary as aux
from .geometry import Geometry


class Interval(Geometry.Base):
    def __init__(self, inf: ArrayLike, sup: ArrayLike):
        inf = (
            inf
            if isinstance(inf, np.ndarray)
            else np.array(inf, dtype=float).reshape(-1)
        )
        sup = (
            sup
            if isinstance(sup, np.ndarray)
            else np.array(sup, dtype=float).reshape(-1)
        )
        assert inf.shape == sup.shape
        assert inf.ndim == 1
        assert np.all(inf <= sup)
        self._inf = inf
        self._sup = sup
        self._type = Geometry.TYPE.INTERVAL

    # =============================================== property
    @property
    def inf(self) -> np.ndarray:
        return self._inf

    @property
    def sup(self) -> np.ndarray:
        return self._sup

    @property
    def dim(self) -> int:
        return None if self.is_empty else self.inf.shape[0]

    @property
    def is_empty(self) -> bool:
        return aux.is_empty(self.inf) or aux.is_empty(self.sup)

    @property
    def c(self) -> np.ndarray:
        assert not self.is_empty
        return (self.inf + self.sup) * 0.5

    @property
    def bd(self) -> np.ndarray:
        return np.vstack([self.inf, self.sup])

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
    def type(self) -> Geometry.TYPE:
        return self._type

    # =============================================== class method
    @classmethod
    def functional(cls):
        return {
            "__add__": cls.__add__,
            "__mul__": cls.__mul__,
            "__matmul__": cls.__matmul__,
        }

    # =============================================== operators
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
        return Interval(inf, sup)

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        if isinstance(other, Interval):
            assert self.dim == other.dim
            return Interval(self.inf + other.inf, self.sup + other.sup)
        elif isinstance(other, (Real, np.ndarray)):
            return Interval(self.inf + other, self.sup + other)
        else:
            raise NotImplementedError

    def __radd__(self, other):
        return self + other

    def __iadd__(self, other):
        return self + other

    def __getitem__(self, item):
        return Interval(self.inf[item], self.sup[item])

    def __setitem__(self, key, value):
        if isinstance(value, Interval):
            self._inf[key] = value.inf
            self._sup[key] = value.sup
        else:
            self._inf[key] = value
            self._sup[key] = value

    def __matmul__(self, other):
        if isinstance(other, Real):
            return self * other
        else:
            raise NotImplementedError

    def __rmatmul__(self, other):
        raise NotImplementedError("Use 'other@interval' instead")

    def __imatmul__(self, other):
        return self @ other

    def __mul__(self, other):
        def __mul_interval(rhs: Interval):
            bd = np.vstack(
                [
                    self.inf * rhs.inf,
                    self.inf * rhs.sup,
                    self.sup * rhs.inf,
                    self.sup * rhs.sup,
                ]
            )
            inf, sup = np.min(bd, axis=0), np.max(bd, axis=0)
            return Interval(inf, sup)

        def __mul_array_like(rhs: ArrayLike):
            rhs = np.array(rhs) if isinstance(rhs, (list, tuple)) else rhs
            infm, supm = self.inf * rhs, self.sup * rhs
            inf, sup = np.minimum(infm, supm), np.maximum(infm, supm)
            return Interval(inf, sup)

        if isinstance(other, (Real, np.ndarray, list, tuple)):
            return __mul_array_like(other)
        elif isinstance(other, Geometry.Base):
            if other.type == Geometry.TYPE.INTERVAL:
                return __mul_interval(other)
            elif other.type == Geometry.TYPE.ZONOTOPE:
                return other * self
        else:
            raise NotImplementedError("Invalid rhs operand")

    def __rmul__(self, other):
        return self * other

    def __imul__(self, other):
        return self * other

    def __pos__(self):
        return self

    def __neg__(self):
        return self * -1

    def __or__(self, other):
        raise NotImplementedError

    def __ror__(self, other):
        raise NotImplementedError

    def __ior__(self, other):
        raise NotImplementedError

    def __sub__(self, other):
        raise NotImplementedError

    def __rsub__(self, other):
        raise NotImplementedError

    def __isub__(self, other):
        raise NotImplementedError

    def __truediv__(self, other):
        raise NotImplementedError

    def __rtruediv__(self, other):
        def __rt_real(rhs: Real):
            infrt, suprt = rhs / self.inf, rhs / self.sup
            inf, sup = np.minimum(infrt, suprt), np.maximum(infrt, suprt)

            ind = (self.inf == 0) & (self.sup == 0)
            inf[ind] = np.nan
            sup[ind] = np.nan

            ind = self.sup == 0
            inf[ind] = -np.inf
            sup[ind] = other / self.sup[ind]

            ind = (self.inf < 0) & (self.sup > 0)
            inf[ind] = -np.inf
            sup[ind] = +np.inf

            return Interval(inf, sup)

        if isinstance(other, Real):
            return __rt_real(other)
        elif isinstance(other, Geometry.Base):
            raise NotImplementedError  # TODO
        else:
            raise NotImplementedError

    def __pow__(self, power, modulo=None):
        def __pow_int(exp):
            if exp >= 0:  # positive integer exponent
                infp, supp = self.inf**exp, self.sup**exp
                inf, sup = np.minimum(infp, supp), np.maximum(infp, supp)
                # modification for even exponent
                if exp % 2 == 0 and exp != 0:
                    ind = (self.inf < 0) & (self.sup > 0)
                    inf[ind] = 0
                return Interval(inf, sup)
            else:  # negative integer exponent
                return (1 / self) ** (-exp)

        def __pow_real(exp):
            if exp >= 0:  # positive real valued exponent
                inf, sup = self.inf**exp, self.sup**exp
                ind = self.inf < 0
                inf[ind] = np.nan
                sup[ind] = np.nan
                return Interval(inf, sup)
            else:  # negative real valued exponent
                return (1 / self) ** (-exp)

        def __pow_num(exp):
            if abs(round(exp) - exp) <= np.finfo(np.float).eps:
                return __pow_int(int(exp))
            else:
                return __pow_real(exp)

        if isinstance(power, Real):
            return __pow_num(power)
        elif isinstance(power, Geometry.Base):
            raise NotImplementedError  # TODO
        else:
            raise NotImplementedError

    def __rpow__(self, other):
        def __rp_interval(denominator):
            raise NotImplementedError

        if isinstance(other, Real):
            inf, sup = other**self.inf, other**self.sup
            return Interval(inf, sup)
        elif isinstance(other, Geometry.Base):
            if other.type == Geometry.TYPE.INTERVAL:
                return __rp_interval(other)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    # =============================================== public method
    def reduce(self, method: str, order: int):
        raise NotImplementedError

    def proj(self, dims):
        raise NotImplementedError

    # =============================================== static method
    @staticmethod
    def empty(dim: int):
        raise NotImplementedError

    @staticmethod
    def rand(dim: int):
        raise NotImplementedError
