from __future__ import annotations

from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from .geometry import Geometry

if TYPE_CHECKING:
    pass


class Interval(Geometry.Base):
    def __init__(self, inf: ArrayLike, sup: ArrayLike):
        inf = inf if isinstance(inf, np.ndarray) else np.asarray(inf, dtype=float)
        sup = sup if isinstance(sup, np.ndarray) else np.asarray(sup, dtype=float)
        assert inf.shape == sup.shape
        mask = np.logical_not(np.isnan(inf) | np.isnan(sup))  # NAN indicates empty
        assert np.all(inf[mask] <= sup[mask])
        self._inf = inf
        self._sup = sup
        self._type = Geometry.TYPE.INTERVAL

    # =============================================== property
    @property
    def c(self) -> np.ndarray:
        """
        center of this interval
        :return:
        """
        return (self._inf + self._sup) * 0.5

    @property
    def rad(self) -> np.ndarray:
        """
        radius of this interval
        :return:
        """
        return (self._sup - self._inf) * 0.5

    @property
    def inf(self) -> np.ndarray:
        return self._inf

    @property
    def sup(self) -> np.ndarray:
        return self._sup

    @property
    def T(self):
        """
        shorthand for transpose the interval
        :return:
        """
        return self.transpose()

    @property
    def dim(self) -> tuple:
        return self._inf.shape

    @property
    def is_empty(self) -> np.ndarray:
        return np.logical_or(np.isnan(self._inf), np.isnan(self._sup))

    @property
    def vertices(self) -> np.ndarray:
        # TODO
        raise NotImplementedError

    @property
    def info(self):
        info = "\n ------------- Interval BEGIN ------------- \n"
        info += ">>> dimension \n"
        info += str(self.dim) + "\n"
        info += str(self.inf) + "\n"
        info += str(self.sup) + "\n"
        info += "\n ------------- Interval END ------------- \n"
        return info

    def __str__(self):
        return self.info

    @property
    def type(self) -> Geometry.TYPE:
        return self._type

    # =============================================== operations
    def __getitem__(self, item):
        inf, sup = self.inf[item], self.sup[item]
        inf = inf if isinstance(inf, np.ndarray) else [inf]
        sup = sup if isinstance(sup, np.ndarray) else [sup]
        return Interval(inf, sup)

    def __setitem__(self, key, value):
        def _setitem_by_interval(x: Interval):
            self._inf[key] = x.inf
            self._sup[key] = x.sup

        def _setitem_by_number(x: (Real, np.ndarray)):
            self._inf[key] = x
            self._sup[key] = x

        if isinstance(value, Interval):
            _setitem_by_interval(value)
        elif isinstance(value, (Real, np.ndarray)):
            _setitem_by_number(value)
        else:
            raise NotImplementedError

    def __add__(self, other):
        def _add_interval(x: Interval):
            assert np.allclose(self.dim, x.dim)
            return Interval(self.inf + x.inf, self.sup + x.sup)

        if isinstance(other, (Real, np.ndarray)):
            return Interval(self.inf + other, self.sup + other)
        elif isinstance(other, Geometry.Base):
            if other.type == Geometry.TYPE.INTERVAL:
                return _add_interval(other)
            else:
                raise NotImplementedError

    def __radd__(self, other):
        return self + other

    def __iadd__(self, other):
        return self + other

    def __sub__(self, other):
        def _sub_interval(x: Interval):
            assert np.allclose(self.dim, x.dim)
            return Interval(self.inf - x.sup, self.sup - x.inf)

        if isinstance(other, (Real, np.ndarray)):
            return Interval(self.inf - other, self.sup - other)
        elif isinstance(other, Geometry.Base):
            if other.type == Geometry.TYPE.INTERVAL:
                return _sub_interval(other)
            else:
                raise NotImplementedError

    def __rsub__(self, other):
        raise NotImplementedError

    def __isub__(self, other):
        return self - other

    def __pos__(self):
        return self

    def __neg__(self):
        return Interval(-self.sup, -self.inf)

    def __mul__(self, other):
        def _mul_interval(x: Interval):
            assert np.allclose(self.dim, x.dim)
            bd = np.stack(
                [self.inf * x.inf, self.inf * x.sup, self.sup * x.inf, self.sup * x.sup]
            )
            inf, sup = np.min(bd, axis=0), np.max(bd, axis=0)
            return Interval(inf, sup)

        def _mul_real(x: (Real, np.ndarray)):
            inff, supp = self.inf * x, self.sup * x
            inf, sup = np.minimum(inff, supp), np.maximum(inff, supp)
            return Interval(inf, sup)

        if isinstance(other, (Real, np.ndarray)):
            return _mul_real(other)
        elif isinstance(other, Geometry.Base):
            if other.type == Geometry.TYPE.INTERVAL:
                return _mul_interval(other)
            elif other.type == Geometry.TYPE.ZONOTOPE:
                return NotImplemented
            else:
                raise NotImplementedError

    def __rmul__(self, other):
        if isinstance(other, (Real, np.ndarray)):
            return self * other
        else:
            raise NotImplementedError

    def __imul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (1 / other)

    def __rtruediv__(self, other):
        def _rtdiv_real(x: (Real, np.ndarray)):
            if x == 1:
                inf = np.full_like(self.inf, np.nan)
                sup = np.full_like(self.sup, np.nan)
                ind0, ind1 = self._inf < 0, self._sup > 0
                # empty set if [0,0] by default

                # [1/u,1/l] if 0 not in [l, u]
                ind = (self._inf > 0) | (self._sup < 0)
                inf[ind] = 1 / self._sup[ind]
                sup[ind] = 1 / self._inf[ind]

                # [1/u,+inf] if l==0 and u>0
                ind = (self._inf == 0) & ind1
                inf[ind] = 1 / self._sup[ind]
                sup[ind] = np.inf

                # [-inf,1/l] if l<0 and u==0
                ind = ind0 & (self._sup == 0)
                inf[ind] = -np.inf
                sup[ind] = 1 / self._inf[ind]

                # [-inf,+inf] if l<0 and u>0
                ind = ind0 & ind1
                inf[ind] = -np.inf
                sup[ind] = np.inf

                return Interval(inf, sup)
            else:
                return x * (1 / self)

        if isinstance(other, (Real, np.ndarray)):
            return _rtdiv_real(other)
        else:
            raise NotImplementedError

    def __itruediv__(self, other):
        return self / other

    # =============================================== non-periodic functions

    def __matmul__(self, other):
        def _matmul_matrix(x: np.ndarray):
            posx, negx = x, np.zeros_like(x, dtype=float)
            posx[x < 0] = 0
            negx[x < 0] = x[x < 0]
            inf = self.inf @ posx + self.sup @ negx
            sup = self.sup @ posx + self.inf @ negx
            return Interval(inf, sup)

        def _matmul_interval(x: Interval):
            def mmm(la, lb, ra, rb):
                lhs = la[..., np.newaxis] * lb[np.newaxis, ...]
                rhs = ra[..., np.newaxis] * rb[np.newaxis, ...]
                c = np.maximum(lhs, rhs)
                return np.sum(c, axis=-2)

            def posneg(m):
                pos, neg = m, -m
                pos[pos < 0] = 0
                neg[neg < 0] = 0
                return pos, neg

            (linfp, linfn), (lsupp, lsupn) = posneg(self.inf), posneg(self.sup)
            (rinfp, rinfn), (rsupp, rsupn) = posneg(x.inf), posneg(x.sup)
            inf = mmm(linfp, rinfp, lsupn, rsupn) - mmm(lsupp, rinfn, linfn, rsupp)
            sup = mmm(lsupp, rsupp, linfn, rinfn) - mmm(linfp, rsupn, lsupn, rinfp)
            return Interval(inf, sup)

        if isinstance(other, np.ndarray):
            return _matmul_matrix(other)
        elif isinstance(other, Interval):
            return _matmul_interval(other)
        else:
            raise NotImplementedError

    def __rmatmul__(self, other):
        def _rmm_matrix(x: np.ndarray):
            posx, negx = x, np.zeros_like(x, dtype=float)
            posx[x < 0] = 0
            negx[x < 0] = x[x < 0]
            inf = posx @ self.inf + negx @ self.sup
            sup = posx @ self.sup + negx @ self.inf
            return Interval(inf, sup)

        if isinstance(other, np.ndarray):
            return _rmm_matrix(other)
        else:
            raise NotImplementedError

    def __imatmul__(self, other):
        return self @ other

    def __abs__(self):
        inf, sup = self.inf, self.sup

        ind = self._sup < 0
        inf[ind], sup[ind] = abs(self._sup[ind]), abs(self._inf[ind])

        ind = (self._inf <= 0) & (self._sup >= 0)
        inf[ind] = 0
        sup[ind] = np.maximum(abs(self._inf[ind]), abs(self._sup[ind]))

        return Interval(inf, sup)

    def __pow__(self, power, modulo=None):
        def _pow_int(x: int):
            if x >= 0:
                inff, supp = self.inf**x, self.sup**x
                inf, sup = np.minimum(inff, supp), np.maximum(inff, supp)
                if x % 2 == 0 and x != 0:
                    ind = (self._inf <= 0) & (self._sup >= 0)
                    inf[ind] = 0
                return Interval(inf, sup)
            else:
                return (1 / self) ** (-x)

        def _pow_real(x):
            if x >= 0:
                inf, sup = self.inf**x, self.sup**x
                ind = self._inf < 0
                inf[ind] = np.nan
                sup[ind] = np.nan
                return Interval(inf, sup)
            else:
                return (1 / self) ** (-x)

        def _pow_num(x):
            if abs(round(x) - x) <= np.finfo(np.float).eps:
                return _pow_int(int(x))
            else:
                return _pow_real(x)

        if isinstance(power, (Real, int)):
            return _pow_num(power)
        else:
            raise NotImplementedError

    def __rpow__(self, other):
        raise NotImplementedError

    def __ipow__(self, other):
        return self**other

    @staticmethod
    def exp(x: Interval):
        return Interval(np.exp(x.inf), np.exp(x.sup))

    @staticmethod
    def log(x: Interval):
        inf, sup = np.log(x.inf), np.log(x.sup)

        ind = (x.inf < 0) & (x.sup >= 0)
        inf[ind] = np.nan

        ind = x.sup < 0
        inf[ind] = np.nan
        sup[ind] = np.nan

        return Interval(inf, sup)

    @staticmethod
    def sqrt(x: Interval):
        inf, sup = np.sqrt(x.inf), np.sqrt(x.sup)

        ind = (x.inf < 0) & (x.sup >= 0)
        inf[ind] = np.nan

        ind = x.sup < 0
        inf[ind] = np.nan
        sup[ind] = np.nan

        return Interval(inf, sup)

    @staticmethod
    def arcsin(x: Interval):

        inf, sup = np.arcsin(x.inf), np.arcsin(x.sup)

        ind = (x.inf >= -1) & (x.inf <= 1) & (x.sup > 1)
        sup[ind] = np.nan

        ind = (x.inf < -1) & (x.sup >= -1) & (x.sup <= 1)
        inf[ind] = np.nan

        ind = (x.inf < -1) & (x.sup > 1)
        inf[ind] = np.nan
        sup[ind] = np.nan

        return Interval(inf, sup)

    @staticmethod
    def arccos(x: Interval):
        inf, sup = np.arccos(x.sup), np.arccos(x.inf)

        ind = (x.inf >= -1) & (x.inf <= 1) & (x.sup > 1)
        sup[ind] = np.nan

        ind = (x.inf < -1) & (x.sup >= -1) & (x.sup <= 1)
        inf[ind] = np.nan

        ind = (x.inf < -1) & (x.sup > 1)
        inf[ind] = np.nan
        sup[ind] = np.nan

        return Interval(inf, sup)

    @staticmethod
    def arctan(x: Interval):
        return Interval(np.arctan(x.inf), np.arctan(x.sup))

    @staticmethod
    def sinh(x: Interval):
        return Interval(np.sinh(x.inf), np.sinh(x.sup))

    @staticmethod
    def cosh(x: Interval):
        inf, sup = np.cosh(x.sup), np.cosh(x.inf)

        ind = (x.inf <= 0) & (x.sup >= 0)
        inf[ind] = 1
        sup[ind] = np.cosh(np.maximum(abs(x.inf[ind]), abs(x.sup[ind])))

        ind = x.inf > 0
        inf[ind] = np.cosh(x.inf[ind])
        sup[ind] = np.cosh(x.sup[ind])

        return Interval(inf, sup)

    @staticmethod
    def tanh(x: Interval):
        return Interval(np.tanh(x.inf), np.tanh(x.sup))

    @staticmethod
    def arcsinh(x: Interval):
        return Interval(np.arcsinh(x.inf), np.arcsinh(x.sup))

    @staticmethod
    def arccosh(x: Interval):
        inf, sup = np.arccosh(x.inf), np.arccosh(x.sup)

        ind = (x.inf < 1) & (x.sup >= 1)
        inf[ind] = np.nan

        ind = x.sup < 1
        inf[ind] = np.nan
        sup[ind] = np.nan

        return Interval(inf, sup)

    @staticmethod
    def arctanh(x: Interval):
        inf, sup = np.arctanh(x.inf), np.arctanh(x.sup)

        ind = (x.inf > -1) & (x.inf < 1) & (x.sup >= 1)
        sup[ind] = np.nan

        ind = (x.inf <= -1) & (x.sup > -1) & (x.sup < 1)
        inf[ind] = np.nan

        ind = (x.inf <= -1) & (x.sup >= 1)
        inf[ind] = np.nan
        sup[ind] = np.nan

        return Interval(inf, sup)

    # =============================================== periodic functions
    @staticmethod
    def sin(x: Interval):
        ind0 = (x.sup - x.inf) >= 2 * np.pi  # xsup -xinf >= 2*pi
        yinf, ysup = np.mod(x.inf, np.pi * 2), np.mod(x.sup, np.pi * 2)

        ind1 = yinf < np.pi * 0.5  # yinf in R1
        ind2 = ysup < np.pi * 0.5  # ysup in R1
        ind3 = np.logical_not(ind1) & (yinf < np.pi * 1.5)  # yinf in R2
        ind4 = np.logical_not(ind2) & (ysup < np.pi * 1.5)  # ysup in R2
        ind5 = yinf >= np.pi * 1.5  # yinf in R3
        ind6 = ysup >= np.pi * 1.5  # ysup in R3
        ind7 = yinf > ysup  # yinf > ysup
        ind8 = np.logical_not(ind7)  # yinf <=ysup

        inf, sup = x.inf, x.sup

        ind = (ind1 & ind2 & ind8) | (ind5 & ind2) | (ind5 & ind6 & ind8)
        inf[ind] = np.sin(yinf[ind])
        sup[ind] = np.sin(ysup[ind])

        ind = (ind1 & ind4) | (ind5 & ind4)
        inf[ind] = np.minimum(np.sin(yinf[ind]), np.sin(ysup[ind]))
        sup[ind] = 1

        ind = (ind3 & ind2) | (ind3 & ind6)
        inf[ind] = -1
        sup[ind] = np.maximum(np.sin(yinf[ind]), np.sin(ysup[ind]))

        ind = ind3 & ind4 & ind8
        inf[ind] = np.sin(ysup[ind])
        sup[ind] = np.sin(yinf[ind])

        ind = (
            ind0
            | (ind1 & ind2 & ind7)
            | (ind1 & ind6)
            | (ind3 & ind4 & ind7)
            | (ind5 & ind6 & ind7)
        )
        inf[ind] = -1
        sup[ind] = 1

        return Interval(inf, sup)

    @staticmethod
    def cos(x: Interval):
        ind0 = (x.sup - x.inf) >= 2 * np.pi  # xsup -xinf >= 2*pi
        yinf, ysup = np.mod(x.inf, np.pi * 2), np.mod(x.sup, np.pi * 2)

        ind1 = yinf < np.pi  # yinf in R1
        ind2 = ysup < np.pi  # ysup in R1
        ind3 = np.logical_not(ind1)  # yinf in R2
        ind4 = np.logical_not(ind2)  # ysup in R2
        ind5 = yinf > ysup  # yinf > ysup
        ind6 = np.logical_not(ind5)  # yinf <= ysup

        inf, sup = x.inf, x.sup

        ind = ind3 & ind4 & ind6
        inf[ind] = np.cos(yinf[ind])
        sup[ind] = np.cos(ysup[ind])

        ind = ind3 & ind2
        inf[ind] = np.minimum(np.cos(yinf[ind]), np.cos(ysup[ind]))
        sup[ind] = 1

        ind = ind1 & ind4
        inf[ind] = -1
        sup[ind] = np.maximum(np.cos(yinf[ind]), np.cos(ysup[ind]))

        ind = ind1 & ind2 & ind6
        inf[ind] = np.cos(ysup[ind])
        sup[ind] = np.cos(yinf[ind])

        ind = ind0 | (ind1 & ind2 & ind5) | (ind3 & ind4 & ind5)
        inf[ind] = -1
        sup[ind] = 1

        return Interval(inf, sup)

    @staticmethod
    def tan(x: Interval):
        ind0 = (x.sup - x.inf) >= np.pi  # xsup -xinf >= pi
        zinf, zsup = np.mod(x.inf, np.pi), np.mod(x.sup, np.pi)

        ind1 = zinf < np.pi * 0.5  # zinf in R1
        ind2 = zsup < np.pi * 0.5  # zsup in R1
        ind3 = np.logical_not(ind1)  # zinf in R2
        ind4 = np.logical_not(ind2)  # zsup in R2
        ind5 = zinf > zsup  # zinf > zsup
        ind6 = np.logical_not(ind5)  # zinf <= zsup

        inf, sup = x.inf, x.sup

        # different from ref ??? TODO need check
        ind = (ind1 & ind2 & ind6) | (ind3 & ind4 & ind6) | ind3 & (ind6 | ind2)
        inf[ind] = np.tan(zinf[ind])
        sup[ind] = np.tan(zsup[ind])

        ind = ind0 | (ind1 & ind2 & ind5) | (ind3 & ind4 & ind5) | (ind1 & ind4)
        inf[ind] = -np.inf
        sup[ind] = np.inf

        return Interval(inf, sup)

    @staticmethod
    def cot(x: Interval):
        # TODO need check
        ind0 = (x.sup - x.inf) >= np.pi  # xsup -xinf >= pi
        zinf, zsup = np.mod(x.inf, np.pi), np.mod(x.sup, np.pi)

        inf, sup = x.inf, x.sup

        ind = zinf <= zsup
        inf[ind] = 1 / np.tan(zsup[ind])
        sup[ind] = 1 / np.tan(zinf[ind])

        ind = ind0 | (zinf > zsup)
        inf[ind] = -np.inf
        sup[ind] = np.inf

        return Interval(inf, sup)

    # =============================================== class method
    @classmethod
    def functional(cls):
        return {
            "exp": cls.exp,
            "log": cls.log,
            "sqrt": cls.sqrt,
            "arcsin": cls.arcsin,
            "arccos": cls.arccos,
            "arctan": cls.arctan,
            "sinh": cls.sinh,
            "cosh": cls.cosh,
            "tanh": cls.tanh,
            "arcsinh": cls.arcsinh,
            "arccosh": cls.arccosh,
            "arctanh": cls.arctanh,
            "sin": cls.sin,
            "cos": cls.cos,
            "tan": cls.tan,
        }

    # =============================================== static method
    @staticmethod
    def empty(dim):
        inf, sup = np.full(dim, np.nan, dtype=float), np.full(dim, np.nan, dtype=float)
        return Interval(inf, sup)

    @staticmethod
    def rand(shape):
        inf, sup = np.random.rand(shape), np.random.rand(shape)
        inf, sup = np.minimum(inf, sup), np.maximum(inf, sup)
        return Interval(inf, sup)

    @staticmethod
    def zeros(shape):
        inf, sup = np.zeros(shape, dtype=float), np.zeros(shape, dtype=float)
        return Interval(inf, sup)

    @staticmethod
    def ones(shape):
        inf, sup = np.ones(shape, dtype=float), np.ones(shape, dtype=float)
        return Interval(inf, sup)

    # =============================================== public method
    def enclose(self, x):
        """
        enclose given data x by a interval
        :param x:
        :return:
        """

        def _enclose_pts(pts: np.ndarray):
            inf, sup = np.minimum(pts), np.maximum(pts)
            return Interval(inf, sup)

        if isinstance(x, np.ndarray):
            return _enclose_pts(x)
        else:
            raise NotImplementedError

    def reduce(self):
        raise NotImplementedError

    def proj(self, dims):
        raise NotImplementedError

    def boundary(self, max_dist: float, element: Geometry.TYPE):
        # TODO
        raise NotImplementedError

    def transpose(self, *axes):
        return Interval(self._inf.transpose(*axes), self._sup.transpose(*axes))

    def union(self, xs: [Interval]):
        """
        get the union of given intevals
        :param xs:
        :return:
        """
        # TODO
        raise NotImplementedError

    def intersection(self, xs: [Interval]):
        """
        get the intersections of given intervals
        :param xs:
        :return:
        """
        # TODO
        raise NotImplementedError

    def contains(self, x):
        """
        check if given data inside the domain specified by this interval
        :param x:
        :return:
        """
        # TODO
        raise NotImplementedError
