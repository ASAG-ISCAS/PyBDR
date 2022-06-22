from __future__ import annotations

from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from .geometry import Geometry

if TYPE_CHECKING:
    pass


class IntervalTensor(Geometry.Base):
    def __init__(self, inf: ArrayLike, sup: ArrayLike):
        inf = inf if isinstance(inf, np.ndarray) else np.asarray(inf, dtype=float)
        sup = sup if isinstance(sup, np.ndarray) else np.asarray(sup, dtype=float)
        assert inf.shape == sup.shape
        mask = np.logical_xor(np.isnan(inf), np.isnan(sup))  # NAN indicates empty
        assert np.all(inf[mask] <= sup[mask])
        self._inf = inf
        self._sup = sup
        self._type = Geometry.TYPE.INTERVAL_TENSOR

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
        info += "\n ------------- Interval END ------------- \n"
        return info

    def __str__(self):
        return self.info

    @property
    def type(self) -> Geometry.TYPE:
        return self._type

    # =============================================== operations
    def __getitem__(self, item):
        return IntervalTensor(self.inf[item], self.sup[item])

    def __setitem__(self, key, value):
        def _setitem_by_interval(x: IntervalTensor):
            self._inf[key] = x.inf
            self._sup[key] = x.sup

        def _setitem_by_number(x: (Real, np.ndarray)):
            self._inf[key] = x
            self._sup[key] = x

        if isinstance(value, IntervalTensor):
            _setitem_by_interval(value)
        elif isinstance(value, (Real, np.ndarray)):
            _setitem_by_number(value)
        else:
            raise NotImplementedError

    def __add__(self, other):
        def _add_interval(x: IntervalTensor):
            assert np.allclose(self.dim == x.dim)
            return IntervalTensor(self.inf + x.inf, self.sup + x.sup)

        if isinstance(other, (Real, np.ndarray)):
            return IntervalTensor(self.inf + other, self.sup + other)
        elif isinstance(other, Geometry.Base):
            if other.type == Geometry.TYPE.INTERVAL_TENSOR:
                return _add_interval(other)
            else:
                raise NotImplementedError

    def __radd__(self, other):
        return self + other

    def __iadd__(self, other):
        return self + other

    def __sub__(self, other):
        def _sub_interval(x: IntervalTensor):
            assert np.allclose(self.dim == x.dim)
            return IntervalTensor(self.inf - x.inf, self.sup - x.sup)

        if isinstance(other, (Real, np.ndarray)):
            return IntervalTensor(self.inf - other, self.sup - other)
        elif isinstance(other, Geometry.Base):
            if other.type == Geometry.TYPE.INTERVAL_TENSOR:
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
        return IntervalTensor(-self.sup, -self.inf)

    def __mul__(self, other):
        def _mul_interval(x: IntervalTensor):
            assert np.allclose(self.dim, x.dim)
            bd = np.stack(
                [self.inf * x.inf, self.inf * x.sup, self.sup * x.inf, self.sup * x.sup]
            )
            inf, sup = np.min(bd, axis=0), np.max(bd, axis=0)
            return IntervalTensor(inf, sup)

        def _mul_real(x: (Real, np.ndarray)):
            inff, supp = self.inf * x, self.sup * x
            inf, sup = np.minimum(inff, supp), np.maximum(inff, supp)
            return IntervalTensor(inf, sup)

        if isinstance(other, (Real, np.ndarray)):
            return _mul_real(other)
        elif isinstance(other, Geometry.Base):
            if other.type == Geometry.TYPE.INTERVAL_TENSOR:
                return _mul_interval(other)
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
                # empty set if [0,0]
                # [1/u,1/l] if 0 not in [l, u]
                # [1/u,+inf] if l==0 and u>0
                # [-inf,1/l] if l<0 and u==0
                # [-inf,+inf] if l<0 and u>0
                raise NotImplementedError
            else:
                return x * (1 / self)

        if isinstance(other, (Real, np.ndarray)):
            return _rtdiv_real(other)
        else:
            raise NotImplementedError

    def __itruediv__(self, other):
        return self / other

    def __matmul__(self, other):
        def _matmul_matrix(x: np.ndarray):
            posx, negx = x, np.zeros_like(x, dtype=float)
            posx[x < 0], negx[x < 0] = 0, x[x < 0]
            inf = self.inf @ posx + self.sup @ negx
            sup = self.sup @ posx + self.inf @ negx
            return IntervalTensor(inf, sup)

        if isinstance(other, np.ndarray):
            return _matmul_matrix(other)
        else:
            raise NotImplementedError

    def __rmatmul__(self, other):
        def _rmm_matrix(x: np.ndarray):
            posx, negx = x, np.zeros_like(x, dtype=float)
            posx[x < 0], negx[x < 0] = 0, x[x < 0]
            inf = posx @ self.inf + negx @ self.sup
            sup = posx @ self.sup + negx @ self.inf
            return IntervalTensor(inf, sup)

        if isinstance(other, np.ndarray):
            return _rmm_matrix(other)
        else:
            raise NotImplementedError

    # =============================================== non-periodic functions

    # =============================================== periodic functions
    @staticmethod
    def sin(x: IntervalTensor):
        # TODO
        raise NotImplementedError

    # =============================================== class method
    @classmethod
    def functional(cls):
        return {
            "sin": cls.sin,
        }

    # =============================================== static method
    @staticmethod
    def empty(dim):
        inf, sup = np.full(dim, np.nan, dtype=float), np.full(dim, np.nan, dtype=float)
        return IntervalTensor(inf, sup)

    @staticmethod
    def rand(shape):
        inf, sup = np.random.rand(shape), np.random.rand(shape)
        return IntervalTensor(inf, sup)

    # =============================================== public method
    def enclose(self, x):
        """
        enclose given data x by a interval tensor
        :param x:
        :return:
        """

        def _enclose_pts(pts: np.ndarray):
            inf, sup = np.minimum(pts), np.maximum(pts)
            return IntervalTensor(inf, sup)

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
        return IntervalTensor(self._inf.transpose(*axes), self._sup.transpose(*axes))

    def union(self, xs: [IntervalTensor]):
        """
        get the union of given IntervalTensors
        :param xs:
        :return:
        """
        # TODO
        raise NotImplementedError

    def intersection(self, xs: [IntervalTensor]):
        """
        get the intersections of given IntervalTensors
        :param xs:
        :return:
        """
        # TODO
        raise NotImplementedError

    def contains(self, x):
        """
        check if given data inside the domain specified by this interval tensor
        :param x:
        :return:
        """
        # TODO
        raise NotImplementedError
