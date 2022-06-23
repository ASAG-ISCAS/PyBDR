from __future__ import annotations

import itertools
from itertools import chain
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

import pyrat.util.functional.auxiliary as aux
from .geometry import Geometry

if TYPE_CHECKING:
    from .interval_matrix import IntervalMatrix  # for type hint, easy coding
    from .zonotope import Zonotope


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
        self._vertices = None
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
    def is_zero(self) -> bool:
        return np.all(abs(self.sup - self.inf) <= np.finfo(float).eps) and np.all(
            abs(self.sup) <= np.finfo(float).eps
        )

    @property
    def c(self) -> np.ndarray:
        assert not self.is_empty
        return (self.inf + self.sup) * 0.5

    @property
    def rad(self) -> np.ndarray:
        assert not self.is_empty
        return (self.sup - self.inf) * 0.5

    @property
    def bd(self) -> np.ndarray:
        return np.vstack([self.inf, self.sup]).T

    @property
    def vertices(self) -> np.ndarray:
        if self._vertices is None:
            col = np.asarray(list(itertools.product(np.arange(2), repeat=self.dim)))
            row = np.tile(np.arange(self.dim), col.shape[0])
            self._vertices = self.bd[row, col.reshape(-1)].reshape((-1, self.dim))

        return self._vertices

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
            "__sub__": cls.__sub__,
            "__mul__": cls.__mul__,
            "__matmul__": cls.__matmul__,
            "sin": cls.sin,
        }

    # =============================================== operators
    def __contains__(self, item):
        def __contains_pts(pts: np.ndarray):
            assert pts.ndim == 1 or pts.ndim == 2
            if pts.ndim == 1:
                if pts.shape[0] != self.dim:
                    return False
                return (self.inf <= pts) & (self.sup >= pts)
            if pts.ndim == 2:
                if pts.shape[1] != self.dim:
                    return np.full(pts.shape[0], False, dtype=bool)
                return (self.inf <= pts[:]) & (self.sup >= pts[:])

        def __contains_interval(other: Interval):
            if self.dim != other.dim:
                return False
            return np.all(self.inf <= other.inf) and np.all(self.sup >= other.sup)

        def __contains_zonotope(other: Zonotope):
            from .operation.convert import cvt2

            return other in cvt2(self, Geometry.TYPE.POLYTOPE)

        if isinstance(item, np.ndarray):
            return __contains_pts(item)
        elif isinstance(item, Geometry.Base):
            if item.type == Geometry.TYPE.INTERVAL:
                return __contains_interval(item)
            elif item.type == Geometry.TYPE.ZONOTOPE:
                return __contains_zonotope(item)
            else:
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
        inf[nn_idx] = abs(self.sup[nn_idx])
        return Interval(inf, sup)

    @staticmethod
    def sin(x: Interval):

        inf, sup = x.inf, x.sup

        ind = x.sup - x.inf >= 2 * np.pi
        inf[ind] = -1
        sup[ind] = 1

        inf = np.mod(inf, 2 * np.pi)
        sup = np.mod(sup, 2 * np.pi)

        # inf in [0,pi/2]
        ind0 = (sup - inf) <= 2 * np.pi
        ind1 = inf <= 0.5 * np.pi
        ind2 = sup <= 0.5 * np.pi
        ind3 = sup <= 1.5 * np.pi
        ind4 = sup < inf

        ind = ind0 & ind1 & ind4
        inf[ind] = -1
        sup[ind] = 1

        ind = ind0 & ind1 & ind2 & (np.logical_not(ind4))
        inf[ind] = np.sin(inf[ind])
        sup[ind] = np.sin(sup[ind])

        ind = ind0 & ind1 & np.logical_not(ind2) & ind3
        inf[ind] = np.minimum(np.sin(inf[ind]), np.sin(sup[ind]))
        sup[ind] = 1

        ind = ind0 & ind1 & np.logical_not(ind4)
        inf[ind] = -1
        sup[ind] = 1

        # inf in [pi/2,3/2*pi]
        ind1 = np.logical_not(ind1)
        ind2 = inf <= 1.5 * np.pi
        ind3 = sup > 0.5 * np.pi
        ind4 = sup <= 1.5 * np.pi
        ind5 = sup < inf

        ind = ind0 & ind1 & ind2 & ind3 & ind5
        inf[ind] = -1
        sup[ind] = 1

        ind = ind0 & ind1 & ind2 & np.logical_not(ind3)
        inf[ind] = -1
        sup[ind] = np.maximum(np.sin(inf[ind]), np.sin(sup[ind]))

        ind = ind0 & ind1 & ind2 & ind3 & ind4 & ind5
        inf[ind] = np.sin(inf[ind])
        sup[ind] = np.sin(sup[ind])

        ind = ind0 & ind1 & ind2 & np.logical_not(ind4) & np.logical_not(ind5)
        inf[ind] = -1
        sup[ind] = np.maximum(np.sin(inf[ind]), np.sin(sup[ind]))

        # inf in [3/2*pi,2*pi]

        ind1 = inf > 1.5 * np.pi
        ind2 = inf <= 2 * np.pi

        ind = ind0 & ind1 & ind2 & np.logical_not(ind4) & ind5
        inf[ind] = -1
        sup[ind] = 1

        ind = ind0 & ind1 & ind2 & np.logical_not(ind3)
        inf[ind] = np.sin(inf[ind])
        sup[ind] = np.sin(sup[ind])

        ind = ind0 & ind1 & ind2 & ind3 & ind4
        inf[ind] = np.minimum(np.sin(inf[ind]), np.sin(sup[ind]))
        sup[ind] = 1

        ind = ind0 & ind1 & ind2 & np.logical_not(ind4) & np.logical_not(ind5)
        inf[ind] = np.sin(inf[ind])
        sup[ind] = np.sin(sup[ind])

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
        def __matmul_interval_matrix(rhs: IntervalMatrix):
            inf = np.repeat(self.inf.reshape((-1, 1)), rhs.inf.shape[1], axis=1)
            sup = np.repeat(self.sup.reshape((-1, 1)), rhs.inf.shape[1], axis=1)
            bd = np.stack(
                [
                    inf * rhs.inf,
                    inf * rhs.sup,
                    sup * rhs.inf,
                    sup * rhs.sup,
                ]
            )
            inf, sup = np.min(bd, axis=0).sum(axis=0), np.max(bd, axis=0).sum(axis=0)
            return Interval(inf, sup)

        def __matmul_interval(rhs: Interval):
            b = self * rhs
            inf, sup = b.inf.sum(), b.sup.sum()
            return Interval(inf, sup)

        if isinstance(other, Real):
            return self * other
        elif isinstance(other, Geometry.Base):
            if other.type == Geometry.TYPE.INTERVAL_MATRIX:
                return __matmul_interval_matrix(other)
            elif other.type == Geometry.TYPE.INTERVAL:
                return __matmul_interval(other)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def __rmatmul__(self, other):
        def __rmatmul_array(lhs: np.ndarray):
            from .interval_matrix import IntervalMatrix

            inf, sup = np.zeros((lhs.shape[0], self.dim)), np.zeros(
                (lhs.shape[0], self.dim)
            )
            for i in range(lhs.shape[0]):
                factor = np.repeat(lhs[i, :].reshape((-1, 1)), self.dim, axis=1)
                b = factor * self
                inf[i] = np.sum(b.inf, axis=0)
                sup[i] = np.sum(b.sup, axis=0)

            return IntervalMatrix(inf, sup)

        if isinstance(other, np.ndarray):
            return __rmatmul_array(other)
        else:
            raise NotImplementedError

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

        def __mul_real(rhs: Real):
            rhs = np.array(rhs) if isinstance(rhs, (list, tuple)) else rhs
            infm, supm = self.inf * rhs, self.sup * rhs
            inf, sup = np.minimum(infm, supm), np.maximum(infm, supm)
            return Interval(inf, sup)

        def __mul_array_like(rhs: ArrayLike):
            from .interval_matrix import IntervalMatrix

            rhs = np.array(rhs) if isinstance(rhs, (list, tuple)) else rhs
            infm, supm = self.inf * rhs, self.sup * rhs
            inf, sup = np.minimum(infm, supm), np.maximum(infm, supm)
            return IntervalMatrix(inf, sup)

        if isinstance(other, Real):
            return __mul_real(other)
        elif isinstance(other, (np.ndarray, list, tuple)):
            return __mul_array_like(other)
        elif isinstance(other, Geometry.Base):
            if other.type == Geometry.TYPE.INTERVAL:
                return __mul_interval(other)
            elif other.type == Geometry.TYPE.ZONOTOPE:
                return other * self
            elif other.type == Geometry.TYPE.INTERVAL_MATRIX:
                raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError("Invalid rhs operand")

    def __rmul__(self, other):
        return self * other

        def __rmul_array(lhs: np.ndarray):
            from .interval_matrix import IntervalMatrix

            infm, supm = self.inf * lhs, self.sup * lhs
            inf, sup = np.minimum(infm, supm), np.maximum(infm, supm)

            print(inf)
            print(sup)
            return IntervalMatrix(inf, sup)

        if isinstance(other, np.ndarray):
            return __rmul_array(other)
        else:
            raise NotImplementedError

    def __imul__(self, other):
        return self * other

    def __pos__(self):
        return self

    def __neg__(self):
        return Interval(-self.sup, -self.inf)

    def __or__(self, other):
        raise NotImplementedError

    def __ror__(self, other):
        raise NotImplementedError

    def __ior__(self, other):
        raise NotImplementedError

    def __sub__(self, other):
        if isinstance(other, (Real, np.ndarray)):
            return Interval(self.inf - other, self.sup - other)
        elif isinstance(other, Geometry.Base):
            if other.type == Geometry.TYPE.INTERVAL:
                return Interval(self.inf - other.sup, self.sup - other.inf)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def __rsub__(self, other):
        raise NotImplementedError

    def __isub__(self, other):
        return self - other

    def __truediv__(self, other):
        def __t_real(rhs: Real):
            inft, supt = self.inf / rhs, self.sup / rhs
            inf, sup = np.minimum(inft, supt), np.maximum(inft, supt)
            if rhs == 0:
                inf[:] = -np.inf
                sup[:] = np.inf
            return Interval(inf, sup)

        if isinstance(other, Real):
            return __t_real(other)
        elif isinstance(other, Geometry.Base):
            if other.type == Geometry.TYPE.INTERVAL:
                return self * (1 / other)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def __rtruediv__(self, other):
        def __rt_real(lhs: Real):
            infrt, suprt = lhs / self.inf, lhs / self.sup
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
            raise NotImplemented
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
    def rectangle(self, dims: ArrayLike):
        dims = dims if isinstance(dims, np.ndarray) else np.asarray(dims, dtype=int)
        assert dims.ndim == 1 and dims.size == 2
        assert dims.dtype == int
        pts = np.zeros((4, 2), dtype=float)
        pts[[0, 3], 0] = self.inf[dims[0]]
        pts[[1, 2], 0] = self.sup[dims[0]]
        pts[[0, 1], 1] = self.inf[dims[1]]
        pts[[2, 3], 1] = self.sup[dims[1]]
        return pts

    def cube(self, dims: ArrayLike) -> (np.ndarray, np.ndarray):
        dims = dims if isinstance(dims, np.ndarray) else np.asarray(dims, dtype=int)
        assert dims.ndim == 1 and dims.size == 3
        # set points
        pts = np.zeros((8, 3), dtype=float)
        pts[:4, 0] = self.inf[dims[0]]
        pts[4:, 0] = self.sup[dims[0]]
        pts[[0, 1, 4, 5], 1] = self.inf[dims[1]]
        pts[[2, 3, 6, 7], 1] = self.sup[dims[1]]
        pts[[0, 2, 4, 6], 2] = self.inf[dims[2]]
        pts[[1, 3, 5, 7], 2] = self.sup[dims[2]]
        # set edges
        edges = np.zeros((12, 2), dtype=int)
        edges[0] = 0, 1
        edges[1] = 0, 2
        edges[2] = 2, 3
        edges[3] = 3, 1
        edges[4] = 0, 4
        edges[5] = 2, 6
        edges[6] = 3, 7
        edges[7] = 1, 5
        edges[8] = 4, 5
        edges[9] = 5, 7
        edges[10] = 7, 6
        edges[11] = 6, 4
        return pts, edges

    def reduce(self, method: str, order: int):
        raise NotImplementedError

    def proj(self, dims):
        return Interval(self.inf[dims], self.sup[dims])

    def boundary(self, max_dist: float, element: Geometry.TYPE):
        def __boundary_interval():
            bd = []
            dims = np.arange(self.dim)
            for i in range(self.dim):
                valid_dims = np.setdiff1d(dims, i)
                g = self.proj(valid_dims).grid(max_dist)
                data = np.zeros((g.shape[0], self.dim * 2, 2), dtype=float)
                # set this dimension inf related boundary
                data[:, valid_dims, :] = g
                data[:, i, :] = self.inf[i]
                # set this dimension sup related boundary
                data[:, valid_dims + self.dim, :] = g
                data[:, i + self.dim, :] = self.sup[i]
                data = data.reshape((-1, self.dim, 2))
                bd.append(
                    [Interval(cur_data[:, 0], cur_data[:, 1]) for cur_data in data]
                )

            return list(chain.from_iterable(bd))

        if element == Geometry.TYPE.INTERVAL:
            return __boundary_interval()
        elif element == Geometry.TYPE.ZONOTOPE:
            raise NotImplementedError
        else:
            raise NotImplementedError

    def enclose(self, other):
        raise NotImplementedError

    def grid(self, max_dist: float) -> np.ndarray:
        def __ll2arr(ll, fill_value: float):
            lens = [lst.shape[0] for lst in ll]
            max_len = max(lens)
            mask = np.arange(max_len) < np.array(lens)[:, None]
            arr = np.ones((len(lens), max_len, 2), dtype=float) * fill_value
            arr[mask] = np.concatenate(ll)
            return arr, mask

        def __get_seg(dim_idx: int, seg_num: int):
            if seg_num <= 1:
                return np.array(
                    [self.inf[dim_idx], self.sup[dim_idx]], dtype=float
                ).reshape((1, -1))
            else:
                samples = np.linspace(
                    self.inf[dim_idx], self.sup[dim_idx], num=seg_num + 1
                )
                this_segs = np.zeros((seg_num, 2), dtype=float)
                this_segs[:, 0] = samples[:-1]
                this_segs[:, 1] = samples[1:]
                return this_segs

        nums = np.floor((self.sup - self.inf) / max_dist).astype(dtype=int) + 1
        segs, _ = __ll2arr([__get_seg(i, nums[i]) for i in range(self.dim)], np.nan)
        idx_list = [np.arange(nums[i]) for i in range(self.dim)]
        ext_idx = np.array(np.meshgrid(*idx_list)).T.reshape((-1, len(idx_list)))
        aux_idx = np.tile(np.arange(self.dim), ext_idx.shape[0])
        return segs[aux_idx, ext_idx.reshape(-1)].reshape((-1, self.dim, 2))

    def split(self, dim: int):
        """
        split this interval into two intervals along specified dimension
        :param dim:
        :return:
        """
        inf, sup = self.inf.copy(), self.sup.copy()
        c = (self.inf[dim] + self.sup[dim]) * 0.5
        inf[dim] = c
        sup[dim] = c
        return Interval(self.inf, sup), Interval(inf, self.sup)

    # =============================================== static method
    @staticmethod
    def empty(dim: int):
        raise NotImplementedError

    @staticmethod
    def rand(dim: int):
        bd = np.sort(np.random.rand(dim, 2), axis=1)
        return Interval(bd[:, 0], bd[:, 1])

    @staticmethod
    def zero(dim: int):
        return Interval(np.zeros(dim), np.zeros(dim))
