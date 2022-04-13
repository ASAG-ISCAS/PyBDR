from __future__ import annotations

import copy
import numbers
from itertools import product

import numpy as np
from scipy.spatial import ConvexHull

import pyrat.util.functional.auxiliary as aux

# from .interval_old import IntervalOld
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .interval import Interval


class VectorZonotope:
    __array_ufunc__ = None
    """
    set it to None in order to override the behavior of NumPy’s ufuncs
    check following links for more:
    https://stackoverflow.com/a/58120561/10450361
    https://stackoverflow.com/questions/40252765/overriding-other-rmul-with-your-classs-mul
    https://numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array_priority__
    https://numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array_ufunc__
    """

    def __init__(self, data: np.ndarray):
        self.__init(data)

    def __init(self, data):
        if isinstance(data, VectorZonotope):
            self.__init_from_vector_zonotope(data)
            return
        elif isinstance(data, np.ndarray):
            self.__init_from_numpy_array(data)
            return
        elif isinstance(data, Interval):
            self.__init_from_interval(data)
            return
        raise NotImplementedError  # TODO

    def __init_from_vector_zonotope(self, data: VectorZonotope):
        self._z = data.z
        self._rank = data.rank
        self._vertices = None
        self._ix = data.ix()
        self.remove_zero_gen()

    def __init_from_numpy_array(self, data: np.ndarray):
        assert data.ndim == 2
        self._z = data
        self._rank = None
        self._vertices = None
        self._ix = None
        self.remove_zero_gen()  # remove zero generators

    def __init_from_interval(self, interval: Interval):
        raise NotImplementedError

    # =============================================== property
    @property
    def c(self) -> np.ndarray:
        return self._z[:, 0]

    @property
    def gen(self) -> np.ndarray:
        return self._z[:, 1:]

    @property
    def z(self) -> np.ndarray:
        return self._z

    @property
    def dim(self) -> int:
        return 0 if self.is_empty else self._z.shape[0]

    @property
    def is_empty(self) -> bool:
        return aux.is_empty(self._z)

    @property
    def is_fulldim(self) -> bool:
        return False if self.is_empty else self.dim == self.rank

    @property
    def gen_num(self) -> int:
        return self._z.shape[1] - 1

    @property
    def rank(self) -> int:
        if self._rank is None:
            self._rank = np.linalg.matrix_rank(self.gen)
        return self._rank

    @property
    def is_interval(self) -> bool:
        raise NotImplementedError
        # TODO

    @property
    def info(self):
        info = "\n ------------- vector zonotope info ------------- \n"
        info += str(self.dim) + "\n"
        info += str(self.gen_num) + "\n"
        info += str(self.c) + "\n"
        return info

    # =============================================== operators
    def __add__(self, other):
        """
        override "+" for Minkowski addition of two zonotopes or a zonotopes with a vector
        :param other: zonotope or vector
        :return:
        """
        if isinstance(other, VectorZonotope):
            z = np.hstack([self._z, other.gen])
            z[:, 0] += other.c
            return VectorZonotope(z)
        elif isinstance(other, (np.ndarray, numbers.Real)):
            z = self._z.copy()
            z[:, 0] += other
            return VectorZonotope(z)
        else:
            raise NotImplementedError

    def __iadd__(self, other):
        return self + other

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, (np.ndarray, numbers.Real)):
            return self + (-other)
        else:
            raise NotImplementedError("TODO")

    def __isub__(self, other):
        return self - other

    def __rsub__(self, other):
        return self - other

    def __matmul__(self, other):
        """
        override "@" for multiplication with a matrix
        :param other: matrix as numpy array
        :return:
        """
        if isinstance(other, np.ndarray):
            z = other @ self._z
            return VectorZonotope(z)
        else:
            raise NotImplementedError("TODO")

    def __imatmul__(self, other):
        return self @ other

    def __rmatmul__(self, other):
        return self @ other

    def __mul__(self, other):
        """
        override "*" for element wise multiplication
        :param other:
        :return:
        """
        if isinstance(other, numbers.Real):
            return VectorZonotope(self.z * other)
        elif isinstance(other, Interval):
            # get symmetric interval matrix
            s = 0.5 * (other.sup - other.inf)
            z_as = np.sum(abs(self._z), axis=1)
            # compute new zonotope
            return VectorZonotope(np.hstack([other.c @ self._z, np.diag(s @ z_as)]))

        else:
            raise NotImplementedError

    def __imul__(self, other):
        return self * other

    def __rmul__(self, other):
        return self * other

    def __and__(self, other):
        raise NotImplementedError
        # TODO

    def __rand__(self, other):
        raise NotImplementedError
        # TODO

    def __str__(self):
        sep = "\n ====================================================== \n"
        return sep + str(self.c) + "\n" + str(self.gen) + sep

    def __abs__(self):
        return VectorZonotope(abs(self._z))

    def __eq__(self, other):
        return self.is_equal(other)

    def __or__(self, other):
        """
        return a zonotope enclosing this instance and other zonotope
        """
        if isinstance(other, VectorZonotope):
            # get generator numbers
            lhs_num, rhs_num = self.gen_num + 1, other.gen_num + 1
            # if first zonotope has more or equal generators
            z_cut, z_add, z_eq = None, None, None
            if rhs_num < lhs_num:
                z_cut = self._z[:, :rhs_num]
                z_add = self._z[:, rhs_num:lhs_num]
                z_eq = other._z
            else:
                z_cut = other._z[:, :lhs_num]
                z_add = other._z[:, lhs_num:rhs_num]
                z_eq = self._z
            return VectorZonotope(
                np.concatenate(
                    [(z_cut + z_eq) * 0.5, (z_cut - z_eq) * 0.5, z_add], axis=1
                )
            )
        else:
            raise NotImplementedError

    def __ror__(self, other):
        return self | other

    # =============================================== static method
    @staticmethod
    def empty(dim: int):
        return VectorZonotope(np.zeros((dim, 0), dtype=float))

    @staticmethod
    def rand(dim: int, gen_num: int):
        assert dim > 0 and gen_num >= 1
        return VectorZonotope(np.random.rand(dim, gen_num))

    # =============================================== private method
    def _picked_generators(self, order) -> (np.ndarray, np.ndarray):
        gur, gr = np.empty((self.dim, 0), dtype=float), np.empty(
            (self.dim, 0), dtype=float
        )

        if not aux.is_empty(self.gen):
            # delete zero-length generators
            self.remove_zero_gen()
            dim, gen_num = self.dim, self.gen_num
            # only reduce if zonotope order is greater than the desired order
            if gen_num > dim * order:
                # compute metric of generators
                h = np.linalg.norm(self.gen, ord=1, axis=0) - np.linalg.norm(
                    self.gen, ord=np.inf, axis=0
                )
                # number of generators that are not reduced
                num_ur = np.floor(self.dim * (order - 1)).astype(dtype=int)
                # number of generators that are reduced
                num_r = self.gen_num - num_ur

                # pick generators with smallest h values to be reduced
                idx_r = np.argpartition(h, num_r)
                gr = self.gen[:, idx_r]
                # unreduced generators
                idx_ur = np.setdiff1d(np.arange(self.gen_num), idx_r)
                gur = self.gen[:, idx_ur]
            else:
                gur = self.gen

        return gur, gr

    def _reduce_girard(self, order: int):
        # pick generators to reduce
        gur, gr = self._picked_generators(order)
        # box remaining generators
        d = np.sum(abs(gr), axis=1)
        gb = np.diag(d)
        # build reduced zonotope
        return VectorZonotope(np.hstack([self.c.reshape((-1, 1)), gur, gb]))

    # =============================================== public method
    def is_contain(self, other) -> bool:
        raise NotImplementedError
        # TODO

    def is_equal(self, other: VectorZonotope, tol=np.finfo(np.float).eps) -> bool:
        if self.dim != other.dim:
            return False
        if np.any(abs(self.c - other.c) > tol):
            return False
        if self.gen_num != other.gen_num:
            return False
        if np.all(abs(self.gen - other.gen) <= tol):
            return True
        g0 = self.gen[:, self.ix()]
        g1 = other.gen[:, other.ix()]
        return True if np.all(abs(g0 - g1) <= tol) else False

    def remove_zero_gen(self):
        if self.gen_num <= 1:
            return  # if only one generator, do nothing??? # TODO
        ng = self.gen[:, abs(self.gen).sum(axis=0) > 0]
        if aux.is_empty(ng):
            ng = self.gen[:, 0:1]  # at least one generator even all zeros inside
        self._z = np.hstack([self.c.reshape((-1, 1)), ng])

    @property
    def vertices(self):
        """
        get possible vertices of this zonotope
        :return:
        """
        if self._vertices is None:
            if self.dim == 2:
                self._vertices = self.polygon()
            else:
                raise NotImplementedError
                # x = np.array([-1, 1], dtype=float)
                # comb = np.array(list(product(x, repeat=self.gen_num)))
                # self._vertices = (
                #     self.gen[None, :, None, :] @ comb[:, None, :, None]
                # ).squeeze() + self._z[:, 0][None, :]
        return self._vertices

    def polygon(self):
        # delete zero generators
        self.remove_zero_gen()
        # obtain size of enclosing interval hull of first two dimensions
        x_max = np.sum(abs(self.gen[0, :]))
        y_max = np.sum(abs(self.gen[1, :]))

        # z with normalized direction: all generators pointing "up"
        g_norm = self.gen.copy()
        g_norm[:, g_norm[1, :] < 0] *= -1

        # compute angles
        angles = np.arctan2(g_norm[1, :], g_norm[0, :])
        angles[angles < 0] += 2 * np.pi

        # sort all generators by their angle
        idx = np.argsort(angles)

        # cumsum the generators in order of angle
        pts = np.zeros((2, self.gen_num + 1), dtype=float)
        for i in range(self.gen_num):
            pts[:, i + 1] = pts[:, i] + 2 * g_norm[:, idx[i]]

        pts[0, :] += x_max - np.max(pts[0, :])
        pts[1, :] -= y_max

        # flip/mirror upper half to get lower half of zonotope (point symmetry)
        pts_sym = (pts[:, -1] + pts[:, 0])[:, None] - pts[:, 1:]
        pts = np.concatenate([pts, pts_sym], axis=1)

        # consider center
        pts[0, :] += self.c[0]
        pts[1, :] += self.c[1]

        return pts.T

    def ix(self) -> np.ndarray:
        """
        get generators (each column vector) indices in ascending order according to rows
        :return: indices of original column vectors in ascending order
        """
        if self._ix is None:
            self._ix = np.lexsort(self.gen[::-1, :])
        return self._ix

    def enclose(self, other: VectorZonotope) -> VectorZonotope:
        """
        return a zonotope enclosing this zonotope and given other zonotope
        :param other: another given zonotope
        :return:
        """
        # TODO
        raise NotImplementedError("Use '|' for enclosing computation instead")

    def reduce(self, method: str, order: int):
        if method == "girard":
            return self._reduce_girard(order)
        else:
            raise NotImplementedError

    def proj(self, dims):
        return VectorZonotope(self._z[dims, :])

    # =============================================== public method
    # =============================================== public method
    # =============================================== public method
