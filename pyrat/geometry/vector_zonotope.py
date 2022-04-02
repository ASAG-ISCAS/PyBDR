from __future__ import annotations

import numbers
from itertools import product

import numpy as np
from scipy.spatial import ConvexHull

import pyrat.util.functional.auxiliary as aux
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

    def __init__(self, z: np.ndarray):
        assert z.ndim == 2
        self._z = z
        self._rank = None
        self._vertices = None
        self._ix = None
        self.remove_zero_gen()  # remove zero generators

    # =============================================== property
    @property
    def center(self) -> np.ndarray:
        return self._z[:, 0]

    @property
    def generator(self) -> np.ndarray:
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
            self._rank = np.linalg.matrix_rank(self.generator)
        return self._rank

    @property
    def is_interval(self) -> bool:
        raise NotImplementedError
        # TODO

    # =============================================== operators
    def __add__(self, other):
        """
        override "+" for Minkowski addition of two zonotopes or a zonotopes with a vector
        :param other: zonotope or vector
        :return:
        """
        if isinstance(other, VectorZonotope):
            z = np.hstack([self._z, other.generator])
            z[:, 0] += other.center
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
            # get minimum and maximum
            inf, sup = other.inf, other.sup
            # get center of interval matrix
            c = other.center
            # get symmetric interval matrix
            s = 0.5 * (sup - inf)
            z_as = np.sum(abs(self._z), axis=1)
            # compute new zonotope
            return VectorZonotope(np.hstack([c @ self._z, np.diag(s @ z_as)]))

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
        return sep + str(self.center) + "\n" + str(self.generator) + sep

    def __abs__(self):
        return VectorZonotope(abs(self._z))

    def __eq__(self, other):
        return self.is_equal(other)

    # =============================================== static method
    @staticmethod
    def empty(dim: int):
        return VectorZonotope(np.zeros((dim, 0), dtype=float))

    @staticmethod
    def rand(dim: int):
        assert dim > 0
        gen_num = np.random.randint(1, 10)
        return VectorZonotope(np.random.rand(dim, gen_num))

    # =============================================== private method
    # =============================================== public method
    def is_contain(self, other) -> bool:
        raise NotImplementedError
        # TODO

    def is_equal(self, other: VectorZonotope, tol=np.finfo(np.float).eps) -> bool:
        if self.dim != other.dim:
            return False
        if np.any(abs(self.center - other.center) > tol):
            return False
        if self.gen_num != other.gen_num:
            return False
        if np.all(abs(self.generator - other.generator) <= tol):
            return True
        g0 = self.generator[:, self.ix()]
        g1 = other.generator[:, other.ix()]
        return True if np.all(abs(g0 - g1) <= tol) else False

    def remove_zero_gen(self):
        if self.gen_num <= 1:
            return  # if only one generator, do nothing??? # TODO
        ng = self.generator[:, abs(self.generator).sum(axis=0) > 0]
        self._z = np.hstack([self.center.reshape((-1, 1)), ng])

    def vertices(self):
        """
        get possible vertices of this zonotope
        :return:
        """
        if self._vertices is None:
            x = np.array([-1, 1], dtype=float)
            comb = np.array(list(product(x, repeat=self.gen_num)))
            self._vertices = (
                self.generator[None, :, None, :] @ comb[:, None, :, None]
            ).squeeze() + self._z[:, 0][None, :]
        return self._vertices

    def polygon(self):
        """
        converts a 2d zonotope to a polygon
        :return: ordered vertices of the final polytope
        """
        assert self.dim == 2  # only care about 2d case
        pts = self.vertices()[ConvexHull(self.vertices()).vertices, :].tolist()
        pts.sort(
            key=lambda p: np.arctan2(p[1] - self.center[0, 1], p[0] - self.center[0, 0])
        )
        return np.array(pts)

    def ix(self) -> np.ndarray:
        """
        get generators (each column vector) indices in ascending order according to rows
        :return: indices of original column vectors in ascending order
        """
        if self._ix is None:
            self._ix = np.lexsort(self.generator[::-1, :])
        return self._ix

    # =============================================== public method
    # =============================================== public method
    # =============================================== public method
