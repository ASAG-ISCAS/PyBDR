from __future__ import annotations

import numpy as np


import pyrat.util.functional.auxiliary as aux


class HalfSpace:
    def __init__(self, c: np.ndarray = None, d: float = None):
        """
        define Halfspace by c@x<=d, where c is column vector and d is float number
        :param c: 1-dimensional vector as numpy array, indicates normal direction
        :param d: float number indicates offset to the origin
        """
        if c is not None:
            assert c.ndim == 1 and np.sum(abs(c)) > 0
        self._c = c
        self._d = d

    # =============================================== property
    @property
    def dim(self) -> int:
        """
        get dimension of this halfspace
        :return: return -1 for empty set
        """
        return -1 if self.is_empty else self._c.shape[0]

    @property
    def c(self) -> np.ndarray:
        """
        get linear part of the constraint
        :return:
        """
        return self._c

    @property
    def d(self) -> float:
        """
        get constant part of the constraint
        :return:
        """
        return self._d

    @property
    def h(self) -> np.ndarray:
        """
        get c and d as one column vector, d as the final element
        :return:
        """
        return np.append(self.c, self.d)

    @property
    def is_empty(self) -> bool:
        """
        check if this halfspace instance is empty or not
        :return:
        """
        if self.c is None or self.d is None:
            return True
        return aux.is_empty(self.c) and self.d == 0

    # =============================================== operator
    def __add__(self, other):
        """
        override "+" for Minkowski addition with a vector, equals to shift the boundary
        :param other: another halfspace instance
        :return:
        """
        if self.is_empty:
            raise Exception("Empty set")
        if isinstance(other, np.ndarray):
            # adding a vector
            assert self.dim == other.shape[0] and other.ndim == 1
            return HalfSpace(self.c, self.d + self.c @ other)
        else:
            raise Exception("Unsupported addition")

    def __iadd__(self, other):
        """
        override operator "+="
        :param other:
        :return:
        """
        return self + other

    def __radd__(self, other):
        raise NotImplementedError(
            "For more details, please check https://stackoverflow.com/a/58120561/10450361"
        )

    def __sub__(self, other):
        return self + (-other)

    def __isub__(self, other):
        return self - other

    def __rsub__(self, other):
        raise NotImplementedError(
            "For more details, please check https://stackoverflow.com/a/58120561/10450361"
        )

    def __pos__(self):
        return self

    def __neg__(self):
        return HalfSpace(-self.c, -self.d)

    def __len__(self):
        """
        dimension of the halfspace
        :return:
        """
        return -1 if self.is_empty else self.c.shape[0]

    def __eq__(self, other):
        """
        override operator "==" for equality checking with default tolerance
        :param other: other halfspace instance
        :return:
        """
        return self.is_equal(other)

    def __ne__(self, other):
        """
        override operator "!= for inequality checking with default tolerance
        :param other:  other halfspace instance
        :return:
        """
        return not self.is_equal(other)

    def __str__(self):
        """
        print this info of this halfspace instance
        :return:
        """
        return str(self.c) + " " + str(self.d)

    def __and__(self, other):
        """
        override operator "&" for computing intersection with other geometry object
        :param other:
        :return:
        """
        return other & self

    def __mul__(self, other):
        raise NotImplementedError("For matrix multiplication, using @ instead")

    def __matmul__(self, other):
        """
        override operator "@" for multiplication with a matrix
        :param other: matrix as numpy array
        :return:
        """
        try:
            c = np.linalg.inv(other).T @ self.c
            return HalfSpace(c, self.d)
        except:
            if self.is_empty:
                raise Exception("Empty set")
            elif abs(np.linalg.det(other)) < np.finfo(np.float).eps:
                raise Exception("Linear transformation with near-singular matrix")
        raise Exception("Matrix multiplication failed")

    def __rmatmul__(self, other):
        return self @ other

    # =============================================== private method
    # TODO

    # =============================================== static method
    @staticmethod
    def empty():
        return HalfSpace()

    @staticmethod
    def rand(dim: int = None):
        dim = np.random.randint(0, 10) if dim is None else dim
        # ranges
        lb, ub = -10, 10
        # instantiate interval
        c = lb + np.random.rand(dim) * (ub - lb)
        d = lb + np.random.rand(1)[0] * (ub - lb)
        return HalfSpace(c, d)

    # =============================================== public method
    def is_equal(self, other: HalfSpace, tol: float = np.finfo(np.float).eps) -> bool:
        """
        check if this halfspace instance equals to another
        :param other: another halfspace instance
        :param tol: tolerance for difference checking, OPTIONAL
        :return:
        """
        return np.all(abs(self.c - other.c) < tol) and abs(self.d - other.d) < tol

    def is_contain(self, other) -> bool:
        """
        check if other geometry object fully inside this halfspace instance
        :param other: another given geometry instance
        :return:
        """
        if isinstance(other, np.ndarray):
            # if given N checking points as matrix in shape (N,dim)
            assert other.ndim == 2 and other.shape[0] > 0
            return (self.c[None, None, :] @ other[:, :, None]).squeeze() <= self.d
        else:
            raise NotImplementedError

    def is_intersecting(self, other) -> bool:
        """
        check if this halfspace instance intersecting with another geometry object
        :param other: another given geometry instance
        :return:
        """
        # TODO
        raise NotImplementedError

    def is_fullspace(self, d: int) -> bool:
        """
        check if this halfspace capture specified full space
        :param d: specified dimension, default is 0
        :return:
        """
        return abs(self._c[d]) <= np.finfo(np.float).eps

    def common_pt(self, other) -> np.ndarray:
        pt = self.intersection_pt(other)
        c = np.vstack([self.c, other.c]).astype(dtype=float).T
        norm = np.linalg.norm(c, axis=1)
        c /= norm[:, None]
        factors = np.random.rand(2)
        return pt - c @ factors

    def intersection_pt(self, other):
        """
        get the intersection of two halfspace boundaries, only work for 2d test
        :param other: other halfspace instance
        :return:
        """
        assert self.dim == 2 and other.dim == 2
        assert not (self.is_empty or other.is_empty)
        mv = np.vstack([self.h, other.h])
        return np.linalg.lstsq(mv[:, :-1], mv[:, -1])[0].reshape((1, -1))

    def rotate(self, d: np.ndarray, rot_pt: np.ndarray):
        """
        rotate a halfspace around a point such that the new normal vector is aligned
        with given direction
        :param d:
        :param rot_pt:
        :return:
        """
        raise NotImplementedError
        # TODO

    def proj(self, d: int, dims: np.ndarray):
        raise NotImplementedError
        # TODO

    def dist2bd(self, other: np.ndarray):
        """
        get distances to this halfspace boundary, negative distance for inside point
        :param other: N points in shape (N,dim)
        :return:
        """
        assert other.ndim == 2
        return np.matmul(
            self.c[None, None], other[:, :, None]
        ).squeeze() - self.d / np.linalg.norm(self.c)
