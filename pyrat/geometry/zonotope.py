from __future__ import annotations

from enum import IntEnum
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import block_diag
import pyrat.util.functional.auxiliary as aux
from .geometry import Geometry

if TYPE_CHECKING:  # for type hint, easy coding ：)
    from .interval import Interval
    from .interval_matrix import IntervalMatrix


class Zonotope(Geometry.Base):
    class METHOD:
        class REDUCE(IntEnum):
            GIRARD = 0

    REDUCE_METHOD = METHOD.REDUCE.GIRARD
    ORDER = 50
    ERROR_ORDER = 20
    INTERMEDIATE_ORDER = 50

    def __init__(self, c: ArrayLike, gen: ArrayLike):
        c = c if isinstance(c, np.ndarray) else np.array(c, dtype=float)
        gen = gen if isinstance(gen, np.ndarray) else np.array(gen, dtype=float)
        assert c.ndim == 1 and gen.ndim == 2
        self._c = c
        self._gen = gen
        self._vertices = None
        self._type = Geometry.TYPE.ZONOTOPE

    # =============================================== property
    @property
    def c(self) -> np.ndarray:
        return self._c

    @property
    def gen(self) -> np.ndarray:
        return self._gen

    @property
    def z(self) -> np.ndarray:
        return np.hstack([self.c.reshape((-1, 1)), self.gen])

    @property
    def dim(self) -> int:
        return None if self.is_empty else self._c.shape[0]

    @property
    def gen_num(self):
        return self._gen.shape[1]

    @property
    def is_empty(self) -> bool:
        return aux.is_empty(self._c)

    @property
    def vertices(self) -> np.ndarray:
        def __vertices_convex_hull():
            from scipy.spatial import ConvexHull

            # first vertex is the center of the zonotope
            v = self.c.reshape((-1, 1))

            # generate further potential vertices
            for i in range(self.gen_num):
                trans = self.gen[:, i].reshape((-1, 1)).repeat(v.shape[1], axis=1)
                v = np.concatenate([v + trans, v - trans], axis=1)

                # remove inner points
                if i >= self.dim:
                    hull = ConvexHull(v.transpose())
                    v = v[:, hull.vertices]
                # else, do nothing

            return v.transpose()

        def __vertices_polytope():
            from .operation import cvt2

            self._vertices = cvt2(self, Geometry.TYPE.POLYTOPE).vertices

        if self._vertices is None:
            if self.dim == 2:
                self._vertices = self.polygon()
            else:
                return __vertices_convex_hull()

        return self._vertices

    @property
    def info(self):
        info = "\n ------------- Zonotope BEGIN ------------- \n"
        info += ">>> dimension -- gen_num -- center\n"
        info += str(self.dim) + "\n"
        info += str(self.gen_num) + "\n"
        info += str(self.c) + "\n"
        info += str(self.gen) + "\n"
        info += "\n ------------- Zonotope END --------------- \n"
        return info

    @property
    def type(self) -> Geometry.TYPE:
        return self._type

    # =============================================== operator
    def __contains__(self, item):
        from .operation import cvt2

        # TODO
        raise NotImplementedError

    def __str__(self):
        return self.info

    def __abs__(self):
        return Zonotope(abs(self.c), abs(self.gen))

    def __add__(self, other):
        if isinstance(other, (np.ndarray, Real)):
            return Zonotope(self.c + other, self.gen)
        elif isinstance(other, Geometry.Base):
            if other.type == Geometry.TYPE.ZONOTOPE:
                return Zonotope(self.c + other.c, np.hstack([self.gen, other.gen]))
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def __radd__(self, other):
        return self + other

    def __iadd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, (np.ndarray, Real)):
            return self + (-other)
        elif isinstance(other, Geometry.Base):
            raise NotImplementedError
        else:
            raise NotImplementedError

    def __rsub__(self, other):
        raise NotImplementedError

    def __isub__(self, other):
        return self - other

    def __pos__(self):
        return self

    def __neg__(self):
        raise NotImplementedError

    def __matmul__(self, other):
        if isinstance(other, np.ndarray):
            raise NotImplementedError(
                "For matrix multiplication, use 'matrix@zonotope' instead"
            )
        else:
            raise NotImplementedError

    def __rmatmul__(self, other):
        if isinstance(other, np.ndarray):
            return Zonotope(other @ self.c, other @ self.gen)
        else:
            raise NotImplementedError

    def __mul__(self, other):
        def __mul_interval(rhs: Interval):
            s = (rhs.sup - rhs.inf) * 0.5
            zas = np.sum(abs(self.z), axis=1)
            print(other.c.shape, self.z.shape)
            z = np.hstack([other.c @ self.z, np.diag(s @ zas)])
            return Zonotope(z[:, 0], z[:, 1:])

        if isinstance(other, Real):
            return Zonotope(self.c * other, self.gen * other)
        elif isinstance(other, Geometry.Base):
            if other.type == Geometry.TYPE.INTERVAL:
                return __mul_interval(other)
            elif other.type == Geometry.TYPE.INTERVAL_MATRIX:
                return __mul_interval(other)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def __rmul__(self, other):
        def __rmul_interval_matrix(lhs: IntervalMatrix):
            zas = np.sum(abs(self.z), axis=1)
            z = None
            s = lhs.rad
            if not np.any(lhs.c):
                # no empty generators if interval matrix is symmetric
                z = np.hstack([0 * self.c.reshape((-1, 1)), np.diag(s @ zas)])
            else:
                z = np.hstack([lhs.c @ self.z, np.diag(s @ zas)])
            return Zonotope(z[:, 0], z[:, 1:])

        if isinstance(other, Real):
            return self * other
        elif isinstance(other, Geometry.Base):
            if other.type == Geometry.TYPE.INTERVAL_MATRIX:
                return __rmul_interval_matrix(other)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def __imul__(self, other):
        return self * other

    def __or__(self, other):
        raise NotImplementedError

    # =============================================== class method
    @classmethod
    def functional(cls):
        raise NotImplementedError

    # =============================================== static method
    @staticmethod
    def empty(dim: int):
        return Zonotope(np.empty(dim, dtype=float), np.empty((dim, 0), dtype=float))

    @staticmethod
    def rand(dim: int, gen_num: int = 0):
        assert dim >= 1 and gen_num >= 0
        return Zonotope(np.random.rand(dim), np.random.rand(dim, gen_num))

    @staticmethod
    def zero(dim: int, gen_num: int = 0):
        assert dim >= 1 and gen_num >= 0
        return Zonotope(np.zeros(dim), np.zeros((dim, gen_num)))

    # =============================================== private method
    def _picked_gen(self) -> (np.ndarray, np.ndarray):
        gur = np.empty((self.dim, 0), dtype=float)
        gr = np.empty((self.dim, 0), dtype=float)

        if not aux.is_empty(self.gen):
            # delete zero-length generators
            self.remove_zero_gen()
            dim, gen_num = self.dim, self.gen_num
            # only reduce if zonotope order is greater than the desired order
            if gen_num > dim * self.ORDER:
                # compute metric of generators
                h = np.linalg.norm(self.gen, ord=1, axis=0) - np.linalg.norm(
                    self.gen, ord=np.inf, axis=0
                )
                # number of generators that are not reduced
                num_ur = np.floor(self.dim * (self.ORDER - 1)).astype(dtype=int)
                # number of generators that are reduced
                num_r = self.gen_num - num_ur

                # pick generators with smallest h values to be reduced
                idx_r = np.argpartition(h, num_r)[:num_r]
                gr = self.gen[:, idx_r]
                # unreduced generators
                idx_ur = np.setdiff1d(np.arange(self.gen_num), idx_r)
                gur = self.gen[:, idx_ur]
            else:
                gur = self.gen

        return gur, gr

    # =============================================== public method
    def remove_zero_gen(self):
        if self.gen_num <= 1:
            return
        ng = self.gen[:, abs(self.gen).sum(axis=0) > 0]
        if aux.is_empty(ng):
            ng = self.gen[:, 0:1]  # at least one generator even all zeros inside
        self._gen = ng

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

    def enclose(self, other: Zonotope) -> Zonotope:
        if isinstance(other, Zonotope):
            # get generator numbers
            lhs_num, rhs_num = self.gen_num + 1, other.gen_num + 1
            # if first zonotope has more or equal generators
            z_cut, z_add, z_eq = None, None, None
            if rhs_num < lhs_num:
                z_cut = self.z[:, :rhs_num]
                z_add = self.z[:, rhs_num:lhs_num]
                z_eq = other.z
            else:
                z_cut = other.z[:, :lhs_num]
                z_add = other.z[:, lhs_num:rhs_num]
                z_eq = self.z
            z = np.concatenate(
                [(z_cut + z_eq) * 0.5, (z_cut - z_eq) * 0.5, z_add], axis=1
            )
            return Zonotope(z[:, 0], z[:, 1:])
        else:
            raise NotImplementedError

    def reduce(self, method: REDUCE_METHOD, order: int):
        def __reduce_girard():
            # pick generators to reduce
            gur, gr = self._picked_gen()
            # box remaining generators
            d = np.sum(abs(gr), axis=1)
            d[abs(d) < 0] = 0
            gb = np.diag(d) if d.shape[0] > 0 else np.empty((self.dim, 0), dtype=float)
            # build reduced zonotope
            return Zonotope(self.c, np.hstack([gur, gb]))

        if self.REDUCE_METHOD == Zonotope.METHOD.REDUCE.GIRARD:
            return __reduce_girard()
        else:
            raise NotImplementedError

    def proj(self, dims):
        return Zonotope(self.c[dims], self.gen[dims, :])

    def boundary(self, max_dist: float, element: Geometry.TYPE):
        raise NotImplementedError

    def card_prod(self, other):
        if isinstance(other, Zonotope):
            c = np.concatenate([self.c, other.c])
            gen = block_diag(self.gen, other.gen)
            z = Zonotope(c, gen)
            z.remove_zero_gen()
            return z
        else:
            raise NotImplementedError

    def quad_map(self, q: [np.ndarray], rz: Zonotope = None):
        def _xTQx():
            dim_q = q[0].shape[0]
            c = np.zeros(dim_q)
            gen_num = int(0.5 * (self.gen_num**2 + self.gen_num)) + self.gen_num
            gens = self.gen_num
            gen = np.zeros((dim_q, gen_num))

            z = self.z

            # count empty matrices
            q_noz = np.zeros(dim_q, dtype=bool)

            # for each dimension, compute generator elements
            for i in range(dim_q):
                q_noz[i] = np.any([np.any(iq[i]) for iq in q])
                if q_noz[i]:
                    # pure quadratic evaluation
                    qi = block_diag(*[iq[i] for iq in q])
                    quad_mat = z.T @ qi @ z
                    # faster method diag elements
                    gen[i, :gens] = 0.5 * np.diag(quad_mat[1 : gens + 1, 1 : gens + 1])
                    # center
                    c[i] = quad_mat[0, 0] + np.sum(gen[i, 0:gens])
                    # off-diagonal elements added, pick via logical indexing
                    quad_mat_off_diag = quad_mat + quad_mat.T
                    k_ind = np.tril(np.ones((gens + 1, gens + 1), dtype=bool), -1)
                    gen[i, gens:] = quad_mat_off_diag[k_ind]

            # generate new zonotope
            if np.sum(q_noz) <= 1:
                return Zonotope(c, np.sum(abs(gen), axis=1).reshape((-1, 1)))
            else:
                z = Zonotope(c, gen)
                z.remove_zero_gen()
                return z

        def _x1TQx2():
            z_mat1 = self.z
            z_mat2 = rz.z
            dim_q = q[0].shape[0]

            # init solution (center + generator matrix)
            z = np.zeros((dim_q, z_mat1.shape[1] * z_mat2.shape[1]))

            # count empty matrices
            q_noz = np.zeros(dim_q, dtype=bool)

            # for each dimension, compute center + generator elements
            for i in range(dim_q):
                q_noz[i] = np.any([np.any(iq[i]) for iq in q])
                if q_noz[i]:
                    # pure quadratic evaluation
                    qi = block_diag(*[iq[i] for iq in q])
                    quad_mat = z_mat1.T @ qi @ z_mat2
                    z[i] = quad_mat.reshape(-1)

            # generate new zonotope
            if np.sum(q_noz) <= 1:
                return Zonotope(z[:, 0], np.sum(abs(z[:, 1:]), axis=1).reshape((-1, 1)))
            else:
                zono = Zonotope(z[:, 0], z[:, 1:])
                zono.remove_zero_gen()
                return zono

        return _xTQx() if rz is None else _x1TQx2()

    def support_func(self, dir: np.ndarray, type: str = "u"):
        """
        calculates the upper or lower bound of this zonotope along given direction
        :param type: type of the calculation, "u" for upper bound, "l" for lower bound
        :param dir: given direction in numpy ndarray format
        :return:

        # TODO speed up this function !!!!!!!!!!
        """
        proj_zono = dir @ self
        c, g = proj_zono.c, proj_zono.gen
        if type == "u":
            val = c + np.sum(abs(g))
            fac = np.sign(g)
        elif type == "l":
            val = c - np.sum(abs(g))
            fac = -np.sign(g)
        else:
            raise NotImplementedError
        return self.c + self.gen * fac, val, fac
