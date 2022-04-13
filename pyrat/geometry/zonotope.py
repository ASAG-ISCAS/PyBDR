from __future__ import annotations

import numbers

import numpy as np
import scipy.sparse
from numpy.typing import ArrayLike
from scipy.sparse import csc_matrix, hstack, diags

import pyrat.util.functional.auxiliary as aux
from .geometry import Geometry, GeoTYPE


class Zonotope(Geometry):
    def __init__(self, c: ArrayLike, gen: ArrayLike):
        c = c if isinstance(c, np.ndarray) else np.array(c, dtype=float)
        gen = gen if isinstance(gen, csc_matrix) else csc_matrix(gen, dtype=float)
        assert c.ndim == 1
        assert c.shape[0] == gen.shape[0] and gen.ndim == 2
        self._c = c
        self._gen = gen
        self._vertices = None
        self._ix = None
        self._type = GeoTYPE.ZONOTOPE

    # =============================================== property
    @property
    def c(self) -> np.ndarray:
        return self._c

    @property
    def z(self) -> csc_matrix:
        return hstack([self._c.reshape((-1, 1)), self._gen], format="csc")

    @property
    def gen(self):
        return self._gen

    @property
    def gen_num(self):
        return self._gen.shape[1]

    @property
    def dim(self) -> int:
        return -1 if self.is_empty else self._c.shape[0]

    @property
    def is_empty(self) -> bool:
        return aux.is_empty(self._c)

    @property
    def vertices(self) -> np.ndarray:
        if self._vertices is None:
            if self.dim == 2:
                self._vertices = self.polygon()
            else:
                raise NotImplementedError  # TODO
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
    def type(self) -> GeoTYPE:
        return self._type

    # =============================================== operator
    def __contains__(self, item):
        raise NotImplementedError

    def __str__(self):
        return self.info

    def __abs__(self):
        return Zonotope(abs(self.c), abs(self.gen))

    def __add__(self, other):
        if isinstance(other, (np.ndarray, list, tuple, numbers.Real)):
            return Zonotope(self.c + other, self.gen)
        elif isinstance(other, Zonotope):
            return Zonotope(self.c + other.c, hstack([self.gen, other.gen]))
        elif isinstance(other, Geometry):
            raise NotImplementedError  # TODO addition according to the priority ???
        raise NotImplementedError

    def __radd__(self, other):
        return self + other

    def __iadd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __pos__(self):
        return self

    def __neg__(self):
        return self * -1

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
        if isinstance(other, numbers.Real):
            return Zonotope(self.c * other, self.gen * other)
        elif isinstance(other, Geometry):
            if other.type == GeoTYPE.INTERVAL:
                s = 0.5 * (other.sup - other.inf)
                zas = csc_matrix(abs(self.z).sum(axis=1))
                gen = diags(
                    (s @ zas).toarray().reshape(-1), format="csc"
                )  # prefer csc_matrix
                z = hstack([other.c @ self.z, gen], format="csc")
                return Zonotope(z[:, 0].toarray().reshape(-1), z[:, 1:])
            else:
                raise NotImplementedError

    def __rmul__(self, other):
        return self * other

    def __imul__(self, other):
        return self * other

    def __or__(self, other):
        raise NotImplementedError  # TODO

    # =============================================== comparator
    def __eq__(self, other):
        raise NotImplementedError

    # =============================================== class method
    @classmethod
    def functional(cls):
        return {
            "__add__": cls.__add__,
            "__sub__": cls.__sub__,
            "__pos__": cls.__pos__,
            "__neg__": cls.__neg__,
        }

    # =============================================== static method
    @staticmethod
    def empty(dim: int):
        raise NotImplementedError

    @staticmethod
    def rand(dim: int, gen_num: int):
        assert dim >= 1 and gen_num >= 1  # at least one generator in 1D space
        return Zonotope(np.random.rand(dim), scipy.sparse.rand(dim, gen_num))

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
        return Zonotope(self.c, hstack([gur, gb]))

    # =============================================== public method
    def reduce(self, method: str, order: int):
        if method == "girard":
            return self._reduce_girard(order)
        else:
            raise NotImplementedError

    def proj(self, dims):
        raise NotImplementedError

    def polygon(self) -> np.ndarray:
        # delete zero generators
        self.remove_zero_gen()
        # obtain size of enclosing interval hull of first two dimensions
        x_max = abs(self.gen[0, :]).sum()
        y_max = abs(self.gen[1, :]).sum()

        # z with normalized direction: all generators pointing "up"
        g_norm = self.gen.copy()
        ind = (g_norm[1, :] < 0).data
        g_norm[:, ind] *= -1

        # compute angles
        angles = np.arctan2(
            g_norm[1, :].toarray().reshape(-1), g_norm[0, :].toarray().reshape(-1)
        )
        angles[angles < 0] += 2 * np.pi

        # sort all generators by their angle
        idx = np.argsort(angles)

        # cumsum the generators in order of angle
        pts = np.zeros((2, self.gen_num + 1), dtype=float)
        for i in range(self.gen_num):
            pts[:, i + 1] = pts[:, i] + 2 * g_norm[:, idx[i]].toarray().reshape(-1)

        pts[0, :] += x_max - np.max(pts[0, :])
        pts[1, :] -= y_max

        # flip/mirror upper half to get lower half of zonotope (point symmetry)
        pts_sym = (pts[:, -1] + pts[:, 0])[:, None] - pts[:, 1:]
        pts = np.concatenate([pts, pts_sym], axis=1)

        # involve the center into the computation
        pts += self.c[:, None]

        return pts.T

    def enclose(self, other):
        if isinstance(other, Zonotope):
            lhs_num, rhs_num = self.gen_num, other.gen_num
            z_cut, z_add, z_eq = None, None, None
            if other.gen_num < self.gen_num:
                z_cut = self.z[:, : rhs_num + 1]
                z_add = self.z[:, rhs_num:lhs_num]
                z_eq = other.z
            else:
                z_cut = other.z[:, : lhs_num + 1]
                z_add = other.z[:, lhs_num:rhs_num]
                z_eq = self.z
            print(z_cut.shape)
            print(z_eq.shape)
            print(z_add.shape)
            z = hstack(
                [(z_cut + z_eq) * 0.5, (z_cut - z_eq) * 0.5, z_add], format="csc"
            )
            return Zonotope(z[:, 0].toarray().reshape(-1), z[:, 1:])
        else:
            raise NotImplementedError

    def remove_zero_gen(self):
        if self.gen_num <= 1:
            return
        idx = np.array(abs(self.gen).sum(axis=0)).reshape(-1) > 0
        ng = self.gen[:, idx]
        if aux.is_empty(ng):
            ng = self.gen[:, 0:1]  # at least one zero generator even all zeros inside
        self._gen = ng
