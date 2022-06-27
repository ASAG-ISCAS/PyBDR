from __future__ import annotations

from enum import IntEnum
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike

import pyrat.util.functional.auxiliary as aux
from pyrat.geometry.geometry import Geometry
from pyrat.geometry.zonotope import Zonotope


class PolyZonotope(Geometry.Base):
    class METHOD:
        class RESTRUCTURE(IntEnum):
            REDUCE_PCA = 0

        class RECUE(IntEnum):
            GIRARD = 0

    MAX_DEPTH_GEN_ORDER = 50
    MAX_POLY_ZONO_RATIO = 0.01
    RESTRUCTURE_METHOD = METHOD.RESTRUCTURE.REDUCE_PCA
    REDUCE_METHOD = METHOD.RECUE.GIRARD

    def __init__(
        self,
        c: ArrayLike,
        gen: ArrayLike = None,
        gen_rst: ArrayLike = None,
        exp_mat: ArrayLike = None,
    ):
        def __input_validation(data, tp=float):
            if data is None or isinstance(data, tp):
                return data
            return np.asarray(data, dtype=tp)

        self._c = __input_validation(c)
        self._gen = __input_validation(gen)
        self._gen_rst = __input_validation(gen_rst)
        self._exp_mat = __input_validation(exp_mat, tp=int)
        self._type = Geometry.TYPE.POLY_ZONOTOPE
        # post check
        assert self._c.ndim == 1
        assert self._gen is None or self._gen.ndim == 2
        assert self._gen_rst is None or self._gen_rst.ndim == 2
        assert self._exp_mat is None or self._exp_mat.ndim == 2

    # =============================================== property
    @property
    def c(self) -> np.ndarray:
        return self._c

    @property
    def gen(self):
        return self._gen

    @property
    def gen_rst(self):
        return self._gen_rst

    @property
    def exp_mat(self):
        return self._exp_mat

    @property
    def dim(self) -> int:
        return None if self.is_empty else self._c.shape[0]

    @property
    def is_empty(self) -> bool:
        return (
            aux.is_empty(self._c)
            and aux.is_empty(self._gen)
            and aux.is_empty(self._gen_rst)
        )

    @property
    def vertices(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def info(self):
        info = "\n ------------- Polynomial Zonotope BEGIN ------------- \n"
        info += ">>> dimension -- gen_num -- center"
        info += str(self.dim) + "\n"
        info += str(self.c) + "\n"
        info += str(self.gen) + "\n"
        info += "\n ------------- Polynomial Zonotope END ------------- \n"
        raise NotImplementedError

    @property
    def type(self) -> Geometry.TYPE:
        return self._type

    # =============================================== operator

    def __contains__(self, item):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def __add__(self, other):
        def __add_zonotope(rhs: Zonotope):
            c = self.c + rhs.c
            gen_rst = (
                rhs.gen
                if self.gen_rst is None
                else np.concatenate([self.gen_rst, rhs.gen], axis=1)
            )
            return PolyZonotope(c, self.gen, gen_rst, self.exp_mat)

        if isinstance(other, (np.ndarray, Real)):
            return PolyZonotope(self.c + other, self.gen, self.gen_rst, self.exp_mat)
        elif isinstance(other, Geometry.Base):
            if other.type == Geometry.TYPE.ZONOTOPE:
                return __add_zonotope(other)
        else:
            raise NotImplementedError

    def __sub__(self, other):
        if isinstance(other, (np.ndarray, Real)):
            return self + (-other)
        else:
            raise NotImplementedError

    def __pos__(self):
        raise NotImplementedError

    def __neg__(self):
        raise NotImplementedError

    def __matmul__(self, other):
        if isinstance(other, np.ndarray):
            raise NotImplementedError(
                "For matrix multiplication, use 'matrix@polyzonotope' instead"
            )
        else:
            raise NotImplementedError

    def __rmatmul__(self, other):
        if isinstance(other, np.ndarray):
            c = other @ self.c
            gen = other @ self.gen
            gen_rst = other @ self.gen_rst if self.gen_rst is not None else None
            return PolyZonotope(c, gen, gen_rst, self.exp_mat)

    def __mul__(self, other):
        raise NotImplementedError

    def __or__(self, other):
        raise NotImplementedError

    # =============================================== class method
    @classmethod
    def functional(cls):
        raise NotImplementedError

    # =============================================== static method
    @staticmethod
    def empty(dim: int):
        raise NotImplementedError

    @staticmethod
    def rand(dim: int):
        raise NotImplementedError

    # =============================================== public method
    def enclose(self, other):
        def __enclose_polyzonotope(rhs: PolyZonotope):
            if np.all(self.exp_mat.shape == rhs.exp_mat.shape) and np.all(
                self.exp_mat == rhs.exp_mat
            ):
                gen = np.concatenate(
                    [
                        0.5 * self.gen + 0.5 * rhs.gen,
                        0.5 * self.gen - 0.5 * rhs.gen,
                        (0.5 * self.c - 0.5 * rhs.c).reshape((-1, 1)),
                    ],
                    axis=1,
                )
                c = 0.5 * self.c + 0.5 * rhs.c

                temp = np.ones((1, self.exp_mat.shape[1]))
                exp_mat = np.concatenate(
                    [
                        np.concatenate([self.exp_mat, rhs.exp_mat], axis=1),
                        np.concatenate([0 * temp, temp], axis=1),
                    ],
                    axis=0,
                )
                temp = np.zeros((exp_mat.shape[0], 1))
                temp[-1] = 1
                exp_mat = np.concatenate([exp_mat, temp], axis=1)

                # compute convex hull of the independent generators by using the
                # enclose function for linear zonotopes
                temp = np.zeros_like(self.c)
                zono1 = Zonotope(
                    temp,
                    np.zeros((temp.shape[0], 1))
                    if self.gen_rst is None
                    else self.gen_rst,
                )
                zono2 = Zonotope(
                    temp,
                    np.zeros((temp.shape[0], 1))
                    if rhs.gen_rst is None
                    else rhs.gen_rst,
                )

                zono = zono1.enclose(zono2)
                gen_rst = zono.gen
                return PolyZonotope(c, gen, gen_rst, exp_mat)
            else:
                raise NotImplementedError

        if isinstance(other, PolyZonotope):
            return __enclose_polyzonotope(other)
        else:
            raise NotImplementedError

    def reduce(self):

        raise NotImplementedError

    def proj(self, dims):
        raise NotImplementedError

    def boundary(self, max_dist: float, element: Geometry.TYPE):
        raise NotImplementedError
