from __future__ import annotations

from numbers import Real
from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import ArrayLike
from enum import IntEnum
from .geometry import Geometry
import pyrat.util.functional.auxiliary as aux


class PolyZonotope(Geometry.Base):
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
        raise NotImplementedError

    def __sub__(self, other):
        raise NotImplementedError

    def __pos__(self):
        raise NotImplementedError

    def __neg__(self):
        raise NotImplementedError

    def __matmul__(self, other):
        raise NotImplementedError

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
    def reduce(self, method: str, order: int):
        raise NotImplementedError

    def proj(self, dims):
        raise NotImplementedError

    def boundary(self, max_dist: float, element: Geometry.TYPE):
        raise NotImplementedError
