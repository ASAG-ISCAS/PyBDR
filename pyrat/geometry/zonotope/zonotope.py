from __future__ import annotations

import numbers
import numpy as np
from .functional import *
from pyrat.util.functional.aux_numpy import *


class Zonotope:
    def __init__(self, z: np.ndarray):
        assert z.ndim == 2
        self.__z = z

    # =============================== properties
    @property
    def dim(self) -> int:
        return 0 if self.is_empty else self.__z.shape[0]

    @property
    def z(self) -> np.ndarray:
        return self.__z

    @property
    def center(self) -> np.ndarray:
        return self.__z[:, :1]

    @property
    def generator(self) -> np.ndarray:
        return self.__z[:, 1:]

    @property
    def gen_num(self) -> int:
        return self.__z.shape[1] - 1

    @property
    def is_empty(self) -> bool:
        return is_empty(self.__z)

    @property
    def rank(self) -> int:
        return np.linalg.matrix_rank(self.generator)

    @property
    def is_full_dim(self) -> bool:
        return False if self.is_empty else self.dim == self.rank

    # =============================== static methods
    @staticmethod
    def random_fix_dim(dimension: int) -> Zonotope:
        assert dimension > 0
        gen_nums = np.random.randint(0, 10)
        return Zonotope(np.random.rand(dimension, gen_nums))

    # =============================== _functional methods
    def remove_empty_gen(self) -> Zonotope:
        ng = self.generator[:, abs(self.generator).sum(axis=0) > 0]
        return Zonotope(np.concatenate([self.center, ng], axis=1))

    # =============================== conversion
    def cvt_as(self, target: str):
        if target == "polyhedron":
            return cvt2polyhedron(self)
        else:
            raise Exception("Unsupported target type")

    # =============================== overload numeric operators
    def __add__(self, other: Zonotope | numbers.Real) -> Zonotope:
        if isinstance(other, numbers.Real):
            z = self.__z.copy()
            z[:, :1] += other
            return Zonotope(z)
        elif isinstance(other, Zonotope):
            z = np.hstack([self.__z, other.generator])
            z[:, :1] += other.center
            return Zonotope(z)
        else:
            raise ValueError("Invalid operand to do Minkowski addition")

    def __iadd__(self, other: Zonotope | numbers.Real) -> Zonotope:
        return self + other

    def __sub__(
        self, other: Zonotope | numbers.Real, method: str = "althoff"
    ) -> Zonotope:
        if isinstance(other, numbers.Real):
            z = self.__z.copy()
            z[:, :1] -= other
            return Zonotope(z)
        elif isinstance(other, Zonotope):
            if method == "althoff":
                return approx_mink_diff_althoff(self, other)
            elif method == "cons_zono":
                return approx_mink_diff_cons_zono(self, other)
            else:
                raise Exception("Invalid specified method to do minkowski difference")
        else:
            raise ValueError("Invalid operand to do Minkowski difference")

    def __isub__(self, other: Zonotope | numbers.Real) -> Zonotope:
        return self - other

    def __abs__(self) -> Zonotope:
        return Zonotope(abs(self.__z))

    def __str__(self) -> str:
        return "center: " + str(self.center) + "\ngenerator: \n" + str(self.generator)
