from __future__ import annotations

from itertools import combinations

import numpy as np

import pyrat.util.functional.aux_numpy as an


class VectorZonotope:
    def __init__(self, z: np.ndarray):
        assert z.ndim == 2
        self._z = z
        self._rank = None

    # =============================================== property
    @property
    def center(self) -> np.ndarray:
        return self._z[:, :1]

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
        return an.is_empty(self._z)

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
        raise NotImplementedError
        # TODO

    def __iadd__(self, other):
        raise NotImplementedError
        # TODO

    def __radd__(self, other):
        raise NotImplementedError
        # TODO

    def __sub__(self, other):
        raise NotImplementedError
        # TODO

    def __isub__(self, other):
        raise NotImplementedError
        # TODO

    def __rsub__(self, other):
        raise NotImplementedError
        # TODO

    def __matmul__(self, other):
        raise NotImplementedError
        # TODO

    def __imatmul__(self, other):
        raise NotImplementedError
        # TODO

    def __rmatmul__(self, other):
        raise NotImplementedError
        # TODO

    def __and__(self, other):
        raise NotImplementedError
        # TODO

    def __rand__(self, other):
        raise NotImplementedError
        # TODO

    def __str__(self):
        raise NotImplementedError
        # TODO

    def __abs__(self):
        raise NotImplementedError
        # TODO

    # =============================================== static method
    @staticmethod
    def empty(dim: int):
        return VectorZonotope(np.zeros((dim, 0), dtype=float))

    @staticmethod
    def rand_fix_dim(dim: int):
        assert dim > 0
        gen_num = np.random.randint(0, 10)
        return VectorZonotope(np.random.rand(dim, gen_num))

    # =============================================== private method
    # =============================================== public method
    def is_contain(self, other) -> bool:
        raise NotImplementedError
        # TODO

    def delete_zeros(self):
        raise NotImplementedError
        # TODO

    def polygon(self):
        """
        converts a 2d zonotope to a polygon
        :return: ordered vertices of the final polytope
        """
        assert self.dim == 2  # only care about 2d case
        # delete zero generators
        # self.delete_zeros()
        # get all potential vertices
        comb = list(combinations([1, -1], self.gen_num))
        # TODO
        print(comb)
        print(self.center)
        print(self.generator)

        raise NotImplementedError
        # TODO

    # =============================================== public method
    # =============================================== public method
    # =============================================== public method
