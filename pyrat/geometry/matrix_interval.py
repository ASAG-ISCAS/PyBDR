from __future__ import annotations

import numpy as np


class MatrixInterval:
    def __init__(self, bound: np.ndarray):
        assert bound.ndim == 3
        self._bd = bound

    # =============================================== property
    @property
    def bd(self) -> np.ndarray:
        return self._bd

    @property
    def inf(self) -> np.ndarray:
        return self._bd[0]

    @property
    def sup(self) -> np.ndarray:
        return self._bd[1]

    # =============================================== operator
    def __add__(self, other):
        if isinstance(other, np.ndarray):
            # TODO
            raise NotImplementedError
        else:
            raise NotImplementedError

    # =============================================== private method
    # =============================================== private method
