from __future__ import annotations
import numpy as np


class MatrixInterval:
    def __init__(self, bound: np.ndarray):
        assert bound.ndim == 3
        self._bd = bound

    # =============================================== property
    # =============================================== operator
    def __add__(self, other):
        raise NotImplementedError

    # =============================================== private method
    # =============================================== private method
