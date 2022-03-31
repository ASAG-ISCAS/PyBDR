from __future__ import annotations

import numpy as np

from .geometry import Geometry


class Interval(Geometry):
    def __init__(self, bd: np.ndarray):
        assert (bd.ndim == 2 or bd.ndim == 3) and bd.shape[0] == 2
        self._bd = bd

    # =============================================== property
    @property
    def bd(self) -> np.ndarray:
        return self._bd

    @property
    def dim(self) -> int:
        raise NotImplementedError

    # =============================================== operator
    def __add__(self, other):
        pass
