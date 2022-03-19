from __future__ import annotations

import numpy as np
import pyrat.util.functional.aux_numpy as an


class HalfSpace:
    def __init__(self, c: np.ndarray = None, d: float = None):
        assert c.ndim == 1
        self._c = c
        self._d = d

    # =============================================== property
    @property
    def dim(self) -> int:
        return self._c.shape[0]

    @property
    def c(self) -> np.ndarray:
        return self._c

    @property
    def d(self) -> float:
        return self._d

    @property
    def is_empty(self) -> bool:
        raise NotImplementedError

    # =============================================== operator
    # =============================================== private method
    # =============================================== public method
