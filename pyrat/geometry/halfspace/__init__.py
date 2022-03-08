from __future__ import annotations

import numpy as np


class HalfSpace:
    from .functional import c, d

    def __init__(self, c: np.ndarray, d: float):
        self._c = c
        self._d = d
