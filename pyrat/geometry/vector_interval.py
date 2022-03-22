from __future__ import annotations
import numpy as np


class VectorInterval:
    def __init__(self, bounds: np.ndarray):
        assert bounds.ndim == 2
        self._bd = bounds
