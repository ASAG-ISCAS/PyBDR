from __future__ import annotations
import numpy as np


class Zonotope:
    def __init__(self, center: np.ndarray, generator: np.ndarray):
        assert np.ndim == 1 and generator.ndim == 2
        self.c = center
        self.g = generator
