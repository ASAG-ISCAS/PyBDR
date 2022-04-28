from __future__ import annotations

import numpy as np

from pyrat.geometry import Geometry
from dataclasses import dataclass


@dataclass
class Set:
    """
    NOTE:
        Set is designed for characterizing an approximation of implicit exact set by
        providing the approximating error
    """

    geo: Geometry.Base = None
    err: np.ndarray = None

    def __post_init__(self):
        if self.geo is not None and self.err is None:
            self.err = np.zeros(self.geo.dim, dtype=float)
