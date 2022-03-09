from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyrat.geometry import HalfSpace


def proj_high(self: HalfSpace, n: int, dims: np.ndarray) -> HalfSpace:
    """
    project a halfspace instance to a higher dimensional space
    :param self: this halfspace instance
    :param n: dimension of the higher dimensional space
    :param dims: states of the high dimensional space that correspond to the states of the low
    dimensional halfspace
    :return:
    """
    # initialize variables
    c = np.zeros((n, 1), dtype=float)
    # insert parameters from the original halfspace object
    c[dims] = self.c
    return self._new(c, self.d)
