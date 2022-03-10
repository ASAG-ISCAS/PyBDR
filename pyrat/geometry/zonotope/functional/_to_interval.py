from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.geometry import Zonotope
from pyrat.geometry import Interval
import numpy as np


def _to_interval(self: Zonotope) -> Interval:
    """
    over approximate this zonotope by an interval
    :param self: this zonotope instance
    :return: interval instance representing this zonotope
    """
    # extract center
    c = self.center
    # determine lower bounds and upper bounds
    delta = np.sum(abs(self.z), axis=1) - abs(self.center)
    lb = c - delta
    ub = c + delta
    return Interval(np.concatenate([lb, ub], axis=1))
