from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.geometry import Interval

import numpy as np


def __and__(self: Interval, other: Interval) -> Interval:
    """
    compute intersection of these two intervals
    :param self: this interval instance
    :param other: other interval instance
    :return: resulting interval instance
    """
    # compute intersection
    inf = np.max(self.inf, other.inf)
    sup = np.min(self.sup, other.sup)
    if np.all(inf - sup <= np.finfo(np.float).eps):
        return self._new(np.concatenate([np.min(inf, sup), np.max([inf, sup])]))
    return self.empty()
