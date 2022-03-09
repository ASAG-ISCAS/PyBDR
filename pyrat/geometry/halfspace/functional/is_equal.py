from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyrat.geometry import HalfSpace


def is_equal(self: HalfSpace, other: HalfSpace, tol=np.finfo(np.float).eps) -> bool:
    """
    check if this halfspace is equal to another halfspace
    :param self: this halfspace instance
    :param other: other halfspace instance
    :param tol: tolerance for difference checking, OPTIONAL
    :return: TRUE if these two halfspace are equal
    """
    return np.all(abs(self.c - other.c) < tol) and abs(self.d - other.d) < tol
