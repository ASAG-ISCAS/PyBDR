from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyrat.geometry import HalfSpace


def __add__(self: HalfSpace, other: HalfSpace | np.ndarray) -> HalfSpace:
    """
    do Minkowski addition with a vector or a halfspace
    :param self: this halfspace instance
    :param other: vector or halfspace instance
    :return: resulting halfspace instance
    """
    if self.is_empty:
        raise Exception("Invalid Minkowski addition with empty halfspace")
    d = self.d + self.c.T @ other
    return HalfSpace(self.c, d)


def __radd__(self: HalfSpace, other: HalfSpace | np.ndarray) -> HalfSpace:
    return self + other


def __iadd__(self: HalfSpace, other: HalfSpace | np.ndarray) -> HalfSpace:
    return self + other
