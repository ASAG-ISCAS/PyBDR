from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyrat.geometry import HalfSpace


def __sub__(self: HalfSpace, other: HalfSpace | np.ndarray) -> HalfSpace:
    """
    do Minkowski subtraction with vector or halfspace
    :param self: this halfspace instance
    :param other: vector or halfspace instance
    :return: resulting halfspace instance
    """
    return self + (-other)


def __isub__(self: HalfSpace, other: HalfSpace | np.ndarray) -> HalfSpace:
    return self - other


def __rsub__(self: HalfSpace, other: HalfSpace | np.ndarray) -> HalfSpace:
    return -self + other
