from __future__ import annotations

import numbers
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyrat.geometry import Zonotope


def __add__(self: Zonotope, other: Zonotope | numbers.Real) -> Zonotope:
    if isinstance(other, numbers.Real):
        z = self._z.copy()
        z[:, :1] += other
        return self._new(self.__class__, z)
    elif isinstance(other, Zonotope):
        z = np.hstack([self._z, other.generator])
        z[:, :1] += other.center
        return self._new(self.__class__, z)


def __iadd__(self: Zonotope, other: Zonotope | numbers.Real) -> Zonotope:
    return self + other
