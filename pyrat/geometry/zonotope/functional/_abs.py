from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.geometry import Zonotope


def __abs__(self: Zonotope) -> Zonotope:
    return self._new(self.__class__, abs(self._z))
