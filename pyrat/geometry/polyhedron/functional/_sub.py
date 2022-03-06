from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.geometry import Polyhedron


def __sub__(self: Polyhedron, other: Polyhedron) -> Polyhedron:
    return self.__class__._mks_sub(self, other)


def __isub__(self: Polyhedron, other: Polyhedron) -> Polyhedron:
    return self.__sub__(other)
