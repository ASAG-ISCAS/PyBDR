from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.geometry import Polyhedron


def __add__(self: Polyhedron, other: Polyhedron) -> Polyhedron:
    return self.__class__._mks_add(self, other)


def __iadd__(self: Polyhedron, other: Polyhedron):
    return self.__add__(other)
