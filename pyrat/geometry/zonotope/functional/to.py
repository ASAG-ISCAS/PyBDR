from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.geometry import Zonotope


def to(self: Zonotope, target: str):
    if target == "polyhedron":
        # return self._to_po
        pass
    pass
