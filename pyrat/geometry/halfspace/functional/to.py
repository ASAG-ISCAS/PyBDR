from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.geometry import HalfSpace


def to(self: HalfSpace, target: str):
    if target == "polyhedron":
        return self._to_polyhedron()
    raise NotImplementedError
    # TODO
