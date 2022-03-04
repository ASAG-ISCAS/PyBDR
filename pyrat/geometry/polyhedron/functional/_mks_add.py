from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.geometry import Polyhedron


def _mks_add(lhs: Polyhedron, rhs: Polyhedron) -> Polyhedron:
    print("calling minkowski addition")
    # TODO
    pass
