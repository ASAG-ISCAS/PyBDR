from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.geometry import Polyhedron


def irr_vrep(self: Polyhedron) -> bool:
    return self._irr_vrep
