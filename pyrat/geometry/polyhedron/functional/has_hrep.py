from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.geometry import Polyhedron
from pyrat.util.functional.aux_python import *


@property
def has_hrep(self: Polyhedron) -> bool:
    return self._has_hrep
