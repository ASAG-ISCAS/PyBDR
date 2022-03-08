from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.geometry import HalfSpace

from pyrat.util.functional.aux_python import *


@reg_property
def d(self: HalfSpace) -> float:
    return self._d
