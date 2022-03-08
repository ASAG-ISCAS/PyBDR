from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.geometry import Zonotope

from pyrat.util.functional.aux_python import *
from pyrat.util.functional.aux_numpy import *


@reg_property
def is_empty(self: Zonotope) -> bool:
    return is_empty(self._z)
