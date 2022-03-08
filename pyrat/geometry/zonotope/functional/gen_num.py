from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.geometry import Zonotope

from pyrat.util.functional.aux_python import *


@reg_property
def gen_num(self: Zonotope) -> int:
    return self._z.shape[1] - 1
