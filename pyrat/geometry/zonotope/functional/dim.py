from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.geometry import Zonotope

from pyrat.util.functional.aux_python import *


@reg_property
def dim(self: Zonotope) -> int:
    return 0 if self.is_empty else self._z.shape[0]
