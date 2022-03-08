from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.geometry import Zonotope

from pyrat.util.functional.aux_python import *


@reg_property
def is_fulldim(self: Zonotope) -> bool:
    return False if self.is_empty else self.dim == self.rank
