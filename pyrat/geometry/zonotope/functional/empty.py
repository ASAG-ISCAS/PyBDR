from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyrat.geometry import Zonotope

from pyrat.util.functional.aux_python import *


@reg_classmethod
def empty(cls: Zonotope, dim: int) -> Zonotope:
    return cls._new(np.zeros((dim, 0), dtype=float))
