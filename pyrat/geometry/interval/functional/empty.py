from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyrat.geometry import Interval

from pyrat.util.functional.aux_python import *


@reg_classmethod
def empty(cls: Interval) -> Interval:
    return cls._new(np.zeros((0, 2), dtype=float))
