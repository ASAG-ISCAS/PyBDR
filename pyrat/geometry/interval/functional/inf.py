from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyrat.geometry import Interval

from pyrat.util.functional.aux_python import *


@reg_property
def inf(self: Interval) -> np.ndarray:
    return self._bd[:, 0]
