from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyrat.geometry import Zonotope
from pyrat.util.functional.aux_python import *


@reg_property
def generator(self: Zonotope) -> np.ndarray:
    return self._z[:, 1:]
