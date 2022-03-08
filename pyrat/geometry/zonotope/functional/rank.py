from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.geometry import Zonotope

from pyrat.util.functional.aux_python import *
import numpy as np


@reg_property
def rank(self: Zonotope) -> int:
    return np.linalg.matrix_rank(self.generator)
