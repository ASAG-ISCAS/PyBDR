from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyrat.geometry import Zonotope


def remove_empty_gen(self: Zonotope):
    ng = self.generator[:, abs(self.generator).sum(axis=0) > 0]
    self._z = np.concatenate([self.center, ng], axis=1)
