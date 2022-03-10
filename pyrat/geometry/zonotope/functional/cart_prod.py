from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import block_diag

if TYPE_CHECKING:
    from pyrat.geometry import Zonotope


def cart_prod(self: Zonotope, other) -> Zonotope:
    """
    cartesian product of this zonotope with other geometry object
    :param self: this zonotope instance
    :param other: other geometry object
    :return: zonotope represent the cartesian product result
    """
    try:
        # assume given two zonotope
        c = np.concatenate([self.center, other.center], axis=0)
        g = block_diag(self.generator, other.generator)
        return self._new(np.concatenate([c, g], axis=1))
    except:
        return self.cart_prod(other.to("zonotope"))
