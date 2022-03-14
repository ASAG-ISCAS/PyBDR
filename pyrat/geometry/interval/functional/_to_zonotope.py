from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyrat.geometry import Interval
    from pyrat.geometry import Zonotope


def _to_zonotope(self: Interval) -> Zonotope:
    """
    convert an interval object into an over approximating zonotope
    :param self: this interval instance
    :return:
    """
    # obtain center
    c = self.center
    # construct generator matrix g
    g = np.diag(self.radius)
    # init zonotope
    return Zonotope(np.concatenate([c, g], axis=1))
