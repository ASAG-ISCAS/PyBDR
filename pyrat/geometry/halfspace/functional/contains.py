from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyrat.geometry import HalfSpace


def contains(self: HalfSpace, other) -> bool:
    """
    check if given other object completely inside this halfspace
    :param self: this halfspace instance
    :param other: other geometry object for containing check
    :return: TRUE if other object completely inside this halfspace
    """
    if isinstance(other, np.ndarray):
        return self.c @ other <= self.d
    raise NotImplementedError
    # TODO
