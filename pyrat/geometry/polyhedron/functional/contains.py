from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyrat.geometry import Polyhedron


def contains(self: Polyhedron, x: np.ndarray) -> bool:
    """
    check if given point inside the domain of this polyhedron
    :param self: this polyhedron instance
    :param x: testing point
    :return: TRUE if given point inside this domain
    """
    raise NotImplementedError
    # TODO
