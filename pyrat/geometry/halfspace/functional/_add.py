from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyrat.geometry import HalfSpace


def __add__(self: HalfSpace, other: HalfSpace | np.ndarray) -> HalfSpace:
    raise NotImplementedError
    # TODO


def __iadd__(self: HalfSpace, other: HalfSpace | np.ndarray) -> HalfSpace:
    raise NotImplementedError
    # TODO
