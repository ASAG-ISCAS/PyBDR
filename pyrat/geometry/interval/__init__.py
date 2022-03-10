from __future__ import annotations

import numpy as np


class Interval:
    from .functional import (
        is_empty,
        inf,
        sup,
        empty,
        __and__,
        radius,
        center,
        to,
        _to_zonotope,
    )

    def __init__(self, bd: np.ndarray):
        assert bd.shape[1] == 2  # give two column vectors as matrix
        assert np.all(bd[:, 0] <= bd[:, 1])
        self._bd = bd

    @classmethod
    def _new(cls, bd: np.ndarray):
        return Interval(bd)
