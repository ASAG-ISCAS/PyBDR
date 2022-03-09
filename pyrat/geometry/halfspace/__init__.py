from __future__ import annotations

import numpy as np


class HalfSpace:
    """
    Halfspace object defined by
    c@x<=d, c is column vector and d is real number
    """

    from .functional import (
        c,
        d,
        contains,
        dim,
        common_pt,
        is_empty,
        is_equal,
        _to_polyhedron,
        to,
        proj_high,
        __str__,
        __and__,
        __add__,
        __iadd__,
        __radd__,
        __sub__,
        __isub__,
        __matmul__,
        __imatmul__,
        __rmatmul__,
        __neg__,
    )

    def __init__(self, c: np.ndarray, d: float):
        assert c.ndim == 1  # pure column vector
        self._c = c
        self._d = d

    @classmethod
    def _new(cls, c: np.ndarray, d: float):
        return HalfSpace(c, d)
