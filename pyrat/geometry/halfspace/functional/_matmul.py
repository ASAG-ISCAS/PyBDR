from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyrat.geometry import HalfSpace


def __matmul__(self: HalfSpace, m: np.ndarray) -> HalfSpace:
    """
    do matrix multiplication, can call by operator '@'
    :param self: this halfspace instance
    :param m: given matrix
    :return:
    """
    try:
        # assume that factor is an invertible matrix
        c = np.matmul(np.linalg.inv(m).T, self.c)
        return self._new(c, self.d)
    except:
        if self.is_empty:
            # empty halfspace
            pass
        elif abs(np.linalg.det(m)) < np.finfo(np.float).eps:
            raise ValueError("Linear transformation with near-singular matrix")
        else:
            raise NotImplementedError


def __rmatmul__(self: HalfSpace, m: np.ndarray) -> HalfSpace:
    return self @ m


def __imatmul__(self: HalfSpace, m: np.ndarray) -> HalfSpace:
    return self @ m
