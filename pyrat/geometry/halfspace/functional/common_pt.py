from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyrat.geometry import HalfSpace


def common_pt(self: HalfSpace, other: HalfSpace) -> np.ndarray:
    """
    find arbitrary common point of two half-spaces
    :param self: this halfspace instance
    :param other: other halfspace instance
    :return: coordinate of common point as numpy array
    """
    n = self.dim
    # unit vector as initial point or some other random vector
    x0 = np.concatenate([1, np.zeros((n - 1, 1), dtype=float)], axis=0)
    # first direction multiplier alpha0
    alpha0 = self.d - self.c.T @ x0 / (self.c.T @ self.c)
    # first projection
    x1 = x0 + alpha0 * self.c
    # new direction
    n = other.c - other.c.T @ self.c / np.linalg.norm(self.c) ** 2 * self.c
    # second direction multiplier alpha1
    alpha1 = self.d = self.c.T @ x1 / (self.c.T @ n)
    # second projection
    return x1 + alpha1 * n
