from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyrat.geometry import HalfSpace


def _null(a, tol=1e-5):
    u, s, v = np.linalg.svd(a)
    rank = (s > tol * s[0]).sum()
    return rank, v[rank:].T.copy()


def _rm(h: HalfSpace, d: np.ndarray) -> np.ndarray:
    """
    compute a rotation matrix to orient the normal vector of a hyperplane to new direction
    :param h: given halfspace object
    :param d: new direction as numpy array
    :return: rotation matrix
    """
    # get dimension
    dim = h.dim
    n = h.c / np.linalg.norm(h.c, ord=2)  # euclidean norm
    b = np.zeros((dim, dim), dtype=float)
    if abs(n.T @ d) != 1:
        # normalize normal vectors
        d = d / np.linalg.norm(d, ord=2)
        # create mapping matrix
        b[:, 0] = n
        # find orthonormal basis for n, u_vec
        in_vec = d - (d.T @ n) @ n
        b[:, 1] = in_vec / np.linalg.norm(in_vec, ord=2)
        # complete mapping matrix b
        if dim > 2:
            _, b[:, 2:] = _null(b[:, :1].T)
        # compute angel between u_vec and n
        angle = np.arccos(d.T @ n)
        # rotation matrix
        r = np.eye(dim, dtype=float)
        r[0, 0] = np.cos(angle)
        r[0, 1] = -np.sin(angle)
        r[1, 0] = np.sin(angle)
        r[1, 1] = np.cos(angle)
        # final rotation matrix
        return b @ r @ np.linalg.inv(b)
    else:
        if n.T @ d == 1:
            return np.eye(dim, dtype=float)
        else:
            return -np.eye(dim, dtype=float)


def rotate(self: HalfSpace, d: np.ndarray, rp: np.ndarray) -> HalfSpace:
    """
    rotate a halfspace around a point such that the new normal vector is aligned with
    given direction
    :param self: this halfspace instance
    :param d: new direction as vector
    :param rp: center point of rotation
    :return: resulting halfspace instance
    """
    # obtain rotation matrix
    rot = _rm(self, d)
    # translate and rotate halfspace
    return rot @ (self - rp) + rp
