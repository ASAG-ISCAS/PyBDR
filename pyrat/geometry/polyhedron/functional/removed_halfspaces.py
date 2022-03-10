from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyrat.geometry import Polyhedron


def removed_halfspaces(self: Polyhedron, h: np.ndarray) -> np.ndarray | None:
    """
    check which halfspaces have been removed from given H
    :param self: this polyhedron instance
    :param h: given H array
    :return: indices of removed halfspaces as numpy array
    """
    # remove redundant halfspaces
    self._compute_min_hrep()
    # check if empty
    if self.is_empty:
        return None
    else:
        # init
        removed_indices = []
        # previous H
        prev_h = h
        # post H; also the mirrored normal vectors are considered since the order how
        # these normal vectors are obtained is unknown
        post_h = self.ieqh[:, :-1]
        # normalize; additional consideration of k value in normalization robustifies
        # the result
        norm = np.linalg.norm(prev_h, ord=2, axis=0)
        prev_h /= norm[:, None]
        norm = np.linalg.norm(post_h, ord=2, axis=0)
        post_h /= norm[:, None]
        for i in range(prev_h.shape[0]):
            diff = abs(prev_h[i, None, :] - post_h[None, :, :]).sum(axis=[0, 2])
            # find minimum difference
            min_diff = np.min(diff)
            if min_diff > prev_h.shape[1] * 1e-10:
                removed_indices.append(i)
        return np.array(removed_indices, dtype=float)
