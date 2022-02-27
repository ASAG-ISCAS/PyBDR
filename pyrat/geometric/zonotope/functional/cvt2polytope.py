from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.geometric.zonotope import Zonotope
    from pyrat.geometric.polytope import Polytope

import numpy as np
from itertools import combinations

"""
This function convert input zonotope to polytope representation

"""


def __cross_ndim(m: np.ndarray) -> np.ndarray:
    v = np.tile(m, (m.shape[0], 1, 1))
    mask = np.ones_like(v, dtype=bool)
    idx = np.arange(m.shape[0])
    mask[idx, idx, :] = False
    v = np.reshape(v[mask], (m.shape[0], v.shape[2], v.shape[2]))
    return (-1) ** (idx + 1) * np.linalg.det(v)


def cvt2polytope(z: Zonotope) -> (Polytope, np.ndarray, bool):
    nz = z.remove_empty_gen()
    c, g = nz.center, nz.generator
    dim, gen_num = g.shape
    is_full_dimensional = False
    cst_m, cst_v = None, None  # constraint matrix and vector define a polytope
    if np.linalg.matrix_rank(g) >= dim:
        if dim > 1:
            comb = np.array(list(combinations(np.arange(gen_num), dim - 1)))
            cst_m = np.zeros((comb.shape[0], dim), dtype=float)
            for comb_idx in range(comb.shape[0]):
                this_comb = comb[comb_idx, :]
                q = g[:, this_comb]
                cst_m[comb_idx, :] = __cross_ndim(q)
            # remove NaN rows due to rank deficiency
            indices = np.where(np.isnan(cst_m.sum(axis=1)))[0]
            cst_m = np.delete(cst_m, indices, axis=0)
        else:
            cst_m = 1
        # build d vector and determine delta d
        delta_d = abs(np.matmul(cst_m, g)).sum(axis=1)
        # compute d_positive and d_negative
        pos_d = np.matmul(cst_m, c) + delta_d
        neg_d = -np.matmul(cst_m, c) + delta_d
        # construct the overall inequality constraints
        cst_m = np.concatenate([cst_m, -cst_m], axis=0)
        cst_v = np.concatenate([pos_d, neg_d], axis=0)
        # catch the case where the zonotope is not full-dimensional
        temp = np.min([abs(cst_m - cst_m[0, :]).sum(axis=1), abs(cst_m - cst_m[0, :]).sum(axis=1)], axis=1)
        if dim > 1 and (np.prod(temp.shape) == 0 or np.all(np.all(np.isnan(cst_m)))
                        or np.all(temp < 1e-12) or np.any(np.max(abs(cst_m), axis=0) < 11e-12)):
            # singular value decomposition
            u, s, vh = np.linalg.svd(g)
            # state space transformation
            z_ = u.T @ np.concatenate([c, g], axis=1)
            # remove dimensions with all zeros
            # TODO
            print(z_.shape)
            exit(False)

        exit(False)

    raise Exception("not supported yet")
