"""
Generate reference results from the legacy pypoman / RealPaver based implementations.

These results are used by the regression tests that check the scipy / codac based
replacements. This script only runs on the commit right before pypoman and RealPaver
were removed (it needs both, and RealPaver only ships macOS / Windows binaries):

    python test/data/make_reference.py
"""

import time
from pathlib import Path

import numpy as np
import pypoman

from pybdr.geometry import Geometry, Polytope, Zonotope
from pybdr.geometry.operation import boundary

OUT = Path(__file__).with_name("reference_legacy.npz")


def polytope_cases():
    """Random full-dimensional polytopes built from random point clouds.

    The Chebyshev center is not stored: it is not unique in general (and pypoman's
    cvxopt backend crashes on some setups), so the tests check its defining property.
    """
    cases = {}
    for dim in (2, 3, 4):
        for seed in range(5):
            rng = np.random.default_rng(100 * dim + seed)
            pts = rng.uniform(-1, 1, size=(rng.integers(8, 20), dim))
            a, b = pypoman.compute_polytope_halfspaces(pts)
            key = f"poly_d{dim}_s{seed}"
            cases[key + "_pts"] = pts
            cases[key + "_a"] = np.asarray(a, dtype=float)
            cases[key + "_b"] = np.asarray(b, dtype=float)
            p = Polytope(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
            cases[key + "_vertices"] = p.vertices
            if dim > 2:
                cases[key + "_polygon01"] = p.polygon([0, 1])
    return cases


def boundary_cases():
    """Boundary boxes computed by RealPaver for polytope / zonotope initial sets."""
    brusselator_a = np.array([[0.84680084, 0.53191008],
                              [0.17672503, 0.98426026],
                              [-0.68443591, 0.72907303],
                              [-0.99997959, -0.00638965],
                              [-0.92101281, -0.3895323],
                              [-0.39691115, -0.91785704],
                              [0.60973787, -0.79260314],
                              [0.93952577, -0.34247821]])
    brusselator_b = np.array([0.40646441, 0.43238234, 0.45970287, 0.41445799,
                              0.40973516, 0.44057138, 0.38044951, 0.45097237])
    brusselator_z = np.array([[0.0, 0.24, -0.10, 0.03, -0.08],
                              [0.0, 0.05, 0.18, -0.12, -0.04]])
    cut_square_a = np.array([[1.0, 0], [-1, 0], [0, 1], [0, -1], [1, 1]])
    cut_square_b = np.array([1.0, 1, 1, 1, 1.5])
    rng = np.random.default_rng(7)
    a3, b3 = pypoman.compute_polytope_halfspaces(rng.uniform(-1, 1, size=(12, 3)))

    sets = {
        "bd_brusselator_poly": (Polytope(brusselator_a, brusselator_b), 0.04),
        "bd_brusselator_zono": (Zonotope(brusselator_z[:, 0], brusselator_z[:, 1:]), 0.04),
        "bd_cut_square": (Polytope(cut_square_a, cut_square_b), 0.1),
        "bd_poly3d": (Polytope(np.asarray(a3, dtype=float), np.asarray(b3, dtype=float)), 0.2),
    }

    cases = {}
    for key, (src, r) in sets.items():
        start = time.perf_counter()
        boxes = boundary(src, r, Geometry.TYPE.INTERVAL)
        elapsed = time.perf_counter() - start
        if src.type == Geometry.TYPE.POLYTOPE:
            cases[key + "_a"], cases[key + "_b"] = src.a, src.b
        else:
            cases[key + "_c"], cases[key + "_gen"] = src.c, src.gen
        cases[key + "_r"] = np.array(r)
        cases[key + "_boxes"] = np.stack([np.stack([box.inf, box.sup], axis=-1) for box in boxes])
        cases[key + "_time"] = np.array(elapsed)
        print(f"{key}: {len(boxes)} boxes in {elapsed:.2f}s")
    return cases


if __name__ == "__main__":
    np.savez_compressed(OUT, **polytope_cases(), **boundary_cases())
    print(f"saved to {OUT}")
