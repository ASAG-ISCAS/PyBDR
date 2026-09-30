"""
Regression tests for the scipy based polytope operations against the legacy pypoman results
stored in test/data/reference_legacy.npz.
"""

from pathlib import Path

import cvxpy as cp
import numpy as np
import pytest
from scipy.spatial import ConvexHull

from pybdr.geometry import Polytope
from pybdr.geometry.polytope import chebyshev_center, halfspaces_to_vertices, vertices_to_halfspaces

REF = np.load(Path(__file__).parents[1] / "data" / "reference_legacy.npz")
CASES = sorted({k.rsplit("_", 1)[0] for k in REF.files if k.startswith("poly_")})


def assert_same_points(x: np.ndarray, y: np.ndarray, tol: float = 1e-7):
    """x and y contain the same points (the legacy results may contain duplicates)"""
    assert x.shape[1] == y.shape[1]
    dist = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)
    assert dist.min(axis=1).max() < tol
    assert dist.min(axis=0).max() < tol


@pytest.mark.parametrize("case", CASES)
def test_vertices_match_legacy(case):
    vs = halfspaces_to_vertices(REF[case + "_a"], REF[case + "_b"])
    assert_same_points(vs, REF[case + "_vertices"])


@pytest.mark.parametrize("case", CASES)
def test_halfspaces_describe_same_polytope(case):
    a, b = vertices_to_halfspaces(REF[case + "_pts"])
    assert_same_points(halfspaces_to_vertices(a, b), REF[case + "_vertices"])


@pytest.mark.parametrize("case", CASES)
def test_chebyshev_center(case):
    a, b = REF[case + "_a"], REF[case + "_b"]
    c, r = chebyshev_center(a, b)
    norms = np.linalg.norm(a, axis=1)
    # the ball is inside the polytope ...
    assert np.all(a @ c + norms * r <= b + 1e-9)
    # ... and has the maximal radius
    x, t = cp.Variable(a.shape[1]), cp.Variable()
    cp.Problem(cp.Maximize(t), [a @ x + norms * t <= b]).solve()
    assert r == pytest.approx(t.value, rel=1e-6)
    assert np.allclose(Polytope(a, b).c, c)


@pytest.mark.parametrize("case", [c for c in CASES if c + "_polygon01" in REF.files])
def test_polygon_matches_legacy(case):
    p = Polytope(REF[case + "_a"], REF[case + "_b"])
    assert_same_points(p.polygon([0, 1]), REF[case + "_polygon01"])


@pytest.mark.parametrize("dims", [[0], [1, 3], [0, 2, 3]])
def test_proj_any_number_of_dims(dims):
    p = Polytope(REF["poly_d4_s0_a"], REF["poly_d4_s0_b"])
    projected = p.proj(dims).vertices
    expected = REF["poly_d4_s0_vertices"][:, dims]
    if len(dims) > 1:
        expected = expected[ConvexHull(expected).vertices]
    else:
        expected = np.array([[expected.max()], [expected.min()]])
    assert_same_points(projected, expected)


def test_no_duplicate_vertices():
    vs = halfspaces_to_vertices(REF["poly_d4_s3_a"], REF["poly_d4_s3_b"])
    assert len(vs) == len(ConvexHull(REF["poly_d4_s3_pts"]).vertices)


def test_rand():
    p = Polytope.rand(3)
    assert p.vertices.shape[1] == 3


def test_unbounded_raises():
    with pytest.raises(ValueError, match="unbounded"):
        halfspaces_to_vertices(np.array([[1.0, 0], [-1, 0], [0, 1]]), np.array([1.0, 1, 1]))


def test_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        chebyshev_center(np.array([[1.0], [-1.0]]), np.array([-1.0, -1.0]))


def test_flat_raises():
    with pytest.raises(ValueError, match="interior"):
        halfspaces_to_vertices(np.array([[1.0, 0], [-1, 0], [0, 1], [0, -1]]), np.array([1.0, 1, 0, 0]))
