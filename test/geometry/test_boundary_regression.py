"""
Regression tests for the codac based boundary extraction against the legacy RealPaver results
stored in test/data/reference_legacy.npz.
"""

from pathlib import Path

import codac
import numpy as np
import pytest
from scipy.spatial import ConvexHull

from pybdr.geometry import Geometry, Interval, Polytope, Zonotope
from pybdr.geometry.operation import boundary
from pybdr.geometry.polytope import halfspaces_to_vertices
from pybdr.util.functional import extract_boundary, function_boundary

REF = np.load(Path(__file__).parents[1] / "data" / "reference_legacy.npz")
CASES = ["bd_brusselator_poly", "bd_brusselator_zono", "bd_cut_square", "bd_poly3d"]


def load_case(case):
    if case + "_a" in REF.files:
        return Polytope(REF[case + "_a"], REF[case + "_b"]), float(REF[case + "_r"])
    return Zonotope(REF[case + "_c"], REF[case + "_gen"]), float(REF[case + "_r"])


def as_array(boxes):
    return np.stack([np.stack([box.inf, box.sup], axis=-1) for box in boxes])


def boundary_samples(vs: np.ndarray, num: int = 4000, seed: int = 0) -> np.ndarray:
    """random points on the facets of the convex hull of vs"""
    hull = ConvexHull(vs)
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(vs.shape[1]), size=(len(hull.simplices), num // len(hull.simplices) + 1))
    return np.einsum("fkd,fdn->fkn", weights, vs[hull.simplices]).reshape(-1, vs.shape[1])


def covered(boxes: np.ndarray, pts: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    inside = (boxes[None, :, :, 0] - tol <= pts[:, None, :]) & (pts[:, None, :] <= boxes[None, :, :, 1] + tol)
    return np.any(np.all(inside, axis=-1), axis=-1)


@pytest.mark.parametrize("case", CASES)
def test_boundary_matches_legacy(case):
    src, r = load_case(case)
    boxes = as_array(boundary(src, r, Geometry.TYPE.INTERVAL))
    legacy = REF[case + "_boxes"]

    # sound: every point on the boundary is covered
    vs = cvt_vertices(src)
    assert covered(boxes, boundary_samples(vs)).all()
    # boxes respect the requested precision
    assert np.all(boxes[:, :, 1] - boxes[:, :, 0] <= r + 1e-12)
    # same magnitude as RealPaver
    assert 0.5 < len(boxes) / len(legacy) < 2
    volume = np.prod(boxes[:, :, 1] - boxes[:, :, 0], axis=1).sum()
    legacy_volume = np.prod(legacy[:, :, 1] - legacy[:, :, 0], axis=1).sum()
    assert 0.5 < volume / legacy_volume < 2


def cvt_vertices(src):
    if src.type == Geometry.TYPE.POLYTOPE:
        return halfspaces_to_vertices(src.a, src.b)
    return src.vertices


@pytest.mark.parametrize("elem", [Geometry.TYPE.INTERVAL, Geometry.TYPE.POLYTOPE, Geometry.TYPE.ZONOTOPE])
def test_boundary_element_types(elem):
    src, r = load_case("bd_brusselator_poly")
    bounds = boundary(src, r, elem)
    assert len(bounds) > 0
    assert all(b.type == elem for b in bounds)


@pytest.mark.parametrize("domain", [None, "interval", "codac"])
def test_extract_boundary_polytope(domain):
    src, r = load_case("bd_brusselator_poly")
    if domain == "interval":
        domain = Interval([-0.5, -0.5], [0.5, 0.5])
    elif domain == "codac":
        domain = codac.IntervalVector([[-0.5, 0.5], [-0.5, 0.5]])
    zonos = extract_boundary(domain, src, eps=r)
    assert all(z.type == Geometry.TYPE.ZONOTOPE for z in zonos)
    boxes = np.stack([np.stack([z.c - np.abs(z.gen).sum(axis=1), z.c + np.abs(z.gen).sum(axis=1)], -1) for z in zonos])
    assert covered(boxes, boundary_samples(cvt_vertices(src))).all()


def test_extract_boundary_zonotope():
    src, r = load_case("bd_brusselator_zono")
    assert len(extract_boundary(None, src, eps=r)) > 0


def test_function_boundary_circle():
    x = codac.VectorVar(2)
    f = codac.AnalyticFunction([x], x[0] ** 2 + x[1] ** 2 - 1)
    boxes = function_boundary(f, Interval([-2, -2], [2, 2]), eps=0.1)
    arr = as_array(boxes)
    theta = np.linspace(0, 2 * np.pi, 2000)
    assert covered(arr, np.stack([np.cos(theta), np.sin(theta)], axis=-1)).all()
    # every box touches the circle
    radius_lo = np.linalg.norm(np.clip(0, arr[:, :, 0], arr[:, :, 1]), axis=1)
    radius_hi = np.linalg.norm(np.maximum(np.abs(arr[:, :, 0]), np.abs(arr[:, :, 1])), axis=1)
    assert np.all((radius_lo <= 1 + 1e-9) & (radius_hi >= 1 - 1e-9))


def test_extract_boundary_function_returns_zonotopes():
    x = codac.VectorVar(2)
    f = codac.AnalyticFunction([x], x[0] ** 2 + x[1] ** 2 - 0.1)
    zonos = extract_boundary(Interval([-0.5, -0.5], [0.5, 0.5]), f, eps=0.04)
    assert len(zonos) > 0 and all(z.type == Geometry.TYPE.ZONOTOPE for z in zonos)
