import numpy as np

from pyrat.geometry import Polytope, cvt2, Geometry
from pyrat.util.visualization import vis2dGeo


def test_construction():
    a = np.array(
        [
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        ]
    )
    b = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 2, 2, 1, 2, 3])
    p = Polytope(a, b)

    vis2dGeo([p], [0, 1])


def test_vis():
    a = np.array([[-3, 0], [2, 4], [1, -2], [1, 1]])
    b = np.array([-1, 14, 1, 4])
    p = Polytope(a, b)
    boxes = p.boundary(0.05, Geometry.TYPE.INTERVAL)
    vis2dGeo([p, *boxes], [0, 1])


def test_construction_from_vertices():
    vs = np.random.rand(100, 2)
    p = cvt2(vs, Geometry.TYPE.POLYTOPE)
    boxes = p.boundary(0.01, Geometry.TYPE.INTERVAL)
    vis2dGeo([p, *boxes], [0, 1])
