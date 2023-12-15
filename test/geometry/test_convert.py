import numpy as np

from pybdr.geometry import Geometry, Interval, Polytope, Zonotope
from pybdr.geometry.operation import cvt2
from pybdr.util.visualization import plot


def test_interval2interval():
    a = Interval.rand(3)
    b = cvt2(a, Geometry.TYPE.INTERVAL)
    plot([a, b], [0, 1])
    plot([a, b], [1, 2])


def test_interval2polytope():
    a = Interval.rand(3)
    b = cvt2(a, Geometry.TYPE.POLYTOPE)
    plot([a, b], [0, 1])
    plot([a, b], [1, 2])


def test_interval2zonotope():
    a = Interval.rand(3)
    b = cvt2(a, Geometry.TYPE.ZONOTOPE)
    plot([a, b], [0, 1])
    plot([a, b], [1, 2])


def test_polytope2interval():
    a = Polytope.rand(3)
    b = cvt2(a, Geometry.TYPE.INTERVAL)
    plot([a, b], [0, 1])
    plot([a, b], [1, 2])


def test_polytope2polytope():
    a = Polytope.rand(3)
    b = cvt2(a, Geometry.TYPE.POLYTOPE)
    plot([a, b], [0, 1])
    plot([a, b], [1, 2])


def test_polytope2zonotope():
    a = Polytope.rand(3)
    b = cvt2(a, Geometry.TYPE.ZONOTOPE)
    plot([a, b], [0, 1])
    plot([a, b], [1, 2])


def test_zonotope2interval():
    a = Zonotope.rand(3, 10)
    b = cvt2(a, Geometry.TYPE.INTERVAL)
    plot([a, b], [0, 1])
    plot([a, b], [1, 2])


def test_zonotope2polytope():
    a = Zonotope.rand(3, 10)
    b = cvt2(a, Geometry.TYPE.POLYTOPE)
    plot([b, a], [0, 1])
    plot([b, a], [1, 2])


def test_zonotope2zonotope():
    a = Zonotope.rand(3, 10)
    b = cvt2(a, Geometry.TYPE.ZONOTOPE)
    plot([b, a], [0, 1])
    plot([b, a], [1, 2])
    c = a.reduce(Zonotope.REDUCE_METHOD.GIRARD, 3)
    plot([a, c], [0, 1])
    plot([a, c], [1, 2])


def test_pts2polytope():
    pts = np.random.rand(100, 2)
    poly = cvt2(pts, Geometry.TYPE.POLYTOPE)
    plot([pts, poly], [0, 1])
