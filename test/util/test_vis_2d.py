import numpy as np

from pybdr.geometry import *
from pybdr.geometry.operation import cvt2
from pybdr.util.visualization import plot


def test_plot_interval():
    a = Interval.rand(2)
    plot([a, a.vertices], [0, 1])
    print(a)


def test_plot_zonotope():
    a = Zonotope.rand(2, 5)
    plot([a], [0, 1])
    print(a)


def test_plot_polytope():
    from pybdr.geometry.operation import cvt2

    pts = np.random.rand(100, 2)
    a = cvt2(pts, Geometry.TYPE.POLYTOPE)

    plot([a, pts], [0, 1])
    print(a)


def test_plot_polytope_level():
    pts = np.random.rand(10, 2)
    p0 = cvt2(pts, Geometry.TYPE.POLYTOPE)
    a = p0.a
    b0 = p0.b
    b1 = b0 + 0.1

    offset = np.arange(len(b0)) * 0.1 + 0.2
    b2 = b0 + offset

    print(offset)

    poly_00 = Polytope(a, b0)
    poly_01 = Polytope(a, b1)
    poly_10 = Polytope(a, b2)

    plot([poly_00, poly_01, poly_10], [0, 1])
