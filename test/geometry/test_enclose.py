import numpy as np

from pybdr.geometry import Zonotope, Interval, Polytope, Geometry
from pybdr.geometry.operation import enclose
from pybdr.util.visualization import plot


def test_zonotope_by_zontope():
    a = Zonotope.rand(2, 5)
    b = Zonotope.rand(2, 10)
    c = enclose(a, b)
    plot([a, b, c], [0, 1])


def test_00():
    a = Zonotope([1.5, 1.5], [[1, 0], [0, 1]])
    m = np.array([[-1, 0], [0, -1]])
    b = m @ a + 0.5 * np.ones(2)
    c = enclose(a, b)
    plot([a, b, c], [0, 1])


def test_01():
    a = Zonotope.rand(2, 10)
    b = a + np.ones(2) * 10
    c = enclose(a, b)
    plot([a, b, c], [0, 1])


def test_02():
    a = Zonotope.rand(2, 10)
    b = Zonotope.rand(2, 3) + 10 * np.ones(2)
    c = enclose(a, b)
    plot([a, b, c], [0, 1])


def test_03():
    a = Zonotope.rand(2, 4)
    # b = Zonotope.rand(2, 5)
    from pybdr.geometry import Interval, Geometry
    from pybdr.geometry.operation import cvt2
    b = Interval.rand(2)
    b = cvt2(b, Geometry.TYPE.ZONOTOPE)
    c = a + b
    d = enclose(a, b)
    plot([a, b, c, d], [0, 1])


def test_04():
    a = Zonotope.rand(2, 10)
    b = 10 * a
    plot([a + b, 11 * a], [0, 1])


def test_05():
    from pybdr.geometry import Interval, Geometry
    from pybdr.geometry.operation import cvt2
    a = Interval.rand(2)
    b = 10 * a
    za = cvt2(a, Geometry.TYPE.ZONOTOPE)
    zb = cvt2(b, Geometry.TYPE.ZONOTOPE)
    d = enclose(za, zb)
    plot([a, b, 11 * a], [0, 1])
    plot([za, zb, za + zb, d], [0, 1])


def test_ii2i():
    a = Interval.rand(2)
    b = Interval.rand(2)
    c = enclose(a, b, Geometry.TYPE.INTERVAL)
    plot([a, b, c], [0, 1])


def test_zz2z():
    a = Zonotope.rand(2, 10)
    b = Zonotope.rand(2, 5)
    c = enclose(a, b, Geometry.TYPE.ZONOTOPE)
    plot([a, b, c], [0, 1])
