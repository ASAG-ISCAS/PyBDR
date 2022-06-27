import numpy as np

from pyrat.geometry import *
from pyrat.util.visualization import plot


def test_plot_interval():
    a = Interval.rand(2)
    plot([a, a.vertices], [0, 1])
    print(a)


def test_plot_zonotope():
    a = Zonotope.rand(2, 5)
    plot([a], [0, 1])
    print(a)


def test_plot_polytope():
    from pyrat.geometry.operation import cvt2

    pts = np.random.rand(100, 2)
    a = cvt2(pts, Geometry.TYPE.POLYTOPE)

    plot([a, pts], [0, 1])
    print(a)
