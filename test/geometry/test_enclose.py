import numpy as np

from pybdr.geometry import Zonotope
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
