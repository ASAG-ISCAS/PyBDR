import numpy as np

from pyrat.geometry import Polyhedron


def test_basic():
    arr = np.random.rand(3, 4)
    p = Polyhedron(arr)
    p.min_affine_rep(np.random.rand(3, 3))
    b = p
    Polyhedron.empty(2)
    Polyhedron.fullspace(2)
