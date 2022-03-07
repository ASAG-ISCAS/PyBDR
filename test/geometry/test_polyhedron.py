import numpy as np

from pyrat.geometry import Polyhedron
from scipy.spatial import ConvexHull


def test_basic():
    arr = np.random.rand(100, 4)
    hull = ConvexHull(arr)
    print(hull.vertices)
    print(hull)
    exit(False)
    p = Polyhedron(arr)
    b = p
    e = Polyhedron.empty(2)
    f = Polyhedron.fullspace(2)
    print(p.dim)
    print(e.dim)
    print(f.dim)
    print(p.has_hrep)
    print(p.has_vrep)
    print(p.eqb)
    print(p.eqa)
    print(p.ieqh)
    print(p.ieqa)
    print(p.irr_hrep)
    print(p.irr_vrep)
    c = b + f
    e += b
    g = b - f
    print(c)
    print(c.is_fullspace)
