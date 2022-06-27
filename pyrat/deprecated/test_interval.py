import numpy as np

from pyrat.geometry import Geometry, IntervalOld, Zonotope
from pyrat.util.visualization import vis2dGeo, vis3dGeo


def test_interval_2d_boundary():
    a = IntervalOld([-2, 4], [2, 5])
    boundary = a.boundary(1, Geometry.TYPE.INTERVAL)
    print(len(boundary))
    # vis2dGeo(boundary, [0, 1])
    # vis2dGeo(boundary, [1, 0])
    print(a.vertices)
    vis2dGeo([a, a.vertices], [0, 1])


def test_interval_3d_boundary():
    a = IntervalOld([-2, 2.5, -3], [1, 3.7, 2.7])
    boundary = a.boundary(1, Geometry.TYPE.INTERVAL)
    vis3dGeo(boundary, [0, 1, 2])


def test_interval_4d_boundary():
    a = IntervalOld([-1, 3, -2, 9], [2, 7, 5, 11])
    boundary = a.boundary(1, Geometry.TYPE.INTERVAL)
    vis3dGeo(boundary, [0, 1, 2])


def test_interval_contains():
    int0 = IntervalOld([-3, -2], [5, 4])
    zono0 = Zonotope([0.5, 0], [[2, 1], [1, -0.7]])
    zono1 = Zonotope([6.5, -3], [[2, 1], [1, -0.7]])
    print(zono0 in int0)
    print(zono1 in int0)


def test_zonotope_contains():
    z = Zonotope(
        [-4, 1, 4], np.array([[-3, -2, -1, 9, 0], [2, 3, 4, 2, 7], [6, -7, 3, 3, 1]])
    )
    print(z.vertices.shape)
    print(z.vertices)


def test_matrix_mul():
    a = IntervalOld([-1, 1, -3], [2, 3, -1])
    b = np.array([[0.8, 0.12, 0.15], [0.9, 0.9, 0.2]])
    print(b)
    print(b * a)
    print(b @ a)
    # TODO need to recheck the implementation
