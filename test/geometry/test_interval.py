from pyrat.geometry import Geometry, Interval
from pyrat.util.visualization import vis2dGeo, vis3dGeo


def test_interval_2d_boundary():
    a = Interval([-2, 4], [2, 5])
    boundary = a.boundary(1, Geometry.TYPE.INTERVAL)
    vis2dGeo(boundary, [0, 1])
    vis2dGeo(boundary, [1, 0])


def test_interval_3d_boundary():

    a = Interval(
        [-5, -2.5, 3],
        [-1, 3.7, 9],
    )
    boundary = a.boundary(1, Geometry.TYPE.INTERVAL)
    vis3dGeo(boundary, [0, 1, 2])


def test_interval_4d_boundary():
    a = Interval([-1, 3, -2, 9], [2, 7, 5, 11])
    boundary = a.boundary(1, Geometry.TYPE.INTERVAL)
    vis3dGeo(boundary, [0, 1, 2])
    vis3dGeo(boundary, [1, 2, 3])
