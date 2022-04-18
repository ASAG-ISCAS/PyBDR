import numpy as np

from pyrat.geometry import Geometry, Interval


def test_basic():
    a = Interval([-1, -2.5, 2.9, 3.1], [-1, 3.7, 3.0, 9])
    boundary = a.boundary(1, Geometry.TYPE.INTERVAL)


def test_mesh_grid():
    arrs_num = 5
    arrs = [np.arange(arrs_num) for idx in range(arrs_num)]
    cp = np.array(np.meshgrid(*arrs)).T.reshape((-1, arrs_num))
    print(cp)


def test_sampling():
    temp = np.linspace(1, 1.2, 2)
    print(temp)
