import numpy as np

from pyrat.geometry import Geometry, Zonotope, IntervalOld, IntervalMatrix
from pyrat.geometry.operation import cvt2
from pyrat.util.visualization import vis2dGeo


def test_np_function():
    v = np.random.rand(10)
    v[[1, 3, 4]] = np.inf
    m = np.random.rand(7, 5)
    a = np.array([[1, 2, 3], [4, 5, 6], [9, 8, 7]])
    print(a)
    b = np.linalg.norm(a, axis=1, ord=2)


def test_construction():
    z = Zonotope.rand(2, 5)
    print(z.info)
    assert z.dim == 2


def test_numeric_operations():
    data = np.array(
        [[3, 2, -4, -6, 3, 5, 0], [3, 1, -7, 3, -5, 2, 0], [-2, 0, 4, -7, 3, 2, 0]],
        dtype=float,
    )
    z = Zonotope(data[:, 0], data[:, 1:])
    abs_z = abs(z)
    z.remove_zero_gen()
    z0 = z + np.array([1, 0, 1])
    z0 += np.ones(3)
    z1 = z + z0
    z2 = np.ones(3) + z0
    z3 = z0 - np.ones(3)
    z4 = z + 1
    z5 = 1 + z
    z6 = np.random.rand(3, 3) @ z5
    print()
    print(z4)
    print(z5)
    print(z2)


def test_python():
    data = np.array(
        [
            [3, 0, 0, 0.24],
            [4, 1, 1, 0.41],
            [2, 1, 1, 0.63],
            [1, 1, 3, 0.38],
            [0, 0, 0, 1],
        ]
    )
    ix = np.lexsort(data[::-1, :])
    print()
    print(data)
    print(ix)
    print(data[:, ix])


def test_auxiliary_functions():
    z = Zonotope.rand(2, 4)
    # print(z)
    # print(z.dim)
    # print(z.center, z.center.shape)
    # print(z.generator, z.generator.shape)
    print(z.z)


def test_polygon():
    data = np.array(
        [
            [0.62573669, 0.10487259, 0.21082717, 0.12907895],
            [0.66141065, 0.87252549, 0.07096709, 0.26345968],
        ],
        dtype=float,
    )
    z = Zonotope(data[:, 0], data[:, 1:])
    print()
    temp = z.polygon()
    print(temp)

    interval = cvt2(z, Geometry.TYPE.INTERVAL)

    print(interval)


def test_vis_2d():
    data = np.array([[0, -2, 3, -7, 9], [0, -9, 6, -8, -5]])
    z = Zonotope(data[:, 0], data[:, 1:])
    vis2dGeo([z], [0, 1])


def test_enclose():
    data = np.array([[0, -2, 3, -7, 9], [0, -9, 6, -8, -5]])
    z0 = Zonotope(data[:, 0], data[:, 1:])
    data = np.array(
        [
            [0.62573669, 0.10487259, 0.21082717, 0.12907895],
            [0.66141065, 0.87252549, 0.07096709, 0.26345968],
        ]
    )
    z1 = Zonotope(data[:, 0], data[:, 1:])
    z0 = Zonotope.rand(2, 2)
    z1 = Zonotope.rand(2, 19)
    z2 = z0.enclose(z1)
    z3 = z1.enclose(z0)
    print()
    print()
    print(z0.z.T)
    print("---------------------")
    print(z1.z.T)
    print("\n")
    print(z3.z.T)


def test_mul():
    a = np.array([[2, 4, 5], [7, 6, 8]])
    a = IntervalMatrix(a.T, a.T)
    b = np.array([[-2, 4, -5, 7], [9, 6, 3, 11]])
    b = Zonotope(b[:, 0], b[:, 1:])
    print(a)
    print(b)
    print(a * b)
    print(b * a)
