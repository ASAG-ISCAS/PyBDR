import numpy as np

from pyrat.geometry import VectorZonotope
from pyrat.util.visualization import vis2d_old
from pyrat.geometry import cvt2


def test_np_function():
    v = np.random.rand(10)
    v[[1, 3, 4]] = np.inf
    m = np.random.rand(7, 5)
    a = np.array([[1, 2, 3], [4, 5, 6], [9, 8, 7]])
    print(a)
    b = np.linalg.norm(a, axis=1, ord=2)


def test_construction():
    z = VectorZonotope.rand(2)
    p = z.to("polyhedron")
    print(p)
    print(z)
    print(z.dim)
    assert z.dim == 2


def test_numeric_operations():
    z = VectorZonotope(
        np.array(
            [[3, 2, -4, -6, 3, 5, 0], [3, 1, -7, 3, -5, 2, 0], [-2, 0, 4, -7, 3, 2, 0]],
            dtype=float,
        )
    )
    abs_z = abs(z)
    assert np.allclose(abs(z.c), abs_z.c)
    assert np.allclose(abs(z.gen), abs_z.gen)
    z.remove_zero_gen()
    z0 = z + np.array([1, 0, 1])
    z0 += np.ones(3)
    z1 = z + z0
    z2 = np.ones(3) + z0
    z3 = z0 - np.ones(3)
    z4 = z + 1
    z5 = 1 + z
    temp = z == z3
    print(temp)
    z6 = np.random.rand(3, 3) @ z5
    print()
    print(z4)
    print(z5)
    print(z2)
    exit(False)
    # z2 = z0 - z1
    # z3=Zonotope.__sub__(z,z1,method=)


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
    z = VectorZonotope.rand(2, 4)
    # print(z)
    # print(z.dim)
    # print(z.center, z.center.shape)
    # print(z.generator, z.generator.shape)
    print(z.z)
    vis2d_old([z])


def test_polygon():
    z = VectorZonotope(
        np.array(
            [
                [0.62573669, 0.10487259, 0.21082717, 0.12907895],
                [0.66141065, 0.87252549, 0.07096709, 0.26345968],
            ],
            dtype=float,
        )
    )
    print(z.polygon())
    vis2d_old([z])

    interval = cvt2(z, "int")
    print()
    print(interval)


def test_vis_2d():
    z = VectorZonotope(np.array([[0, -2, 3, -7, 9], [0, -9, 6, -8, -5]]))
    vis2d_old([z.c.reshape((-1, 2)), z.polygon(), z])


def test_enclose():
    z0 = VectorZonotope(np.array([[0, -2, 3, -7, 9], [0, -9, 6, -8, -5]]))
    z1 = VectorZonotope(
        np.array(
            [
                [0.62573669, 0.10487259, 0.21082717, 0.12907895],
                [0.66141065, 0.87252549, 0.07096709, 0.26345968],
            ]
        )
    )
    z0 = VectorZonotope.rand(2, 2)
    z1 = VectorZonotope.rand(2, 19)
    z2 = z0 | z1
    z3 = z1 | z0
    print()
    print()
    print(z0.z.T)
    print("---------------------")
    print(z1.z.T)
    print("\n")
    print(z3.z.T)


if __name__ == "__main__":
    test_construction()
    test_numeric_operations()
    test_auxiliary_functions()
