import numpy as np

from pyrat.geometry import VectorZonotope
from pyrat.util.visualization import vis2d


def test_np_function():
    v = np.random.rand(10)
    v[[1, 3, 4]] = np.inf
    m = np.random.rand(7, 5)
    a = np.array([[1, 2, 3], [4, 5, 6], [9, 8, 7]])
    print(a)
    b = np.linalg.norm(a, axis=1, ord=2)


def test_construction():
    z = VectorZonotope.rand_fix_dim(2)
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
    assert np.allclose(abs(z.center), abs_z.center)
    assert np.allclose(abs(z.generator), abs_z.generator)
    nz = z.delete_zeros()
    z0 = z + 1.2
    z0 += 2.5
    z1 = z + z0
    # z2 = z0 - z1
    # z3=Zonotope.__sub__(z,z1,method=)


def test_auxiliary_functions():
    z = VectorZonotope.rand_fix_dim(2)
    print(z)
    print(z.dim)
    print(z.center, z.center.shape)
    print(z.generator, z.generator.shape)


def test_vis_2d():
    z = VectorZonotope(np.array([[0, 2, 3], [0, 9, 6]]))
    vis2d([z])


if __name__ == "__main__":
    test_construction()
    test_numeric_operations()
    test_auxiliary_functions()
