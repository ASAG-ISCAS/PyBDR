import numpy as np
from pyrat.geometric.zonotope import Zonotope


def test_construction():
    z = Zonotope.random_fix_dim(2)
    print(z)
    print(z.dimension)
    assert z.dimension == 2


def test_numeric_operations():
    z = Zonotope(np.array([[3, 2, -4, -6, 3, 5, 0],
                           [3, 1, -7, 3, -5, 2, 0],
                           [-2, 0, 4, -7, 3, 2, 0]], dtype=float))
    abs_z = abs(z)
    assert np.allclose(abs(z.center), abs_z.center)
    assert np.allclose(abs(z.generator), abs_z.generator)
    nz = z.remove_empty_gen()
    z0 = z + 1.2
    z0 += 2.5
    z1 = z + z0
    z2 = z0 - z1
    # z3=Zonotope.__sub__(z,z1,method=)


def test_auxiliary_functions():
    z = Zonotope.random_fix_dim(2)
    print(z)
    print(z.dimension)
    print(z.center, z.center.shape)
    print(z.generator, z.generator.shape)


if __name__ == '__main__':
    test_construction()
    test_numeric_operations()
    test_auxiliary_functions()
