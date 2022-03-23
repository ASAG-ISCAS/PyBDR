import numpy as np


def test_sympy():
    import sympy as sy

    x, t, z, nu = sy.symbols("x t z nu")
    expr = sy.diff(sy.sin(x) * sy.exp(x), x, 100)
    print(expr)
    f = sy.lambdify(x, expr, "numpy")
    print(f(np.ones(3)))

    # TODO
    pass


def test_interval_matrix():
    import mpmath as mm

    a = np.array((2, 2), dtype=object)
    a = mm.iv.zeros(2, 2)
    a[0, 0].pm = -1000
    print(a[0, 0])
    print(mm.expm(a))
    print(a)

    pass


def test_nonlinear_reach_01_tank():
    # TODO
    pass


def test_nonlinear_reach_02_vanDerPol_polyZonotope():
    # TODO
    pass
