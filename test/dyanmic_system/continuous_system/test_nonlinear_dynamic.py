import numpy as np
from inspect import signature


def test_sympy():
    import sympy as sy

    x, t, z, nu = sy.symbols("x t z nu")
    x = sy.symbols("x:3")
    print(x)
    dxdt = sy.symbols(("dxdt:5"))
    print(dxdt)
    expr = sy.sin(x[0]) * sy.exp(x[0]) + 1
    print(expr)
    f = sy.lambdify(x, expr, "numpy")
    ivf = sy.lambdify(x, expr, "mpmath")
    print(signature(f))
    print(signature(ivf))
    print(f(*np.ones(3)))
    import mpmath as mm

    x = mm.iv.matrix(3, 1)
    print(x)
    print(ivf(*x))

    # TODO
    pass


def test_interval_matrix():
    import mpmath as mm

    a = np.array((2, 2), dtype=object)
    v = mm.iv.mpf(1)
    print(v.a)
    a = mm.iv.zeros(2, 2)
    a[0, 0] = mm.iv.mpf([])
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
