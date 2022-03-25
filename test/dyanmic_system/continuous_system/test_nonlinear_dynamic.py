from inspect import signature

import numpy as np
import scipy.special

from pyrat.dynamic_system import NonLinearSystem as nls
from pyrat.geometry import VectorZonotope
from pyrat.model import tank6Eq


def test_case_0():
    """
    test case for tank6eq model using nonlinear system over-reach analysis
    :return:
    """

    """
    init nonlinear system --------------------------------------------------------------
    """
    system = nls.System(tank6Eq())

    """
    init parameters for reachability analysis problem definition -----------------------
    """
    params = nls.Parameters()
    params.time_end = 400
    params.set_r0 = VectorZonotope(
        np.vstack([np.array([2, 4, 4, 2, 10, 4]), 0.2 * np.eye(6)]).T
    )
    params.set_ru = VectorZonotope(np.array([0, 0.005]).reshape((1, -1)))

    """
    init options for the computation ---------------------------------------------------
    """
    options = nls.Options()
    options.time_step = 0.1
    options.taylor_terms = 4
    options.zonotope_order = 50
    options.algo = "standard"
    options.tensor_order = 2
    options.lagrange_rem["simplify"] = "simplify"

    """
    over approximating reachability analysis -------------------------------------------
    """
    reachable_results = system.reach_over(params, options)

    """
    simulation -------------------------------------------------------------------------
    """
    simulate_results = system.simulate_rand(params, options)

    """
    results visualization --------------------------------------------------------------
    """


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
