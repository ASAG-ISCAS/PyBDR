from inspect import signature

import numpy as np
import scipy.special

from pyrat.dynamic_system import NonLinSys
from pyrat.geometry import VectorZonotope
from pyrat.misc import Reachable
from pyrat.model import Tank6Eq
from pyrat.util.visualization import vis2d


def test_case_0():
    """
    test case for tank6eq model using nonlinear system over-reach analysis
    :return:
    """

    """
    init nonlinear system --------------------------------------------------------------
    """
    system = NonLinSys.Sys(Tank6Eq())

    """
    init parameters for reachability analysis problem definition -----------------------
    """
    # params = NonLinSys.Parameters()
    # params.time_end = 400
    # params.set_r0 = VectorZonotope(
    #     np.vstack([np.array([2, 4, 4, 2, 10, 4]), 0.2 * np.eye(6)]).T
    # )
    # params.set_ru = VectorZonotope(np.array([0, 0.005]).reshape((1, -1)))

    """
    init options for the computation ---------------------------------------------------
    """
    option = NonLinSys.Option()
    option.t_end = 40
    option.steps = 41
    option.taylor_terms = 4
    option.zonotope_order = 50
    option.algo = "lin"
    option.tensor_order = 2
    option.lagrange_rem["simplify"] = "simplify"
    option.r_init = [Reachable.Element()]
    option.r_init[0].set = VectorZonotope(
        np.vstack([np.array([2, 4, 4, 2, 10, 4]), 0.2 * np.eye(6)]).T
    )
    option.r_init[0].err = np.zeros(option.r_init[0].set.dim, dtype=float)
    option.u = VectorZonotope(np.array([0, 0.005]).reshape((1, -1)))
    option.u_trans = np.zeros(1)

    """
    over approximating reachability analysis -------------------------------------------
    """
    reachable_results = system.reach(option)

    """
    visualization the results
    """

    vis2d(reachable_results, [0, 1])
    vis2d(reachable_results, [2, 3])
    vis2d(reachable_results, [4, 5])

    """
    simulation -------------------------------------------------------------------------
    """
    # simulation_results = system.simulate(option)

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

    v = mm.iv.matrix(2, 1)
    v[0, 0] = mm.iv.mpf([-1, 1])
    print(v + 1)
    print(v)
    print(mm.asin(v))
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
