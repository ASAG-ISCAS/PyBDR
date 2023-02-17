import numpy as np
from pyrat.algorithm import ALTHOFF2013HSCC
from pyrat.dynamic_system import NonLinSys
from pyrat.geometry import Geometry, Zonotope
from pyrat.geometry.operation import cvt2, boundary
from pyrat.model import *
from pyrat.util.visualization import plot


def test_case_0():
    # init dynamic system
    system = NonLinSys(Model(vanderpol, [2, 1]))

    # settings for the computation
    options = ALTHOFF2013HSCC.Options()
    options.t_end = 6.74
    options.step = 0.005
    options.taylor_terms = 4
    options.tensor_order = 2
    options.r0 = [Zonotope([1.4, 2.4], np.diag([0.17, 0.06]))]
    options.u = Zonotope.zero(1, 1)
    options.u_trans = np.zeros(1)

    # settings for using Zonotope
    Zonotope.ORDER = 50
    Zonotope.INTERMEDIATE_ORDER = 50
    Zonotope.ERROR_ORDER = 20

    # reachable sets
    ti, tp, _, _ = ALTHOFF2013HSCC.reach(system, options)

    # visualize the results
    plot(tp, [0, 1])


def test_vanderpol_bound_reach():
    # init dynamic system
    system = NonLinSys(Model(vanderpol, [2, 1]))

    # settings for the computation
    options = ALTHOFF2013HSCC.Options()
    options.t_end = 6.74
    options.step = 0.005
    options.taylor_terms = 4
    options.tensor_order = 2
    z = Zonotope([1.4, 2.4], np.diag([0.17, 0.06]))
    options.r0 = boundary(z, 1, Geometry.TYPE.ZONOTOPE)
    options.u = Zonotope.zero(1, 1)
    options.u_trans = np.zeros(1)

    # settings for using Zonotope
    Zonotope.ORDER = 50
    Zonotope.INTERMEDIATE_ORDER = 50
    Zonotope.ERROR_ORDER = 20

    # reachable sets
    ti, tp, _, _ = ALTHOFF2013HSCC.reach(system, options)

    # visualize the results
    plot(tp, [0, 1])


def test_van_der_pol_using_zonotope():
    # init dynamic system
    system = NonLinSys(Model(vanderpol, [2, 1]))

    # settings for the computation
    options = ALTHOFF2013HSCC.Options()
    options.t_end = 6.74
    options.step = 0.005
    options.taylor_terms = 4
    options.tensor_order = 3
    options.r0 = [Zonotope([1.4, 2.4], np.diag([0.17, 0.06]))]
    options.u = Zonotope.zero(1, 1)
    options.u_trans = np.zeros(1)

    # settings for using Zonotope
    Zonotope.ORDER = 50
    Zonotope.INTERMEDIATE_ORDER = 50
    Zonotope.ERROR_ORDER = 20

    # reachable sets
    ti, tp, _, _ = ALTHOFF2013HSCC.reach(system, options)

    # visualize the results
    plot(tp, [0, 1])


def test_van_der_pol_using_polyzonotope():
    # init dynamic system
    system = NonLinSys(Model(vanderpol, [2, 1]))

    # settings for the computation
    options = ALTHOFF2013HSCC.Options()
    options.t_end = 6.74
    options.step = 0.005
    options.taylor_terms = 4
    options.tensor_order = 3
    poly_zono = cvt2(
        Zonotope([1.4, 2.4], np.diag([0.17, 0.06])), Geometry.TYPE.POLY_ZONOTOPE)
    options.r0 = [poly_zono]
    options.u = Zonotope.zero(1, 1)
    options.u_trans = np.zeros(1)

    # reachable sets
    ti, tp, _, _ = ALTHOFF2013HSCC.reach(system, options)

    # visualize the results
    plot(tp, [0, 1])


def test_tank6eq():
    # init dynamic system
    system = NonLinSys(Model(tank6eq, [6, 1]))

    # settings for the computations
    options = ALTHOFF2013HSCC.Options()
    options.t_end = 400
    options.step = 4
    options.tensor_order = 3
    options.taylor_terms = 4
    options.r0 = [Zonotope([2, 4, 4, 2, 10, 4], np.eye(6) * 0.2)]
    options.u = Zonotope([0], [[0.005]])
    options.u = Zonotope.zero(1, 1)
    options.u_trans = np.zeros(1)

    # settings for using geometry
    Zonotope.REDUCE_METHOD = Zonotope.METHOD.REDUCE.GIRARD
    Zonotope.ORDER = 50
    Zonotope.INTERMEDIATE_ORDER = 50
    Zonotope.ERROR_ORDER = 20

    ti, tp, _, _ = ALTHOFF2013HSCC.reach(system, options)

    plot(tp, [0, 1])
    plot(tp, [2, 3])
    plot(tp, [4, 5])
