import numpy as np
from pyrat.algorithm import ALTHOFF2013HSCC
from pyrat.dynamic_system import NonLinSys
from pyrat.geometry import Geometry, Zonotope
from pyrat.geometry.operation import cvt2, boundary
from pyrat.model import *
from pyrat.util.visualization import plot, plot_cmp


def test_case_0():
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


def test_pi_controller_with_disturbance_cmp():
    # init dynamic system
    system = NonLinSys(Model(pi_controller_with_disturbance, [2, 1]))

    # settings for the computation
    options = ALTHOFF2013HSCC.Options()
    options.t_end = 2
    options.step = 0.005
    options.tensor_order = 3
    options.taylor_terms = 4

    options.u = Zonotope([0], np.diag([0]))
    options.u_trans = options.u.c

    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    z = Zonotope([1, 0], np.diag([0.1, 0.1]))

    # reachable sets computation without boundary analysis
    options.r0 = [z]
    ti_whole, tp_whole, _, _ = ALTHOFF2013HSCC.reach(system, options)

    with_bound = False

    tp_bound = []
    if with_bound:
        # reachable sets computation with boundary analysis
        options.r0 = boundary(z, 1, Geometry.TYPE.ZONOTOPE)
        ti_bound, tp_bound, _, _ = ALTHOFF2013HSCC.reach(system, options)

    # visualize the results
    plot_cmp([tp_whole, tp_bound], [0, 1], cs=['#FF5722', '#303F9F'])


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
    options.r0 = [Zonotope([1.4, 2.4], np.diag([0.17, 0.06]))]
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
