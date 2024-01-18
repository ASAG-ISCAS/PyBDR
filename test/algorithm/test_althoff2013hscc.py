import numpy as np
from pybdr.algorithm import ALTH2013HSCC
from pybdr.dynamic_system import NonLinSys
from pybdr.geometry import Geometry, Zonotope, Interval
from pybdr.geometry.operation import cvt2, boundary
from pybdr.model import *
from pybdr.util.visualization import plot, plot_cmp
from pybdr.util.functional import performance_counter_start, performance_counter


def test_case_0():
    # settings for the computation
    options = ALTH2013HSCC.Options()
    options.t_end = 6.74
    options.step = 0.005
    options.taylor_terms = 4
    options.tensor_order = 3
    # options.r0 = [Zonotope([1.4, 2.4], np.diag([0.17, 0.06]))]
    options.u = Zonotope.zero(1, 1)
    options.u_trans = np.zeros(1)
    x0 = Zonotope([1.4, 2.4], np.diag([0.17, 0.06]))

    # settings for using Zonotope
    Zonotope.ORDER = 50
    Zonotope.INTERMEDIATE_ORDER = 50
    Zonotope.ERROR_ORDER = 20

    # reachable sets
    ri, rp = ALTH2013HSCC.reach(vanderpol, [2, 1], options, x0)

    # visualize the results
    plot(rp, [0, 1])


def test_case_1():
    # settings for the computation
    options = ALTH2013HSCC.Options()
    options.t_end = 6.74
    options.step = 0.005
    options.taylor_terms = 4
    options.tensor_order = 3
    options.u = Zonotope.zero(1, 1)
    options.u_trans = np.zeros(1)

    x0 = Interval([1.23, 2.34], [1.57, 2.46])
    xs = boundary(x0, 1, Geometry.TYPE.ZONOTOPE)

    # settings for using Zonotope
    Zonotope.ORDER = 50
    Zonotope.INTERMEDIATE_ORDER = 50
    Zonotope.ERROR_ORDER = 20

    # reachable sets
    ri, rp = ALTH2013HSCC.reach_parallel(vanderpol, [2, 1], options, xs)

    # visualize the results
    plot(rp, [0, 1])


def test_pi_controller_with_disturbance_cmp():
    # settings for the computation
    options = ALTH2013HSCC.Options()
    options.t_end = 4
    options.step = 0.01
    options.tensor_order = 3
    options.taylor_terms = 4

    options.u = Zonotope([0], np.diag([0]))
    options.u_trans = options.u.c

    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    z = Interval([0.9, -0.1], [1.1, 0.1])
    x0 = cvt2(z, Geometry.TYPE.ZONOTOPE)
    xs = boundary(z, 1, Geometry.TYPE.ZONOTOPE)

    ri_without_bound, rp_without_bound = ALTH2013HSCC.reach(pi_controller_with_disturbance, [2, 1], options, x0)
    ri_with_bound, rp_with_bound = ALTH2013HSCC.reach_parallel(pi_controller_with_disturbance, [2, 1], options, xs)

    # visualize the results
    plot_cmp([ri_without_bound, ri_with_bound], [0, 1], cs=["#FF5722", "#303F9F"], filled=True)


def test_vanderpol_cmp():
    # settings for the computation
    options = ALTH2013HSCC.Options()
    options.t_end = 6.74
    options.step = 0.005
    options.taylor_terms = 4
    options.tensor_order = 3
    options.u = Zonotope.zero(1, 1)
    options.u_trans = np.zeros(1)

    # settings for using Zonotope
    Zonotope.ORDER = 50
    Zonotope.INTERMEDIATE_ORDER = 50
    Zonotope.ERROR_ORDER = 20

    z = Interval([1.23, 2.34], [1.57, 2.46])

    x0 = cvt2(z, Geometry.TYPE.ZONOTOPE)
    xs = boundary(z, 1, Geometry.TYPE.ZONOTOPE)

    ri_without_bound, rp_without_bound = ALTH2013HSCC.reach(vanderpol, [2, 1], options, x0)
    ri_with_bound, rp_with_bound = ALTH2013HSCC.reach_parallel(vanderpol, [2, 1], options, xs)

    # visualize the results
    plot_cmp([ri_without_bound, ri_with_bound], [0, 1], cs=["#FF5722", "#303F9F"], filled=True)


def test_tank6eq():
    # settings for the computation
    options = ALTH2013HSCC.Options()
    options.t_end = 400
    options.step = 4
    options.taylor_terms = 4
    options.tensor_order = 3
    options.u = Zonotope.zero(1, 1)
    options.u_trans = np.zeros(1)

    # settings for using Zonotope
    Zonotope.ORDER = 50
    Zonotope.INTERMEDIATE_ORDER = 50
    Zonotope.ERROR_ORDER = 20

    c = np.array([2, 4, 4, 2, 10, 4])
    z = Interval(c - 0.2, c + 0.2)

    x0 = cvt2(z, Geometry.TYPE.ZONOTOPE)
    xs = boundary(z, 1, Geometry.TYPE.ZONOTOPE)

    time_tag = performance_counter_start()

    ri_without_bound, rp_without_bound = ALTH2013HSCC.reach(tank6eq, [6, 1], options, x0)
    time_tag = performance_counter(time_tag, 'reach')

    ri_with_bound, rp_with_bound = ALTH2013HSCC.reach_parallel(tank6eq, [6, 1], options, xs)
    time_tag = performance_counter(time_tag, 'reach_parallel')

    # visualize the results
    plot_cmp([ri_without_bound, ri_with_bound], [0, 1], cs=["#FF5722", "#303F9F"], filled=True)
    plot_cmp([ri_without_bound, ri_with_bound], [2, 3], cs=["#FF5722", "#303F9F"], filled=True)
    plot_cmp([ri_without_bound, ri_with_bound], [4, 5], cs=["#FF5722", "#303F9F"], filled=True)
