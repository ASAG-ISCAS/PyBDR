import matplotlib.pyplot as plt
import numpy as np
import sympy

from pybdr.algorithm import ASB2008CDC
from pybdr.dynamic_system import NonLinSys
from pybdr.geometry import Zonotope, Interval, Geometry
from pybdr.geometry.operation import boundary, cvt2
from pybdr.model import *
from pybdr.util.visualization import plot, plot_cmp


def test_case_00():
    # settings for the computation
    options = ASB2008CDC.Options()
    options.t_end = 5.4
    options.step = 0.02
    options.tensor_order = 2
    options.taylor_terms = 4

    options.u = Zonotope([0], np.diag([0]))
    options.u_trans = options.u.c

    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    z = Interval([0.1, 0.1], [0.3, 0.3])
    x0 = cvt2(z, Geometry.TYPE.ZONOTOPE)
    xs = boundary(z, 1, Geometry.TYPE.ZONOTOPE)

    ri_without_bound, rp_without_bound = ASB2008CDC.reach(brusselator, [2, 1], options, x0)
    ri_with_bound, rp_with_bound = ASB2008CDC.reach_parallel(brusselator, [2, 1], options, xs)

    # visualize the results
    plot_cmp([ri_without_bound, ri_with_bound], [0, 1], cs=["#FF5722", "#303F9F"], filled=True)


def test_case_01():
    # settings for the computation
    options = ASB2008CDC.Options()
    options.t_end = 5.8
    options.step = 0.02
    options.tensor_order = 2
    options.taylor_terms = 4

    options.u = Zonotope([0], np.diag([0]))
    options.u_trans = options.u.c

    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    z = Interval([0.1, 0.1], [0.3, 0.3])
    x0 = cvt2(z, Geometry.TYPE.ZONOTOPE)
    xs = boundary(z, 1, Geometry.TYPE.ZONOTOPE)

    ri_without_bound, rp_without_bound = ASB2008CDC.reach(brusselator, [2, 1], options, x0)
    ri_with_bound, rp_with_bound = ASB2008CDC.reach_parallel(brusselator, [2, 1], options, xs)

    # visualize the results
    plot_cmp([ri_without_bound, ri_with_bound], [0, 1], cs=["#FF5722", "#303F9F"], filled=True)


def test_case_02():
    # settings for the computation
    options = ASB2008CDC.Options()
    options.t_end = 3.2
    options.step = 0.01
    options.tensor_order = 2
    options.taylor_terms = 4

    options.u = Zonotope.zero(4)
    options.u_trans = options.u.c

    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    z = Interval([1.15, 3.4, -0.02], [1.35, 3.6, 0.02])

    x0 = cvt2(z, Geometry.TYPE.ZONOTOPE)
    xs = boundary(z, 0.5, Geometry.TYPE.ZONOTOPE)

    ri_without_bound, rp_without_bound = ASB2008CDC.reach(ltv, [3, 4], options, x0)
    ri_with_bound, rp_with_bound = ASB2008CDC.reach_parallel(ltv, [3, 4], options, xs)

    # visualize the results
    plot_cmp([ri_without_bound, ri_with_bound], [0, 1], cs=["#FF5722", "#303F9F"], filled=True)
    plot_cmp([ri_without_bound, ri_with_bound], [2, 0], cs=["#FF5722", "#303F9F"], filled=True)
    plot_cmp([ri_without_bound, ri_with_bound], [2, 1], cs=["#FF5722", "#303F9F"], filled=True)


def test_case_03():
    # settings for the computation
    options = ASB2008CDC.Options()
    options.t_end = 10
    options.step = 0.01
    options.tensor_order = 2
    options.taylor_terms = 4

    options.u = Zonotope([0], np.diag([0]))
    options.u_trans = options.u.c

    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    z = Interval([0.8, 0.8], [1.2, 1.2])
    x0 = cvt2(z, Geometry.TYPE.ZONOTOPE)
    xs = boundary(z, 1, Geometry.TYPE.ZONOTOPE)

    ri_without_bound, rp_without_bound = ASB2008CDC.reach(jet_engine, [2, 1], options, x0)
    ri_with_bound, rp_with_bound = ASB2008CDC.reach_parallel(jet_engine, [2, 1], options, xs)

    # visualize the results
    plot_cmp([ri_without_bound, ri_with_bound], [0, 1], cs=["#FF5722", "#303F9F"], filled=True)


def test_case_04():
    # settings for the computation
    options = ASB2008CDC.Options()
    options.t_end = 2
    options.step = 0.01
    options.tensor_order = 2
    options.taylor_terms = 4

    options.u = Zonotope([0], np.diag([0]))
    options.u_trans = options.u.c

    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    z = Interval([0.9, -0.1], [1.1, 0.1])
    x0 = cvt2(z, Geometry.TYPE.ZONOTOPE)
    xs = boundary(z, 1, Geometry.TYPE.ZONOTOPE)

    ri_without_bound, rp_without_bound = ASB2008CDC.reach(pi_controller_with_disturbance, [2, 1], options, x0)
    ri_with_bound, rp_with_bound = ASB2008CDC.reach_parallel(pi_controller_with_disturbance, [2, 1], options, xs)

    # visualize the results
    plot_cmp([ri_without_bound, ri_with_bound], [0, 1], cs=["#FF5722", "#303F9F"], filled=True)


def test_case_05():
    # settings for the computation
    options = ASB2008CDC.Options()
    options.t_end = 0.2
    options.step = 0.001
    options.tensor_order = 3
    options.taylor_terms = 4

    options.u = Zonotope([0], np.diag([0]))
    options.u_trans = options.u.c

    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    z = Interval([9.9, 9.99, -0.01], [10.1, 10.01, 0.01])
    x0 = cvt2(z, Geometry.TYPE.ZONOTOPE)
    xs = boundary(z, 1, Geometry.TYPE.ZONOTOPE)

    ri_without_bound, rp_without_bound = ASB2008CDC.reach(rossler_attractor, [3, 1], options, x0)
    ri_with_bound, rp_with_bound = ASB2008CDC.reach_parallel(rossler_attractor, [3, 1], options, xs)

    # visualize the results
    plot_cmp([ri_without_bound, ri_with_bound], [0, 1], cs=["#FF5722", "#303F9F"], filled=True)
    plot_cmp([ri_without_bound, ri_with_bound], [0, 2], cs=["#FF5722", "#303F9F"], filled=True)
    plot_cmp([ri_without_bound, ri_with_bound], [1, 2], cs=["#FF5722", "#303F9F"], filled=True)


def test_case_06():
    # settings for the computation
    options = ASB2008CDC.Options()
    options.t_end = 6.5
    options.step = 0.01
    options.tensor_order = 3
    options.taylor_terms = 4

    options.u = Zonotope([0], np.diag([0]))
    options.u_trans = options.u.c

    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    z = Interval([14.99, 14.99, 35.99], [15.01, 15.01, 36.01])
    x0 = cvt2(z, Geometry.TYPE.ZONOTOPE)
    xs = boundary(z, 1, Geometry.TYPE.ZONOTOPE)

    ri_without_bound, rp_without_bound = ASB2008CDC.reach(lorentz, [3, 1], options, x0)
    ri_with_bound, rp_with_bound = ASB2008CDC.reach_parallel(lorentz, [3, 1], options, xs)

    # visualize the results
    plot_cmp([ri_without_bound, ri_with_bound], [0, 1], cs=["#FF5722", "#303F9F"], filled=True)
    plot_cmp([ri_without_bound, ri_with_bound], [0, 2], cs=["#FF5722", "#303F9F"], filled=True)
    plot_cmp([ri_without_bound, ri_with_bound], [1, 2], cs=["#FF5722", "#303F9F"], filled=True)


def test_case_07():
    # settings for the computation
    options = ASB2008CDC.Options()
    options.t_end = 2.2
    options.step = 0.005
    options.tensor_order = 3
    options.taylor_terms = 4

    options.u = Zonotope([0], np.diag([0]))
    options.u_trans = options.u.c

    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    z = Interval([2.5, 2.5], [3.5, 3.5])
    x0 = cvt2(z, Geometry.TYPE.ZONOTOPE)
    xs = boundary(z, 1, Geometry.TYPE.ZONOTOPE)

    ri_without_bound, rp_without_bound = ASB2008CDC.reach(lotka_volterra_2d, [2, 1], options, x0)
    ri_with_bound, rp_with_bound = ASB2008CDC.reach_parallel(lotka_volterra_2d, [2, 1], options, xs)

    # visualize the results
    plot_cmp([ri_without_bound, ri_with_bound], [0, 1], cs=["#FF5722", "#303F9F"], filled=True)


def test_case_08():
    # settings for the computation
    options = ASB2008CDC.Options()
    options.t_end = 3
    options.step = 0.01
    options.tensor_order = 3
    options.taylor_terms = 4

    options.u = Zonotope([0], np.diag([0]))
    options.u_trans = options.u.c

    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    z = Interval([-0.2, 2.8], [0.2, 3.2])
    x0 = cvt2(z, Geometry.TYPE.ZONOTOPE)
    xs = boundary(z, 1, Geometry.TYPE.ZONOTOPE)

    ri_without_bound, rp_without_bound = ASB2008CDC.reach(synchronous_machine, [2, 1], options, x0)
    ri_with_bound, rp_with_bound = ASB2008CDC.reach_parallel(synchronous_machine, [2, 1], options, xs)

    # visualize the results
    plot_cmp([ri_without_bound, ri_with_bound], [0, 1], cs=["#FF5722", "#303F9F"], filled=True)


def test_case_09():
    # settings for the computation
    options = ASB2008CDC.Options()
    options.t_end = 3
    options.step = 0.01
    options.tensor_order = 3
    options.taylor_terms = 4

    options.u = Zonotope([0], np.diag([0]))
    options.u_trans = options.u.c

    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    z = Interval([9.9, -7], [10.1, -3])
    x0 = cvt2(z, Geometry.TYPE.ZONOTOPE)
    xs = boundary(z, 1, Geometry.TYPE.ZONOTOPE)

    ri_without_bound, rp_without_bound = ASB2008CDC.reach(ode2d, [2, 1], options, x0)
    ri_with_bound, rp_with_bound = ASB2008CDC.reach_parallel(ode2d, [2, 1], options, xs)

    # visualize the results
    plot_cmp([ri_without_bound, ri_with_bound], [0, 1], cs=["#FF5722", "#303F9F"], filled=True)


def test_case_10():
    # settings for the computation
    options = ASB2008CDC.Options()
    options.t_end = 6.74
    options.step = 0.005
    options.tensor_order = 3
    options.taylor_terms = 4

    options.u = Zonotope([0], np.diag([0]))
    options.u_trans = options.u.c

    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    z = Interval([1.23, 2.34], [1.57, 2.46])
    x0 = cvt2(z, Geometry.TYPE.ZONOTOPE)
    xs = boundary(z, 1, Geometry.TYPE.ZONOTOPE)

    ri_without_bound, rp_without_bound = ASB2008CDC.reach(vanderpol, [2, 1], options, x0)
    ri_with_bound, rp_with_bound = ASB2008CDC.reach_parallel(vanderpol, [2, 1], options, xs)

    # visualize the results
    plot_cmp([ri_without_bound, ri_with_bound], [0, 1], cs=["#FF5722", "#303F9F"], filled=True)


def test_case_11():
    # settings for the computation
    options = ASB2008CDC.Options()
    options.t_end = 400
    options.step = 10
    options.tensor_order = 3
    options.taylor_terms = 4

    options.u = Zonotope([0], np.diag([0]))
    options.u_trans = options.u.c

    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    c = np.array([2, 4, 4, 2, 10, 4])
    z = Interval(c - 0.2, c + 0.2)
    x0 = cvt2(z, Geometry.TYPE.ZONOTOPE)
    xs = boundary(z, 1, Geometry.TYPE.ZONOTOPE)

    ri_without_bound, rp_without_bound = ASB2008CDC.reach(tank6eq, [6, 1], options, x0)
    ri_with_bound, rp_with_bound = ASB2008CDC.reach_parallel(tank6eq, [6, 1], options, xs)

    # visualize the results
    plot_cmp([ri_without_bound, ri_with_bound], [0, 1], cs=["#FF5722", "#303F9F"], filled=True)
    plot_cmp([ri_without_bound, ri_with_bound], [2, 3], cs=["#FF5722", "#303F9F"], filled=True)
    plot_cmp([ri_without_bound, ri_with_bound], [4, 5], cs=["#FF5722", "#303F9F"], filled=True)


def test_case_12():
    # settings for the computation
    options = ASB2008CDC.Options()
    options.t_end = 20
    options.step = 0.04
    options.tensor_order = 2
    options.taylor_terms = 4

    options.u = Zonotope([0], np.diag([0]))
    options.u_trans = options.u.c

    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    c = np.array([1.2, 1.05, 1.5, 2.4, 1, 0.1, 0.45])
    z = Interval(c - 0.01, c + 0.01)
    x0 = cvt2(z, Geometry.TYPE.ZONOTOPE)
    xs = boundary(z, 1, Geometry.TYPE.ZONOTOPE)

    ri_without_bound, rp_without_bound = ASB2008CDC.reach(laubloomis, [7, 1], options, x0)
    ri_with_bound, rp_with_bound = ASB2008CDC.reach_parallel(laubloomis, [7, 1], options, xs)

    # visualize the results
    plot_cmp([ri_without_bound, ri_with_bound], [0, 1], cs=["#FF5722", "#303F9F"], filled=True)
    plot_cmp([ri_without_bound, ri_with_bound], [2, 4], cs=["#FF5722", "#303F9F"], filled=True)
    plot_cmp([ri_without_bound, ri_with_bound], [5, 6], cs=["#FF5722", "#303F9F"], filled=True)
    plot_cmp([ri_without_bound, ri_with_bound], [1, 5], cs=["#FF5722", "#303F9F"], filled=True)
    plot_cmp([ri_without_bound, ri_with_bound], [4, 6], cs=["#FF5722", "#303F9F"], filled=True)


def test_case_13():
    # settings for the computation
    options = ASB2008CDC.Options()
    options.t_end = 1
    options.step = 0.01
    options.tensor_order = 2
    options.taylor_terms = 2

    options.u = Zonotope([0], np.diag([0]))
    options.u_trans = options.u.c

    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    z = Interval([0, -0.5], [1, 0.5])
    x0 = cvt2(z, Geometry.TYPE.ZONOTOPE)
    xs = boundary(z, 1, Geometry.TYPE.ZONOTOPE)

    ri_without_bound, rp_without_bound = ASB2008CDC.reach(neural_ode_spiral1, [2, 1], options, x0)
    ri_with_bound, rp_with_bound = ASB2008CDC.reach_parallel(neural_ode_spiral1, [2, 1], options, xs)

    # visualize the results
    plot_cmp([ri_without_bound, ri_with_bound], [0, 1], cs=["#FF5722", "#303F9F"], filled=True)
