import numpy as np

from pybdr.geometry import Zonotope, Geometry, Interval
from pybdr.dynamic_system import LinearSystemSimple
from pybdr.algorithm import ALK2011HSCC
from pybdr.geometry.operation import cvt2, boundary
from pybdr.util.visualization import plot, plot_cmp


def test_case_0():
    a = np.array([[-0.7, -2], [2, -0.7]])
    b = np.identity(2)

    lin_dyn = LinearSystemSimple(a, b)

    # settings for the computation
    options = ALK2011HSCC.Options()
    options.t_end = 5
    options.step = 0.01
    options.taylor_terms = 4
    options.tensor_order = 3

    options.u = Zonotope.zero(2, 1)
    options.u_trans = np.zeros(2)

    # settings for using Zonotope
    Zonotope.ORDER = 50
    Zonotope.INTERMEDIATE_ORDER = 50
    Zonotope.ERROR_ORDER = 20

    z = Interval([1.23, 2.34], [1.57, 2.46])

    x0 = cvt2(z, Geometry.TYPE.ZONOTOPE)
    xs = boundary(z, 0.01, Geometry.TYPE.ZONOTOPE)

    ri_without_bound, rp_without_bound = ALK2011HSCC.reach(lin_dyn, options, x0)
    ri_with_bound, rp_with_bound = ALK2011HSCC.reach_parallel(lin_dyn, options, xs)

    # visualize the results
    plot_cmp([ri_without_bound, ri_with_bound], [0, 1], cs=["#FF5722", "#303F9F"], filled=True)
