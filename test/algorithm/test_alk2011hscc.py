import numpy as np

from pybdr.geometry import Zonotope
from pybdr.dynamic_system import LinSys
from pybdr.algorithm import ALK2011HSCC
from pybdr.util.visualization import plot


def test_00():
    a = np.array([[-0.7, -2], [2, -0.7]])
    b = np.identity(2)
    lin_dyn = LinSys(a, ub=b)

    # settings for the computation
    options = ALK2011HSCC.Options()
    options.t_end = 6.74
    options.step = 0.005
    options.taylor_terms = 4
    options.tensor_order = 3
    options.r0 = [Zonotope([1.4, 2.4], np.diag([0.17, 0.06]))]
    options.u = Zonotope.zero(2, 1)
    options.u_trans = np.zeros(2)

    # settings for using Zonotope
    Zonotope.ORDER = 50
    Zonotope.INTERMEDIATE_ORDER = 50
    Zonotope.ERROR_ORDER = 20

    # reachable sets
    ti, tp, _, _ = ALK2011HSCC.reach(lin_dyn, options)

    # visualize the results
    plot(tp, [0, 1])

# def test_case_0():
#     import numpy as np
#     from pybdr.geometry import Zonotope
#     from pybdr.dynamic_system import LinSys
#     from pybdr.model import Model, vanderpol
#     from pybdr.algorithm import ALK2011HSCC
#     from pybdr.util.visualization import plot
#
#     # init dynamic system
#     system = LinSys(Model(vanderpol, [2, 1]))
#
#     # settings for the computation
#     options = ALK2011HSCC.Options()
#     options.t_end = 6.74
#     options.step = 0.005
#     options.taylor_terms = 4
#     options.tensor_order = 3
#     options.r0 = [Zonotope([1.4, 2.4], np.diag([0.17, 0.06]))]
#     options.u = Zonotope.zero(1, 1)
#     options.u_trans = np.zeros(1)
#
#     # settings for using Zonotope
#     Zonotope.ORDER = 50
#     Zonotope.INTERMEDIATE_ORDER = 50
#     Zonotope.ERROR_ORDER = 20
#
#     # reachable sets
#     ti, tp, _, _ = ALK2011HSCC.reach(system, options)
#
#     # visualize the results
#     plot(tp, [0, 1])
