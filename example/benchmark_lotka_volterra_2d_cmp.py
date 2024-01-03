import matplotlib.pyplot as plt
import numpy as np
import sympy

from pybdr.algorithm import ASB2008CDC
from pybdr.dynamic_system import NonLinSys
from pybdr.geometry import Zonotope, Interval, Geometry
from pybdr.geometry.operation import boundary, cvt2
from pybdr.model import *
from pybdr.util.visualization import plot, plot_cmp

if __name__ == '__main__':
    # init dynamic system
    system = NonLinSys(Model(lotka_volterra_2d, [2, 1]))

    # settings for the computation
    options = ASB2008CDC.Options()
    options.t_end = 2.2
    options.step = 0.005
    options.tensor_order = 3
    options.taylor_terms = 4

    options.u = Zonotope.zero(1, 1)
    options.u_trans = np.zeros(1)

    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    z = Interval([2.5, 2.5], [3.5, 3.5])

    # reachable sets computation without boundary analysis
    options.r0 = [cvt2(z, Geometry.TYPE.ZONOTOPE)]
    ti_whole, tp_whole, _, _ = ASB2008CDC.reach(system, options)

    # reachable sets computation with boundary analysis
    options.r0 = boundary(z, 1, Geometry.TYPE.ZONOTOPE)
    ti_bound, tp_bound, _, _ = ASB2008CDC.reach(system, options)

    # visualize the results
    plot_cmp([tp_whole, tp_bound], [0, 1], cs=["#FF5722", "#303F9F"])
