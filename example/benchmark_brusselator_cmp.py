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
    system = NonLinSys(Model(brusselator, [2, 1]))

    # settings for the computation
    options = ASB2008CDC.Options()
    options.t_end = 6.0
    options.step = 0.02
    options.tensor_order = 2
    options.taylor_terms = 4

    options.u = Zonotope([0], np.diag([0]))
    options.u_trans = options.u.c

    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    z = Interval.identity(2) * 0.1 + 0.2;

    no_boundary_analysis = False

    tp_whole, tp_bound = None, None
    xlim, ylim = [0, 2], [0, 2.4]

    if no_boundary_analysis:
        # reachable sets computation without boundary analysis
        options.r0 = [cvt2(z, Geometry.TYPE.ZONOTOPE)]
        ti_whole, tp_whole, _, _ = ASB2008CDC.reach(system, options)
    else:
        # reachable sets computation with boundary analysis
        options.r0 = boundary(z, 0.3, Geometry.TYPE.ZONOTOPE)
        ti_bound, tp_bound, _, _ = ASB2008CDC.reach(system, options)

    # visualize the results
    if no_boundary_analysis:
        plot(tp_whole, [0, 1], xlim=xlim, ylim=ylim)
    else:
        plot(tp_bound, [0, 1], xlim=xlim, ylim=ylim)
