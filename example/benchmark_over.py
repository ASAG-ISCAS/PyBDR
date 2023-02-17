from __future__ import annotations
import numpy as np

np.seterr(divide='ignore', invalid='ignore')
from pyrat.geometry import Geometry, Zonotope, Interval
from pyrat.dynamic_system import NonLinSys
from pyrat.model import vanderpol, Model
from pyrat.algorithm import ASB2008CDC
from pyrat.geometry.operation import boundary
from pyrat.util.visualization import plot_cmp

if __name__ == '__main__':
    # init dynamic system
    system = NonLinSys(Model(vanderpol, [2, 1]))

    # settings for the computation
    options = ASB2008CDC.Options()
    options.t_end = 6.74
    options.step = 0.005
    options.tensor_order = 3
    options.taylor_terms = 4

    options.u = Zonotope.zero(1, 1)
    options.u_trans = np.zeros(1)

    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    z = Zonotope([1.4, 2.4], np.diag([0.17, 0.06]))

    # reachable sets computation without boundary analysis
    options.r0 = [z]
    ti_whole, tp_whole, _, _ = ASB2008CDC.reach(system, options)

    # reachable sets computation with boundary analysis
    options.r0 = boundary(z, 1, Geometry.TYPE.ZONOTOPE)
    ti_bound, tp_bound, _, _ = ASB2008CDC.reach(system, options)

    # visualize the results
    plot_cmp([tp_whole, tp_bound], [0, 1], cs=['#FF5722', '#303F9F'])
