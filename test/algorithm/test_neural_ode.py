import numpy as np


np.seterr(divide='ignore',invalid='ignore')
import sys
sys.path.append("./../../")
from pyrat.algorithm import ASB2008CDC
from pyrat.dynamic_system import NonLinSys
from pyrat.geometry import Zonotope, Interval, Geometry
from pyrat.model import *
from pyrat.util.visualization import plot
from pyrat.neural_ode.model_generate import neuralODE

if __name__=="__main__":
    # init neural ODE
    system = NonLinSys(Model(neuralODE, [2,1]))

    # settings for the computation
    options = ASB2008CDC.Options()
    options.t_end = 8
    options.step = 0.02
    options.tensor_order = 2
    options.taylor_terms = 2
    options.r0 = [Zonotope([1.4, 2.4], np.diag([0.17, 0.06]))]
    from pyrat.geometry.operation import boundary

    c = np.array([1.4, 2.4], dtype=float)
    inf = c - [0.17, 0.06]
    sup = c + [0.17, 0.06]
    box = Interval(inf, sup)

    options.u = Zonotope.zero(1, 1)
    options.u_trans = np.zeros(1)

    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    # reachable sets computation
    ti, tp, _, _ = ASB2008CDC.reach(system, options)

    # visualize the results
    plot(tp, [0, 1])