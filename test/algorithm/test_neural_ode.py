import numpy as np


np.seterr(divide='ignore',invalid='ignore')
import sys
sys.path.append("./../../")
from pyrat.algorithm import ASB2008CDC
from pyrat.dynamic_system import NonLinSys
from pyrat.geometry import Zonotope, Interval, Geometry
from pyrat.model import *
from pyrat.util.visualization import plot
from pyrat.util.functional.neural_ode_generate import neuralODE
from pyrat.geometry.operation import boundary

if __name__=="__main__":
    # init neural ODE
    system = NonLinSys(Model(neuralODE, [2,1]))

    # settings for the computation
    options = ASB2008CDC.Options()
    options.t_end = 0.1
    options.step = 0.01
    options.tensor_order = 2
    options.taylor_terms = 2
    Z = Zonotope([0.5, 0], np.diag([0.5, 0.5]))
    
    # Reachable sets computed with boundary analysis
    # options.r0 = boundary(Z,1,Geometry.TYPE.ZONOTOPE)

    # Reachable sets computed without boundary analysis
    options.r0 = [Z]

    options.u = Zonotope.zero(1, 1)
    options.u_trans = np.zeros(1)

    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    # reachable sets computation
    ti, tp, _, _ = ASB2008CDC.reach(system, options)

    # visualize the results
    plot(tp, [0, 1])