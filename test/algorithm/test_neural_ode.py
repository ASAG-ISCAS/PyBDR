import numpy as np

np.seterr(divide="ignore", invalid="ignore")
import sys

# sys.path.append("./../../") # add directory 'pybdr' into system path
from pybdr.algorithm import ASB2008CDC
from pybdr.dynamic_system import NonLinSys
from pybdr.geometry import Zonotope, Interval, Geometry
from pybdr.model import *
from pybdr.util.visualization import plot
from pybdr.util.functional.neural_ode_generate import neuralODE
from pybdr.geometry.operation import boundary

if __name__ == "__main__":
    # init neural ODE
    system = NonLinSys(Model(neuralODE, [2, 1]))

    # settings for the computation
    options = ASB2008CDC.Options()
    options.t_end = 1
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
