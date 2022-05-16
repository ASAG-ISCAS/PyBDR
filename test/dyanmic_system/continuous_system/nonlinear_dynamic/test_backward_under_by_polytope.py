import numpy as np
from pyrat.dynamic_system import NonLinSys
from pyrat.model import *
from pyrat.geometry import Interval
from pyrat.util.visualization import vis2d


def test_synchronous_machine():
    system = NonLinSys.Entity(SynchronousMachine())

    # setting for the computation
    option = NonLinSys.Option.BackUnder()
    option.t_end = 10
    option.steps = 20
    option.tensor_order = 2
    option.taylor_terms = 4
    option.r0 = [Interval([-0.1, -0.1], [0.1, 0.1])]
    option.u = Interval.zero(1)
    option.u_trans = np.zeros(1)
    option.epsilon_m = 0.001
    option.epsilon = 0.5

    # setting for using geometry
    # TODO nothing to set

    results = system.reach(option)
    vis2d(results, [0, 1])
