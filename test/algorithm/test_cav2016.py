import numpy as np
from pyrat.algorithm import CAV2016, CDC2008
from pyrat.dynamic_system import NonLinSys
from pyrat.geometry import Interval, Zonotope
from pyrat.model import *
from pyrat.util.visualization import vis2d


def test_synchronous_machine():
    # init dynamic system
    system = NonLinSys.Entity(SynchronousMachine())

    # settings for the under approximation computation
    options = CAV2016.Options()
    options.t_end = 10
    options.step = 0.5
    options.r0 = Interval([-0.1, 0.1], [-0.1, 0.1])
    options.epsilon = 0.5
    options.epsilon_m = 0.001

    # settings for one step backward over approximation computation
    options_back_one_step = CDC2008.Options()
    options_back_one_step.t_end = 0.5
    options_back_one_step.step = 0.5
    options_back_one_step.tensor_order = 2
    options_back_one_step.taylor_terms = 4
    options_back_one_step.u = Zonotope.zero(1, 1)
    options_back_one_step.u_trans = np.zeros(1)

    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    # reachable sets computation
    results = CAV2016.reach(system, options, options_back_one_step)

    # visualize the results
    vis2d(results, [0, 1])
