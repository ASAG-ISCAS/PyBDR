import numpy as np

from pybdr.algorithm import XSE2016CAV, ASB2008CDC
from pybdr.dynamic_system import NonLinSys
from pybdr.geometry import Interval, Zonotope
from pybdr.model import *
from pybdr.util.visualization import plot


def test_case_00():
    # settings for the under approximation computation
    options = XSE2016CAV.Options()
    options.t_end = 2
    options.step = 0.3
    options.r0 = Interval([-0.1, 0.1], [2.9, 3.1])
    options.epsilon = 0.5
    options.epsilon_m = 0.1

    # settings for one step backward over approximation computation
    options_back_one_step = ASB2008CDC.Options()
    options_back_one_step.t_end = options.step
    options_back_one_step.step = options.step / 3
    options_back_one_step.tensor_order = 2
    options_back_one_step.taylor_terms = 4
    options_back_one_step.u = Zonotope.zero(1, 1)
    options_back_one_step.u_trans = np.zeros(1)

    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    # reachable sets computation
    rs, _ = XSE2016CAV.reach(synchronous_machine, [2, 1], options, options_back_one_step)

    # visualize the results
    plot(rs, [0, 1])
