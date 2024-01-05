from pybdr.algorithm import ASB2008CDCParallel
from pybdr.geometry import Interval, Zonotope, Geometry
from pybdr.dynamic_system import NonLinSys
from pybdr.model import *
from pybdr.geometry.operation import cvt2, boundary
from pybdr.util.visualization import plot
from pybdr.util.functional import performance_counter, performance_counter_start
import numpy as np


def test_case_00():
    time_tag = performance_counter_start()
    # settings for the computation
    opts = ASB2008CDCParallel.Options()
    opts.t_end = 5.4
    opts.step = 0.02
    opts.tensor_order = 2
    opts.taylor_terms = 4

    opts.u = Zonotope([0], np.diag([0]))
    opts.u_trans = opts.u.c

    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    z = Interval([0.1, 0.1], [0.3, 0.3])

    z_bounds = boundary(z, 0.01, Geometry.TYPE.ZONOTOPE)

    print(len(z_bounds))

    ri, rp = ASB2008CDCParallel.reach_parallel(brusselator, [2, 1], opts, z_bounds)

    performance_counter(time_tag, "ASB2008CDCParallel")

    plot(ri, [0, 1])
