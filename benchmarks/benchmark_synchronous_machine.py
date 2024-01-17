import numpy as np
from pybdr.algorithm import ASB2008CDC, ASB2008CDC
from pybdr.geometry import Zonotope, Interval, Geometry
from pybdr.geometry.operation import boundary
from pybdr.model import *

if __name__ == '__main__':
    # settings for the computation
    options = ASB2008CDC.Options()
    options.t_end = 7.0
    options.step = 0.01
    options.tensor_order = 3
    options.taylor_terms = 4
    options.u = Zonotope.zero(1, 1)
    options.u_trans = np.zeros(1)

    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    z = Interval([-0.7, 2.3], [0.7, 3.7])

    cell_nums = {12: 0.5, 16: 0.4, 20: 0.3, 24: 0.25}
    xs = boundary(z, cell_nums[24], Geometry.TYPE.ZONOTOPE)

    ri_with_bound, rp_with_bound = ASB2008CDC.reach_parallel(synchronous_machine, [2, 1], options, xs)

    # visualize the results
    # plot(ri_with_bound, [0, 1])
