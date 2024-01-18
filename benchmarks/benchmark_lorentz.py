import numpy as np
from pybdr.algorithm import ASB2008CDC
from pybdr.geometry import Zonotope, Interval, Geometry
from pybdr.geometry.operation import boundary
from pybdr.model import *
from pybdr.util.visualization import plot

if __name__ == '__main__':
    # settings for the computation
    options = ASB2008CDC.Options()
    options.t_end = 1.0
    options.step = 0.02
    options.tensor_order = 3
    options.taylor_terms = 4
    options.u = Zonotope.zero(1, 1)
    options.u_trans = np.zeros(1)

    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    z = Interval([-11, -3, -3], [3, 11, 11])

    cell_nums = {216: 2.5, 294: 2.3, 384: 2}
    xs = boundary(z, cell_nums[384], Geometry.TYPE.ZONOTOPE)

    ri_with_bound, rp_with_bound = ASB2008CDC.reach_parallel(lorentz, [3, 1], options, xs)

    # visualize the results
    # plot(ri_with_bound, [0, 1])
