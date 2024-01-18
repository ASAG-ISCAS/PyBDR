import numpy as np

from pybdr.algorithm import ASB2008CDC, ASB2008CDC
from pybdr.util.functional import performance_counter, performance_counter_start
from pybdr.geometry import Zonotope, Interval, Geometry
from pybdr.geometry.operation import boundary
from pybdr.model import *
from pybdr.util.visualization import plot_cmp, plot
# from save_results import save_result
from pybdr.geometry.operation.convert import cvt2

if __name__ == '__main__':
    # settings for the computation
    options = ASB2008CDC.Options()
    options.t_end = 5.0
    options.step = 0.01
    options.tensor_order = 3
    options.taylor_terms = 4
    options.u = Zonotope.zero(1, 1)
    options.u_trans = np.zeros(1)

    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    z = Interval([-0.1, 3.9], [0.1, 4.1])

    cell_nums = {12: 0.05, 16: 0.06, 20: 0.045}
    xs = boundary(z, cell_nums[20], Geometry.TYPE.ZONOTOPE)

    ri_with_bound, rp_with_bound = ASB2008CDC.reach_parallel(brusselator, [2, 1], options, xs)

    # visualize the results
    plot(ri_with_bound, [0, 1])
