import numpy as np

from pybdr.geometry import Interval, Zonotope, Geometry
from pybdr.dynamic_system import LinSys
from pybdr.geometry.operation import boundary
from pybdr.algorithm import IntervalTensorReachLinear
from pybdr.util.visualization import plot
from pybdr.geometry.operation import cvt2


def test_tensor_reach_linear_case_00():
    # init settings
    xa = np.array([[-1, -4], [4, -1]])
    ub = np.array([[1], [1]])
    lin_sys = LinSys(xa=xa)
    opt = IntervalTensorReachLinear.Options()
    opt.u = ub @ Interval(-0.1, 0.1)
    opt.taylor_terms = 3
    opt.t_end = 10
    opt.step = 0.02

    r0 = Interval([0.9, 0.9], [1.1, 1.1])
    # get the boundary of the initial set and make it as interval tensor
    r0_bounds = boundary(r0, 0.05, Geometry.TYPE.INTERVAL)

    # plot(r0_bounds, [0, 1])
    # make all the bounding boxes into interval tensor
    r0_bounds_tensor = Interval.stack(r0_bounds)
    print(r0_bounds_tensor.shape)
    # exit(False)
    opt.r0 = r0_bounds_tensor
    # compute the reachable set for this interval tensor with specified linear system
    _, tp_set, _, _ = IntervalTensorReachLinear.reach(lin_sys, opt)

    # rearrange the elements of the resulting reachable set to make it suitable for initial problem
    vis_set = []
    for this_tp_set in tp_set:
        this_boxes = Interval.split(this_tp_set, this_tp_set.shape[0])
        this_vs = [this_box.vertices for this_box in this_boxes]
        this_vs = np.stack(this_vs).reshape((-1, 2))
        vis_set.append(this_vs)
        # this_poly = cvt2(this_vs, Geometry.TYPE.POLYTOPE)
        # plot([this_vs], [0, 1])
        # print(this_poly.a)
        # print(this_poly.b)
        # plot([this_vs, this_poly], [0, 1])
        # print('---------------------------------------')
        # vis_set.append(cvt2(this_vs, Geometry.TYPE.POLYTOPE))

    # vis_set = [Interval.split(this_tp_set, this_tp_set.shape[0]) for this_tp_set in tp_set]
    # visualization
    plot(vis_set, [0, 1])


def test_tensor_reach_linear_case_01():
    xa = np.array([[-1, -4, 0, 0, 0], [4. - 1, 0, 0, 0], [0, 0, -3, 1, 0], [0, 0, -1, -3, 0], [0, 0, 0, 0, -2]])
    lin_sys = LinSys(xa=xa)
    options = IntervalTensorReachLinear.Options()
    r0 = Interval(0.9 * np.ones(5), 1.1 * np.ones(5))
    u = Interval([0.9, -0.25, -0.1, 0.25, -0.75], [1.1, 0.25, 0.1, 0.75, -0.25])
    # get the boundary of the inital set and make it as interval tensor
    r0_bounds = boundary(r0, 0.05, Geometry.TYPE.INTERVAL)
