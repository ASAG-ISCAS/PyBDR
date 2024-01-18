from pybdr.util.functional import Simulator
from pybdr.dynamic_system import LinSys, NonLinSys
from pybdr.model import *
from pybdr.util.visualization import plot
import numpy as np


def test_case_00():
    """
    single x, single u
    @return:
    """
    xa = [[-1, -4], [4, -1]]
    ub = [[1], [1]]
    t_end = 5
    step = 0.01

    init_x = [1, 1]
    init_u = [0]

    lin_sys = LinSys(xa, ub)
    trajs = Simulator.simulate(lin_sys, t_end, step, init_x, init_u)

    plot(trajs, [0, 1])


def test_case_01():
    """
    single x, multiple u
    @return:
    """
    xa = [[-1, -4], [4, -1]]
    ub = [[1], [1]]
    t_end = 5
    step = 0.01

    init_x = [1, 1]
    init_u = np.random.rand(100, 1) * 0.1

    lin_sys = LinSys(xa, ub)
    trajs = Simulator.simulate(lin_sys, t_end, step, init_x, init_u)

    plot(trajs, [0, 1])


def test_case_02():
    """
    multiple x, single u
    @return:
    """
    xa = [[-1, -4], [4, -1]]
    ub = [[1], [1]]
    t_end = 5
    step = 0.01

    init_x = np.random.rand(100, 2) * 0.2 + 1
    init_u = [0]

    lin_sys = LinSys(xa, ub)
    trajs = Simulator.simulate(lin_sys, t_end, step, init_x, init_u)

    plot(trajs, [0, 1])


def test_case_03():
    """
    multiple x, multiple u
    @return:
    """
    xa = [[-1, -4], [4, -1]]
    ub = [[1], [1]]
    t_end = 5
    step = 0.01

    init_x = np.random.rand(100, 2) * 0.2 + 1
    init_u = np.random.rand(100, 1) * 0.1

    lin_sys = LinSys(xa, ub)
    trajs = Simulator.simulate(lin_sys, t_end, step, init_x, init_u)

    plot(trajs, [0, 1])


def test_case_04():
    """
    5-dimensional x, 5-dimensional u
    @return:
    """
    xa = [[-1, -4, 0, 0, 0], [4, -1, 0, 0, 0], [0, 0, -3, 1, 0], [0, 0, -1, -3, 0], [0, 0, 0, 0, -2]]
    ub = np.eye(5)
    t_end = 5
    step = 0.01

    init_x = np.ones((1, 5))
    init_u = [1, 0, 0, 0.5, -0.5]

    lin_sys = LinSys(xa, ub)
    trajs = Simulator.simulate(lin_sys, t_end, step, init_x, init_u)

    plot(trajs, [1, 2])
    plot(trajs, [3, 4])


def test_case_05():
    """
    multiple x, multiple u
    @return:
    """
    xa = [[-1, -4, 0, 0, 0], [4, -1, 0, 0, 0], [0, 0, -3, 1, 0], [0, 0, -1, -3, 0], [0, 0, 0, 0, -2]]
    ub = np.eye(5)
    t_end = 5
    step = 0.01

    init_x = np.random.rand(100, 5) * 0.2 + 1
    init_u = np.random.rand(100, 5) * 0.1 + [1, 0, 0, 0.5, -0.5]

    lin_sys = LinSys(xa, ub)
    trajs = Simulator.simulate(lin_sys, t_end, step, init_x, init_u)

    plot(trajs, [1, 2])
    plot(trajs, [3, 4])


def test_case_06():
    """
    nonlinear
    @return:
    """
    nonlin_sys = NonLinSys(Model(brusselator, [2, 1]))
    t_end = 5.4
    step = 0.02

    init_x = np.random.rand(2) * 0.2 + 0.1
    init_u = np.random.rand(1) * 0.1

    trajs = Simulator.simulate(nonlin_sys, t_end, step, init_x, init_u)

    plot(trajs, [0, 1])


def test_case_07():
    nonlin_sys = NonLinSys(Model(vanderpol, [2, 1]))
    t_end = 10
    step = 0.005

    init_x = [1.4, 2.4]
    init_u = 0

    trajs = Simulator.simulate(nonlin_sys, t_end, step, init_x, init_u)

    plot(trajs, [0, 1])


if __name__ == '__main__':
    pass
