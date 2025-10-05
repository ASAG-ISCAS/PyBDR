import numpy as np
from sympy import *

from pybdr.model import Model


def test_case_00():
    import pybdr.util.functional.auxiliary as aux

    def f(x, u):
        # parameters
        k0, k1, g = 0.015, 0.01, 9.81
        # dynamic
        dxdt = [None] * 6

        dxdt[0] = u[0] + 0.1 + k1 * (4 - x[5]) - k0 * sqrt(2 * g) * sqrt(x[0])
        dxdt[1] = k0 * sqrt(2 * g) * (sqrt(x[0]) - sqrt(x[1]))
        dxdt[2] = k0 * sqrt(2 * g) * (sqrt(x[1]) - sqrt(x[2]))
        dxdt[3] = k0 * sqrt(2 * g) * (sqrt(x[2]) - sqrt(x[3]))
        dxdt[4] = k0 * sqrt(2 * g) * (sqrt(x[3]) - sqrt(x[4]))
        dxdt[5] = k0 * sqrt(2 * g) * (sqrt(x[4]) - sqrt(x[5]))
        return Matrix(dxdt)

    modelref = Model(f, [6, 1])
    start = aux.performance_counter_start()

    x, u = np.random.rand(6), np.random.rand(1)
    start = aux.performance_counter(start, "xu")
    pre_temp = modelref.evaluate((x, u), "numpy", 3, 1)
    start = aux.performance_counter(start, "pre_temp")
    suc_temp = modelref.evaluate((x, u), "numpy", 3, 0)
    print(suc_temp.shape)
    start = aux.performance_counter(start, "suc_temp")
    tt = modelref.evaluate((x, u), "numpy", 0, 0)
    print(tt.shape)

    from pybdr.geometry import Interval

    x, u = Interval.rand(6), Interval.rand(1)

    temp0 = modelref.evaluate((x, u), "interval", 3, 0)
    start = aux.performance_counter(start, "modref1")
    temp1 = modelref.evaluate((x, u), "interval", 2, 0)
    start = aux.performance_counter(start, "modref2")
    temp2 = modelref.evaluate((x, u), "interval", 2, 0)
    start = aux.performance_counter(start, "modref3")
    temp3 = modelref.evaluate((x, u), "interval", 0, 1)
    print(temp0.shape)
    print(temp1.shape)
    print(temp2.shape)
    print(temp3.shape)


def test_case_01():
    import numpy as np

    from pybdr.geometry import Interval
    from pybdr.model import Model, tank6eq
    from pybdr.util.functional import performance_counter, performance_counter_start

    m = Model(tank6eq, [6, 1])

    time_start = performance_counter_start()
    x, u = np.random.random(6), np.random.rand(1)

    np_derivative_0 = m.evaluate((x, u), "numpy", 3, 0)
    np_derivative_1 = m.evaluate((x, u), "numpy", 3, 1)
    np_derivative_2 = m.evaluate((x, u), "numpy", 0, 0)

    x, u = Interval.rand(6), Interval.rand(1)
    int_derivative_0 = m.evaluate((x, u), "interval", 3, 0)
    int_derivative_1 = m.evaluate((x, u), "interval", 2, 0)
    int_derivative_2 = m.evaluate((x, u), "interval", 2, 0)
    int_derivative_3 = m.evaluate((x, u), "interval", 0, 1)

    performance_counter(time_start, "sym_derivative")


def test_case_02():
    from pybdr.geometry import Interval
    from pybdr.model import Model, tank6eq

    m = Model(tank6eq, [6, 1])
    x, u = Interval.rand(6), Interval.rand(1)
    Jx = m.evaluate((x, u), "interval", 1, 0)
    pts_iv = np.random.rand(100, 6, 1)

    quad_iv = pts_iv.transpose(0, 2, 1) @ Jx[None, ...] @ pts_iv
    quad_iv = Interval.squeeze(quad_iv)
    print(quad_iv.shape)


if __name__ == "__main__":
    # test_case_00()
    # test_case_01()
    test_case_02()
