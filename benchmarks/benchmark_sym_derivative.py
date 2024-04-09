import numpy as np
from pybdr.geometry import Interval
from pybdr.model import tank6eq, Model
from pybdr.util.functional import performance_counter, performance_counter_start


def test_sym_derivative_case_00():
    sys_test = tank6eq
    dimes = [6, 1]

    m = Model(sys_test, dimes)

    x_np, u_np = np.random.rand(6), np.random.rand(1)

    print()

    time_cur = performance_counter_start()

    # ----------------------------------------------------------------------
    # order 0, for variable 0

    np_derivative_00 = m.evaluate((x_np, u_np), 'numpy', 0, 0)  # Run 1

    time_cur = performance_counter(time_cur, 'derivative 00 Run 1')

    # 10 runs
    np_derivative_00 = m.evaluate((x_np, u_np), 'numpy', 0, 0)
    np_derivative_00 = m.evaluate((x_np, u_np), 'numpy', 0, 0)
    np_derivative_00 = m.evaluate((x_np, u_np), 'numpy', 0, 0)
    np_derivative_00 = m.evaluate((x_np, u_np), 'numpy', 0, 0)
    np_derivative_00 = m.evaluate((x_np, u_np), 'numpy', 0, 0)
    np_derivative_00 = m.evaluate((x_np, u_np), 'numpy', 0, 0)
    np_derivative_00 = m.evaluate((x_np, u_np), 'numpy', 0, 0)
    np_derivative_00 = m.evaluate((x_np, u_np), 'numpy', 0, 0)
    np_derivative_00 = m.evaluate((x_np, u_np), 'numpy', 0, 0)
    np_derivative_00 = m.evaluate((x_np, u_np), 'numpy', 0, 0)

    time_cur = performance_counter(time_cur, 'derivative 00 Run 2+ AVG', 10)

    # ----------------------------------------------------------------------
    # order 0, for variable 1

    np_derivative_01 = m.evaluate((x_np, u_np), 'numpy', 0, 1)  # Run 1

    time_cur = performance_counter(time_cur, 'derivative 01 Run 1')

    # 10 runs
    np_derivative_01 = m.evaluate((x_np, u_np), 'numpy', 0, 1)
    np_derivative_01 = m.evaluate((x_np, u_np), 'numpy', 0, 1)
    np_derivative_01 = m.evaluate((x_np, u_np), 'numpy', 0, 1)
    np_derivative_01 = m.evaluate((x_np, u_np), 'numpy', 0, 1)
    np_derivative_01 = m.evaluate((x_np, u_np), 'numpy', 0, 1)
    np_derivative_01 = m.evaluate((x_np, u_np), 'numpy', 0, 1)
    np_derivative_01 = m.evaluate((x_np, u_np), 'numpy', 0, 1)
    np_derivative_01 = m.evaluate((x_np, u_np), 'numpy', 0, 1)
    np_derivative_01 = m.evaluate((x_np, u_np), 'numpy', 0, 1)
    np_derivative_01 = m.evaluate((x_np, u_np), 'numpy', 0, 1)

    time_cur = performance_counter(time_cur, 'derivative 01 Run 2+ AVG', 10)

    # ----------------------------------------------------------------------
    # order 1, for variable 0

    np_derivative_10 = m.evaluate((x_np, u_np), 'numpy', 1, 0)  # Run 1

    time_cur = performance_counter(time_cur, 'derivative 10 Run 1')

    # 10 runs
    np_derivative_10 = m.evaluate((x_np, u_np), 'numpy', 1, 0)
    np_derivative_10 = m.evaluate((x_np, u_np), 'numpy', 1, 0)
    np_derivative_10 = m.evaluate((x_np, u_np), 'numpy', 1, 0)
    np_derivative_10 = m.evaluate((x_np, u_np), 'numpy', 1, 0)
    np_derivative_10 = m.evaluate((x_np, u_np), 'numpy', 1, 0)
    np_derivative_10 = m.evaluate((x_np, u_np), 'numpy', 1, 0)
    np_derivative_10 = m.evaluate((x_np, u_np), 'numpy', 1, 0)
    np_derivative_10 = m.evaluate((x_np, u_np), 'numpy', 1, 0)
    np_derivative_10 = m.evaluate((x_np, u_np), 'numpy', 1, 0)
    np_derivative_10 = m.evaluate((x_np, u_np), 'numpy', 1, 0)

    time_cur = performance_counter(time_cur, 'derivative 10 Run 2+ AVG', 10)

    # ----------------------------------------------------------------------
    # order 1, for variable 1

    np_derivative_11 = m.evaluate((x_np, u_np), 'numpy', 1, 1)  # Run 1

    time_cur = performance_counter(time_cur, 'derivative 10 Run 1')

    np_derivative_11 = m.evaluate((x_np, u_np), 'numpy', 1, 1)
    np_derivative_11 = m.evaluate((x_np, u_np), 'numpy', 1, 1)
    np_derivative_11 = m.evaluate((x_np, u_np), 'numpy', 1, 1)
    np_derivative_11 = m.evaluate((x_np, u_np), 'numpy', 1, 1)
    np_derivative_11 = m.evaluate((x_np, u_np), 'numpy', 1, 1)
    np_derivative_11 = m.evaluate((x_np, u_np), 'numpy', 1, 1)
    np_derivative_11 = m.evaluate((x_np, u_np), 'numpy', 1, 1)
    np_derivative_11 = m.evaluate((x_np, u_np), 'numpy', 1, 1)
    np_derivative_11 = m.evaluate((x_np, u_np), 'numpy', 1, 1)
    np_derivative_11 = m.evaluate((x_np, u_np), 'numpy', 1, 1)

    time_cur = performance_counter(time_cur, 'derivative 11 Run 2+ AVG', 10)

    # ----------------------------------------------------------------------
    # order 3, for variable 0

    np_derivative_30 = m.evaluate((x_np, u_np), 'numpy', 3, 0)  # Run 1

    time_cur = performance_counter(time_cur, 'derivative 10 Run 1')

    # 10 runs
    np_derivative_30 = m.evaluate((x_np, u_np), 'numpy', 3, 0)
    np_derivative_30 = m.evaluate((x_np, u_np), 'numpy', 3, 0)
    np_derivative_30 = m.evaluate((x_np, u_np), 'numpy', 3, 0)
    np_derivative_30 = m.evaluate((x_np, u_np), 'numpy', 3, 0)
    np_derivative_30 = m.evaluate((x_np, u_np), 'numpy', 3, 0)
    np_derivative_30 = m.evaluate((x_np, u_np), 'numpy', 3, 0)
    np_derivative_30 = m.evaluate((x_np, u_np), 'numpy', 3, 0)
    np_derivative_30 = m.evaluate((x_np, u_np), 'numpy', 3, 0)
    np_derivative_30 = m.evaluate((x_np, u_np), 'numpy', 3, 0)
    np_derivative_30 = m.evaluate((x_np, u_np), 'numpy', 3, 0)

    time_cur = performance_counter(time_cur, 'derivative 30 Run 2+ AVG', 10)

    # ----------------------------------------------------------------------
    # order 3, for variable 1

    np_derivative_31 = m.evaluate((x_np, u_np), 'numpy', 3, 1)  # Run 1

    time_cur = performance_counter(time_cur, 'derivative 10 Run 1')

    # 10 runs
    np_derivative_31 = m.evaluate((x_np, u_np), 'numpy', 3, 1)
    np_derivative_31 = m.evaluate((x_np, u_np), 'numpy', 3, 1)
    np_derivative_31 = m.evaluate((x_np, u_np), 'numpy', 3, 1)
    np_derivative_31 = m.evaluate((x_np, u_np), 'numpy', 3, 1)
    np_derivative_31 = m.evaluate((x_np, u_np), 'numpy', 3, 1)
    np_derivative_31 = m.evaluate((x_np, u_np), 'numpy', 3, 1)
    np_derivative_31 = m.evaluate((x_np, u_np), 'numpy', 3, 1)
    np_derivative_31 = m.evaluate((x_np, u_np), 'numpy', 3, 1)
    np_derivative_31 = m.evaluate((x_np, u_np), 'numpy', 3, 1)
    np_derivative_31 = m.evaluate((x_np, u_np), 'numpy', 3, 1)

    time_cur = performance_counter(time_cur, 'derivative 31 Run 2+ AVG', 10)


def test_sym_derivative_case_01():
    print("test_sym_derivative_case_01")


def test_sym_derivative_case_02():
    print("test_sym_derivative_case_02")


if __name__ == "__main__":
    test_sym_derivative_case_00()
    test_sym_derivative_case_01()
    test_sym_derivative_case_02()
