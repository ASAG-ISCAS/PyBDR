import numpy as np
from pybdr.geometry import Interval
from pybdr.model import tank6eq, ltv, Model, quadrocopterControlledEq
from pybdr.util.functional import performance_counter, performance_counter_start


def sym_derivative_case00_NUM_test():
    # sys_test = tank6eq
    # dimes = [6, 1]
    # sys_test = ltv
    # dimes = [3, 4]
    sys_test = quadrocopterControlledEq
    dimes = [12, 3]

    m = Model(sys_test, dimes)

    x_np, u_np = np.random.rand(12), np.random.rand(3)

    print()
    print('NUM Derivative >>>>>>>>>>>>>>>>>>>>>>>>>>>')
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

    time_cur = performance_counter(time_cur, 'derivative 11 Run 1')

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

    time_cur = performance_counter(time_cur, 'derivative 30 Run 1')

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

    time_cur = performance_counter(time_cur, 'derivative 31 Run 1')

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


def sym_derivative_case00_INT_test():
    sys_test = tank6eq
    dimes = [6, 1]

    m = Model(sys_test, dimes)

    x_int, u_int = Interval.rand(6), Interval.rand(1)

    print()
    print('INT Derivative >>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print()

    time_cur = performance_counter_start()

    # ----------------------------------------------------------------------
    # order 0, for variable 0

    int_derivative_00 = m.evaluate((x_int, u_int), 'interval', 0, 0)  # Run 1

    time_cur = performance_counter(time_cur, 'derivative 00 Run 1')

    # 10 runs
    int_derivative_00 = m.evaluate((x_int, u_int), 'interval', 0, 0)
    int_derivative_00 = m.evaluate((x_int, u_int), 'interval', 0, 0)
    int_derivative_00 = m.evaluate((x_int, u_int), 'interval', 0, 0)
    int_derivative_00 = m.evaluate((x_int, u_int), 'interval', 0, 0)
    int_derivative_00 = m.evaluate((x_int, u_int), 'interval', 0, 0)
    int_derivative_00 = m.evaluate((x_int, u_int), 'interval', 0, 0)
    int_derivative_00 = m.evaluate((x_int, u_int), 'interval', 0, 0)
    int_derivative_00 = m.evaluate((x_int, u_int), 'interval', 0, 0)
    int_derivative_00 = m.evaluate((x_int, u_int), 'interval', 0, 0)
    int_derivative_00 = m.evaluate((x_int, u_int), 'interval', 0, 0)

    time_cur = performance_counter(time_cur, 'derivative 00 Run 2+ AVG', 10)

    # ----------------------------------------------------------------------
    # order 0, for variable 1

    int_derivative_01 = m.evaluate((x_int, u_int), 'interval', 0, 1)  # Run 1

    time_cur = performance_counter(time_cur, 'derivative 01 Run 1')

    # 10 runs
    int_derivative_01 = m.evaluate((x_int, u_int), 'interval', 0, 1)
    int_derivative_01 = m.evaluate((x_int, u_int), 'interval', 0, 1)
    int_derivative_01 = m.evaluate((x_int, u_int), 'interval', 0, 1)
    int_derivative_01 = m.evaluate((x_int, u_int), 'interval', 0, 1)
    int_derivative_01 = m.evaluate((x_int, u_int), 'interval', 0, 1)
    int_derivative_01 = m.evaluate((x_int, u_int), 'interval', 0, 1)
    int_derivative_01 = m.evaluate((x_int, u_int), 'interval', 0, 1)
    int_derivative_01 = m.evaluate((x_int, u_int), 'interval', 0, 1)
    int_derivative_01 = m.evaluate((x_int, u_int), 'interval', 0, 1)
    int_derivative_01 = m.evaluate((x_int, u_int), 'interval', 0, 1)

    time_cur = performance_counter(time_cur, 'derivative 01 Run 2+ AVG', 10)

    # ----------------------------------------------------------------------
    # order 1, for variable 0

    int_derivative_10 = m.evaluate((x_int, u_int), 'interval', 1, 0)  # Run 1

    time_cur = performance_counter(time_cur, 'derivative 10 Run 1')

    # 10 runs
    int_derivative_10 = m.evaluate((x_int, u_int), 'interval', 1, 0)
    int_derivative_10 = m.evaluate((x_int, u_int), 'interval', 1, 0)
    int_derivative_10 = m.evaluate((x_int, u_int), 'interval', 1, 0)
    int_derivative_10 = m.evaluate((x_int, u_int), 'interval', 1, 0)
    int_derivative_10 = m.evaluate((x_int, u_int), 'interval', 1, 0)
    int_derivative_10 = m.evaluate((x_int, u_int), 'interval', 1, 0)
    int_derivative_10 = m.evaluate((x_int, u_int), 'interval', 1, 0)
    int_derivative_10 = m.evaluate((x_int, u_int), 'interval', 1, 0)
    int_derivative_10 = m.evaluate((x_int, u_int), 'interval', 1, 0)
    int_derivative_10 = m.evaluate((x_int, u_int), 'interval', 1, 0)

    time_cur = performance_counter(time_cur, 'derivative 10 Run 2+ AVG', 10)

    # ----------------------------------------------------------------------
    # order 1, for variable 1

    int_derivative_11 = m.evaluate((x_int, u_int), 'interval', 1, 1)  # Run 1

    time_cur = performance_counter(time_cur, 'derivative 11 Run 1')

    int_derivative_11 = m.evaluate((x_int, u_int), 'interval', 1, 1)
    int_derivative_11 = m.evaluate((x_int, u_int), 'interval', 1, 1)
    int_derivative_11 = m.evaluate((x_int, u_int), 'interval', 1, 1)
    int_derivative_11 = m.evaluate((x_int, u_int), 'interval', 1, 1)
    int_derivative_11 = m.evaluate((x_int, u_int), 'interval', 1, 1)
    int_derivative_11 = m.evaluate((x_int, u_int), 'interval', 1, 1)
    int_derivative_11 = m.evaluate((x_int, u_int), 'interval', 1, 1)
    int_derivative_11 = m.evaluate((x_int, u_int), 'interval', 1, 1)
    int_derivative_11 = m.evaluate((x_int, u_int), 'interval', 1, 1)
    int_derivative_11 = m.evaluate((x_int, u_int), 'interval', 1, 1)

    time_cur = performance_counter(time_cur, 'derivative 11 Run 2+ AVG', 10)

    # ----------------------------------------------------------------------
    # order 3, for variable 0

    int_derivative_30 = m.evaluate((x_int, u_int), 'interval', 3, 0)  # Run 1

    time_cur = performance_counter(time_cur, 'derivative 30 Run 1')

    # 10 runs
    int_derivative_30 = m.evaluate((x_int, u_int), 'interval', 3, 0)
    int_derivative_30 = m.evaluate((x_int, u_int), 'interval', 3, 0)
    int_derivative_30 = m.evaluate((x_int, u_int), 'interval', 3, 0)
    int_derivative_30 = m.evaluate((x_int, u_int), 'interval', 3, 0)
    int_derivative_30 = m.evaluate((x_int, u_int), 'interval', 3, 0)
    int_derivative_30 = m.evaluate((x_int, u_int), 'interval', 3, 0)
    int_derivative_30 = m.evaluate((x_int, u_int), 'interval', 3, 0)
    int_derivative_30 = m.evaluate((x_int, u_int), 'interval', 3, 0)
    int_derivative_30 = m.evaluate((x_int, u_int), 'interval', 3, 0)
    int_derivative_30 = m.evaluate((x_int, u_int), 'interval', 3, 0)

    time_cur = performance_counter(time_cur, 'derivative 30 Run 2+ AVG', 10)

    # ----------------------------------------------------------------------
    # order 3, for variable 1

    int_derivative_31 = m.evaluate((x_int, u_int), 'interval', 3, 1)  # Run 1

    time_cur = performance_counter(time_cur, 'derivative 31 Run 1')

    # 10 runs
    int_derivative_31 = m.evaluate((x_int, u_int), 'interval', 3, 1)
    int_derivative_31 = m.evaluate((x_int, u_int), 'interval', 3, 1)
    int_derivative_31 = m.evaluate((x_int, u_int), 'interval', 3, 1)
    int_derivative_31 = m.evaluate((x_int, u_int), 'interval', 3, 1)
    int_derivative_31 = m.evaluate((x_int, u_int), 'interval', 3, 1)
    int_derivative_31 = m.evaluate((x_int, u_int), 'interval', 3, 1)
    int_derivative_31 = m.evaluate((x_int, u_int), 'interval', 3, 1)
    int_derivative_31 = m.evaluate((x_int, u_int), 'interval', 3, 1)
    int_derivative_31 = m.evaluate((x_int, u_int), 'interval', 3, 1)
    int_derivative_31 = m.evaluate((x_int, u_int), 'interval', 3, 1)

    time_cur = performance_counter(time_cur, 'derivative 31 Run 2+ AVG', 10)


def sym_derivative_case01_test():
    print("test_sym_derivative_case_01")


def sym_derivative_case02_test():
    print("test_sym_derivative_case_02")


if __name__ == "__main__":
    sym_derivative_case00_NUM_test()
    sym_derivative_case00_INT_test()
