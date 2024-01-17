import numpy as np
from pybdr.geometry import Interval
from pybdr.model import tank6eq, Model
from pybdr.util.functional import performance_counter, performance_counter_start

if __name__ == "__main__":
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
    