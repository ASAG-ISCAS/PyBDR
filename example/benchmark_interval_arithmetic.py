import numpy as np
from pyrat.geometry import Interval
from pyrat.util.functional import performance_counter_start, performance_counter

if __name__ == '__main__':
    time_start = performance_counter_start()

    a = Interval.rand(100, 100, 10) + Interval.rand(100, 100, 10)
    b = Interval.rand(100, 100, 10) - Interval.rand(100, 100, 10)
    c = Interval.rand(100, 100, 10) * Interval.rand(100, 100, 10)
    d = Interval.rand(100, 100, 10) / Interval.rand(100, 100, 10)
    e = Interval.rand(100, 100, 10) @ Interval.rand(100, 10, 10)
    f = Interval.rand(100, 100, 10) @ np.random.rand(10, 10)

    performance_counter(time_start, 'interval arithmetic')
