import numpy as np
from pybdr.geometry import Interval
from pybdr.model import tank6eq, Model
from pybdr.util.functional import performance_counter, performance_counter_start
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial


def parallel_model_evaluate(f, dims, xs, cond, o, v):
    m = Model(f, dims)
    return m.evaluate(xs, cond, o, v)


def parallel_model_evaluate_parallel(f, dims, xs, cond, o, v):
    return parallel_model_evaluate(f, dims, xs, cond, o, v)


if __name__ == '__main__':
    # m = Model(tank6eq, [6, 1])

    time_start = performance_counter_start()
    num_tasks = 1000
    x, u = np.random.rand(num_tasks, 6), np.random.rand(num_tasks, 1)
    oo = np.random.randint(0, 4, size=num_tasks)
    vv = np.random.randint(0, 1, size=num_tasks)
    # print(oo)
    # print(vv)

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(parallel_model_evaluate_parallel, tank6eq, [6, 1], (x[i], u[i]), 'numpy', oo[i], vv[i])
            for
            i in
            range(num_tasks)]

        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                raise e

    # np_derivative_0 = m.evaluate((x, u), "numpy", 3, 0)
    # np_derivative_1 = m.evaluate((x, u), "numpy", 3, 1)
    # np_derivative_2 = m.evaluate((x, u), "numpy", 0, 0)
    #
    # x, u = Interval.rand(6), Interval.rand(1)
    # int_derivative_0 = m.evaluate((x, u), "interval", 3, 0)
    # int_derivative_1 = m.evaluate((x, u), "interval", 2, 0)
    # int_derivative_2 = m.evaluate((x, u), "interval", 2, 0)
    # int_derivative_3 = m.evaluate((x, u), "interval", 0, 1)

    performance_counter(time_start, "sym_derivative")
