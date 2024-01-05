import inspect

import numpy as np
from pybdr.model import Model
from sympy import *


def test_fxu():
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
    tt = modelref.evaluate((x, u), "interval", 0, 0)
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


# def parallel_evaluate()


def test_case_00():
    from concurrent.futures import ProcessPoolExecutor, as_completed

    modelref = Model(f, [6, 1])

    from pybdr.geometry import Interval

    x, u = Interval.rand(6), Interval.rand(1)

    orders = [0, 1, 2, 3, 4]
    v = [0, 0, 0, 1, 1]

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(modelref.evaluate, (x, u), "interval", orders[i], v[i]) for i in range(len(orders))]

        result = []

        for future in as_completed(futures):
            try:
                result.append(future.result())
            except Exception as e:
                print(e)


def test_case_01():
    from concurrent.futures import ThreadPoolExecutor, as_completed

    modelref = Model(f, [6, 1])

    from pybdr.geometry import Interval

    x, u = Interval.rand(6), Interval.rand(1)

    orders = [0, 1, 2, 3, 4]
    v = [0, 0, 0, 1, 1]

    with ThreadPoolExecutor as executor:
        result = executor.map(modelref.evaluate, (x, u))


def test_case_02():
    import threading
    import time

    class SomeObject:
        def __init__(self, value):
            self.value = value
            self._lock = threading.Lock()

        def some_method(self, additional_value):
            with self._lock:  # 使用锁来保护临界区
                # 模拟一些耗时操作
                time.sleep(1)
                self.value += additional_value  # 修改内部变量时受到锁的保护
                return self.value

    from concurrent.futures import ThreadPoolExecutor

    num_proj = 100

    # 创建SomeObject的实例
    some_objects = [SomeObject(i) for i in range(num_proj)]

    # 这是我们想要并行传递给some_method的附加值列表
    additional_values = np.arange(num_proj)

    # 使用ThreadPoolExecutor来并行执行方法
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(obj.some_method, val) for obj, val in zip(some_objects, additional_values)]

        # 获取结果
        results = [future.result() for future in futures]

    # 输出结果
    print(results)
