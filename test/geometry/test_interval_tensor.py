import numpy as np

from pyrat.geometry import IntervalTensor

np.set_printoptions(precision=3, suppress=True)


def test_interval():
    inf, sup = np.random.rand(4, 5), np.random.rand(4, 5)
    inf[0, 0] *= -1
    sup[0, 0] *= -1
    inf, sup = np.minimum(inf, sup), np.maximum(inf, sup)
    interval = IntervalTensor(inf, sup)
    temp = 1 / interval
    print()
    print(temp.inf)
    print(temp.sup)
    print(1 / interval)
    print(interval.info)
    print(interval.dim)
    print(interval.inf)
    print(interval.sup)
    print(interval.dim)
    a = np.random.rand(3, 3)
    b = np.random.rand(3)
    print(a**b)
    print(b**a)
    exit(False)
    print(a @ b)
    print(b @ a)
    print(b @ b)
    print(a / b)
    print(b / a)
    print(a * b)
    print(b * a)

    c = np.random.rand(2, 4)
    print(c @ interval)
    print(interval @ np.random.rand(5))
    temp = interval * interval
    np.set_printoptions(precision=3, suppress=True)
    print(temp)
    print(temp.inf)
    print(temp.sup)
    interval = interval.T
    print(interval)
    e = IntervalTensor.empty(2)
    r = IntervalTensor.rand(1)
    print(r.is_empty)
    print(interval.is_empty)
    print(e.is_empty)
