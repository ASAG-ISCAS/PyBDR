import numpy as np

from pyrat.geometry import IntervalOld


def test_basic():
    bd = np.sort(np.random.rand(2, 3), axis=0)
    i = IntervalOld(bd)
    print()
    print(bd[0])
    print(bd[1])
    print(i.dim)
    print(i.c)
    print(i)

    assert np.all(bd[0] <= bd[1])


def test_new_interval():
    from pyrat.geometry import Interval

    a = Interval(-1, 0)
    print()
    print(a**3)
    print(a**2)
