from pyrat.geometry import Interval
import numpy as np


def test_basic():
    bd = np.sort(np.random.rand(2, 3), axis=0)
    i = Interval(bd)
    print()
    print(bd[0])
    print(bd[1])
    print(i.dim)
    print(i.center)
    print(i)

    assert np.all(bd[0] <= bd[1])
