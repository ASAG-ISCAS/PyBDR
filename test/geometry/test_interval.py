import numpy as np
import scipy.sparse

from pyrat.geometry import Interval


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

    a = np.random.rand(3)

    print(a)
    print(2**a)
    print(a**2)
    a = scipy.sparse.rand(3, 1)
    print(a.toarray())

    a = Interval([-1, 1, 3], [2, 3, 9])
    print(isinstance(a, Geometry))
    print(isinstance(a, Interval))
    print(isinstance(a, IntervalOld))

    exit(False)
    temp = 2**a
    print(temp.inf.toarray())
    print(temp.sup.toarray())
    exit(False)
    bounds = [
        [-1.7929, 2.2128],
        [3.7566, 4.2047],
        [3.7963, 4.2034],
        [1.7967, 2.2419],
        [9.6800, 10.2047],
        [3.7979, 4.2784],
    ]
    c = Interval(bounds[:3], bounds[3:])
    print()
    temp0 = a**2
    temp1 = a**3
    temp2 = c**-2
    print(temp0.inf.toarray())
    print(temp0.sup.toarray())
    print(temp1.inf.toarray())
    print(temp1.sup.toarray())
    print(temp2.inf.toarray())
    print(temp2.sup.toarray())
