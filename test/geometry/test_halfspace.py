import random

from pyrat.geometry import HalfSpace
import numpy as np


def test_python():
    print(np.finfo(np.float).eps)
    pass


def test_basic():
    c = np.random.rand(6)
    d = random.random()
    h = HalfSpace(c, d)
    print(h)
    pass
