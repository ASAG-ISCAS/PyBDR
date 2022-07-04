import numpy as np

from pyrat.geometry import ZonoTensor

np.set_printoptions(precision=5, suppress=True)


def test_zono_rand():
    b = np.random.rand(2)
    print(b.shape)
    a = ZonoTensor.zeros(5, 3)
    print(a.shape)
    print(a.c)
    print(a.gen.shape)
    print(a.transpose().shape)
