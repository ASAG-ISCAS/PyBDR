import numpy as np

from pyrat.geometry import ZonoTensor, Interval

np.set_printoptions(precision=5, suppress=True)


def test_zono_rand():
    b = np.random.rand(2)
    print(b.shape)
    a = ZonoTensor.zeros(5, 3)
    print(a.shape)
    print(a.c)
    print(a.gen.shape)
    print(a.transpose().shape)


def test_zono_add():
    a = ZonoTensor.rand(4, 2, 5)
    b = ZonoTensor.rand(7, 2, 5)
    c = a + b
    print(c.shape)
    print(c.gen_num)
    a = ZonoTensor.rand(4, 2, 20, 5)
    d = ZonoTensor.rand(11, 5)
    e = a + d
    print(e.shape)
    print(e.gen_num)


def test_zono_mul():
    a = ZonoTensor.rand(10, 3, 4)
    b = ZonoTensor.rand(2, 5)
    print(a.shape)
    print(b.shape)
    c = np.random.rand(11, 5, 4)
    d = np.random.rand(5, 4)
    e = c * d
    f = d * c
    print(e.shape)
    print(f.shape)
