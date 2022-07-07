import numpy as np

from pyrat.geometry import ZonoTensor

np.set_printoptions(precision=5, suppress=True)


def out_zono(z, name):
    zc = z.c.reshape(np.append(z.c.shape, 1))
    cg = np.concatenate([zc, z.gen], axis=-1)
    info = name + "z=["
    for row in range(cg.shape[0]):
        info += "["
        for col in range(cg.shape[1]):
            info += str(cg[row][col]) + ", "
        info += "];\n"
    info += "];"
    print(info)


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
    c = np.random.rand(3, 2)
    d = np.random.rand(2)
    e = c * d
    f = d * c
    print(e.shape)
    print(f.shape)
    print("------------------------")

    a = ZonoTensor.rand(3, 2)
    b = ZonoTensor.rand(2, 2)
    out_zono(a, "a")
    print()
    print()
    out_zono(b, "b")
    from pyrat.geometry import Interval

    c = Interval([-1, 2], [3, 5])
    d = c * a
    print(d.c)
    print(d.gen)
