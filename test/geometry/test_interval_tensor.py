import numpy as np

from pyrat.geometry import IntervalTensor

np.set_printoptions(precision=5, suppress=True)


def random_interval(m, n, neg: bool = True):
    inf, sup = np.random.rand(m, n), np.random.rand(m, n)
    row = np.random.choice(inf.shape[0], 2)
    col = np.random.choice(inf.shape[1], 2)
    if neg:
        inf[row, col] *= -1
        sup[row, col] *= -1
    inf, sup = np.minimum(inf, sup), np.maximum(inf, sup)
    return IntervalTensor(inf, sup)


def out_interval(a, name):
    info = name + "inf=["
    for row in range(a.inf.shape[0]):
        info += "["
        for col in range(a.inf.shape[1]):
            info += str(a.inf[row][col]) + ", "
        info += "];\n"
    info += "];"
    print(info)
    info = name + "sup=["
    for row in range(a.sup.shape[0]):
        info += "["
        for col in range(a.sup.shape[1]):
            info += str(a.sup[row][col]) + ", "
        info += "];\n"
    info += "];"
    print(info)


def out_matrix(a, name):
    info = name + "=["
    for row in range(a.shape[0]):
        info += "["
        for col in range(a.shape[1]):
            info += str(a[row][col]) + ", "
        info += "];\n"
    info += "];"
    print(info)


def test_interval():
    a = random_interval(3, 5)
    print(a.inf)
    print(a.sup)


def test_addition():
    print()
    a = random_interval(2, 4)
    out_interval(a, "a")
    b = random_interval(2, 4)
    out_interval(b, "b")
    print("--------------------------")
    c = 1 + b
    print(c)


def test_subtraction():
    print()
    a = random_interval(2, 4)
    out_interval(a, "a")
    b = random_interval(2, 4)
    out_interval(b, "b")
    print("--------------------------")
    c = 1 - b
    print(c)


def test_multiplication():
    print()
    a = random_interval(2, 4)
    out_interval(a, "a")
    b = random_interval(2, 4)
    out_interval(b, "b")
    print("--------------------------")
    c = 1 - b
    print(c)


def test_division():
    print()
    a = random_interval(2, 4)
    out_interval(a, "a")
    b = random_interval(2, 4)
    out_interval(b, "b")
    print("--------------------------")
    c = 1 - b
    print(c)


def test_matrix_multiplication_case_0():
    print()
    a = np.random.rand(2, 4)
    out_matrix(a, "a")
    b = random_interval(4, 5)
    out_interval(b, "b")
    print("--------------------------")
    c = a @ b
    print(c)


def test_matrix_multiplication_case_1():
    print()
    a = random_interval(2, 4, False)
    out_interval(a, "a")
    b = random_interval(4, 5, False)
    out_interval(b, "b")
    print("--------------------------")
    c = a @ b
    print(c)


def test_matrix():
    a = np.random.rand(2, 2, 4, 3)
    b = np.random.rand(3)
    c = a @ b
    d = np.random.rand(4) @ np.random.rand(4)
    print(d.shape)
    # d = np.sum(a[:, :, None] * b[None, :, :], axis=1)
    need_squeeze = b.ndim <= 1
    if need_squeeze:
        b = np.expand_dims(b, -1)
    print(b.shape)
    print(np.expand_dims(a, -1).shape)
    print(np.expand_dims(b, 0).shape)
    e = np.sum(np.expand_dims(a, -1) * np.expand_dims(b, 0), axis=-2)
    if need_squeeze:
        e = e.squeeze()
        print(e.shape)
    # assert np.allclose(c, d)
    print(c.shape)
    print(e.shape)
    assert np.allclose(c, e)
