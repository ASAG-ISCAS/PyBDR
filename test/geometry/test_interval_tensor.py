import numpy as np

from pyrat.geometry import Interval

np.set_printoptions(precision=5, suppress=True)


def random_interval(m, n, neg: bool = True):
    inf, sup = np.random.rand(m, n), np.random.rand(m, n)
    row = np.random.choice(inf.shape[0], 2)
    col = np.random.choice(inf.shape[1], 2)
    if neg:
        inf[row, col] *= -1
        sup[row, col] *= -1
    inf, sup = np.minimum(inf, sup), np.maximum(inf, sup)
    return Interval(inf, sup)


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
    c = 1 / b
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
    a = random_interval(2, 4)
    out_interval(a, "a")
    b = random_interval(4, 1)
    out_interval(b, "b")
    print("--------------------------")
    c = a @ b
    print(c)


def test_abs():
    print()
    a = random_interval(2, 4)
    out_interval(a, "a")
    print("--------------------------")
    print(abs(a))


def test_pow():
    print()
    a = random_interval(2, 4, False)
    out_interval(a, "a")
    print("--------------------------")
    print(a ** (2.1))
    print(1 / a)


def test_exp():
    print()
    a = random_interval(2, 4, False)
    out_interval(a, "a")
    print("--------------------------")
    print(Interval.exp(a))


def test_log():
    print()
    a = random_interval(2, 4, False)
    out_interval(a, "a")
    print("--------------------------")
    print(Interval.log(a))


def test_sqrt():
    print()
    a = random_interval(2, 4, False)
    out_interval(a, "a")
    print("--------------------------")
    print(Interval.sqrt(a))


def test_arcsin():
    print()
    a = random_interval(2, 4, False)
    out_interval(a, "a")
    print("--------------------------")
    print(Interval.arcsin(a))


def test_arccos():
    print()
    a = random_interval(2, 4, False)
    out_interval(a, "a")
    print("--------------------------")
    print(Interval.arccos(a))


def test_arctan():
    print()
    a = random_interval(2, 4)
    out_interval(a, "a")
    print("--------------------------")
    print(Interval.arctan(a))


def test_sinh():
    print()
    a = random_interval(2, 4)
    out_interval(a, "a")
    print("--------------------------")
    print(Interval.sinh(a))


def test_cosh():
    print()
    a = random_interval(2, 4)
    out_interval(a, "a")
    print("--------------------------")
    print(Interval.cosh(a))


def test_tanh():
    print()
    a = random_interval(2, 4)
    out_interval(a, "a")
    print("--------------------------")
    print(Interval.tanh(a))


def test_arcsinh():
    print()
    a = random_interval(2, 4)
    out_interval(a, "a")
    print("--------------------------")
    print(Interval.arcsinh(a))


def test_arccosh():
    print()
    a = random_interval(2, 4, False) + 1
    out_interval(a, "a")
    print("--------------------------")
    print(Interval.arccosh(a))


def test_arctanh():
    print()
    a = random_interval(2, 4)
    out_interval(a, "a")
    print("--------------------------")
    print(Interval.arctanh(a))


def test_sin():
    print()
    a = random_interval(2, 4) * 10
    out_interval(a, "a")
    print("--------------------------")
    print(Interval.sin(a))


def test_cos():
    print()
    a = random_interval(2, 4) * 10
    out_interval(a, "a")
    print("--------------------------")
    print(Interval.cos(a))


def test_tan():
    print()
    a = random_interval(2, 4) * 10
    out_interval(a, "a")
    print("--------------------------")
    print(Interval.tan(a))
    b = Interval(5.781807046304337, 7.156265610661695)
    print(Interval.tan(b))


def test_cot():
    print()
    a = random_interval(2, 4) * 10
    out_interval(a, "a")
    print("--------------------------")
    print(Interval.cot(a))
    b = Interval(5.781807046304337, 7.156265610661695)
    print(Interval.cot(b))
