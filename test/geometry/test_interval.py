import numpy as np

import pybdr
from pybdr.geometry import Interval, Geometry
from pybdr.util.visualization import plot

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


def test_addition_00():
    print()
    a = random_interval(2, 4)
    out_interval(a, "a")
    b = random_interval(2, 4)
    out_interval(b, "b")
    print("--------------------------")
    c = 1 + b
    print(c)


def test_addition_01():
    a = Interval.rand(2, 3, 4)
    b = a + 1
    print(b)


def test_addition_case_01():
    from pybdr.geometry import Zonotope
    a = Interval.rand(2)
    b = Zonotope.rand(2, 3)
    c = a + b
    plot([a, b, c], [0, 1])


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
    a = np.random.rand(2, 3, 4)
    out_matrix(a, "a")
    b = random_interval(4, 5)
    b = Interval.rand(4)
    # out_interval(b, "b")
    print("--------------------------")
    c = a @ b
    print(c)


def test_matrix_multiplication_case_1():
    print()
    # a = random_interval(2, 3, 4)
    a = Interval.rand(5)
    # out_interval(a, "a")
    b = Interval.rand(2, 3, 5, 4)
    # out_interval(b, "b")
    print("--------------------------")
    c = a @ b
    print(c.shape)
    # print(c)


def test_matrix_multiplication_case_2():
    # Z = zonotope([1 1 0; 0 0 1]);
    # Z.center
    # Z.generators
    # I = interval([0 1; 1 0], [1 2; 2 1]);
    from pybdr.geometry import Zonotope
    a = Interval([[0, 1], [1, 0]], [[1, 2], [2, 1]])
    b = Zonotope([1, 0], [[1, 0], [0, 1]])
    c = a @ b
    plot([b, c], [0, 1])


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


def test_mm():
    def mm(la, lb):
        if la.ndim == 1 and lb.ndim == 1:
            return np.sum(la * lb, axis=0)
        elif la.ndim == 1 and lb.ndim == 2:
            return np.sum(la[..., None] * lb, axis=0)
        elif la.ndim == 1 and lb.ndim > 2:
            return np.sum(la[..., None] * lb, axis=-2)
        elif la.ndim >= 2 and lb.ndim == 1:
            return np.sum(la * lb[None, ...], axis=-1)
        else:
            return np.sum(la[..., np.newaxis] * lb[..., np.newaxis, :, :], axis=-2)

    a = np.random.rand(2, 3, 4, 7, 5)
    b = np.random.rand(2, 3, 4, 5, 9)

    c = a @ b
    print(c.shape)
    d = mm(a, b)
    print(c.shape, d.shape)
    assert np.allclose(c, d)
    print("DONE")


def test_partition():
    a = Interval.rand(2)
    from pybdr.util.visualization import plot

    plot([a], [0, 1])
    from pybdr.geometry import Geometry
    from pybdr.geometry.operation import partition

    parts = partition(a, 0.1, Geometry.TYPE.INTERVAL)
    plot(parts, [0, 1])
    zonos = partition(a, 0.1, Geometry.TYPE.ZONOTOPE)
    plot(zonos, [0, 1])


def test_boundary():
    a = Interval.rand(2)
    from pybdr.geometry.operation import boundary

    bounds = boundary(a, 0.01, pybdr.Geometry.TYPE.INTERVAL)
    print(len(bounds))
    from pybdr.util.visualization import plot

    plot(bounds, [0, 1])


def test_temp():
    from pybdr.geometry import Interval
    from pybdr.util.visualization import plot
    from pybdr.geometry import Geometry
    from pybdr.geometry.operation import partition

    a = Interval([-1, -2], [3, 5])
    parts = partition(a, 0.1, Geometry.TYPE.INTERVAL)
    plot([*parts, a], [0, 1])


def test_proj():
    from pybdr.geometry import Interval
    from pybdr.util.visualization import plot

    a = Interval.rand(5)
    print(a)
    a0 = a.proj([0, 1])
    a1 = a.proj([1, 2])
    a2 = a.proj([2, 3])
    a3 = a.proj([3, 4])
    plot([a0, a1, a2, a3], [0, 1])


def test_decompose():
    a = Interval.rand(2)
    al, ar = a.decompose(0)
    print(a)
    print(al)
    print(ar)
    plot([a, al, ar], [0, 1])
    ad, au = a.decompose(1)
    plot([a], [0, 1])
    plot([a, al, ar], [0, 1])
    plot([a, ad, au], [0, 1])


def test_split():
    box = Interval.rand(10, 2, 30)
    boxes_00 = Interval.split(box, box.shape[0], 0)
    boxes_01 = Interval.split(box, box.shape[1], 1)
    boxes_02 = Interval.split(box, box.shape[2], 2)

    print(boxes_00[0].shape, len(boxes_00))
    print(boxes_01[0].shape, len(boxes_01))
    print(boxes_02[0].shape, len(boxes_02))


def test_concatenate():
    box = Interval.rand(10, 2, 30)
    boxes_00 = Interval.split(box, box.shape[0], 0)
    boxes_01 = Interval.split(box, box.shape[1], 1)
    boxes_02 = Interval.split(box, box.shape[2], 2)

    conc_box_00 = Interval.concatenate(boxes_00, 0)
    conc_box_01 = Interval.concatenate(boxes_01, 0)
    conc_box_02 = Interval.concatenate(boxes_02, 0)

    print(conc_box_00.shape)
    print(conc_box_01.shape)
    print(conc_box_02.shape)


def test_stack():
    box = Interval.rand(10, 2, 30)
    boxes_00 = Interval.split(box, box.shape[0], 0)

    stack_box_00 = Interval.stack(boxes_00, axis=0)
    stack_box_01 = Interval.stack(boxes_00, axis=1)
    stack_box_02 = Interval.stack(boxes_00, axis=2)
    print(stack_box_00.shape)
    print(stack_box_01.shape)
    print(stack_box_02.shape)


def test_vstack():
    box = Interval.rand(10, 2, 30)
    boxes = Interval.split(box, box.shape[2], axis=2)

    vstack_box = Interval.stack(boxes)
    print(vstack_box.shape)


def test_hstack():
    box = Interval.rand(10, 2, 30, 50)
    boxes = Interval.split(box, box.shape[2], axis=2)
    hstack_box = Interval.hstack(boxes)
    print(hstack_box.shape)


def test_contains():
    a = Interval(0, 1)
    print(a.contains(0.5))
    print(a.contains(2))
    b = Interval([-1, -1], [2, 2])
    print(b.contains([0, 0]))
    print(b.contains([0, 3]))
    print(b.contains(np.array([0, 0])))
    print(b.contains(np.array([0, 3])))


if __name__ == '__main__':
    pass
