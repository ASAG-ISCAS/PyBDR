import numpy as np

from pyrat.geometry import HalfSpace
from pyrat.util.visualization import vis2d


def test_python():
    a = np.random.rand(2)
    b = np.random.rand(10, 2)
    c = np.matmul(a[None, None], b[:, :, None]).squeeze()
    print(c.shape)
    pass


def test_constructor():
    h = HalfSpace()  # empty constructor
    assert h.is_empty
    h = HalfSpace(np.random.rand(1), 0)
    h_rand = HalfSpace.rand(2)
    assert h_rand.dim == 2
    assert h.dim == 1


def test_operators():
    h = HalfSpace(np.array([5, 2]), 1)
    h1 = HalfSpace(np.array([-2, 3]), 1)
    m = np.array([[2, 3], [1, 2]], dtype=float)
    h2 = h @ m
    pt = h.common_pt(h1)
    int_pt = h.intersection_pt(h1)
    vis2d([h, h1, pt, int_pt, np.vstack([h.c, h1.c])])


def test_common_pt():
    h0 = HalfSpace.rand(2)
    h1 = HalfSpace.rand(2)
    pt = h0.common_pt(h1)
    int_pt = h0.intersection_pt(h1)
    print(h0)
    print(h1)
    pt1 = np.append(np.zeros(1, dtype=float), 1).reshape((1, -1))
    vis2d([h0, h1, pt, int_pt])


# if __name__ == "__main__":
#     test_python()
