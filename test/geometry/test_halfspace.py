import numpy as np

from pyrat.geometry import HalfSpace


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
    h = HalfSpace(np.random.rand(1), 0)
    print(h + 0.1)
    print(0.1 + h)


# if __name__ == "__main__":
#     test_python()
