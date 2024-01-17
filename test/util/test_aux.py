import pybdr.util.functional.auxiliary as aux
import numpy as np


def test_power_2d_00():
    a = np.array([[1, 2], [3, 4]])
    b = aux.mat_powers_2d(a, 4)
    print(a)
    print(b.shape)
    print(b)


def test_power_2d_01():
    a = np.array([[1, 2], [3, 4]])
    n = 4
    t = 0.1
    b = aux.mat_powers_2d(a * t, 4)
    c = aux.mat_powers_2d(a, 4) * np.power(t, np.arange(n + 1))[:, None, None]
    diff = b - c
    print(abs(diff).sum())
    assert np.allclose(b, c, rtol=1.e-10, atol=1.e-10)


if __name__ == '__main__':
    pass
