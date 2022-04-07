import numpy as np

from pyrat.geometry import HalfSpace
from pyrat.util.visualization import vis2d_old


def test_vis_rand_halfspace():
    objs = [HalfSpace.rand(2) for _ in range(3)]
    vis2d_old(objs, eq_axis=False)


def test_vis_halfspace():
    h = HalfSpace(np.array([1, 2]), 1)
    h0 = h + np.array([1, 3.5])
    h -= np.array([1, 2])
    print(h)
    m = np.array([[2, 3], [1, 2]])
    temp = h @ m
    print(temp)
    vh = HalfSpace(np.array([-1, 0]), 10)
    hh = HalfSpace(np.array([0, -1]), 5)
    pt = vh.intersection_pt(hh).reshape((1, -1))
    vis2d_old([vh, hh, h, pt, np.random.rand(5, 2) * 100], eq_axis=True)


if __name__ == "__main__":
    test_vis_rand_halfspace()
