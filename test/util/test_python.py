from dataclasses import dataclass

import numpy as np


def test_is_field():
    @dataclass
    class Demo:
        a = 0
        b = 1

    d = Demo()
    print(hasattr(d, "a"))
    print(hasattr(d, "b"))
    print(hasattr(d, "c"))
    # update filed inside the Demo class on fly


def test_abs():
    from abc import ABC, abstractmethod

    class Geo(ABC):
        @abstractmethod
        def __init__(self):
            raise NotImplementedError

        @abstractmethod
        def __str__(self):
            raise NotImplementedError

        @abstractmethod
        def __add__(self, other):
            return 0

    class Polygon(Geo):
        def __init__(self):
            self._data = 1

        def __str__(self):
            return str(self._data)

        def __add__(self, other):
            return 1

    class Circle(Geo):
        def __init__(self):
            self._data = "data"

        def __str__(self):
            return str(self._data)

        def __add__(self, other):
            return 2

    p = Polygon()
    c = Circle()
    d = c + p
    e = p + c
    print(d)
    print(e)


def test_numpy_container():
    class Interval:
        def __init__(self, a, b):
            self._a = a
            self._b = b

        def __str__(self):
            return str(self._a) + " " + str(self._b)

        def __add__(self, other):
            return Interval(min(self._a, other._a), max(self._b, other._b))

    a = np.array([Interval(0, 1), Interval(2, 3)], dtype=Interval)
    b = np.array([Interval(4, 5), Interval(6, 7)], dtype=Interval)
    c = a + b
    print(2.222 % 2)
    print(2.0 % 2)
    print(3 % 2)


def test_basic_numpy():
    a = np.random.rand(2, 3, 7)
    b = np.random.rand(7)
    a0 = np.random.rand(2, 3, 7, 4)
    b0 = np.random.rand(7, 9)
    bb = np.broadcast_to(b0, np.append(a0.shape[:-1], b0.shape[-1]))
    print(bb.shape)
    c = a + b
    print(c.shape)
    d = np.concatenate([a0, bb], axis=-1)
    print(d.shape)
    e = np.random.rand(7)
    f = np.random.rand(9)
    g = np.outer(e, f)
    print(g.shape)
    temp = np.any(np.isnan(a > 0.1), axis=-1)
    print(temp.shape)


def test_static_class_variables():
    class Foo:
        _var0 = 0

        def __init__(self):
            self.__var1 = self.__class__._var0

        def var(self):
            return self.__var1

    f0 = Foo()
    Foo._var0 = 2
    f1 = Foo()
    print(f0.var())
    print(f1.var())


def test_random_pts():
    from pybdr.geometry import Zonotope, Geometry
    from pybdr.geometry.operation import cvt2
    from pybdr.util.visualization import plot
    from scipy import interpolate

    z = Zonotope.rand(2, 8)
    pts = z.vertices
    i = cvt2(pts, Geometry.TYPE.INTERVAL)
    print(pts.shape)
    index = np.random.choice(z.vertices.shape[0], 15)
    pts[index] *= 0.8
    index = np.random.choice(z.vertices.shape[0], 10)
    pts[index] *= 0.7

    p = cvt2(pts, Geometry.TYPE.POLYTOPE)
    x, y = pts[:, 0], pts[:, 1]

    tck, u = interpolate.splprep([x, y], s=0.3, per=True)
    xi, yi = interpolate.splev(np.linspace(0, 1, 1000), tck)
    temp = np.vstack([xi, yi]).T
    pp = cvt2(temp, Geometry.TYPE.POLYTOPE)

    # plot([p, temp], [0, 1])
    # plot([z, temp], [0, 1])
    # plot([i, temp], [0, 1])
    plot([p], [0, 1])
    plot([z], [0, 1])
    plot([i], [0, 1])

    import matplotlib.pyplot as plt

    dims = [0, 1]
    width, height = 800, 800
    assert len(dims) == 2
    px = 1 / plt.rcParams["figure.dpi"]
    fig, ax = plt.subplots(figsize=(width * px, height * px), layout="constrained")
    ax.plot(xi, yi, c="black", linewidth=3)
    ax.autoscale_view()
    ax.axis("equal")
    ax.axis("off")
    ax.set_xlabel("x" + str(dims[0]))
    ax.set_ylabel("x" + str(dims[1]))

    plt.show()
