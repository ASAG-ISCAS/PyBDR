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
    a = np.random.rand(2)
    print((-2) ** a)
    inf = np.nan
    sup = np.nan
    assert inf <= sup
