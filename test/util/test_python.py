from dataclasses import dataclass


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
