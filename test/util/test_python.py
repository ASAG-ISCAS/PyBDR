import numpy as np
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
