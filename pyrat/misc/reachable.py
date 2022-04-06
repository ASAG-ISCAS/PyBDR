from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from pyrat.geometry import Geometry


class Reachable:
    @dataclass
    class Element:
        set: Geometry = None
        err: np.ndarray = None

    @dataclass
    class Result:
        ti: Reachable.Element = None
        tp: Reachable.Element = None
        loc: int = -1
        pre: int = -1
