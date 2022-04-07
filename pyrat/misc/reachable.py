from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from pyrat.geometry import Geometry


class Reachable:
    @dataclass
    class Element:
        set: Geometry = None
        err: np.ndarray = None
        pre: int = -1

    @dataclass
    class Result:
        ti: [Reachable.Element]
        tp: [Reachable.Element]
        ti_time: np.ndarray = None
        tp_time: np.ndarray = None
        loc: int = -1
        pre: int = -1
