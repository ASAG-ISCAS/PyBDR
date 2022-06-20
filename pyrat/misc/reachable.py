from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from pyrat.geometry import Geometry
from .set import Set


class Reachable:
    @dataclass
    class Result:
        tis: [Set]
        tps: [Set]
        tit: np.ndarray = None
        tpt: np.ndarray = None
