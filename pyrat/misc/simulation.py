from __future__ import annotations
from dataclasses import dataclass

import numpy as np


class Simulation:
    @dataclass
    class Element:
        x: np.ndarray = None
        t: float = -1

    @dataclass
    class Result:
        traj: [Simulation.Element] = None
        loc: int = -1
