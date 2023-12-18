from __future__ import annotations

import numpy as np
from pybdr.dynamic_system import LinearSystemSimple
from pybdr.geometry import Interval, Geometry
from pybdr.geometry.operation import enclose
from dataclasses import dataclass


class IntervalReachLinearAlgo1:
    @dataclass
    class Options:
        pass

    @classmethod
    def reach_one_step(cls):
        raise NotImplementedError

    @classmethod
    def reach(cls, system: LinearSystemSimple, x0: Interval):
        raise NotImplementedError
