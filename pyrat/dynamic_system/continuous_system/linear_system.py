from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .continuous_system import ContSys, Option
from pyrat.geometry import Geometry
from pyrat.misc import Reachable, Simulation, Specification


class LinSys:
    @dataclass
    class Option(Option):
        algo: str = "standard"
        origin_contained: bool = False

        def validate(self) -> bool:
            raise NotImplementedError

    class Sys(ContSys):
        def __init__(
            self,
            xa,
            ub: np.ndarray = None,
            c: float = None,
            xc: np.ndarray = None,
            ud: np.ndarray = None,
            k: float = None,
        ):
            self._xa = xa
            self._ub = ub
            self._c = c
            self._xc = xc
            self._ud = ud
            self._k = k

        # =============================================== operator
        def __str__(self):
            raise NotImplementedError

        # =============================================== property
        @property
        def dim(self) -> int:
            return self._xa.shape[1]

        # =============================================== private method
        def _exponential(self, op: LinSys.Option):
            raise NotImplementedError

        def _reach_init_euclidean(self, r: Geometry, op: LinSys.Option):
            # compute exponential matrix
            # TODO
            raise NotImplementedError

        # =============================================== public method
        def reach_init(self, r_init: Geometry, op: LinSys.Option):
            print("NOTHING")
            raise NotImplementedError

        def reach(self, op) -> Reachable.Result:
            raise NotImplementedError

        def simulate(self, op) -> Simulation.Result:
            raise NotImplementedError
