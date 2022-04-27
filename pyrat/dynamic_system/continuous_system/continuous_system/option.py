from __future__ import annotations
from abc import ABC, abstractmethod
from pyrat.misc import Reachable
from pyrat.geometry import Geometry


class Option:
    class Base(ABC):
        t_start = float = 0
        t_end: float = 0
        steps: int = 10
        step_size: float = None
        r0 = [Reachable.Element] = []
        u: Geometry.Base = None

        @abstractmethod
        def validate(self) -> bool:
            raise NotImplementedError
