from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pyrat.geometry import Geometry
from pyrat.misc import Reachable, Simulation, Specification
import numpy as np


@dataclass
class Option(ABC):
    t_start: float = 0
    t_end: float = 0
    cur_t: float = 0
    steps: int = 10
    step_size: float = None
    r_init: [Reachable.Element] = None
    r_unsafe: [Geometry] = None
    u: Geometry = None
    u_trans: np.ndarray = None
    algo: str = None
    mod: str = "over"  # otherwise "under"
    specs: [Specification] = None

    @abstractmethod
    def validate(self, dim) -> bool:
        raise NotImplementedError


@dataclass
class RunTime(ABC):
    cur_t: float = 0
    lin_err_f0: np.ndarray = None
    lin_err_p: dict = None


class ContSys(ABC):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    # =============================================== operator
    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    # =============================================== property
    @property
    @abstractmethod
    def dim(self) -> int:
        raise NotImplementedError

    # =============================================== public method
    @abstractmethod
    def reach(self, op) -> Reachable.Result:
        raise NotImplementedError

    @abstractmethod
    def simulate(self, op) -> Simulation.Result:
        raise NotImplementedError
