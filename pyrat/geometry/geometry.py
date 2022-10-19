from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np
from enum import IntEnum
from dataclasses import dataclass


class Geometry:
    class TYPE(IntEnum):
        INTERVAL = 0
        ZONOTOPE = 1
        POLYTOPE = 2
        POLY_ZONOTOPE = 3

    class Base(ABC):
        __array_ufunc__ = None

        @abstractmethod
        def __init__(self):
            raise NotImplementedError

        # =============================================== property
        @property
        @abstractmethod
        def c(self) -> np.ndarray:
            raise NotImplementedError

        @property
        @abstractmethod
        def type(self) -> Geometry.TYPE:
            raise NotImplementedError
