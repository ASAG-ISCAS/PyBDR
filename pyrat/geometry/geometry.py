from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np
from enum import IntEnum
from dataclasses import dataclass


class Geometry:
    class TYPE(IntEnum):
        INTERVAL = 0
        INTERVAL_MATRIX = 1
        ZONOTOPE = 2
        POLYTOPE = 3
        POLY_ZONOTOPE = 4

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
        def dim(self) -> int:
            raise NotImplementedError

        @property
        @abstractmethod
        def is_empty(self) -> bool:
            raise NotImplementedError

        @property
        @abstractmethod
        def vertices(self) -> np.ndarray:
            raise NotImplementedError

        @property
        @abstractmethod
        def info(self):
            raise NotImplementedError

        @property
        @abstractmethod
        def type(self) -> Geometry.TYPE:
            raise NotImplementedError

        # =============================================== operator
        @abstractmethod
        def __contains__(self, item):
            raise NotImplementedError

        @abstractmethod
        def __str__(self):
            raise NotImplementedError

        @abstractmethod
        def __add__(self, other):
            raise NotImplementedError

        @abstractmethod
        def __sub__(self, other):
            raise NotImplementedError

        @abstractmethod
        def __pos__(self):
            raise NotImplementedError

        @abstractmethod
        def __neg__(self):
            raise NotImplementedError

        @abstractmethod
        def __matmul__(self, other):
            raise NotImplementedError

        @abstractmethod
        def __mul__(self, other):
            raise NotImplementedError

        @abstractmethod
        def __or__(self, other):
            raise NotImplementedError

        # =============================================== class method

        @classmethod
        @abstractmethod
        def functional(cls):
            """
            public functional for supporting general arithmetic
            :return:
            """
            raise NotImplementedError

        # =============================================== static method
        @staticmethod
        @abstractmethod
        def empty(dim: int):
            raise NotImplementedError

        @staticmethod
        @abstractmethod
        def rand(dim: int):
            raise NotImplementedError

        # =============================================== public method
        def enclose(self, other):
            raise NotImplementedError

        @abstractmethod
        def reduce(self):
            raise NotImplementedError

        @abstractmethod
        def proj(self, dims):
            raise NotImplementedError

        @abstractmethod
        def boundary(self, max_dist: float, element: Geometry.TYPE):
            raise NotImplementedError
