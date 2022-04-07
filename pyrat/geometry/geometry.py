from abc import ABC, abstractmethod

import numpy as np


class Geometry(ABC):
    __array_ufunc__ = None

    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    # =============================================== property
    @property
    @abstractmethod
    def center(self) -> np.ndarray:
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
    @abstractmethod
    def reduce(self, method: str, order: int):
        raise NotImplementedError

    @abstractmethod
    def proj(self, dims):
        raise NotImplementedError
