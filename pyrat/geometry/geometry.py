from abc import ABC, abstractmethod

import numpy as np


class Geometry(ABC):
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
    def data(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def vertices(self):
        raise NotImplementedError

    # =============================================== operator
    @abstractmethod
    def __contains__(self, item):
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
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
