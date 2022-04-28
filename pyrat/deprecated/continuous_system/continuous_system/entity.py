from __future__ import annotations
from abc import ABC, abstractmethod


class Entity(ABC):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def reach(self, option):
        raise NotImplementedError
