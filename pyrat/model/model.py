from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Model(ABC):
    @property
    @abstractmethod
    def f(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def vars(self):
        raise NotImplementedError
