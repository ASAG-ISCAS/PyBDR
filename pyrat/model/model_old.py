from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ModelOld(ABC):
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

    @property
    @abstractmethod
    def dim(self):
        raise NotImplementedError
