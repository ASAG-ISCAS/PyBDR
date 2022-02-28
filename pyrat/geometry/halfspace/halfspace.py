import numpy as np


class HalfSpace:
    def __init__(self, c: np.ndarray, d: float):
        assert c.ndim == 1
        self.__c = c
        self.__d = d

    @property
    def c(self) -> np.ndarray:
        return self.__c

    @property
    def d(self) -> float:
        return self.__d
