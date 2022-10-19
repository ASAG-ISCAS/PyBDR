from __future__ import annotations
import numpy as np


class LinSys:
    def __init__(
        self,
        xa,
        ub: np.ndarray = None,
        c: float = None,
        xc: np.ndarray = None,
        ud: np.ndarray = None,
        k: float = None,
    ):
        self._xa = xa
        self._ub = ub
        self._c = c
        self._xc = xc
        self._ud = ud
        self._k = k

    @property
    def dim(self) -> int:
        return self._xa.shape[1]

    @property
    def xa(self) -> np.ndarray:
        return self._xa

    @property
    def ub(self) -> np.ndarray:
        return self._ub

    @property
    def c(self) -> float:
        return self._c

    @property
    def xc(self) -> np.ndarray:
        return self._xc

    @property
    def ud(self) -> np.ndarray:
        return self._ud

    @property
    def k(self) -> float:
        return self._k
