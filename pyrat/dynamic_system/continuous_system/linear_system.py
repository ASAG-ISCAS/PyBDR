from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Options:
    time_step: float = None
    max_zono_order: int = None
    algo: str = "standard"
    reduction_tec: str = None
    taylor_terms: int = None
    partition: np.ndarray = None
    krylov_error: float = None
    krylov_step: float = None
    error: float = None
    factors: np.ndarray = None
    u_track_vec: np.ndarray = None


class LinearSystem:
    def __init__(
        self,
        xa: np.ndarray,
        xb: np.ndarray,
        c: float,
        xc: np.ndarray,
        xd: np.ndarray,
        k: float,
        name: str = "Linear System",
    ):
        """
        Linear system of the form
        x'(t)=xa@x(t)+xb@u(t)+c
        y(t)=xc@x(t)+xd@u(t)+k
        where x(t) in R^n is the system state, u(t) in R^m is the system input, y(t)
        is the system output
        :param xa: state matrix
        :param xb: input matrix
        :param c: constant input
        :param xc: output matrix
        :param xd: throughput matrix
        :param k: output offset
        :param name: name of the system
        """
        self._xa = xa
        self._xb = xb
        self._c = c
        self._xc = xc
        self._xd = xd
        self._k = k
        self._name = name
        # ========= properties for inner algorithm
        self._taylor = []
        self._krylov = []
        self._over_reach_algo = {
            "standard": self._over_reach_standard,
            "adaptive": self._over_reach_adaptive,
            "wrapping_free": self._over_reach_wrapping_free,
            "from_start": self._over_reach_from_start,
            "decomposition": self._over_reach_decomposition,
            "krylov": self._over_reach_krylov,
        }

    # =============================================== private method
    def _over_reach_standard(self, op):
        raise NotImplementedError
        # TODO

    def _over_reach_adaptive(self, op):
        raise NotImplementedError
        # TODO

    def _over_reach_wrapping_free(self, op):
        raise NotImplementedError
        # TODO

    def _over_reach_from_start(self, op):
        raise NotImplementedError
        # TODO

    def _over_reach_decomposition(self, op):
        raise NotImplementedError
        # TODO

    def _over_reach_krylov(self, op):
        raise NotImplementedError
        # TODO

    # =============================================== public method
    def over_reach(self, op: Options):
        return self._over_reach_algo[op.algo](op)

    def under_reach(self, op: Options):
        raise NotImplementedError
        # TODO
