import numpy as np


class LinearSystem:
    from .functional import (
        _reach_standard,
        _reach_decomp,
        _reach_krylov,
        _reach_adaptive,
        _reach_from_start,
        _reach_wrapping_free,
        under_reach,
        over_reach,
        init_reach_euclidean,
        dim,
    )

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
        where x(t) \in R^n is the system state, u(t) \in R^m is the system input, y(t)
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
        self._taylor = []
        self._krylov = []
        self._name = name
