from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm
from scipy.special import factorial

from pyrat.geometry import VectorInterval, MatrixInterval


@dataclass
class LSOptions:
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
    u = None
    is_rv = None


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
        self._taylor = {"taylor": None, "F": None}
        self._krylov = {}
        self._over_reach_algo = {
            "standard": self._over_reach_standard,
            "adaptive": self._over_reach_adaptive,
            "wrapping_free": self._over_reach_wrapping_free,
            "from_start": self._over_reach_from_start,
            "decomposition": self._over_reach_decomposition,
            "krylov": self._over_reach_krylov,
        }

    # =============================================== property
    @property
    def dim(self) -> int:
        raise NotImplementedError
        # TODO

    # =============================================== private method
    # =============================================== private method
    # =============================================== private method
    # =============================================== private method
    def _tie(self, op: LSOptions):
        """
        compute time interval error; computes the error done by building the convex hull
        of time point solutions
        :param op: options for the computation
        :return:
        """
        apower = self._taylor["power"]
        taylor_terms = op.taylor_terms
        rbyfac = op.factors
        dim = self.dim
        # initialize Asum
        asum_pos = np.zeros(dim, dtype=float)
        asum_neg = np.zeros(dim, dtype=float)

        for i in range(1, taylor_terms):
            # compute factor
            exp1, exp2 = -i / (i - 1), -1 / (i - 1)
            factor = (i**exp1 - i**exp2) * rbyfac[i]
            # init apos, aneg
            apos = np.zeros(dim)
            aneg = np.zeros(dim)
            # obtain positive and negative parts
            pos_ind = apower[i] > 0
            neg_ind = apower[i] < 0
            apos[pos_ind] = apower[i](pos_ind)
            aneg[pos_ind] = apower[i](neg_ind)
            # compute powers; factor is always negative
            asum_pos += factor * aneg
            asum_neg += factor * apos
        # instantiate interval matrix
        asum = MatrixInterval(np.vstack([asum_neg, asum_pos]))
        # write to object structure
        self._taylor["F"] = asum + self._taylor["error"]

    def _exponential(self, op: LSOptions):
        """
        compute the over approximation of the exponential of a system matrix up to a
        certain accuracy
        :param op: options for the computation
        :return:
        """
        m = np.eye(self.dim, dtype=float)
        xa_power = [self._xa]
        xa_power_abs = [abs(self._xa)]
        xa_abs = abs(self._xa)
        # compute powers for each term and sum of these
        for i in range(op.taylor_terms):
            # compute powers
            xa_power.append(xa_power[-1] @ self._xa)
            xa_power_abs.append(xa_power_abs[-1] @ xa_abs)
            m += xa_power_abs[-1] * op.factors[i]
        # determine error due the finite taylor series, see Prop.(2) in [1]
        w = expm(xa_abs * op.time_step) - m
        # compute absolute value of w for numerical stability
        w = abs(w)
        e = VectorInterval(np.vstack([w, -w]))
        # write to object structure
        self._taylor["power"] = xa_power
        self._taylor["error"] = e

    def _input_solution(self, op: LSOptions):
        """
        compute the bloating due the input
        :param op: options for the computation
        :return:
        """
        # set of possible inputs
        v = self._xb * op.u
        op.isrv = True
        # if np.all()
        pass

    def _init_reach_euclidean(self, r_init, op: LSOptions):
        """
        computes the reachable continuous set for the first time step in the
        untransformed space
        :param r_init: init reachable set
        :param op: options for the computation
        :return:
        """
        # compute exponential matrix
        self._exponential(op)
        # compute time interval error (tie)
        self._tie(op)
        # compute reachable set due to input
        # TODO

        pass

    def _over_reach_standard(self, op: LSOptions):
        """
        compute the reachable set for linear systems using the standard
        (non-wrapping-free) reachability algorithm for linear systems
        :param op: options for the computation
        :return:
        """
        # obtain factors for initial state and input solution
        for i in range(op.taylor_terms):
            # compute initial state factor
            op.factors[i] = op.time_step**i / factorial(i)

        # if a trajectory should be tracked
        if op.u_track_vec is None:
            input_corr = 0
        # init reachable set computations

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
    def over_reach(self, op: LSOptions):
        return self._over_reach_algo[op.algo](op)

    def under_reach(self, op: LSOptions):
        raise NotImplementedError
        # TODO
