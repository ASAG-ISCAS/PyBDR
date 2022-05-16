from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from enum import IntEnum
from .continuous_system import ContSys
from pyrat.model import Model
from pyrat.misc import Set
from pyrat.geometry import Geometry, IntervalMatrix, Zonotope, cvt2
from dataclasses import dataclass
from scipy.linalg import expm


class ALGORITHM(IntEnum):
    EUCLIDEAN = 0
    KRYLOV = 1


class LinSys:
    class Option:
        @dataclass
        class Base(ContSys.Option.Base):
            algorithm: ALGORITHM = ALGORITHM.EUCLIDEAN
            u_trans: np.ndarray = None
            factors: np.ndarray = None

            @abstractmethod
            def validation(self):
                # TODO
                return NotImplemented

        @dataclass
        class Euclidean(Base):
            taylor_powers = None
            taylor_err = None
            taylor_f = None
            taylor_input_f = None
            taylor_v = None
            taylor_rv = None
            taylor_r_trans = None
            taylor_input_corr = None
            taylor_ea_int = None
            taylor_ea_t = None
            is_rv = None
            origin_contained = None

            def validation(self):
                # TODO
                return True

        @dataclass
        class KRYLOV(Base):
            def validation(self):
                # TODO
                return False

    class Entity(ContSys.Entity):
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

        def __str__(self):
            raise NotImplementedError

        def __exponential(self, option: LinSys.Option.Euclidean):
            xa_abs = abs(self._xa)
            xa_power = [self._xa]
            xa_power_abs = [xa_abs]
            m = np.eye(self.dim, dtype=float)
            # compute powers for each term and sum of these
            for i in range(option.taylor_terms):
                # compute powers
                xa_power.append(xa_power[i] @ self._xa)
                xa_power_abs.append(xa_power_abs[i] @ xa_abs)
                m += xa_power_abs[i] * option.factors[i]
            # determine error due the finite taylor series, see Prop.(2) in [1]
            w = expm(xa_abs * option.step_size) - m
            # compute absolute value of w for numerical stability
            w = abs(w)
            e = IntervalMatrix(-w, w)
            # write to object structure
            option.taylor_powers = xa_power
            option.taylor_err = e

        def __compute_time_interval_err(self, option: LinSys.Option.Euclidean):
            # initialize asum
            asum_pos = np.zeros((self.dim, self.dim), dtype=float)
            asum_neg = np.zeros((self.dim, self.dim), dtype=float)

            for i in range(1, option.taylor_terms):
                # compute factor
                exp1, exp2 = -(i + 1) / i, -1 / i
                factor = ((i + 1) ** exp1 - (i + 1) ** exp2) * option.factors[i]
                # init apos, aneg
                apos = np.zeros((self.dim, self.dim), dtype=float)
                aneg = np.zeros((self.dim, self.dim), dtype=float)
                # obtain positive and negative parts
                pos_ind = option.taylor_powers[i] > 0
                neg_ind = option.taylor_powers[i] < 0
                apos[pos_ind] = option.taylor_powers[i][pos_ind]
                aneg[neg_ind] = option.taylor_powers[i][neg_ind]
                # compute powers; factor is always negative
                asum_pos += factor * aneg
                asum_neg += factor * apos
            # instantiate interval matrix
            asum = IntervalMatrix(asum_neg, asum_pos)
            # write to object structure
            option.taylor_f = asum + option.taylor_err

        def __input_tie(self, option: LinSys.Option.Euclidean):
            # initialize asum
            asum_pos = np.zeros((self.dim, self.dim), dtype=float)
            asum_neg = np.zeros((self.dim, self.dim), dtype=float)

            for i in range(1, option.taylor_terms + 1):
                # compute factor
                exp1, exp2 = -(i + 1) / i, -1 / i
                factor = ((i + 1) ** exp1 - (i + 1) ** exp2) * option.factors[i]
                # init apos, aneg
                apos = np.zeros((self.dim, self.dim), dtype=float)
                aneg = np.zeros((self.dim, self.dim), dtype=float)
                # obtain positive and negative parts
                pos_ind = option.taylor_powers[i - 1] > 0
                neg_ind = option.taylor_powers[i - 1] < 0
                apos[pos_ind] = option.taylor_powers[i - 1][pos_ind]
                aneg[neg_ind] = option.taylor_powers[i - 1][neg_ind]
                # compute powers; factor is always negative
                asum_pos += factor * aneg
                asum_neg += factor * apos
            # instantiate interval matrix
            asum = IntervalMatrix(asum_neg, asum_pos)
            # compute error due to finite taylor series according to interval document
            # "Input Error Bounds in Reachability Analysis"
            e_input = option.taylor_err * option.step_size
            # write to object structure
            option.taylor_input_f = asum + e_input

        def __input_solution(self, option: LinSys.Option.Euclidean):
            v = option.u if self._ub is None else self._ub @ option.u
            # compute vTrans
            option.is_rv = True
            if np.all(v.c == 0) and v.z.shape[1] == 1:
                option.is_rv = False
            v_trans = option.u_trans if self._ub is None else self._ub @ option.u

            input_solv, input_corr = None, None
            if option.is_rv:
                # init v_sum
                v_sum = option.step_size * v
                a_sum = option.step_size * np.eye(self.dim)
                # compute higher order terms
                for i in range(option.taylor_terms):
                    v_sum += option.taylor_powers[i] @ (option.factors[i + 1] * v)
                    a_sum += option.taylor_powers[i] * option.factors[i + 1]

                # compute overall solution
                input_solv = v_sum + option.taylor_err * option.step_size * v
            else:
                # only a_sum, since v == origin(0)
                a_sum = option.step_size * np.eye(self.dim)
                # compute higher order terms
                for i in range(option.taylor_terms):
                    # compute sum
                    a_sum += option.taylor_powers[i] * option.factors[i + 1]

            # compute solution due to constant input
            ea_int = a_sum + option.taylor_err * option.step_size
            input_solv_trans = ea_int * cvt2(v_trans, Geometry.TYPE.ZONOTOPE)
            # compute additional uncertainty if origin is not contained in input set
            if option.origin_contained:
                raise NotImplementedError  # TODO
            else:
                # compute inputF
                self.__input_tie(option)
                input_corr = option.taylor_input_f * cvt2(
                    v_trans, Geometry.TYPE.ZONOTOPE
                )

            # write to object structure
            option.taylor_v = v
            if option.is_rv and input_solv.z.sum().astype(bool):  # need refine ???
                option.taylor_rv = input_solv
            else:
                option.taylor_rv = Zonotope.zero(self.dim)

            if input_solv_trans.z.sum().astype(bool):
                option.taylor_r_trans = input_solv_trans
            else:
                option.taylor_rv = Zonotope.zero(self.dim)

            option.taylor_input_corr = input_corr
            option.taylor_ea_int = ea_int

        def __reach_init_euclidean(
            self, r: Geometry.Base, option: LinSys.Option.Euclidean
        ):
            self.__exponential(option)
            self.__compute_time_interval_err(option)
            self.__input_solution(option)
            option.taylor_ea_t = expm(self._xa * option.step_size)
            rhom_tp = option.taylor_ea_t @ r + option.taylor_r_trans
            rhom = r.enclose(rhom_tp) + option.taylor_f * r + option.taylor_input_corr
            rhom = rhom.reduce(Zonotope.REDUCE_METHOD, Zonotope.ORDER)
            rhom_tp = rhom_tp.reduce(Zonotope.REDUCE_METHOD, Zonotope.ORDER)
            rv = option.taylor_rv.reduce(Zonotope.REDUCE_METHOD, Zonotope.ORDER)

            r_total_ti = rhom + rv
            r_total_tp = rhom_tp + rv

            return r_total_ti, r_total_tp

        def delta_reach(self, r: Geometry.Base, option: LinSys.Option.Euclidean):
            rhom_tp_delta = (
                option.taylor_ea_t - np.eye(self.dim)
            ) @ r + option.taylor_r_trans

            if r.type == Geometry.TYPE.ZONOTOPE:
                o = Zonotope.zero(self.dim, 0)
                rhom = (
                    o.enclose(rhom_tp_delta)
                    + option.taylor_f * r
                    + option.taylor_input_corr
                )
            elif r.type == Geometry.TYPE.POLY_ZONOTOPE:
                raise NotImplementedError
            else:
                raise NotImplementedError
            # reduce zonotope
            rhom = rhom.reduce(Zonotope.REDUCE_METHOD, Zonotope.INTERMEDIATE_ORDER)
            rv = option.taylor_rv.reduce(
                Zonotope.REDUCE_METHOD, Zonotope.INTERMEDIATE_ORDER
            )

            # final result
            return rhom + rv

        def error_solution(
            self, option: LinSys.Option.Euclidean, v_dyn: Geometry.Base, v_stat=None
        ):
            err_stat = 0 if v_stat is None else option.taylor_ea_int * v_stat
            asum = option.step_size * v_dyn

            for i in range(option.taylor_terms):
                # compute powers
                asum += option.factors[i + 1] * option.taylor_powers[i] @ v_dyn

            # get error due to finite taylor series
            f = option.taylor_err * v_dyn * option.step_size

            # compute error solution (dyn + stat)
            return asum + f + err_stat

        def reach_init(self, r0: [Set], option):
            assert option.validation()
            if option.algorithm == ALGORITHM.EUCLIDEAN:
                return self.__reach_init_euclidean(r0, option)
            elif option.algorithm == ALGORITHM.KRYLOV:
                raise NotImplementedError
            else:
                raise NotImplementedError

        def reach(self, option: ContSys.Option.Base):
            raise NotImplementedError
