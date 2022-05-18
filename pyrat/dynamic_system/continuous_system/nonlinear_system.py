from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from pyrat.geometry import Geometry, Interval, Zonotope
from pyrat.geometry.operation import cvt2
from pyrat.misc import Set
from .continuous_system import ContSys
from .linear_system import LinSys


class ALGORITHM(IntEnum):
    LINEAR = 1
    POLYNOMIAL = 2
    BACK_UNDER = 3


class NonLinSys:
    class Option:
        @dataclass
        class Base(ContSys.Option.Base):
            algorithm: ALGORITHM = ALGORITHM.LINEAR
            tensor_order: int = 2
            u_trans: np.ndarray = None
            factors: np.ndarray = None
            max_err: np.ndarray = None

            def _validate_misc(self, dim: int):
                assert self.tensor_order == 2 or self.tensor_order == 3  # only 2/3
                if self.max_err is None:
                    self.max_err = np.full(dim, np.inf)
                return True

            @abstractmethod
            def validation(self, dim: int):
                return NotImplemented

        @dataclass
        class Linear(Base):
            # runtime variables
            px = None
            pu = None
            lin_err_px = None
            lin_err_pu = None
            lin_err_f0 = None

            def validation(self, dim: int):
                assert self._validate_time_related()
                assert self._validate_inputs()
                assert self._validate_misc(dim)
                return True

        @dataclass
        class Polynomial(Base):
            algorithm: ALGORITHM = ALGORITHM.POLYNOMIAL
            tensor_order: int = 3

            def validation(self, dim: int):
                assert self._validate_time_related()
                assert self._validate_inputs()
                assert self._validate_misc(dim)
                assert 3 <= self.tensor_order <= 7
                return True

        @dataclass
        class BackUnder(Base):
            algorithm: ALGORITHM = ALGORITHM.BACK_UNDER
            algorithm_boundary: ALGORITHM = ALGORITHM.LINEAR
            epsilon_m: float = np.inf  # for boundary sampling
            epsilon: float = np.inf  # for backward verification

            def validation(self, dim: int):
                assert self._validate_time_related()
                assert self._validate_inputs()
                assert self._validate_misc(dim)
                assert not np.isinf(self.epsilon) and self.epsilon >= 0
                assert not np.isinf(self.epsilon_m) and self.epsilon_m > 0
                return True

    class Entity(ContSys.Entity):
        def __str__(self):
            raise NotImplementedError

        @property
        def dim(self):
            return self._model.dim

        def linearize_old(self, r: Geometry.Base, option):
            option.lin_err_pu = option.u_trans if option.u is not None else None
            f0_pre = self.evaluate((r.c, option.lin_err_pu))
            option.lin_err_px = r.c + f0_pre * 0.5 * option.step_size
            option.lin_err_f0 = self.evaluate((option.lin_err_px, option.lin_err_pu))
            a, b = self.jacobian((option.lin_err_px, option.lin_err_pu))
            assert not (np.any(np.isnan(a))) or np.any(np.isnan(b))
            lin_sys = LinSys.Entity(xa=a)
            lin_op = LinSys.Option.Euclidean(
                u_trans=option.u_trans,
                step_size=option.step_size,
                taylor_terms=option.taylor_terms,
                factors=option.factors,
            )

            lin_op.u = b @ (option.u + lin_op.u_trans - option.lin_err_pu)
            lin_op.u -= lin_op.u.c
            lin_op.u_trans = Zonotope(
                option.lin_err_f0 + lin_op.u.c,
                np.zeros((option.lin_err_f0.shape[0], 1)),
            )
            return lin_sys, lin_op

        def __abstract_err_lin(self, r: Geometry.Base, option: NonLinSys.Option.Linear):
            # compute interval of reachable set
            ihx = cvt2(r, Geometry.TYPE.INTERVAL)
            # compute intervals of total reachable set
            total_int_x = ihx + option.lin_err_px

            # compute intervals of input
            ihu = cvt2(option.u, Geometry.TYPE.INTERVAL)
            # translate intervals by linearization point
            total_int_u = ihu + option.lin_err_pu

            if option.tensor_order == 2:
                # obtain maximum absolute values within ihx, ihu
                dx = np.maximum(abs(ihx.inf), abs(ihx.sup))
                du = np.maximum(abs(ihu.inf), abs(ihu.sup))

                # evaluate the hessian matrix with the selected range-bounding technique
                hx, hu = self.hessian((total_int_x, total_int_u), "interval")

                # calculate the Lagrange remainder (second-order error)
                err_lagrange = np.zeros(self.dim, dtype=float)

                for i in range(self.dim):
                    abs_hx, abs_hu = abs(hx[i]), abs(hu[i])
                    hx_ = np.maximum(abs_hx.inf, abs_hx.sup)
                    hu_ = np.maximum(abs_hu.inf, abs_hu.sup)
                    err_lagrange[i] = 0.5 * (dx @ hx_ @ dx + du @ hu_ @ du)

                v_err_dyn = Zonotope(np.zeros(self.dim), np.diag(err_lagrange))
                return err_lagrange, v_err_dyn
            elif option.tensor_order == 3:
                raise NotImplementedError  # TODO
            else:
                raise Exception("unsupported tensor order")

        def __abstract_err_poly(
            self, option, r_all, r_diff, h, zd, verr_stat, t, ind3, zd3
        ):
            # compute interval of reachable set
            dx = cvt2(r_all, Geometry.TYPE.INTERVAL)
            total_int_x = dx + option.lin_err_px
            # compute intervals of input
            du = cvt2(option.u, Geometry.TYPE.INTERVAL)
            total_int_u = du + option.lin_err_pu

            # compute zonotope of state and input
            r_red_diff = cvt2(r_diff, Geometry.TYPE.ZONOTOPE).reduce(
                Zonotope.REDUCE_METHOD, Zonotope.ERROR_ORDER
            )
            z_diff = r_red_diff.card_prod(option.u)

            # second order error
            err_dyn_sec = 0.5 * (
                zd.quad_map(h, z_diff) + z_diff.quad_map(h, zd) + z_diff.quad_map(h)
            )

            if option.tensor_order == 3:
                tx, tu = self.third_order((total_int_x, total_int_u), mod="interval")

                # calculate the lagrange remainder term
                err_dyn_third = Interval.zero(self.dim)

                # error relates to tx
                for row, col in tx[0]:
                    err_dyn_third[row] += dx @ tx[1][row][col] @ dx * dx[col]

                # error relates to tu
                for row, col in tu[0]:
                    err_dyn_third[row] += du @ tu[1][row][col] @ du * du[col]

                err_dyn_third *= 1 / 6
                err_dyn_third = cvt2(err_dyn_third, Geometry.TYPE.ZONOTOPE)

                # no terms of order >=4, max 3 for now
                remainder = Zonotope.zero(self.dim, 1)

            else:
                raise NotImplementedError
            verr_dyn = err_dyn_sec + err_dyn_third + remainder
            verr_dyn = verr_dyn.reduce(
                Zonotope.REDUCE_METHOD, Zonotope.INTERMEDIATE_ORDER
            )

            err_ih_abs = abs(
                cvt2(verr_dyn, Geometry.TYPE.INTERVAL)
                + cvt2(verr_stat, Geometry.TYPE.INTERVAL)
            )
            true_err = err_ih_abs.sup
            return true_err, verr_dyn, verr_stat

        def __linear_reach(self, r: Set, option):
            lin_sys, lin_op = self.linearize_old(r.geometry, option)
            r_delta = r.geometry - option.lin_err_px
            r_ti, r_tp = lin_sys.reach_init(r_delta, lin_op)
            h, zd, err_stat, t, ind3, zd3 = None, None, 0, None, None, None
            if option.algorithm == ALGORITHM.POLYNOMIAL:
                r_diff = lin_sys.delta_reach(r_delta, lin_op)
                if option.tensor_order > 2:
                    h, zd, err_stat, t, ind3, zd3 = self.__pre_stat_err(r_delta, option)

            perf_ind_cur, perf_ind = np.inf, 0
            applied_err, abstract_err, v_err_dyn, v_err_stat = None, r.err, None, None

            while perf_ind_cur > 1 and perf_ind <= 1:
                # estimate the abstraction error
                applied_err = 1.1 * abstract_err
                v_err = Zonotope(0 * applied_err, np.diag(applied_err))
                r_all_err = lin_sys.error_solution(lin_op, v_err)

                # compute the abstraction error using the conservative linearization
                # approach described in [1]
                if option.algorithm == ALGORITHM.LINEAR:
                    # compute overall reachable set including linearization error
                    r_max = r_ti + r_all_err
                    # compute linearization error
                    true_err, v_err_dyn = self.__abstract_err_lin(r_max, option)
                elif option.algorithm == ALGORITHM.POLYNOMIAL:
                    r_max = r_delta + cvt2(r_diff, Geometry.TYPE.ZONOTOPE) + r_all_err
                    true_err, v_err_dyn, v_err_stat = self.__abstract_err_poly(
                        option, r_max, r_diff + r_all_err, h, zd, err_stat, t, ind3, zd3
                    )
                else:
                    raise NotImplementedError

                # compare linearization error with the maximum allowed error
                temp = true_err / applied_err
                temp[np.isnan(temp)] = -np.inf
                perf_ind_cur = np.max(temp)
                perf_ind = np.max(true_err / option.max_err)
                abstract_err = true_err

                # exception for set explosion
                if np.any(abstract_err > 1e100):
                    raise Exception("Set Explosion")
            # translate reachable sets by linearization point
            r_ti += option.lin_err_px
            r_tp += option.lin_err_px

            # compute the reachable set due to the linearization error
            r_err = lin_sys.error_solution(lin_op, v_err_dyn, v_err_stat)

            # add the abstraction error to the reachable sets
            r_ti += r_err
            r_tp += r_err
            # determine the best dimension to split the set in order to reduce the
            # linearization error
            dim_for_split = []
            if perf_ind > 1:
                raise NotImplementedError  # TODO
            # store the linearization error
            return r_ti, Set(r_tp, abstract_err), dim_for_split

        def __reach_init(self, r0: [Set], option):
            r_ti, r_tp, r = [], [], []
            for i in range(len(r0)):
                temp_r_ti, temp_r_tp, dims = self.__linear_reach(r0[i], option)
                # check if initial set has to be split
                if len(dims) <= 0:
                    r_tp.append(temp_r_tp)
                    r_ti.append(temp_r_ti)
                    r.append(r0[i])
                else:
                    raise NotImplementedError  # TODO

                # store the result
            return r_ti, r_tp, r

        def __post(self, r: [Set], option: NonLinSys.Option.Base):
            next_ti, next_tp, next_r0 = self.__reach_init(r, option)

            for i in range(len(next_tp)):
                if not next_tp[i].geometry.is_empty:
                    next_tp[i].geometry.reduce(Zonotope.REDUCE_METHOD, Zonotope.ORDER)
                    next_ti[i].reduce(Zonotope.REDUCE_METHOD, Zonotope.ORDER)

            return next_ti, next_tp, next_r0

        def reach(self, option: NonLinSys.Option.Base):
            assert option.validation(self.dim)  # ensure valid options
            if option.algorithm == ALGORITHM.LINEAR:
                return self.__reach_over_standard(option)
            elif option.algorithm == ALGORITHM.POLYNOMIAL:
                return self.__reach_over_standard(option)
            elif option.algorithm == ALGORITHM.BACK_UNDER:
                return self.__reach_under_standard(option)
            else:
                raise NotImplementedError
