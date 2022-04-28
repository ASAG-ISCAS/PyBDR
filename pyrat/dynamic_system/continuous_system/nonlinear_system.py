from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
import numpy as np
from sympy import derive_by_array, ImmutableDenseNDimArray, lambdify
from pyrat.model import Model
from pyrat.geometry import Geometry, Interval, IntervalMatrix, Zonotope, cvt2
from pyrat.misc import Reachable, Set
from .continuous_system import ContSys
from .linear_system import LinSys
from enum import IntEnum


class ALGORITHM(IntEnum):
    LINEAR = 1
    POLYNOMIAL = 2


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
            tensor_order: int = 3

            def validation(self, dim: int):
                assert self._validate_time_related()
                assert self._validate_inputs()
                assert self._validate_misc(dim)
                return True

    class Entity(ContSys.Entity):
        def __init__(self, model: Model):
            assert 1 <= len(model.vars) <= 2  # only support f(x) or f(x,u) as input
            self._model = model
            self._jaco = None
            self._hess = None
            self.__init_symbolic()  # init symbolic jacobian and hessian

        def __init_symbolic(self):
            def _take_derivative(f, x):
                d = np.asarray(derive_by_array(f, x))
                return np.moveaxis(d, 0, -2)

            self._jaco, self._hess = [], []
            for var in self._model.vars:
                j = _take_derivative(self._model.f, var)
                h = _take_derivative(j, var)
                j = j.squeeze(axis=-1)
                h = h.squeeze(axis=-1)
                self._jaco.append(ImmutableDenseNDimArray(j))
                self._hess.append(ImmutableDenseNDimArray(h))

        def _evaluate(self, xs: tuple, mod: str = "numpy"):
            f = lambdify(self._model.vars, self._model.f, mod)
            return np.squeeze(f(*xs))

        def _jacobian(self, xs: tuple, mod: str = "numpy"):
            def _eval_jacob(jc):
                f = lambdify(self._model.vars, jc, mod)
                return f(*xs)

            return [_eval_jacob(j) for j in self._jaco]

        def _hessian(self, xs: tuple, ops: dict):
            """
            NOTE: only support interval matrix currently
            """

            def _eval_hessian_interval(he):
                def __eval_element(x):
                    if x.is_number:
                        return Interval(float(x), float(x))
                    else:
                        f = lambdify(self._model.vars, x, ops)
                        return f(*xs)

                v = np.vectorize(__eval_element)(he)
                inf = np.vectorize(lambda x: x.inf)(v)
                sup = np.vectorize(lambda x: x.sup)(v)
                return [
                    IntervalMatrix(inf[idx], sup[idx]) for idx in range(he.shape[0])
                ]

            return [_eval_hessian_interval(h) for h in self._hess]

        def __str__(self):
            raise NotImplementedError

        @property
        def dim(self):
            return self._model.dim

        def __linearize(self, r: Geometry.Base, option: NonLinSys.Option.Linear):
            option.lin_err_pu = option.u_trans if option.u is not None else None
            f0_pre = self._evaluate((r.c, option.lin_err_pu))
            option.lin_err_px = r.c + f0_pre * 0.5 * option.step_size
            option.lin_err_f0 = self._evaluate((option.lin_err_px, option.lin_err_pu))
            a, b = self._jacobian((option.lin_err_px, option.lin_err_pu))
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
                hx, hu = self._hessian(
                    (total_int_x, total_int_u), Interval.functional()
                )

                # calculate the Lagrange remainder (second-order error)
                err_lagrange = np.zeros(self.dim, dtype=float)

                for i in range(self.dim):
                    abs_hx, abs_hu = abs(hx[i]), abs(hu[i])
                    hx_ = np.maximum(abs_hx.inf, abs_hx.sup)
                    hu_ = np.maximum(abs_hu.inf, abs_hu.sup)
                    err_lagrange[i] = 0.5 * (dx @ hx_ @ dx + du @ hu_ @ du)

                v_err_dyn = Zonotope(
                    0 * err_lagrange.reshape(-1), np.diag(err_lagrange)
                )
                return err_lagrange, v_err_dyn
            elif option.tensor_order == 3:
                raise NotImplementedError  # TODO
            else:
                raise Exception("unsupported tensor order")

        def __linear_reach(self, r: Set, option):
            lin_sys, lin_op = self.__linearize(r.geo, option)
            r_delta = r.geo - option.lin_err_px
            r_ti, r_tp = lin_sys.reach_init(r_delta, lin_op)
            perf_ind_cur, perf_ind = np.inf, 0
            applied_err, abstract_err, v_err_dyn = None, r.err, None

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
                    raise NotImplementedError
                else:
                    raise NotImplementedError

                # compare linearization error with the maximum allowed error
                perf_ind_cur = np.max(true_err / applied_err)
                perf_ind = np.max(true_err / option.max_err)
                abstract_err = true_err
            # translate reachable sets by linearization point
            r_ti += option.lin_err_px
            r_tp += option.lin_err_px

            # compute the reachable set due to the linearization error
            r_err = lin_sys.error_solution(lin_op, v_err_dyn)

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
                if not next_tp[i].geo.is_empty:
                    next_tp[i].geo.reduce()
                    next_ti[i].reduce()

            return next_ti, next_tp, next_r0

        def reach(self, option: NonLinSys.Option.Base):
            assert option.validation(self.dim)  # ensure valid options
            if option.algorithm == ALGORITHM.LINEAR:
                return self.__reach_over_linear(option)
            elif option.algorithm == ALGORITHM.POLYNOMIAL:
                raise NotImplementedError
            else:
                raise NotImplementedError
