from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from scipy.special import factorial
from sympy import derive_by_array, ImmutableDenseNDimArray, lambdify

from pyrat.geometry import Geometry, Zonotope, cvt2, Interval, IntervalMatrix
from pyrat.misc import Set, Reachable
from pyrat.model import Model


class ContSys:
    class TYPE(IntEnum):
        LINEAR_SYSTEM = 0
        NON_LINEAR_SYSTEM = 1

    class Option:
        @dataclass
        class Base(ABC):
            t_start: float = 0
            t_end: float = 0
            steps: int = 10
            step_size: float = None
            r0: [Set] = None
            u: Geometry.Base = None
            taylor_terms: int = 4

            def _validate_time_related(self):
                assert self.t_start <= self.t_end
                assert self.steps >= 1
                self.step_size = (self.t_end - self.t_start) / self.steps
                return True

            def _validate_inputs(self):
                for idx in range(len(self.r0)):  # confirm valid inputs
                    if isinstance(self.r0[idx], Geometry.Base):
                        self.r0[idx] = Set(self.r0[idx])
                    else:
                        assert isinstance(self.r0[idx], Set)
                return True

            @abstractmethod
            def validation(self, dim: int):
                raise NotImplementedError

    class Entity(ABC):
        def __init__(self, model: Model):
            assert 1 <= len(model.vars) <= 2  # f(x) may be a good alternative
            self._model = model
            self._jaco = None
            self._hess = None
            self._third = None
            self.__init_symbolic()

        def __init_symbolic(self):
            def _take_derivative(f, x):
                d = np.asarray(derive_by_array(f, x))
                return np.moveaxis(d, 0, -2)

            self._jaco, self._hess, self._third = [], [], []
            for var in self._model.vars:
                j = _take_derivative(self._model.f, var)
                h = _take_derivative(j, var)
                t = _take_derivative(h, var)
                j = j.squeeze(axis=-1)
                h = h.squeeze(axis=-1)
                t = t.squeeze(axis=-1)
                self._jaco.append(ImmutableDenseNDimArray(j))
                self._hess.append(ImmutableDenseNDimArray(h))
                self._third.append(ImmutableDenseNDimArray(t))

        def _evaluate(self, xs: tuple, mod: str = "numpy"):
            f = lambdify(self._model.vars, self._model.f, mod)
            return np.squeeze(f(*xs))

        def _jacobian(self, xs: tuple, mod: str = "numpy"):
            def _eval_jacob(jc):
                f = lambdify(self._model.vars, jc, mod)
                return f(*xs)

            return [_eval_jacob(j) for j in self._jaco]

        def _hessian(self, xs: tuple, mod: str = "numpy"):
            """
            NOTE: only support interval matrix currently and numpy
            """

            def _eval_hessian_interval(he):
                def __eval_element(x):
                    if x.is_number:
                        return Interval(float(x), float(x))
                    else:
                        f = lambdify(self._model.vars, x, Interval.functional())
                        return f(*xs)

                v = np.vectorize(__eval_element)(he)
                inf = np.vectorize(lambda x: x.inf)(v)
                sup = np.vectorize(lambda x: x.sup)(v)
                return [
                    IntervalMatrix(inf[idx], sup[idx]) for idx in range(he.shape[0])
                ]

            def _eval_hessian_numpy(he):
                f = lambdify(self._model.vars, he, "numpy")
                return np.asarray(f(*xs)).squeeze()

            def _eval(he):
                if mod == "numpy":
                    return _eval_hessian_numpy(he)
                elif mod == "interval":
                    return _eval_hessian_interval(he)
                else:
                    raise NotImplementedError

            return [_eval(h) for h in self._hess]

        def _third_order(self, xs: tuple, mod: str = "numpy"):
            """
            NOTE: only support interval and numpy currently
            """

            def _eval_third_interval(te):
                def __eval_element(x):
                    if x.is_number:
                        return Interval(float(x), float(x))
                    else:
                        f = lambdify(self._model.vars, x, Interval.functional())
                        return f(*xs)

                v = np.vectorize(__eval_element)(te)
                inf = np.vectorize(lambda x: x.inf)(v)
                sup = np.vectorize(lambda x: x.sup)(v)

                ret = []
                for row in range(te.shape[0]):
                    row_ret = []
                    for col in range(te.shape[1]):
                        row_ret.append(IntervalMatrix(inf[row][col], sup[row][col]))
                    ret.append(row_ret)
                ind = np.argwhere(np.vectorize(lambda x: not x.is_zero)(ret))
                return ind, np.asarray(ret)

            def _eval_third_numpy(te):
                raise NotImplementedError

            def _eval(te):
                if mod == "numpy":
                    return _eval_third_numpy(te)
                elif mod == "interval":
                    return _eval_third_interval(te)
                else:
                    raise NotImplementedError

            return [_eval(te) for te in self._third]

        @abstractmethod
        def __str__(self):
            return NotImplemented

        def __reach_init(self, r0: [Set], option):
            return NotImplemented

        def __post(self, r: [Set], option):
            return NotImplemented

        def __pre_stat_err(self, r_delta, option):
            r_red = cvt2(r_delta, Geometry.TYPE.ZONOTOPE).reduce()
            # extend teh sets byt the input sets
            u_stat = Zonotope.zero(option.u.dim)
            z = r_red.card_prod(u_stat)
            z_delta = r_delta.card_prod(u_stat)
            # compute hessian
            h = self._hessian((option.lin_err_px, option.lin_err_pu), "numpy")
            t, ind3, zd3 = None, None, None

            # calculate the quadratic map == static second order error
            err_stat_sec = 0.5 * z.quad_map(h)
            err_stat = None
            # third order tensor
            if option.tensor_order >= 4:
                raise NotImplementedError
            else:
                err_stat = err_stat_sec
            return h, z_delta, err_stat.reduce(), t, ind3, zd3

        def __reach_over_standard(self, option) -> Reachable.Result:
            # obtain factors for initial state and input solution time step
            i = np.arange(1, option.taylor_terms + 2)
            option.factors = np.power(option.step_size, i) / factorial(i)
            # set current time
            option.cur_t = option.t_start
            # init containers for storing the results
            ti_set, ti_time, tp_set, tp_time = [], [], [], []

            # init reachable set computation
            next_ti, next_tp, next_r0 = self.__reach_init(option.r0, option)

            time_pts = np.linspace(option.t_start, option.t_end, option.steps)

            # loop over all reachability steps
            for i in range(option.steps - 1):
                # save reachable set
                ti_set.append(next_ti)
                ti_time.append(time_pts[i : i + 2])
                tp_set.append(next_tp)
                tp_time.append(time_pts[i + 1])

                # increment time
                option.cur_t = time_pts[i + 1]

                # compute next reachable set
                next_ti, next_tp, next_r0 = self.__post(next_tp, option)

            # save the last reachable set in cell structure
            return Reachable.Result(
                ti_set, tp_set, np.vstack(ti_time), np.array(tp_time)
            )

        @abstractmethod
        def reach(self, option: ContSys.Option.Base):
            raise NotImplementedError
