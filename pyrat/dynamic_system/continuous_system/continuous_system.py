from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from scipy.special import factorial

from pyrat.geometry import Geometry
from pyrat.misc import Set, Reachable


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
        def __init__(self):
            raise NotImplementedError

        @abstractmethod
        def __str__(self):
            return NotImplemented

        def __reach_init(self, r0: [Set], option):
            return NotImplemented

        def __post(self, r: [Set], option):
            return NotImplemented

        def __reach_over_linear(self, option) -> Reachable.Result:
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
