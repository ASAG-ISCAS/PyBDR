from __future__ import annotations

from abc import ABC

from pybdr.geometry import Geometry


class Algorithm:
    class Options(ABC):
        t_start: float = 0
        t_end: float = 0
        steps_num: int = 10
        step: float = None
        step_idx: int = 0  # index of current step
        r0: [Geometry.Base] = []  # but for some algorithms, this maybe only one set
        u: Geometry.Base = None

        def _validate_time_related(self):
            assert 0 <= self.t_start <= self.t_end
            assert 0 < self.step <= self.t_end - self.t_start
            self.steps_num = round((self.t_end - self.t_start) / self.step)
            self.step_idx = 0
            return True

        def validation(self, dim: int):
            raise NotImplemented
