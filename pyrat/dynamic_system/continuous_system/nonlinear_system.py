from __future__ import annotations

from dataclasses import dataclass

from pyrat.model import Model
from .continuous_system import ContSys, Option


class NonLinSys:
    @dataclass
    class Option(Option):
        algo: str = "standard"
        lagrange_rem = {}

        def validate(self) -> bool:
            #  TODO
            return True

    class Sys(ContSys):
        def __init__(self, model: Model):
            self._model = model
            self._misc = {}

        # =============================================== operator
        def __str__(self):
            raise NotImplementedError

        # =============================================== property
        def dim(self) -> int:
            raise NotImplementedError

        # =============================================== private method
        def _reach_over_standard(self, op: NonLinSys.Option):
            raise NotImplementedError

        # TODO

        # =============================================== public method
        def reach(self, op: NonLinSys.Option):
            assert op.validate()
            print("do something")
            raise NotImplementedError

        def simulate(self, op: NonLinSys.Option):
            raise NotImplementedError
