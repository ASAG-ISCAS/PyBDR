from __future__ import annotations

from dataclasses import dataclass

from .continuous_system import ContSys, Option


class LinSys:
    @dataclass
    class Option(Option):
        algo: str = "standard"

        def validate(self) -> bool:
            raise NotImplementedError

    class Sys(ContSys):
        def __init__(self):
            raise NotImplementedError
