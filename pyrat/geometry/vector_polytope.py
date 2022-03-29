from __future__ import annotations

from abc import ABC

from .geometry import Geometry


class VectorPolytope(Geometry, ABC):
    def __init__(self):
        raise NotImplementedError

    # =============================================== property
    @property
    def dim(self) -> int:
        raise NotImplementedError

    @property
    def is_empty(self) -> bool:
        raise NotImplementedError
