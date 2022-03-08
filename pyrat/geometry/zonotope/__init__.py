from __future__ import annotations

import numpy as np


class Zonotope:
    from .functional import (
        dim,
        is_empty,
        generator,
        rank,
        center,
        is_fulldim,
        rand_fix_dim,
        remove_empty_gen,
        __abs__,
        __add__,
        __iadd__,
        __sub__,
        __isub__,
        __str__,
        gen_num,
        to,
        z,
        _to_polyhedron,
        _approx_mink_diff_althoff,
        _approx_mink_diff_cons_zono,
    )

    def __init__(self, z: np.ndarray):
        assert z.ndim == 2
        self._z = z

    @classmethod
    def _new(cls, z: np.ndarray) -> Zonotope:
        return Zonotope(z)
