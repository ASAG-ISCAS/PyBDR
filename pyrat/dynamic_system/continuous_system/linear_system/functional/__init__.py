from dataclasses import dataclass

import numpy as np


@dataclass
class Options:
    time_step: float = None
    max_zono_order: int = None
    algo: str = None
    reduction_tec: str = None
    taylor_terms: int = None
    partition: np.ndarray = None
    krylov_error: float = None
    krylov_step: float = None
    error: float = None
    factors: np.ndarray = None
    u_track_vec: np.ndarray = None


from ._reach_standard import _reach_standard
from ._reach_adaptive import _reach_adaptive
from ._reach_decomp import _reach_decomp
from ._reach_from_start import _reach_from_start
from ._reach_krylov import _reach_krylov
from ._reach_wrapping_free import _reach_wrapping_free
from .over_reach import over_reach
from .under_reach import under_reach
from .init_reach_euclidean import init_reach_euclidean
from .dim import dim

__all__ = [
    "_reach_krylov",
    "_reach_decomp",
    "_reach_adaptive",
    "_reach_standard",
    "_reach_from_start",
    "_reach_wrapping_free",
    "under_reach",
    "over_reach",
    "dim",
]
