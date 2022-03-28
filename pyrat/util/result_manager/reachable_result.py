import numpy as np
from dataclasses import dataclass


class ReachableElement:
    set: None
    error: np.ndarray
    prev: -1
    parent: -1


class ReachableResult:
    def __init__(self, num_interval: int = 0, num_pt: int = 0):
        raise NotImplementedError

    # =============================================== property
    # =============================================== operator
    def __getitem__(self, item):
        pass
