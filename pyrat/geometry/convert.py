from .interval_old import IntervalOld
from .vector_zonotope import VectorZonotope
import numpy as np


def _interval2vector_zonotope(source: IntervalOld):
    raise NotImplementedError  # TODO


def _vector_zonotope2interval(source: VectorZonotope):
    """
    over approximate a zonotope by an interval
    :param source: given zonotope
    :return:
    """
    c = source.c
    # determine left and right limit, specially designed for high performance
    delta = np.sum(abs(source.z), axis=1) - abs(c)
    left_limit = c - delta
    right_limit = c + delta
    return IntervalOld(np.vstack([left_limit, right_limit]))


def cvt2(source, target: str):
    if isinstance(source, IntervalOld) and target == "vz":
        return _interval2vector_zonotope(source)
    elif isinstance(source, VectorZonotope) and target == "int":
        return _vector_zonotope2interval(source)
    else:
        raise NotImplementedError
