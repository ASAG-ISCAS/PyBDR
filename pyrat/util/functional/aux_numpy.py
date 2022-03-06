import numpy as np


def is_empty(arr: np.ndarray) -> bool:
    """
    check if given array is empty or not, same as MATLAB isempty function
    :param arr: given array
    :return: returns True if X is an empty array and 0 otherwise.
    An empty array has no elements, that is prod(size(X))==0.
    """
    return np.prod(arr.shape) == 0


def cross_ndim(m: np.ndarray) -> np.ndarray:
    """
    compute n-dimensional cross product
    :param m: matrix storing column vectors
    :return: orthogonal vector
    """
    v = np.tile(m, (m.shape[0], 1, 1))
    mask = np.ones_like(v, dtype=bool)
    idx = np.arange(m.shape[0])
    mask[idx, idx, :] = False
    v = np.reshape(v[mask], (m.shape[0], v.shape[2], v.shape[2]))
    return np.power(-1, idx + 1) * np.linalg.det(v)


def min_affine(arr: np.ndarray) -> np.ndarray:
    """
    compute a minimum representation for the given affine set as column vectors
    :param arr:
    :return:
    """
    if arr.shape[0] == 0:
        return arr
    r = np.linalg.matrix_rank(arr)
    if r == arr.shape[0]:
        return arr
    # choose r linearly independent rows of the given array
    eig_v, _ = np.linalg.eig(arr)
    return arr[:, eig_v == 0]
