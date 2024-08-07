import numbers
import datetime
import numpy as np
from scipy.sparse import coo_matrix
import time


def is_empty(arr: (np.ndarray, coo_matrix)) -> bool:
    """
    check if a given array is empty or not, same as MATLAB isempty function
    :param arr: given array
    :return: returns True if X is an empty array and 0 otherwise.
    An empty array has no elements, that is prod(size(X))==0.
    """
    return True if arr is None else np.prod(arr.shape) == 0


def cross_ndim(m: np.ndarray) -> np.ndarray:
    """
    compute n-dimensional cross-product
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


def mat_powers_2d(arr: np.ndarray, n: int):
    assert arr.ndim == 2
    m = np.eye(*arr.shape)
    mat_powers = [m]
    for idx in range(n):
        mat_powers.append(mat_powers[-1] @ arr)
    return np.stack(mat_powers)


def mask_lt(m: coo_matrix, v: numbers.Real) -> coo_matrix:
    mask = m.data < v
    data = m.data[mask]
    row = m.row[mask]
    col = m.col[mask]
    return coo_matrix((data, (row, col)), shape=m.shape, dtype=float)


def mask_le(m: coo_matrix, v: numbers.Real) -> coo_matrix:
    mask = m.data <= v
    data = m.data[mask]
    row = m.row[mask]
    col = m.col[mask]
    return coo_matrix((data, (row, col)), shape=m.shape, dtype=float)


def mask_eq(m: coo_matrix, v: numbers.Real, tol: float = 1e-30) -> coo_matrix:
    mask = abs(m.data - v) <= tol
    data = m.data[mask]
    row = m.row[mask]
    col = m.col[mask]
    return coo_matrix((data, (row, col)), shape=m.shape, dtype=float)


def mask_gt(m: coo_matrix, v: numbers.Real) -> coo_matrix:
    mask = m.data > v
    data = np.ones_like(m.data[mask])
    row = m.row[mask]
    col = m.col[mask]
    return coo_matrix((data, (row, col)), shape=m.shape, dtype=float)


def mask_ge(m: coo_matrix, v: numbers.Real) -> coo_matrix:
    mask = m.data >= v
    data = np.ones_like(m.data[mask])
    row = m.row[mask]
    col = m.col[mask]
    return coo_matrix((data, (row, col)), shape=m.shape, dtype=bool)


def mask_condition(m: coo_matrix, inr_mask: np.ndarray) -> coo_matrix:
    assert inr_mask.ndim == 1
    data = np.ones_like(m.data[inr_mask])
    row = m.row[inr_mask]
    col = m.col[inr_mask]
    return coo_matrix((data, (row, col)), shape=m.shape, dtype=bool)


def performance_counter_start():
    return time.perf_counter_ns()


def performance_counter(start: time.perf_counter_ns(), event: str, runs: int = 1):
    assert runs >= 1
    end = time.perf_counter_ns()
    print(event + " cost: {}s".format(((end - start) / runs) * 1e-9))
    return end


def time_stamp():
    t = time.time()
    return datetime.datetime.fromtimestamp(t).strftime("%Y%m%d_%H%M%S")


def get_system():
    import platform
    this_platform = platform.platform().lower()
    if 'windows' in this_platform:
        return 'windows'
    elif 'linux' in this_platform:
        return 'linux'
    elif 'macos' in this_platform:
        return 'macos'
    else:
        raise NotImplementedError
