import numpy as np


def min_affine_rep(arr: np.ndarray) -> np.ndarray:
    """
    compute a minimum representation for the given affine set
    :param arr:
    :return:
    """
    if arr.shape[0] == 0:
        return arr
    r = np.linalg.matrix_rank(arr)
    if r == arr.shape[0]:
        return arr
    # choose r linearly independent rows of the given array
    eig_v, _ = np.linalg.eig(arr.T)
    return arr[eig_v == 0, :]
