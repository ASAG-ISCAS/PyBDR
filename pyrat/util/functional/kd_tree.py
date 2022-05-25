import sys

from scipy import spatial


def rnn(dataset, query, r, n_jobs=-1, is_sorted=True, is_length=False):
    """
    :param dataset: dataset to be indexed
    :param query: dataset to query
    :param r: radius
    :param n_jobs: number of processors
    :param is_sorted: sorts returned indices if TRUE
    :param is_length: return the number of points inside the radius instead of the
    neighboring indices
    :return: neighboring indices
    """
    kd_tree = spatial.cKDTree(dataset)
    return kd_tree.query_ball_point(
        query, r, workers=n_jobs, return_sorted=is_sorted, return_length=is_length
    )


def knn(dataset, query, k, processors_num=-1, distance_upper_bound=sys.float_info.max):
    """
    :param dataset: dataset to be indexed
    :param query: dataset to query
    :param k: k nearest neighbors shall be returned
    :param processors_num: number of processors
    :param distance_upper_bound: upper bound of the distance
    :return: tuple of neighboring indices and related distances
    """
    kd_tree = spatial.cKDTree(dataset)
    return kd_tree.query(
        query, k, workers=processors_num, distance_upper_bound=distance_upper_bound
    )


def kdtree(dataset):
    """

    :param dataset: dataset shall be indexed
    :return: kd tree used for searching
    """
    return spatial.cKDTree(dataset)


def rnn_query(kd, query, r, n_jobs=-1, is_sorted=True, is_length=False):
    """

    :param kd: given kd tree used for radius nearest neighbor searching
    :param query: data to query
    :param r: radius
    :param n_jobs: number of processors
    :param is_sorted: sorts returned indices if TRUE
    :param is_length: return the number of points inside the radius instead of the
    neighboring indices
    :return: neighboring indices
    """
    return kd.query_ball_point(
        query, r, workers=n_jobs, return_sorted=is_sorted, return_length=is_length
    )


def knn_query(kd, query, k, processors_num=-1, distance_upper_bound=sys.float_info.max):
    """

    :param kd: given kd tree used for the nearest neighboring searching
    :param query: data to query
    :param k: k nearest neighbors shall be returned
    :param processors_num: number of processors
    :param distance_upper_bound: upper bound of the distance
    :return: tuple of neighboring indices and related distances
    """
    return kd.query(
        query, k, workers=processors_num, distance_upper_bound=distance_upper_bound
    )
