from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np


def _halfspace_intersection(hs) -> np.ndarray:
    if len(hs) <= 1:
        return np.zeros((1, 2), dtype=float)
    # get intersection pts from each two pairs
    combs = list(combinations(np.arange(len(hs)), 2))
    pts = []
    for comb in combs:
        pts.append(hs[comb[0]].intersection_pt(hs[comb[1]]))
    return np.vstack(pts)


def _vh_minmax(hs, cur_r_min, cur_r_max):
    # update range according to vertical/horizontal condition
    r_min, r_max = cur_r_min, cur_r_max
    for h in hs:
        # if horizontal
        if h.is_fullspace(0):
            bound = h.d / h.c[1]
            r_min = min(r_min, bound)
            r_max = max(r_max, bound)
        # if vertical
        if h.is_fullspace(1):
            bound = h.d / h.c[0]
            r_min = min(r_min, bound)
            r_max = max(r_max, bound)
    return r_min, r_max


def _min_max(objs):
    # default set center as origin and range as [-1,1]
    r_min, r_max = -1, 1
    hs = []
    for obj in objs:
        if obj.__class__.__name__ == "HalfSpace":
            assert obj.dim == 2
            hs.append(obj)
        elif isinstance(obj, np.ndarray):
            assert obj.ndim == 2 and obj.shape[1] == 2
            r_min = min(np.min(obj), r_min)
            r_max = max(np.max(obj), r_max)
        else:
            raise NotImplementedError
    # check intersections among halfspaces
    intersections = _halfspace_intersection(hs)
    r_min = min(np.min(intersections), r_min)
    r_max = max(np.max(intersections), r_max)
    r_min, r_max = _vh_minmax(hs, r_min, r_max)
    ext_bd = min((r_max - r_min) * 0.5, 30)
    return np.floor(r_min - ext_bd), np.ceil(r_max + ext_bd), ext_bd


def _vis_halfspace(h, ax, r_min, r_max):
    assert h.dim == 2
    v = np.eye(2) @ h.c
    xy = np.array([r_min, r_max])
    if np.count_nonzero(v == 0) > 0:  # handle special condition
        if v[1] == 0:  # if vertical condition
            if v[1] > 0:  # draw left region
                ax.fill_betweenx(xy, x1=r_min, x2=h.d / h.c[0], alpha=0.3)
            else:  # else draw right region
                ax.fill_betweenx(xy, x1=h.d / h.c[0], x2=r_max, alpha=0.3)
        else:  # else horizontal condition
            if v[0] > 0:  # draw lower region
                ax.fill_between(xy, y1=r_min, y2=h.d / h.c[1], alpha=0.3)
            else:  # else draw upper region
                ax.fill_between(xy, y1=h.d / h.c[1], y2=r_max, alpha=0.3)
    else:  # handle general condition
        y = (h.d - h.c[0] * xy) / h.c[1]
        if v[1] > 0:  # draw lower region
            ax.fill_between(xy, y1=r_min, y2=y, alpha=0.3)
        else:
            ax.fill_between(xy, y1=y, y2=r_max, alpha=0.3)


def _vis_halfspace_old(h, ax, r_min, r_max):
    assert h.dim == 2
    eps = np.finfo(np.float).eps
    pts = np.zeros((4, 2), dtype=float)
    # handle vertical condition
    if abs(h.c[1]) <= eps:
        bound_val = h.d / h.c[0]
        if h.c[0] > 0:
            # c0*x<=d, draw left region
            pts[0, :] = r_min
            pts[1, :] = r_min, r_max
            pts[2, :] = bound_val, r_max
            pts[3, :] = bound_val, r_min
        else:
            # c0*x>=d, draw right region
            pts[0, :] = bound_val, r_min
            pts[1, :] = bound_val, r_max
            pts[2, :] = r_max
            pts[3, :] = r_max, r_min
        polygon = plt.Polygon(pts, True, alpha=0.3)
        ax.add_patch(polygon)
        return
    # handle horizontal condition
    if abs(h.c[0]) <= eps:
        bound_val = h.d / h.c[1]
        if h.c[1] > 0:
            # c1*y<=d, draw lower region
            pts[0, :] = r_min
            pts[1, :] = r_min, bound_val
            pts[2, :] = r_max, bound_val
            pts[3, :] = r_max, r_min
        else:
            # c1*y>=d, draw upper region
            pts[0, :] = r_min, bound_val
            pts[1, :] = r_min, r_max
            pts[2, :] = r_max
            pts[3, :] = r_max, bound_val
        polygon = plt.Polygon(pts, True, alpha=0.3)
        ax.add_patch(polygon)
        return
    # handle general case
    x = np.array([r_min, r_max])
    y = (h.d - h.c[0] * x) / h.c[1]
    y[y < r_min] = r_min
    y[y > r_max] = r_max
    v = h.c @ np.array([0, 1])
    if v >= 0:
        # draw upper region
        ax.fill_between(x, y1=r_min, y2=y, alpha=0.3)
    else:
        # draw lower region
        ax.fill_between(x, y1=y, y2=r_max, alpha=0.3)


def _vis_pts(pts, ax, pt_size=3):
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        s=np.pi * pt_size**2,
        alpha=1,
    )


def vis2d(objs, width=800, height=800, eq_axis=True):
    """
    # NOTE
    all inputs must be in 2d space, dimensional issue must be confirmed by the caller
    :param objs:
    :param width:
    :param height:
    :param eq_axis:
    :return:
    """
    px = 1 / plt.rcParams["figure.dpi"]
    fig, ax = plt.subplots(figsize=(width * px, height * px), layout="constrained")
    r_min, r_max, ext_bd = _min_max(objs)
    print(r_min, r_max)
    for obj in objs:
        if obj.__class__.__name__ == "HalfSpace":
            _vis_halfspace(obj, ax, r_min, r_max)
        elif isinstance(obj, np.ndarray):
            _vis_pts(obj, ax)
        else:
            raise NotImplementedError
    if eq_axis:
        plt.axis("equal")
    plt.xlim([r_min + ext_bd, r_max - ext_bd])
    plt.ylim([r_min + ext_bd, r_max - ext_bd])
    ax.axhline(y=0, color="k")
    ax.axvline(x=0, color="k")
    plt.show()
