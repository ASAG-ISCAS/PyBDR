import matplotlib.pyplot as plt
import numpy as np


def _y_min_max(objs, x):
    y_min = np.inf
    y_max = -np.inf
    for obj in objs:
        if obj.__class__.__name__ == "HalfSpace":
            assert obj.dim == 2
            # handle vertical condition
            if abs(obj.c[1]) <= np.finfo(np.float).eps:
                return x.min(), x.max()
            # handle horizontal condition
            if abs(obj.c[0]) <= np.finfo(np.float).eps:
                if obj.c[1] > 0:
                    # shall draw the lower region
                    y_max = obj.d / obj.c[1]
                    y_min = y_max + x.min() - x.max()
                else:
                    # shall draw the upper region
                    y_min = obj.d / obj.c[1]
                    y_max = y_min + x.max() - x.min()
            else:
                y = (obj.d - obj.c[0] * x) / obj.c[1]
                y_min = min(np.min(y), y_min)
                y_max = max(np.max(y), y_max)
        elif isinstance(obj, np.ndarray):
            assert obj.ndim == 2 and obj.shape[1] == 2
            y_min = min(np.min(obj[:, 1]), y_min)
            y_max = max(np.max(obj[:, 1]), y_max)
        else:
            raise NotImplementedError
    return y_min, y_max


def _vis_halfspace(h, ax, x, y_min, y_max):
    assert h.dim == 2
    # handle the vertical condition
    if abs(h.c[1]) <= np.finfo(np.float).eps:
        check_pts = np.zeros((2, 2), dtype=float)
        check_pts[0, 0] = x.min()
        check_pts[1, 0] = x.max()
        min_idx = np.argmin(h.dist2bd(check_pts))
        if min_idx == 0:
            # draw the left region
            ax.fill_betweenx(
                [y_min, y_max], x1=x.min(), x2=min(h.d / h.c[0], x.max()), alpha=0.3
            )
        else:
            # draw the right region
            ax.fill_betweenx(
                [y_min, y_max], x1=max(h.d / h.c[0], x.min()), x2=x.max(), alpha=0.3
            )
        return
    # handle the horizontal condition
    if abs(h.c[0]) <= np.finfo(np.float).eps:
        if h.c[1] > 0:
            # draw the lower region
            ax.fill_between(x, y1=y_min, y2=h.d / h.c[1], alpha=0.3)
        else:
            # draw the upper region
            ax.fill_between(x, y1=h.d / h.c[1], y2=y_max, alpha=0.3)
        return
    y = (h.d - h.c[0] * x) / h.c[1]
    # check draw above region or under region
    check_pts = np.zeros((4, 2), dtype=float)  # four corner points
    check_pts[:2, 0] = x.min()
    check_pts[2:, 0] = x.max()
    check_pts[[0, 2], 1] = y_min
    check_pts[[1, 3], 1] = y_max
    min_idx = np.argmin(h.dist2bd(check_pts))
    if min_idx == 0 or min_idx == 2:
        # draw under region
        ax.fill_between(x, y1=y_min, y2=y, alpha=0.3)
    else:
        # draw above region
        ax.fill_between(x, y1=y, y2=y_max, alpha=0.3)


def _vis_pts(pts, ax, pt_size=3):
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        s=np.pi * pt_size**2,
        alpha=1,
    )


def vis2d(objs, width=800, height=800, x_lb=0, x_ub=1, granularity=10, eq_axis=True):
    """
    # NOTE
    all inputs must be in 2d space, dimensional issue must be confirmed by the caller
    :param objs:
    :param width:
    :param height:
    :param x_lb:
    :param x_ub:
    :param granularity:
    :param eq_axis:
    :return:
    """
    px = 1 / plt.rcParams["figure.dpi"]
    fig, ax = plt.subplots(figsize=(width * px, height * px), layout="constrained")
    x = np.linspace(0, 1, num=granularity)
    y_min, y_max = _y_min_max(objs, x)
    print(y_min, y_max)
    for obj in objs:
        if obj.__class__.__name__ == "HalfSpace":
            assert not (np.isinf(y_min) and np.isinf(y_max))
            _vis_halfspace(obj, ax, x, y_min, y_max)
        elif isinstance(obj, np.ndarray):
            _vis_pts(obj, ax)
        else:
            raise NotImplementedError
    if eq_axis:
        plt.axis("equal")
    plt.show()
