def test_case_0():
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Polygon

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    width, height = 800, 800
    px = 1 / plt.rcParams["figure.dpi"]
    fig, ax = plt.subplots(figsize=(width * px, height * px), layout="constrained")

    N = 5
    patches = []

    for i in range(N):
        pts = np.array([[0, 0], [2, 1], [1, 1]])
        # polygon = Polygon(pts, closed=False, alpha=0.7, fill=True, linewidth=10)
        plt.plot(pts[:, 0], pts[:, 1])
        # ax.add_patch(polygon)
        # patches.append(polygon)

    # colors = 100 * np.random.rand(len(patches))
    # p = PatchCollection(patches)
    # p.set_array(colors)
    # ax.add_collection(p)
    ax.autoscale_view()
    plt.show()


def test_case_1():
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    delta = 0.005
    y, x = np.ogrid[-4:4:1000j, -6:6:1000j]

    def f(_x, _y):
        return _x**2 + _y**2

    plt.contour(x.ravel(), y.ravel(), f(x, y), [1, 2])
    plt.show()
