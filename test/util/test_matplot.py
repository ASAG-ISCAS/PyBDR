import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt

# Fixing random state for reproducibility
np.random.seed(19680801)

width, height = 800, 800
px = 1 / plt.rcParams["figure.dpi"]
fig, ax = plt.subplots(figsize=(width * px, height * px), layout="constrained")


N = 5
patches = []


for i in range(N):
    polygon = Polygon(np.random.rand(N, 2) * 1000, closed=True, alpha=0.7, fill=False)
    ax.add_patch(polygon)
    # patches.append(polygon)

# colors = 100 * np.random.rand(len(patches))
# p = PatchCollection(patches)
# p.set_array(colors)
# ax.add_collection(p)
ax.autoscale_view()


def test_case_0():
    plt.show()
