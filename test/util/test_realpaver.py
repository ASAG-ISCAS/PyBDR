import numpy as np

from pybdr.util.functional import RealPaver
from pybdr.util.visualization import plot


def test_00():
    real_paver = RealPaver()
    # add constants
    real_paver.add_constant("d", 1.25)
    real_paver.add_constant("x0", 1.5)
    real_paver.add_constant("y0", 0.5)

    # add variables
    real_paver.add_variable("x", -10, 10, "[", upper_bracket="]")
    real_paver.add_variable("y", -np.inf, np.inf, lower_bracket="]", upper_bracket="[")

    # add constraints
    real_paver.add_constraint("d^2 = (x-x0)^2 + (y-y0)^2")
    real_paver.add_constraint("min(y,x-y) <= 0")
    real_paver.set_branch(precision=0.05)

    boxes = real_paver.solve()
    vis_boxes = [b[2] for b in boxes]
    plot(vis_boxes, [0, 1])


def test_01():
    from pybdr.geometry import Polytope

    a = np.array([[-3, 0 + 1e-10], [2, 4], [1, -2], [1, 1]])
    b = np.array([-1, 14, 1, 4])
    p = Polytope(a, b)

    num_const, num_var = a.shape
    print(num_const, num_var)

    real_paver = RealPaver()
    assert a.ndim == 2 and b.ndim == 1
    assert a.shape[0] == b.shape[0]
    num_var = a.shape[1]

    for idx in range(num_var):
        real_paver.add_variable("x" + str(idx), -np.inf, np.inf, "]", "[")

    num_const = a.shape[0]
    for idx_const in range(num_const):
        this_const = ""
        for idx_var in range(num_var):
            this_const += (
                    "{:.20e}".format(a[idx_const, idx_var]) + "*x" + str(idx_var) + "+"
            )
            # this_const += str(a[idx_const, idx_var]) + "*x" + str(idx_var) + "+"
        this_const = this_const[:-1] + "<=" + "{:.20e}".format(b[idx_const])
        real_paver.add_constraint(this_const)

    real_paver.set_branch(precision=0.1)
    boxes = real_paver.solve()
    bound_boxes = []
    for b in boxes:
        if b[0] == "OUTER":
            bound_boxes.append(b[2])
    all_boxes = [b[2] for b in boxes]
    plot([p, *bound_boxes], [0, 1])
    plot([p, *all_boxes], [0, 1])


def test_03():
    from pybdr.geometry import Geometry, Polytope
    from pybdr.geometry.operation import boundary
    from pybdr.util.visualization import plot

    p = Polytope.rand(2)
    plot([p], [0, 1])
    boxes = boundary(p, 0.1, Geometry.TYPE.INTERVAL)

    plot([*boxes, p], [0, 1])


if __name__ == "__main__":
    import sys

    sys.path.append("../../")

    import numpy as np

    from pybdr.geometry import Geometry, Polytope
    from pybdr.geometry.operation import boundary
    from pybdr.util.visualization import plot

    p = Polytope.rand(2)
    plot([p], [0, 1])
    boxes = boundary(p, 0.02, Geometry.TYPE.INTERVAL)

    plot([*boxes, p], [0, 1])

    # a = Interval.rand(3)
    # # print(a)
    # # plot([a], [0, 1])
    # b = np.random.rand(2, 3)
    # print(b.shape)
    # print(b)
    # ind = np.where(np.abs(b) <= 0.5)
    # print(ind)
    # b[ind] = 10
    # print(b)
