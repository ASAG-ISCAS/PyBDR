import os

import numpy as np
import sympy

import pyrat
from pyrat.geometry import Interval, Zonotope
from pyrat.util.visualization import plot
from sympy import symbols
from pyrat.util.functional import RealPaver


def test_00():
    real_paver = RealPaver()
    # add constants
    real_paver.add_constant('d', 1.25)
    real_paver.add_constant('x0', 1.5)
    real_paver.add_constant('y0', 0.5)

    # add variables
    real_paver.add_variable('x', -10, 10, '[', upper_bracket=']')
    real_paver.add_variable('y', -np.inf, np.inf, lower_bracket=']', upper_bracket='[')

    # add constraints
    real_paver.add_constraint('d^2 = (x-x0)^2 + (y-y0)^2')
    real_paver.add_constraint('min(y,x-y) <= 0')
    real_paver.set_branch(precision=0.05)

    boxes = real_paver.solve()
    vis_boxes = [b[2] for b in boxes]
    plot(vis_boxes, [0, 1])


def test_01():
    from pyrat.geometry import Polytope
    a = np.array([[-3, 0], [2, 4], [1, -2], [1, 1]])
    b = np.array([-1, 14, 1, 4])
    p = Polytope(a, b)
    plot([p], [0, 1])
    print(a.shape)
    print(b.shape)

    real_paver = RealPaver()
    assert (a.ndim == 2 and b.ndim == 1)
    assert a.shape[0] == b.shape[0]
    num_var = a.shape[1]

    for idx in range(num_var):
        real_paver.add_variable('x' + str(idx), -np.inf, np.inf, ']', '[')

    num_const = a.shape[0]
    for idx_const in range(num_const):
        this_const = ''
        for idx_var in range(num_var):
            this_const += str(a[idx_const, idx_var]) + "*x" + str(idx_var) + "+"
        this_const = this_const[:-1] + "<=" + str(b[idx_const])
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
