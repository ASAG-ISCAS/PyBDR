import numpy as np

from pybdr.geometry import Interval, Polytope, Zonotope, Geometry
from pybdr.geometry.operation import boundary
from pybdr.util.visualization import plot


def test_interval2interval():
    box = Interval.rand(2)
    bound_boxes = boundary(box, 0.02, Geometry.TYPE.INTERVAL)
    print(len(bound_boxes))
    plot([box, *bound_boxes], [0, 1])
    plot([box], [0, 1])
    plot(bound_boxes, [0, 1])


def test_interval2polytope():
    box = Interval.rand(3)
    bound_boxes = boundary(box, 0.02, Geometry.TYPE.POLYTOPE)
    # there is a bug when visualize the result if a polytope is essentially a box with 0 width in some dimension


def test_interval2zonotope():
    box = Interval.rand(2)
    bound_zonotopes = boundary(box, 0.02, Geometry.TYPE.ZONOTOPE)
    print(len(bound_zonotopes))
    plot([box, *bound_zonotopes], [0, 1])
    plot([box], [0, 1])
    plot(bound_zonotopes, [0, 1])


def test_polytope2interval():
    poly = Polytope.rand(2)
    bound_boxes = boundary(poly, 0.05, Geometry.TYPE.INTERVAL)
    plot([poly, *bound_boxes], [0, 1])


def test_polytope2polytope():
    poly = Polytope.rand(2)
    bound_boxes = boundary(poly, 0.05, Geometry.TYPE.POLYTOPE)
    plot([*bound_boxes, poly], [0, 1])


def test_polytope2zonotope():
    poly = Polytope.rand(2)
    bound_boxes = boundary(poly, 0.05, Geometry.TYPE.ZONOTOPE)
    plot([poly, *bound_boxes], [0, 1])


def test_zonotope2interval():
    z = Zonotope.rand(2, 5)
    bound_boxes = boundary(z, 0.1, Geometry.TYPE.INTERVAL)
    plot([z, *bound_boxes], [0, 1])


def test_zonotope2polytope():
    z = Zonotope.rand(2, 5)
    bound_boxes = boundary(z, 0.1, Geometry.TYPE.POLYTOPE)
    plot([z, *bound_boxes], [0, 1])


def test_zonotope2zonotope():
    z = Zonotope.rand(2, 5)
    bound_boxes = boundary(z, 0.01, Geometry.TYPE.ZONOTOPE)
    plot([z, *bound_boxes], [0, 1])
