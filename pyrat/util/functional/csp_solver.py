import numpy as np


class CSPSolver:
    MAX_BOXES = 10000

    @classmethod
    def __split(cls, box, epsilon):
        d = box.sup - box.inf
        dim = np.where(d > epsilon)[0]
        if dim.shape[0] > 0:
            return dim[0]
        return None

    @classmethod
    def solve(cls, constraint, lb: np.ndarray, ub: np.ndarray, epsilon: float):
        """

        :param constraint: function used to check if box is valid or not
        :param lb:
        :param ub:
        :param epsilon:
        :return:
        """
        assert epsilon >= 0
        from pyrat.geometry import Interval

        boxes = []

        active_boxes = {Interval(lb, ub)}
        while len(boxes) < CSPSolver.MAX_BOXES and len(active_boxes) > 0:
            box = active_boxes.pop()
            if constraint(box):
                dim = cls.__split(box, epsilon)
                if dim is None:
                    boxes.append(box)
                else:
                    sub_boxes = box.split(dim)
                    for sub_box in sub_boxes:
                        active_boxes.add(sub_box)

        return boxes
