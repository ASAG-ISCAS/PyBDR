from __future__ import annotations
import codac


def extract_boundary(init_interval, init_set, eps: float = 0.04,
                     simplify_result: bool = True, simplify_eps: float = 0.005):
    from pybdr.geometry import Geometry, Interval, Polytope, Zonotope
    from pybdr.geometry.operation.convert import cvt2

    if isinstance(init_set, codac.AnalyticFunction):
        return extract_boundary_analytic_function(init_interval, init_set, eps, simplify_result, simplify_eps)
    elif isinstance(init_set, Polytope) or isinstance(init_set, Zonotope):
        if isinstance(init_set, Zonotope):
            init_set = cvt2(init_set, Geometry.TYPE.POLYTOPE)
        return extract_boundary_polytope(init_interval, init_set, eps)
    else:
        raise NotImplementedError()


def extract_boundary_polytope(init_interval, init_set, eps: float = 0.04):
    from pybdr.geometry import Geometry, Polytope, Interval, Zonotope
    from pybdr.geometry.operation.convert import cvt2
    assert isinstance(init_set, Polytope)
    n = init_set.a.shape[1]
    m = init_set.a.shape[0]

    x = codac.VectorVar(n)

    funcs = [sum(init_set.a[i, j] * x[j] for j in range(n)) - init_set.b[i] for i in range(m)]
    g = codac.AnalyticFunction([x], funcs)

    Y = codac.IntervalVector([[float("-inf"), 0.0]] * m)
    ctc = codac.CtcInverse(g, Y)

    boxes = []
    pave_boxes(init_interval, ctc, eps, boxes)

    boundary_boxes = []
    for box in boxes:
        if box.is_empty():
            continue
        vals = g.eval(box)
        for i in range(m):
            if vals[i].lb() <= 0.0 <= vals[i].ub():
                boundary_boxes.append(box)
                break

    x_init = []
    for i in boundary_boxes:
        if not i.is_empty():
            i_interval = Interval(i.lb(), i.ub())
            # x_init.append(i_interval)
            x_init.append(cvt2(i_interval, Geometry.TYPE.ZONOTOPE))
    return x_init


def extract_boundary_analytic_function(init_interval, init_set, eps: float = 0.04,
                                       simplify_result: bool = True, simplify_eps: float = 0.005):
    from pybdr.geometry import Geometry, Interval
    from pybdr.geometry.operation.convert import cvt2

    ctc = codac.CtcInverse(init_set, [0.0])
    # ctc = codac.CtcInverse(init_set_func, codac.IntervalVector([codac.Interval(0, 1e9)]))

    init_sub_interval_list = []
    for i in range(init_interval.shape[0]):
        init_sub_interval_list.append(codac.Interval(init_interval[i].inf[0], init_interval[i].sup[0]))
    init_domain = codac.IntervalVector(init_sub_interval_list)
    result = []
    pave_boxes(init_domain, ctc, eps, result)

    if simplify_result:
        isolated_boxes = extract_isolated_boxes(result)

        for i in isolated_boxes:
            boxes_i = []
            i_copy = codac.IntervalVector(i)
            pave_boxes(i_copy, ctc, simplify_eps, boxes_i)

            result.remove(i)
            result += boxes_i

    x_init = []
    for i in result:
        if not i.is_empty():
            i_interval = Interval(i.lb(), i.ub())
            # x_init.append(i_interval)
            x_init.append(cvt2(i_interval, Geometry.TYPE.ZONOTOPE))
    return x_init


def pave_boxes(box, ctc: codac.CtcInverse, eps, result):
    ctc.contract(box)
    if box.is_empty():
        return

    if box.max_diam() <= eps:
        result.append(box)
        return

    i = box.max_diam_index()
    left, right = box.bisect(i)
    pave_boxes(left, ctc, eps, result)
    pave_boxes(right, ctc, eps, result)


def boxes_touch(b1: codac.IntervalVector, b2: codac.IntervalVector, tol=1e-15):
    n = b1.size()
    for i in range(n):
        if b1[i].ub() < b2[i].lb() - tol or b2[i].ub() < b1[i].lb() - tol:
            return False
    return True


def extract_isolated_boxes(boxes, tol=1e-12):
    isolated = []
    for i, b in enumerate(boxes):
        connected = False
        for j, b2 in enumerate(boxes):
            if i == j:
                continue
            if boxes_touch(b, b2, tol):
                connected = True
                break
        if not connected:
            isolated.append(b)
    return isolated


