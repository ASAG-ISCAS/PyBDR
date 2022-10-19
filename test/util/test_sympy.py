import inspect
import numbers
from inspect import signature

import numpy
import numpy as np
from sympy import *


def test_case_0():
    x = symbols("x:3", seq=True)
    eq = Matrix([x[2] ** 2 + sin(x[1]), tan(x[0])])
    print(x)
    print(hessian(eq[1], x))
    temp = eq.jacobian(x)
    print(temp.rows, temp.cols)
    print(eq.jacobian(x))
    print(series(eq[1], x[0]).removeO())
    # f = lambdify(eq)


def test_case_1():
    # define a tank6Eq system
    x, dxdt, u = symbols(("x:6", "dxdt:6", "u:1"))
    expr0 = parse_expr("u0+0.1+k1*(4-x5)-k*sqrt(2*g)*sqrt(x0)")
    expr1 = parse_expr("k*sqrt(2*g)*(sqrt(x0)-sqrt(x1))")
    expr2 = parse_expr("k*sqrt(2*g)*(sqrt(x1)-sqrt(x2))")
    expr3 = parse_expr("k*sqrt(2*g)*(sqrt(x2)-sqrt(x3))")
    expr4 = parse_expr("k*sqrt(2*g)*(sqrt(x3)-sqrt(x4))")
    expr5 = parse_expr("k*sqrt(2*g)*(sqrt(x4)-sqrt(x5))")
    system = Matrix([expr0, expr1, expr2, expr3, expr4, expr5])
    f = lambdify((x, u), system, "numpy")
    init_printing()
    print(f)
    xv = np.ones(6)
    uv = np.ones(1) * 2
    print(signature(f))
    jac_x = system.jacobian(x)
    jac_u = system.jacobian(u)

    print(jac_x)
    print(jac_u)


def test_case_2():
    from sympy import Symbol, Matrix, Function, simplify

    eta = Symbol("eta")
    xi = Symbol("xi")
    sigma = Symbol("sig")

    x = Matrix([[xi], [eta]])

    h = [Function("h_" + str(i + 1))(x[0], x[1]) for i in range(3)]
    z = [Symbol("z_" + str(i + 1)) for i in range(3)]

    lamb = 0
    for i in range(3):
        lamb += 1 / (2 * sigma**2) * (z[i] - h[i]) ** 2
    simplify(lamb)
    print(lamb)


def test_case_3():
    def f(_x, _u, _g, _k0, _k1):
        expr0 = -sqrt(2) * sqrt(_g) * _k0 * sqrt(_x[0]) + _k1 * (4 - _x[5]) + _u + 0.1
        expr1 = sqrt(2) * sqrt(g) * k0 * (sqrt(_x[0]) - sqrt(_x[1]))

        return Matrix([expr0, expr1])

    x, (u, g, k0, k1) = symbols(("x:6", "u,g,k0,k1"))
    s = f(x, u, g, k0, k1)
    sv = lambdify([x, u, g, k0, k1], s, "numpy")
    siv = lambdify([x, u, g, k0, k1], s, "mpmath")

    # x = mm.iv.matrix(6, 1)
    # x[0] = mm.iv.mpf([-1, 1])
    # x[1] = mm.iv.mpf(2)
    # print(x)
    # u, g, k0, k1 = mm.iv.mpf(1), mm.iv.mpf(2), mm.iv.mpf(3), mm.iv.mpf(4)
    # v1 = siv(x, u, g, k0, k1)
    # print(v1)

    x = np.ones(6)
    u, g, k0, k1 = 1, 2, 3, 4
    print(sv(x, u, g, k0, k1))


def test_sympy_using_custom_interval_arithmetic():
    class Interval:
        def __init__(self, a, b):
            self._a = a
            self._b = b

        @classmethod
        def op_dict(cls):
            return {
                "__add__": cls.__add__,
                "__mul__": cls.__mul__,
                "__rmul__": cls.__rmul__,
            }

        def __add__(self, other):
            return Interval(min(self._a, other._a), max(self._b, other._b))

        def __mul__(self, other):
            if isinstance(other, Interval):
                a = min(
                    [
                        self._a * other._a,
                        self._a * other._b,
                        self._b * other._a,
                        self._b * other._b,
                    ]
                )
                b = max(
                    [
                        self._a * other._a,
                        self._a * other._b,
                        self._b * other._a,
                        self._b * other._b,
                    ]
                )
                return Interval(a, b)
            elif isinstance(other, numbers.Real):
                return Interval(self._a * other, self._b * other)
            else:
                raise NotImplementedError

        def __rmul__(self, other):
            return self * other

        def __str__(self):
            return "[ " + str(self._a) + ", " + str(self._b) + " ]"

        def __matmul__(self, other):
            raise NotImplementedError

        def __pow__(self, power, modulo=None):
            print("+++++++++++++++++++++")
            raise NotImplementedError

    x, y = symbols(("x", "y"))
    expr = x + y**2
    f = lambdify((x, y), expr, dict(inspect.getmembers(Interval)))

    temp = f(Interval(-1, 1), Interval(2, 3))
    # temp = Interval(-1, 1) * 2
    print(temp)


def test_sympy_derivative_tensor():
    from pyrat.model import Tank6Eq, RandModel, LaubLoomis, VanDerPol

    model = VanDerPol()
    j = np.asarray(derive_by_array(model.f, model.vars[0])).squeeze()
    print(j)
    h = np.asarray(derive_by_array(j, model.vars[0])).squeeze()
    tag = np.vectorize(lambda x: x.is_number and not x.is_zero)(j)
    print(j.shape)
    print(h.shape)
    print(tag)
    exit(False)
    tag = j.applyfunc(lambda x: x.is_number and not x.is_zero)
    tag = np.asarray(tag, dtype=bool).squeeze()
    for this_tag in tag:
        print(this_tag)
    for this_h in h:
        print(this_h)
    exit(False)
    x = np.ones(6, dtype=float) * 10
    u = np.ones(1, dtype=float) * 3
    fj = lambdify(model.vars, j, ["numpy", {"sqrt": numpy.sin}])
    fh = lambdify(model.vars, h, "numpy")
    vj = np.asarray(fj(x, u))
    vh = np.asarray(fh(x, u))
    print(vj.shape, vh.shape)
