from inspect import signature

import numpy as np
from sympy import *
import mpmath as mm


def test_case_0():
    x = symbols("x:3", seq=True)
    eq = x[2] ** 2
    print(x)
    print(hessian(eq, x))
    print(eq.jacobian(x))
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
    print(f(xv, uv))
    print(signature(f))
    jac = system.jacobian(x)
    print(jac.rows, jac.cols)
    for i in range(jac.rows):
        for j in range(jac.cols):
            print(jac[i, j])
        print("--------------------------")


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
        system = []
        expr0 = -sqrt(2) * sqrt(_g) * _k0 * sqrt(_x[0]) + _k1 * (4 - _x[5]) + _u + 0.1
        expr1 = sqrt(2) * sqrt(g) * k0 * (sqrt(_x[0]) - sqrt(_x[1]))
        system.append([expr0, expr1])

        return Matrix(system)

    x, (u, g, k0, k1) = symbols(("x:6", "u,g,k0,k1"))
    s = f(x, u, g, k0, k1)
    sv = lambdify([x, u, g, k0, k1], s, "numpy")
    siv = lambdify([x, u, g, k0, k1], s, "mpmath")

    x = mm.iv.matrix(6, 1)
    x[0] = mm.iv.mpf([-1, 1])
    x[1] = mm.iv.mpf(2)
    print(x)
    u, g, k0, k1 = mm.iv.mpf(1), mm.iv.mpf(2), mm.iv.mpf(3), mm.iv.mpf(4)
    v1 = siv(x, u, g, k0, k1)
    print(v1)

    x = np.ones(6)
    u, g, k0, k1 = 1, 2, 3, 4
    print(sv(x, u, g, k0, k1))
