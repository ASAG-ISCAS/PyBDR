"""
Soundness of the set operations: for random points x in the input sets, the result of the
operation applied to x must lie in the output set.
"""

import numpy as np
import pytest

from pybdr.geometry import Geometry, Interval, Zonotope
from pybdr.geometry.operation import cvt2

RNG = np.random.default_rng(0)
TOL = 1e-9


def sample_interval(x: Interval, num: int = 500) -> np.ndarray:
    u = RNG.uniform(size=(num, *x.inf.shape))
    # include the bounds, where most errors show up
    u[0], u[1] = 0, 1
    return x.inf + (x.sup - x.inf) * u


def sample_zonotope(z: Zonotope, num: int = 500) -> np.ndarray:
    u = RNG.uniform(-1, 1, size=(num, z.gen_num))
    u[: min(num, 2 ** min(z.gen_num, 8))] = np.sign(u[: min(num, 2 ** min(z.gen_num, 8))])  # vertices
    return z.c + u @ z.gen.T


def assert_in_interval(pts: np.ndarray, x: Interval):
    assert np.all(pts >= x.inf - TOL) and np.all(pts <= x.sup + TOL)


def assert_in_zonotope(pts: np.ndarray, z: Zonotope):
    p = cvt2(z, Geometry.TYPE.POLYTOPE)
    assert np.all(pts @ p.a.T <= p.b + 1e-7)


def rand_interval(lo, hi, shape=(4,)):
    a, b = RNG.uniform(lo, hi, size=shape), RNG.uniform(lo, hi, size=shape)
    return Interval(np.minimum(a, b), np.maximum(a, b))


# ================================================== interval arithmetic

BINARY = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
}


@pytest.mark.parametrize("op", BINARY)
def test_interval_binary(op):
    x, y = rand_interval(-3, 3), rand_interval(-3, 3)
    px, py = sample_interval(x), sample_interval(y)
    assert_in_interval(BINARY[op](px, py), BINARY[op](x, y))


def test_interval_div():
    x, y = rand_interval(-3, 3), rand_interval(0.5, 3)
    assert_in_interval(sample_interval(x) / sample_interval(y), x / y)


@pytest.mark.parametrize("power", [2, 3, 4])
def test_interval_pow(power):
    x = rand_interval(-2, 2)
    assert_in_interval(sample_interval(x) ** power, x ** power)


UNARY = {
    # name: (numpy function, domain)
    "exp": (np.exp, (-3, 3)),
    "log": (np.log, (0.1, 5)),
    "sqrt": (np.sqrt, (0, 5)),
    "sin": (np.sin, (-10, 10)),
    "cos": (np.cos, (-10, 10)),
    "arcsin": (np.arcsin, (-1, 1)),
    "arccos": (np.arccos, (-1, 1)),
    "arctan": (np.arctan, (-5, 5)),
    "sinh": (np.sinh, (-3, 3)),
    "cosh": (np.cosh, (-3, 3)),
    "tanh": (np.tanh, (-3, 3)),
    "arcsinh": (np.arcsinh, (-5, 5)),
    "arccosh": (np.arccosh, (1, 5)),
    "arctanh": (np.arctanh, (-0.9, 0.9)),
    "sigmoid": (lambda v: 1 / (1 + np.exp(-v)), (-5, 5)),
}


@pytest.mark.parametrize("name", UNARY)
def test_interval_unary(name):
    f, (lo, hi) = UNARY[name]
    x = rand_interval(lo, hi, shape=(50,))
    assert_in_interval(f(sample_interval(x)), getattr(Interval, name)(x))


def test_interval_matmul():
    a = RNG.uniform(-2, 2, size=(3, 4))
    x = rand_interval(-1, 1)
    assert_in_interval(sample_interval(x) @ a.T, a @ x)
    assert_in_interval(sample_interval(x) @ a.T, x @ a.T)


def test_interval_matmul_interval():
    x = rand_interval(-1, 1, shape=(3, 4))
    y = rand_interval(-1, 1, shape=(4,))
    px, py = sample_interval(x), sample_interval(y)
    assert_in_interval(np.einsum("kij,kj->ki", px, py), x @ y)


def test_matmul_keeps_operands():
    a = RNG.uniform(-2, 2, size=(3, 4))
    x, y = rand_interval(-1, 1), rand_interval(-1, 1, shape=(4, 2))
    saved = a.copy(), x.inf.copy(), x.sup.copy(), y.inf.copy(), y.sup.copy()
    _ = a @ x, x @ a.T, x @ y
    for before, after in zip(saved, (a, x.inf, x.sup, y.inf, y.sup)):
        assert np.array_equal(before, after)


# ================================================== zonotope operations

def rand_zonotope(dim=2, gen_num=5):
    return Zonotope(RNG.uniform(-1, 1, dim), RNG.uniform(-1, 1, (dim, gen_num)))


def test_zonotope_minkowski_sum():
    z1, z2 = rand_zonotope(), rand_zonotope()
    assert_in_zonotope(sample_zonotope(z1) + sample_zonotope(z2), z1 + z2)


def test_zonotope_linear_map():
    z, a = rand_zonotope(), RNG.uniform(-2, 2, size=(2, 2))
    assert_in_zonotope(sample_zonotope(z) @ a.T, a @ z)


@pytest.mark.parametrize("method", [Zonotope.REDUCE_METHOD.GIRARD])
def test_zonotope_reduce(method):
    z = rand_zonotope(gen_num=20)
    assert_in_zonotope(sample_zonotope(z), z.reduce(method, 2))


@pytest.mark.parametrize("target", [Geometry.TYPE.INTERVAL, Geometry.TYPE.POLYTOPE])
def test_zonotope_conversion(target):
    z = rand_zonotope(dim=3, gen_num=6)
    converted = cvt2(z, target)
    pts = sample_zonotope(z)
    if target == Geometry.TYPE.INTERVAL:
        assert_in_interval(pts, converted)
    else:
        assert np.all(pts @ converted.a.T <= converted.b + 1e-7)


def test_interval_to_zonotope_is_exact():
    x = rand_interval(-1, 1, shape=(3,))
    z = cvt2(x, Geometry.TYPE.ZONOTOPE)
    assert_in_zonotope(sample_interval(x), z)
    back = cvt2(z, Geometry.TYPE.INTERVAL)
    assert np.allclose(back.inf, x.inf) and np.allclose(back.sup, x.sup)
