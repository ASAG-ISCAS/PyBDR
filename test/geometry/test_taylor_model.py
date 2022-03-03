from pyrat.geometry import TaylorModel


def test_basic():
    t = TaylorModel()
    t.demo_func()
    print(t._index)
