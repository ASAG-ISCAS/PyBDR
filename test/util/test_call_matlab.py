import matlab.engine

eng = matlab.engine.start_matlab()
eng.addpath("/home/jqlab/Downloads/CORA", nargout=0)


def test_case0():
    t = eng.gcd(100.0, 80.0, nargout=3)
    print(t)


def test_case1():
    import numpy as np

    y = np.array([1, 2, 3])
    print(y)
    print(y.tolist())
    x = matlab.double(y.tolist())
    print(x)
    print(x.size)
    a = np.array([[-1, -1, -3], [2, 3, -1]])
    a = np.random.rand(2, 3)
    a = np.sort(a, axis=0)
    b = np.array([[0.8, 0.12, 0.15], [0.9, 0.9, 0.2]])
    a = matlab.double(a.tolist())
    b = matlab.double(b.tolist())
    inf, sup = eng.test_func(a, nargout=2)
    print(inf)
    print(sup)
