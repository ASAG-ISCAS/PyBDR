import matplotlib.pyplot as plt
import numpy as np


def fourier_coefficients(s):
    """Calculates the first 2*N+1 Fourier series coeff. of a periodic function.

    Given a periodic, function f(t) with period T, this function returns the
    coefficients a0, {a1,a2,...},{b1,b2,...} such that:

    f(t) ~= a0/2+ sum_{k=1}^{N} ( a_k*cos(2*pi*k*t/T) + b_k*sin(2*pi*k*t/T) )

    If return_complex is set to True, it returns instead the coefficients
    {c0,c1,c2,...}
    such that:

    f(t) ~= sum_{k=-N}^{N} c_k * exp(i*2*pi*k*t/T)

    where we define c_{-n} = complex_conjugate(c_{n})

    Refer to wikipedia for the relation between the real-valued and complex
    valued coeffs at http://en.wikipedia.org/wiki/Fourier_series.

    Parameters
    ----------
    f : the periodic function, a callable like f(t)
    T : the period of the function f, so that f(0)==f(T)
    N_max : the function will return the first N_max + 1 Fourier coeff.

    Returns
    -------
    if return_complex == False, the function returns:

    a0 : float
    a,b : numpy float arrays describing respectively the cosine and sine coeff.

    if return_complex == True, the function returns:

    c : numpy 1-dimensional complex-valued array of size N+1

    """
    # From Shanon theoreom we must use a sampling freq. larger than the maximum
    # frequency you want to catch in the signal.
    # f_sample = 2 * N
    # we also need to use an integer sampling frequency, or the
    # points will not be equispaced between 0 and 1. We then add +2 to f_sample
    # t = np.linspace(0, T, f_sample + 2, endpoint=False)
    y = np.fft.rfft(s) / s.size
    y *= 2
    return y[0].real, y[1:-1].real, -y[1:-1].imag


def fftr(a0, a, b, T, N):
    i = np.arange(1, a.shape[0] + 1)
    omega = 2 * np.pi * i / (2 * np.pi)
    t = np.linspace(0, T, 2 * N + 2, endpoint=False)

    m = np.outer(omega, t)
    am = a[:, None] * np.cos(m)
    bm = b[:, None] * np.sin(m)
    return a0 * 0.5 + np.sum(am, axis=0) + np.sum(bm, axis=0)


def test_fftr_1d():
    def f(t):
        return np.cos(t) * np.sin(3 * t) ** 3

    f_sample = 100
    t = np.linspace(0, 2 * np.pi, f_sample + 2, endpoint=False)
    a0, a, b = fourier_coefficients(f(t))

    N = 2300
    rx = fftr(a0, a[:N], b[:N], 2 * np.pi, 100)

    t0 = np.linspace(0, 2 * np.pi, 1000)
    plt.plot(t0, f(t0))
    t1 = np.linspace(0, 2 * np.pi, rx.shape[0])
    plt.plot(t1, rx)
    plt.show()


def test_fftr_2d():

    # sampling along the boundary of box in 2d
    N = 100
    lb, ub = [-2, 3], [5, 7]
    # sampling along line 0
    x0 = np.ones(N) * lb[0]
    y0 = np.linspace(ub[0], ub[1], N, endpoint=True)
    # sampling along line 1
    x1 = np.ones(N) * lb[1]
    y1 = np.linspace(ub[0], ub[1], N, endpoint=True)
    # sampling along line 2
    x2 = np.linspace(lb[0], lb[1], N, endpoint=True)
    y2 = np.ones(N) * ub[0]
    # sampling along line 3
    x3 = np.linspace(lb[0], lb[1], N, endpoint=True)
    y3 = np.ones(N) * ub[1]

    x = np.concatenate([x0, x2, x1, x3])
    y = np.concatenate([y0, y2, y1, y3])

    # reconstruct
    a00, a0, b0 = fourier_coefficients(x)
    a01, a1, b1 = fourier_coefficients(y)

    N = 20

    rx = fftr(a00, a0[:N], b0[:N], 2 * np.pi, 100)
    ry = fftr(a01, a1[:N], b1[:N], 2 * np.pi, 100)

    plt.plot(x, y)
    plt.plot(rx, ry)
    plt.show()
