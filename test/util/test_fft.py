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
        return np.sqrt(2) * 0.5 * np.cos(t) + np.sqrt(2) * 0.5 * np.sin(t)

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


def test_fftr_1d_fg():
    def f(t):
        return np.sqrt(2) * np.cos(t)

    def g(t):
        return 3 * np.sin(t - 5)

    def fg(t):
        return f(t) * g(t)

    f_sample = 1000
    t = np.linspace(0, 2 * np.pi, f_sample + 2, endpoint=False)
    fa0, fa, fb = fourier_coefficients(f(t))
    ga0, ga, gb = fourier_coefficients(g(t))
    fga0, fga, fgb = fourier_coefficients(fg(t))

    N = 2300
    rx0 = fftr(fa0 * ga0, fa * ga, fb * gb, 2 * np.pi, 100)
    t0 = np.linspace(0, 2 * np.pi, rx0.shape[0])
    plt.plot(t0, rx0)
    t0 = np.linspace(0, 2 * np.pi, 1000)
    plt.plot(t0, fg(t0))
    plt.show()


def test_conv_therom():
    import numpy as np
    import matplotlib.pyplot as p

    def Convolution(array, kernel):
        return np.real(np.fft.ifft(np.fft.fft(array) * np.fft.fft(kernel)))

    a_flat = [1.0, 2.0, 3.0, 0.0, 0.0, 4.0, 5.0, 6.0, 0.0, 0.0, 7.0, 8.0, 9.0]
    k_flat = [1, 2, 1, 0, 0, 0, 0, 0, 0, 0, -1, -2, -1]

    a_flat = np.pad(a_flat, (25, 25), "constant", constant_values=(0, 0)).tolist()
    k_flat = np.pad(k_flat, (25, 25), "constant", constant_values=(0, 0)).tolist()

    my_convolution = Convolution(a_flat, k_flat)
    np_convolution = np.convolve(a_flat, k_flat)

    fig, ax = p.subplots(3, figsize=(12, 5))

    ax[0].plot(a_flat)
    ax[1].plot(k_flat)
    ax[2].plot(np.roll(my_convolution, 30), ".-", lw=0.5, label="myconv")
    # arbitrary shift here
    ax[2].plot(np.roll(np_convolution, 0), ".-", lw=3, alpha=0.3, label="npconv")
    p.legend()
    p.show()


def test_fftr_2d():
    SN = 1000
    lb, ub = [-2, 3], [-5, 7]
    # x
    x0 = np.ones(SN - 1) * ub[0]
    x1 = np.linspace(ub[0], lb[0], 2 * SN, endpoint=False)
    x2 = np.ones(2 * SN - 1) * lb[0]
    x3 = np.linspace(lb[0], ub[0], 2 * SN, endpoint=False)
    x4 = np.ones(SN) * ub[0]
    x = np.concatenate([x0, x1, x2, x3, x4])

    # y
    c = (lb[1] + ub[1]) * 0.5
    y0 = np.linspace(c, ub[1], SN - 1, endpoint=False)
    y1 = np.ones(2 * SN) * ub[1]
    y2 = np.linspace(ub[1], lb[1], 2 * SN - 1, endpoint=False)
    y3 = np.ones(2 * SN) * lb[1]
    y4 = np.linspace(lb[1], c, SN, endpoint=False)
    y = np.concatenate([y0, y1, y2, y3, y4])

    # reconstruct
    a00, a0, b0 = fourier_coefficients(x)
    a01, a1, b1 = fourier_coefficients(y)

    RN = 15

    rx = fftr(a00, a0[:RN], b0[:RN], 2 * np.pi, x.shape[0])
    ry = fftr(a01, a1[:RN], b1[:RN], 2 * np.pi, y.shape[0])

    # plt.plot(x, y)
    # plt.plot(rx, ry)
    plt.plot(np.arange(x.shape[0]), x)
    plt.plot(np.arange(y.shape[0]), y)
    plt.show()
