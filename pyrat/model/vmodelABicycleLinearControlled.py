from sympy import *
from pyrat.deprecated.model_old import ModelOld


def _f(x, u):
    """
    NOTE: Bicycle Model with
        - normal force equilibrium for pitching-moments
        - linear tyre model
        state x=[x,y,psi,vx,vy,omega]
        input u=[delta,omega_f,omega_r]
    """

    # body
    m = 1750
    j = 2500
    ll = 2.7
    lf = 1.43
    lr = ll - lf
    h = 0.5

    # street
    mu0 = 1
    g = 9.81

    # tires
    cf = 10.4 * 1.3
    cr = 21.4 * 1.1

    # position
    x_pos = x[0]
    y_pos = x[1]
    psi = x[2]

    # velocity
    vx = x[3]
    vy = x[4]
    omega = x[5]

    # acceleration
    fb = x[6] * m
    delta = x[7]

    # control action
    """
    input are values of the state feedback matrix R, the reference state xn, and the
    feedforward value w
    """

    r = Matrix(
        [
            [u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7]],
            [u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]],
        ]
    )
    xn = Matrix(
        [[u[16]], [u[17]], [u[18]], [u[19]], [u[20]], [u[21]], [u[22]], [u[23]]]
    )
    w = Matrix([[u[24]], [u[25]]])
    v = -r * (x_pos - xn) + w

    # calculate normal forces
    fzf = (vy + lf * m * g - h * fb) / (lr + lf)
    fzr = m * g - fzf

    # side-slip
    sf = (vy + lf * omega) / vx - delta
    sr = (vy - lr * omega) / vx

    # forces
    fyf = -cf * fzf * sf
    fyr = -cr * fzr * sr

    # accelerations
    dvx = fb / m + vy * omega
    dvy = (fyf + fyr) / m - vx * omega
    d_omega = (lf * fyf - lr * fyr) / j

    # position related
    cp = cos(psi)
    sp = sin(psi)

    """
    dynamic
    """
    dxdt = [None] * 8

    # position
    dxdt[0] = cp * vx - sp * vy
    dxdt[1] = sp * vx + cp * vy
    dxdt[2] = omega
    # velocity
    dxdt[3] = dvx
    dxdt[4] = dvy
    dxdt[5] = d_omega
    # acceleration
    dxdt[6] = v[0]
    dxdt[7] = v[1]
    return Matrix(dxdt)


class VModelABicycleLinearControlled(ModelOld):
    vars = symbols(("x:8", "u:26"))
    f = _f(*vars)
    name = "vmodel_a_bicycle_linear_controlled"
    dim = f.rows
