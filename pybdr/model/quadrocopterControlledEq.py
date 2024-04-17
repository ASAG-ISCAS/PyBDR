from sympy import *

"""
NOTE: R. Beard, "Quadrotor Dynamics and Control", Tech Report Bringham Young University
"""


def quadrocopterControlledEq(x, u):
    dxdt = [None] * 12
    g = 9.81
    R = 0.1
    l = 0.5
    M_rotor = 0.1
    M = 1
    m = M + 4*M_rotor

    J_x = 2*M*R**2/5 + 2*l**2*M_rotor
    J_y = J_x
    J_z = 2*M*R**2/5 + 4*l**2*M_rotor

    F = m*g - 10*(x[2] - u[0]) + 3*x[5]
    tau_phi = -(x[6] - u[1]) - x[9]

    tau_theta = -(x[7] - u[2]) - x[10]

    tau_psi = 0

    dxdt[0] = cos(x[7])*cos(x[8])*x[3] + (sin(x[6])*sin(x[7])*cos(x[8]) - cos(x[6])*sin(x[8]))*x[4] + (cos(x[6])*sin(x[7])*cos(x[8]) + sin(x[6])*sin(x[8]))*x[5]
    dxdt[1] = cos(x[7])*sin(x[8])*x[3] + (sin(x[6])*sin(x[7])*sin(x[8]) + cos(x[6])*cos(x[8]))*x[4] + (cos(x[6])*sin(x[7])*sin(x[8]) - sin(x[6])*cos(x[8]))*x[5]
    dxdt[2] = sin(x[7])*x[3] - sin(x[6])*cos(x[7])*x[4] - cos(x[6])*cos(x[7])*x[5]

    dxdt[3] = x[11]*x[4] - x[10]*x[5] - g*sin(x[7])
    dxdt[4] = x[9]*x[5] - x[11]*x[3] + g*cos(x[7])*sin(x[6])
    dxdt[5] = x[10]*x[3] - x[9]*x[4] + g*cos(x[7])*cos(x[6]) - F/m

    dxdt[6] = x[9] + (sin(x[6])*tan(x[7]))*x[10] + (cos(x[6])*tan(x[7]))*x[11]
    dxdt[7] = cos(x[6])*x[10] - sin(x[6])*x[11]
    dxdt[8] = sin(x[6])/cos(x[7])*x[10] + cos(x[6])/cos(x[7])*x[11]

    dxdt[9] = (J_y - J_z)/J_x*x[10]*x[11] + 1/J_x*tau_phi
    dxdt[10] = (J_z - J_x)/J_y*x[9]*x[11] + 1/J_y*tau_theta
    dxdt[11] = (J_x - J_y)/J_z*x[9]*x[10] + 1/J_z*tau_psi

    return Matrix(dxdt)
