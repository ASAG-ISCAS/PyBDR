from sympy import *
import sys

sys.path.append("../")
import numpy as np


def get_param():
    # the weight, bias and activation function list

    w1 = [[0.2911133, 0.12008807],
          [-0.24582624, 0.23181419],
          [-0.25797904, 0.21687193],
          [-0.19282854, -0.2602416],
          [0.26780415, -0.20697702],
          [0.23462369, 0.2294843],
          [-0.2583547, 0.21444395],
          [-0.04514714, 0.29514763],
          [-0.15318371, -0.275755],
          [0.24873598, 0.21018365]]

    b1 = [0.0038677, -0.00026365, -0.007168970, 0.02469357, 0.01338706,
          0.00856025, -0.00888401, 0.00516089, -0.00634514, -0.01914518]

    w2 = [[-0.58693904, -0.814841, -0.8175157, 0.97060364, 0.6908913,
           -0.92446184, -0.79249185, -1.1507587, 1.2072723, -0.7983982],
          [1.1564877, -0.8991244, -1.0774536, -0.6731967, 1.0154784,
           0.8984464, -1.0766245, -0.238209, -0.5233613, 0.8886671]]

    b2 = [-0.04129209, -0.01508532]

    act1 = "sigmoid"

    weight = [w1, w2]
    bias = [b1, b2]
    func_list = [act1]

    return weight, bias, func_list


def sigmoid(x):
    return 1 / (1 + exp(-x))


def purelin(x):
    return x


def neural_ode_spiral1(x, u):
    weight, bias, func_list = get_param()

    layer_num = len(bias)
    dim = np.shape(weight[0])[1]

    x = Matrix(x)
    dxdt = [None] * dim

    for i in range(layer_num):
        if i < layer_num - 1:
            w = Matrix(weight[i])
            b = Matrix(bias[i])
            act_func = func_list[i]
            if act_func == "tanh":
                vecF = np.vectorize(tanh)
            elif act_func == "sigmoid":
                vecF = np.vectorize(sigmoid)
            elif act_func == "purelin":
                vecF = np.vectorize(purelin)
            else:
                raise NotImplementedError
            if i == 0:
                dxdt = vecF(w * x + b)
            else:
                dxdt = vecF(w * dxdt + b)

        else:
            w = Matrix(weight[i])
            b = Matrix(bias[i])
            dxdt = w * dxdt + b
    return dxdt
