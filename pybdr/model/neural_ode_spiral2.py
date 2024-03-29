from sympy import *
import sys

sys.path.append("../")
import numpy as np


def get_param():
    # the weight, bias and activation function list

    w1 = [[-0.32294768, 0.59955627],
          [0.47014388, -0.39748120],
          [-0.56326932, 0.33752987],
          [0.45147443, 0.31528524],
          [0.41403031, -0.47271276],
          [-0.12952870, -0.62095606],
          [-0.41343114, -0.45678866],
          [-0.33266136, 0.29245856],
          [0.50114638, 0.39612201],
          [0.47665390, 0.55137879]]

    b1 = [0.0038923009, 0.0037905588, 0.0017197595, -0.0033185149, 0.0024190384, -0.0013056855, 0.0011365928,
          -0.00042518601, -0.0025141449, 0.0010660964]

    w2 = [[-0.50525320, 0.34800902, -0.34015974, -0.40054744, 0.39193857, 0.59363592, 0.56743664, -0.33811751,
           -0.36945280, -0.46805024],
          [-0.41715327, 0.56257814, -0.56921810, 0.60423535, 0.53992182, -0.14412111, -0.45906776, -0.35295558,
           0.49238238, 0.43526673]]

    b2 = [-0.0013696412, 0.00060380378]

    act1 = "sigmoid"
    # act1 = "purelin"

    weight = [w1, w2]
    bias = [b1, b2]
    func_list = [act1]

    return weight, bias, func_list


def sigmoid(x):
    return 1 / (1 + exp(-x))


def purelin(x):
    return x


def neural_ode_spiral2(x, u):
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
