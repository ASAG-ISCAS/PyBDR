from sympy import *
import sys

sys.path.append("./../../")
import numpy as np
import math

# import model_description as md
import pybdr.model.neural_ode.spiral1 as md

def sigmoid(x):
    return 1 / (1 + exp(-x))


def purelin(x):
    return x


def neuralODE(x, u):
    weight, bias, func_list = md.get_param()

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
