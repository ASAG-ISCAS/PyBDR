from sympy import *
import sys 
import numpy as np
import math 
# import model_description as md
import pyrat.neural_ode.model_description as md

def sigmoid(x):
    return (1 / (1 + exp(-x)))

def neuralODE (x, u):
    weight, bias, func_list = md.get_param()

    layer_num = np.shape(bias)[0]
    dim = np.shape(weight[0])[1]

    x = Matrix(x)
    dxdt = [None] * dim
    
    for i in range(layer_num):
        if (i < layer_num -1):
            w = Matrix(weight[i])
            b = Matrix(bias[i])
            act_func = func_list[i]
            if act_func == "tanh":
                vecF = np.vectorize(tanh)
            elif act_func == "sigmoid":
                vecF = np.vectorize(sigmoid)
            else:
                raise NotImplementedError
            if (i == 0):
                dxdt = vecF(w*x+b)
            else:
                dxdt = vecF(w*dxdt+b)
            
        else:
            w = Matrix(weight[i])
            b = Matrix(bias[i])
            dxdt = w*dxdt+b
    return dxdt