import numpy as np
from pybdr.geometry import Interval
from pybdr.util.functional import performance_counter_start, performance_counter
from scipy.io import savemat


def generate_data(sz, I_inf, I_sup, I_epsilon_inf, I_epsilon_sup):
    if type(sz) == int:
        sz = [sz]
    inf = np.random.rand(*sz) * (I_sup - I_inf) + I_inf
    sup = np.random.rand(*sz) * (I_sup - I_inf) + I_inf
    inf, sup = np.minimum(inf, sup), np.maximum(inf, sup)
    delta = np.random.rand(*sz) * (I_epsilon_sup - I_epsilon_inf) + I_epsilon_inf
    return Interval(inf, sup + delta)


def addition_eval():
    runs = 10 ** 4

    I_lower_bound = -10 ** 2
    I_upper_bound = 10 ** 2
    delta_I_lower_bound = 0
    delta_I_upper_bound = 10 ** 2

    I = generate_data(runs, I_lower_bound, I_upper_bound, delta_I_lower_bound, delta_I_upper_bound)

    time_cur = performance_counter_start()

    res = I + I

    performance_counter(time_cur, 'addition ' + str(runs) + ' Runs AVG.', runs)

    return {'I_inf': I.inf, 'I_sup': I.sup, 'res_inf': res.inf, 'res_sup': res.sup}


def subtraction_eval():
    runs = 10 ** 4

    I_lower_bound = -10 ** 2
    I_upper_bound = 10 ** 2
    delta_I_lower_bound = 0
    delta_I_upper_bound = 10 ** 2

    I = generate_data(runs, I_lower_bound, I_upper_bound, delta_I_lower_bound, delta_I_upper_bound)

    time_cur = performance_counter_start()

    res = I - I

    performance_counter(time_cur, 'subtraction ' + str(runs) + ' Runs AVG.', runs)

    return {'I_inf': I.inf, 'I_sup': I.sup, 'res_inf': res.inf, 'res_sup': res.sup}


def multiplication_eval():
    runs = 10 ** 4

    I_lower_bound = -10 ** 2
    I_upper_bound = 10 ** 2
    delta_I_lower_bound = 0
    delta_I_upper_bound = 10 ** 2

    I = generate_data(runs, I_lower_bound, I_upper_bound, delta_I_lower_bound, delta_I_upper_bound)

    time_cur = performance_counter_start()

    res = I * I

    performance_counter(time_cur, 'multiplication ' + str(runs) + ' Runs AVG.', runs)

    return {'I_inf': I.inf, 'I_sup': I.sup, 'res_inf': res.inf, 'res_sup': res.sup}


def division_eval():
    runs = 10 ** 4

    I_lower_bound = -10 ** 2
    I_upper_bound = 10 ** 2
    delta_I_lower_bound = 0
    delta_I_upper_bound = 10 ** 2

    I = generate_data(runs, I_lower_bound, I_upper_bound, delta_I_lower_bound, delta_I_upper_bound)

    time_cur = performance_counter_start()

    res = I / I

    performance_counter(time_cur, 'division ' + str(runs) + ' Runs AVG.', runs)

    return {'I_inf': I.inf, 'I_sup': I.sup, 'res_inf': res.inf, 'res_sup': res.sup}


def power_eval():
    runs = 10 ** 4

    I_lower_bound = -10 ** 2
    I_upper_bound = 10 ** 2
    delta_I_lower_bound = 0
    delta_I_upper_bound = 10 ** 2

    I = generate_data(runs, I_lower_bound, I_upper_bound, delta_I_lower_bound, delta_I_upper_bound)
    NUM = 3

    time_cur = performance_counter_start()

    res = I ** NUM

    performance_counter(time_cur, 'power ' + str(runs) + ' Runs AVG.', runs)

    return {'I_inf': I.inf, 'I_sup': I.sup, 'NUM': NUM, 'res_inf': res.inf, 'res_sup': res.sup}


def absolute_eval():
    runs = 10 ** 4

    I_lower_bound = -10 ** 2
    I_upper_bound = 10 ** 2
    delta_I_lower_bound = 0
    delta_I_upper_bound = 10 ** 2

    I = generate_data(runs, I_lower_bound, I_upper_bound, delta_I_lower_bound, delta_I_upper_bound)

    time_cur = performance_counter_start()

    res = abs(I)

    performance_counter(time_cur, 'power ' + str(runs) + ' Runs AVG.', runs)

    return {'I_inf': I.inf, 'I_sup': I.sup, 'res_inf': res.inf, 'res_sup': res.sup}


def left_matrix_multiplication_eval():
    runs = 10 ** 4

    I_lower_bound = -10 ** 2
    I_upper_bound = 10 ** 2
    delta_I_lower_bound = 0
    delta_I_upper_bound = 10 ** 2

    I = generate_data((runs, 5, 7), I_lower_bound, I_upper_bound, delta_I_lower_bound, delta_I_upper_bound)
    m = np.random.rand(7, 10)

    time_cur = performance_counter_start()

    res = I @ m

    performance_counter(time_cur, 'left_matrix_multiplication ' + str(runs) + ' Runs AVG.', runs)

    return {'I_inf': I.inf, 'I_sup': I.sup, 'm': m, 'res_inf': res.inf, 'res_sup': res.sup}


def right_matrix_multiplication_eval():
    runs = 10 ** 4

    I_lower_bound = -10 ** 2
    I_upper_bound = 10 ** 2
    delta_I_lower_bound = 0
    delta_I_upper_bound = 10 ** 2

    I = generate_data((runs, 7, 10), I_lower_bound, I_upper_bound, delta_I_lower_bound, delta_I_upper_bound)
    m = np.random.rand(5, 7)

    time_cur = performance_counter_start()

    res = m @ I

    performance_counter(time_cur, 'right_matrix_multiplication ' + str(runs) + ' Runs AVG.', runs)

    return {'I_inf': I.inf, 'I_sup': I.sup, 'm': m, 'res_inf': res.inf, 'res_sup': res.sup}


def exponential_eval():
    runs = 10 ** 4

    I_lower_bound = -10 ** 2
    I_upper_bound = 10 ** 2
    delta_I_lower_bound = 0
    delta_I_upper_bound = 10 ** 2

    I = generate_data(runs, I_lower_bound, I_upper_bound, delta_I_lower_bound, delta_I_upper_bound)

    time_cur = performance_counter_start()

    res = Interval.exp(I)

    performance_counter(time_cur, 'exponential ' + str(runs) + ' Runs AVG.', runs)

    return {'I_inf': I.inf, 'I_sup': I.sup, 'res_inf': res.inf, 'res_sup': res.sup}


def log_eval():
    runs = 10 ** 4

    I_lower_bound = 0
    I_upper_bound = 10 ** 2
    delta_I_lower_bound = 0
    delta_I_upper_bound = 10 ** 2

    I = generate_data(runs, I_lower_bound, I_upper_bound, delta_I_lower_bound, delta_I_upper_bound)

    time_cur = performance_counter_start()

    res = Interval.log(I)

    performance_counter(time_cur, 'log ' + str(runs) + ' Runs AVG.', runs)

    return {'I_inf': I.inf, 'I_sup': I.sup, 'res_inf': res.inf, 'res_sup': res.sup}


def sqrt_eval():
    runs = 10 ** 4

    I_lower_bound = 0
    I_upper_bound = 10 ** 2
    delta_I_lower_bound = 0
    delta_I_upper_bound = 10 ** 2

    I = generate_data(runs, I_lower_bound, I_upper_bound, delta_I_lower_bound, delta_I_upper_bound)

    time_cur = performance_counter_start()

    res = Interval.sqrt(I)

    performance_counter(time_cur, 'sqrt ' + str(runs) + ' Runs AVG.', runs)

    return {'I_inf': I.inf, 'I_sup': I.sup, 'res_inf': res.inf, 'res_sup': res.sup}


def arcsin_eval():
    runs = 10 ** 4

    I_lower_bound = -1
    I_upper_bound = 0
    delta_I_lower_bound = 0
    delta_I_upper_bound = 1

    I = generate_data(runs, I_lower_bound, I_upper_bound, delta_I_lower_bound, delta_I_upper_bound)

    time_cur = performance_counter_start()

    res = Interval.arcsin(I)

    performance_counter(time_cur, 'arcsin ' + str(runs) + ' Runs AVG.', runs)

    return {'I_inf': I.inf, 'I_sup': I.sup, 'res_inf': res.inf, 'res_sup': res.sup}


def arccos_eval():
    runs = 10 ** 4

    I_lower_bound = -1
    I_upper_bound = 0
    delta_I_lower_bound = 0
    delta_I_upper_bound = 1

    I = generate_data(runs, I_lower_bound, I_upper_bound, delta_I_lower_bound, delta_I_upper_bound)

    time_cur = performance_counter_start()

    res = Interval.arccos(I)

    performance_counter(time_cur, 'arccos ' + str(runs) + ' Runs AVG.', runs)

    return {'I_inf': I.inf, 'I_sup': I.sup, 'res_inf': res.inf, 'res_sup': res.sup}


def arctan_eval():
    runs = 10 ** 4

    I_lower_bound = -10 ** 2
    I_upper_bound = 10 ** 2
    delta_I_lower_bound = 0
    delta_I_upper_bound = 10 ** 2

    I = generate_data(runs, I_lower_bound, I_upper_bound, delta_I_lower_bound, delta_I_upper_bound)

    time_cur = performance_counter_start()

    res = Interval.arctan(I)

    performance_counter(time_cur, 'arctan ' + str(runs) + ' Runs AVG.', runs)

    return {'I_inf': I.inf, 'I_sup': I.sup, 'res_inf': res.inf, 'res_sup': res.sup}


def sinh_eval():
    runs = 10 ** 4

    I_lower_bound = -10 ** 2
    I_upper_bound = 10 ** 2
    delta_I_lower_bound = 0
    delta_I_upper_bound = 10 ** 2

    I = generate_data(runs, I_lower_bound, I_upper_bound, delta_I_lower_bound, delta_I_upper_bound)

    time_cur = performance_counter_start()

    res = Interval.sinh(I)

    performance_counter(time_cur, 'sinh ' + str(runs) + ' Runs AVG.', runs)

    return {'I_inf': I.inf, 'I_sup': I.sup, 'res_inf': res.inf, 'res_sup': res.sup}


def cosh_eval():
    runs = 10 ** 4

    I_lower_bound = -10 ** 2
    I_upper_bound = 10 ** 2
    delta_I_lower_bound = 0
    delta_I_upper_bound = 10 ** 2

    I = generate_data(runs, I_lower_bound, I_upper_bound, delta_I_lower_bound, delta_I_upper_bound)

    time_cur = performance_counter_start()

    res = Interval.cosh(I)

    performance_counter(time_cur, 'cosh ' + str(runs) + ' Runs AVG.', runs)

    return {'I_inf': I.inf, 'I_sup': I.sup, 'res_inf': res.inf, 'res_sup': res.sup}


def tanh_eval():
    runs = 10 ** 4

    I_lower_bound = -1
    I_upper_bound = 1
    delta_I_lower_bound = 0
    delta_I_upper_bound = 1

    I = generate_data(runs, I_lower_bound, I_upper_bound, delta_I_lower_bound, delta_I_upper_bound)

    time_cur = performance_counter_start()

    res = Interval.tanh(I)

    performance_counter(time_cur, 'tanh ' + str(runs) + ' Runs AVG.', runs)

    return {'I_inf': I.inf, 'I_sup': I.sup, 'res_inf': res.inf, 'res_sup': res.sup}


def arcsinh_eval():
    runs = 10 ** 4

    I_lower_bound = -10 ** 2
    I_upper_bound = 10 ** 2
    delta_I_lower_bound = 0
    delta_I_upper_bound = 10 ** 2

    I = generate_data(runs, I_lower_bound, I_upper_bound, delta_I_lower_bound, delta_I_upper_bound)

    time_cur = performance_counter_start()

    res = Interval.arcsinh(I)

    performance_counter(time_cur, 'arcsinh ' + str(runs) + ' Runs AVG.', runs)

    return {'I_inf': I.inf, 'I_sup': I.sup, 'res_inf': res.inf, 'res_sup': res.sup}


def arccosh_eval():
    runs = 10 ** 4

    I_lower_bound = 1
    I_upper_bound = 10 ** 1
    delta_I_lower_bound = 0
    delta_I_upper_bound = 10 ** 1

    I = generate_data(runs, I_lower_bound, I_upper_bound, delta_I_lower_bound, delta_I_upper_bound)

    time_cur = performance_counter_start()

    res = Interval.arccosh(I)

    performance_counter(time_cur, 'arccosh ' + str(runs) + ' Runs AVG.', runs)

    return {'I_inf': I.inf, 'I_sup': I.sup, 'res_inf': res.inf, 'res_sup': res.sup}


def arctanh_eval():
    runs = 10 ** 4

    I_lower_bound = -1
    I_upper_bound = 0
    delta_I_lower_bound = 0
    delta_I_upper_bound = 1

    I = generate_data(runs, I_lower_bound, I_upper_bound, delta_I_lower_bound, delta_I_upper_bound)

    time_cur = performance_counter_start()

    res = Interval.arctanh(I)

    performance_counter(time_cur, 'arctanh ' + str(runs) + ' Runs AVG.', runs)

    return {'I_inf': I.inf, 'I_sup': I.sup, 'res_inf': res.inf, 'res_sup': res.sup}


def sin_eval():
    runs = 10 ** 4

    I_lower_bound = -10 ** 2
    I_upper_bound = 10 ** 2
    delta_I_lower_bound = 0
    delta_I_upper_bound = 10 ** 2

    I = generate_data(runs, I_lower_bound, I_upper_bound, delta_I_lower_bound, delta_I_upper_bound)

    time_cur = performance_counter_start()

    res = Interval.sin(I)

    performance_counter(time_cur, 'sin ' + str(runs) + ' Runs AVG.', runs)

    return {'I_inf': I.inf, 'I_sup': I.sup, 'res_inf': res.inf, 'res_sup': res.sup}


def cos_eval():
    runs = 10 ** 4

    I_lower_bound = -10 ** 2
    I_upper_bound = 10 ** 2
    delta_I_lower_bound = 0
    delta_I_upper_bound = 10 ** 2

    I = generate_data(runs, I_lower_bound, I_upper_bound, delta_I_lower_bound, delta_I_upper_bound)

    time_cur = performance_counter_start()

    res = Interval.cos(I)

    performance_counter(time_cur, 'cos ' + str(runs) + ' Runs AVG.', runs)

    return {'I_inf': I.inf, 'I_sup': I.sup, 'res_inf': res.inf, 'res_sup': res.sup}


def tan_eval():
    runs = 10 ** 4

    I_lower_bound = -0.5 * np.pi + 10 ** -2
    I_upper_bound = 0
    delta_I_lower_bound = 0
    delta_I_upper_bound = 0.5 * np.pi - 10 ** -2

    I = generate_data(runs, I_lower_bound, I_upper_bound, delta_I_lower_bound, delta_I_upper_bound)

    time_cur = performance_counter_start()

    res = Interval.tan(I)

    performance_counter(time_cur, 'tan ' + str(runs) + ' Runs AVG.', runs)

    return {'I_inf': I.inf, 'I_sup': I.sup, 'res_inf': res.inf, 'res_sup': res.sup}


def cot_eval():
    runs = 10 ** 4

    I_lower_bound = 0 + 10 ** -2
    I_upper_bound = 0.5 * np.pi
    delta_I_lower_bound = 0
    delta_I_upper_bound = 0.5 * np.pi - 10 ** -2

    I = generate_data(runs, I_lower_bound, I_upper_bound, delta_I_lower_bound, delta_I_upper_bound)

    time_cur = performance_counter_start()

    res = Interval.cot(I)

    performance_counter(time_cur, 'cot ' + str(runs) + ' Runs AVG.', runs)

    return {'I_inf': I.inf, 'I_sup': I.sup, 'res_inf': res.inf, 'res_sup': res.sup}


if __name__ == "__main__":
    data_dict = {'addition_data': addition_eval(),
                 'subtraction_data': subtraction_eval(),
                 'multiplication_data': multiplication_eval(),
                 'division_data': division_eval(),
                 'power_data': power_eval(),
                 'absolute_data': absolute_eval(),
                 'left_matrix_multiplication_data': left_matrix_multiplication_eval(),
                 'right_matrix_multiplication_data': right_matrix_multiplication_eval(),
                 'exponential_data': exponential_eval(),
                 'log_data': log_eval(),
                 'sqrt_data': sqrt_eval(),
                 'sin_data': sin_eval(),
                 'cos_data': cos_eval(),
                 'tan_data': tan_eval(),
                 'cot_data': cot_eval(),
                 'arcsin_data': arcsin_eval(),
                 'arccos_data': arccos_eval(),
                 'arctan_data': arctan_eval(),
                 'sinh_data': sinh_eval(),
                 'cosh_data': cosh_eval(),
                 'tanh_data': tanh_eval(),
                 'arcsinh_data': arcsinh_eval(),
                 'arccosh_data': arccosh_eval(),
                 'arctanh_data': arctanh_eval()}

    savemat('data.mat', data_dict)
