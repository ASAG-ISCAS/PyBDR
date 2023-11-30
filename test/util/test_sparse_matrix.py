from scipy.sparse import csc_matrix, csr_matrix

from pybdr.util.functional.auxiliary import *


def test_conditional_getting():
    data = np.arange(7) + 11
    row = np.arange(data.size)
    col = row * 2 + 1
    m = coo_matrix(
        (data, (row, col)), shape=(data.size * 2, data.size * 2), dtype=float
    ).tocsc()
    temp_mask = m < 15
    m[temp_mask] *= -1
    print(m[0])
    print(m.min(axis=0))
    m.eliminate_zeros()
    abs_m = abs(m)
    abs_m[temp_mask] = -2
    abs_m[:, 0] = -5

    # abs_m.eliminate_zeros()
    print(type(abs_m))
    print(abs_m)

    print(m.toarray())


def test_conditional_setting():
    a = [1, 2, 3, 0]
    c = [3, -1, 4, 9]
    b = [[1, 3, 4], [2, 5, 7]]
    temp0 = csc_matrix(a)
    temp = csc_matrix(c)
    kkk = temp.minimum(temp0)
    temp1 = csr_matrix(a)
    temp2 = csc_matrix(b)
    temp3 = csr_matrix(b)
    print(temp0.power(2))
    print(temp0)
    print(temp1)
