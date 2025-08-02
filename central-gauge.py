import numpy as np
from scipy.linalg import polar
import opt_einsum as oe
from copy import copy

def right_normalize(A):
    a, d, d , b = A.shape
    A_mat = A.reshape(a * d * d, b)
    U, P = polar(A_mat, side = 'right' )
    B = P #.reshape(b, b)
    A_c = U.reshape(a, d, d, b)
    return B, A_c

def left_normalize(A):

    a, d, d, b = A.shape
    A_mat = A.reshape(a, d * d * b)
    U, P = polar(A_mat, side = 'left')
    A = P
    A_c = U.reshape(a, d, d, b)
    return A, A_c


chi_1 = 4
chi_2 = 4
d_1 = 2
d_2 = 2
T = np.random.randn(chi_1, d_1, d_2, chi_2)

# X, Y = left_normalize(T)
# O, J = right_normalize(T)
# print(X.shape)
# print(Y.shape)
# print(O.shape)
# print(J.shape)
# exit()

def central_gauge(T):

    A_c = copy(T)
    error = np.inf
    # iter = 0
    # for i in range(50):
    while error > 1e-10:
        A, A_c = left_normalize(A_c)
        B, A_c = right_normalize(A_c)

        error_1 = np.linalg.norm(np.eye(A_c.shape[3]) - oe.contract('aijb, aijc -> bc', A_c, A_c.conj()))
        error_2 = np.linalg.norm(np.eye(A_c.shape[0]) - oe.contract('aijb, cijb -> ac', A_c, A_c.conj()))
        # print(error_1, error_2)
        error = max(error_1, error_2)
        # iter += 1
    # print("Number of iteration:", iter)
    return A, A_c, B
    
A, A_c, B = central_gauge(T)
# print(A.shape)
# print(A_c.shape)
# print(B.shape)