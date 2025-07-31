import numpy as np
from scipy.linalg import polar
import opt_einsum as oe
from copy import copy

def polar_decomposition(M):
    W, S, V = np.linalg.svd(M, full_matrices=False)
    U = W @ V
    P = V.T @ np.diag(S) @ V
    return U, P # A_c = U * P

# exit()

def right_normalize(A):

    a, d , b = A.shape
    A_mat = A.reshape(a * d, b)
    U, P = polar(A_mat, side = 'right' )
    B = P #.reshape(b, b)
    A_c = U.reshape(a, d, b)
    return B, A_c

# chi = 4
# d = 2
# T = np.random.randn(d, chi, chi)
# right_normalize(T)
# print(T)
# exit()

def left_normalize(A):

    a, d, b = A.shape
    A_mat = A.reshape(a, d * b)
    U, P = polar(A_mat, side = 'left')
    A = P
    A_c = U.reshape(a, d, b)
    return A, A_c


chi_1 = 4
chi_2 = 4
d = 2
T = np.random.randn(chi_1, d, chi_2)
# A, A_c1 = left_normalize(T)
# B, A_c2 = right_normalize(T)
# print(A.shape)
# print(B.shape)
# print(A_c1.shape)
# print(A_c2.shape)
# exit()

def central_gauge(T):

    #for all A such that left_normalize(A) @ right_normalize (A) = Identity
    #left_normalize(A) @ right_normalize (A) = np.identity(A.shape)
    A_c = copy(T)
    error = np.inf
    # for i in range(50):
    while error > 1e-6:
        A, A_c = left_normalize(A_c)
        B, A_c = right_normalize(A_c)

        error_1 = np.linalg.norm(np.eye(A_c.shape[2]) - oe.contract('aib,aic->bc', A_c, A_c.conj()))
        error_2 = np.linalg.norm(np.eye(A_c.shape[0]) - oe.contract('aib,cib->ac', A_c, A_c.conj()))
        print(error_1, error_2)
        error = max(error_1, error_2)

    # A @ A_c @ B @ A @ A_c @ B
    # np.identity(A.shape) = oe.contract('a a, a i b, b b, a a, a i b, b b  -> a b',  A, A_c, B, A, A_c, B)

    return A, A_c, B

A, A_c, B = central_gauge(T)