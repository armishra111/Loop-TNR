import numpy as np
from scipy.linalg import polar
from scipy.linalg import svd
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
T = (np.random.randn(chi_1, d_1, d_2, chi_2) + 1j*np.random.randn(chi_1, d_1, d_2, chi_2)) / np.sqrt(2)

# X, Y = left_normalize(T)
# O, J = right_normalize(T)
# print(X.shape)
# print(Y.shape)
# print(O.shape)
# print(J.shape)
# exit()

def central_gauge(T):
    A, A_c, B = None, None, None
    A_c = copy(T)
    error = np.inf
    # iter = 0
    # for i in range(50):
    while error > 1e-10:
        A, A_c = left_normalize(A_c)
        B, A_c = right_normalize(A_c)

        error_1 = np.linalg.norm(np.eye(A_c.shape[3]) - oe.contract('aijb, aijc -> bc', A_c, A_c.conj()))
        error_2 = np.linalg.norm(np.eye(A_c.shape[0]) - oe.contract('aijb, cijb -> ac', A_c, A_c.conj()))
        #print(error_1, error_2)
        error = max(error_1, error_2)
        #iter += 1
        #print("Number of iteration:", iter)

    if A is None or B is None:
        raise ValueError("internal error: A or B was not assigned in compute_gauge")

    return A, A_c, B

A, A_c, B = central_gauge(T)

def svd_redefining(A, A_c, B, tol=1e-9):
    A_c_copy = A_c.copy()
    a, d1, d2, b = A_c_copy.shape
    A_c_mat = A_c_copy.reshape(a * d1 * d2, b)

    U, sigma, V = svd(A_c_mat, full_matrices=False)
    rank = sigma.shape[0]
    U_tensor = U.reshape(a, d1, d2, rank)

    A_updated = np.tensordot(A, U_tensor, axes=1)
    B_updated = np.tensordot(V, B, axes=1)

    A_clean = A_updated.copy()
    sigma_clean = sigma.copy()
    B_clean = B_updated.copy()

    A_clean[np.abs(A_clean) < tol] = 0.0
    sigma_clean[np.abs(sigma_clean) < tol] = 0.0
    B_clean[np.abs(B_clean) < tol] = 0.0

    return A_clean, sigma_clean, B_clean


A_clean, sigma_clean, B_clean = svd_redefining(A, A_c, B)
print(B_clean)