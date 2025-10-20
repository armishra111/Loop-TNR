import numpy as np
import random
from itertools import product
import opt_einsum as oe

def get_allowed_quadruples(allowed_pairs, vertical_allowed_pairs):
    vertical = set(vertical_allowed_pairs)
    if not vertical or not allowed_pairs:
        return set()

    allowed_quadruples = set()
    for (A, B) in allowed_pairs:
        for (D, E) in allowed_pairs:
            if (A, D) in vertical and (B, E) in vertical:
                allowed_quadruples.add((A, B, D, E))

    return allowed_quadruples

# def get_allowed_nonuples(allowed_pairs, vertical_allowed_pairs, modules):
    
#     allowed_nonuple = set()
#     allowed_set = set(allowed_pairs)
#     vertical_set = set(vertical_allowed_pairs)     

#     for (A, B), (D, E), (G, H) in product(allowed_set, repeat = 3):
#         for (C, F, I) in product(modules, repeat = 3):

#             horizontal_pairs = [(A, B), (B, C), (D, E), (E, F), (G, H), (H, I)]
#             if any(pair not in allowed_set for pair in horizontal_pairs):
#                 continue

#             verticle_pairs = [(A, D), (D, G), (B, E), (E, H), (C, F), (F, I)]
#             if any(pair not in vertical_set for pair in verticle_pairs):
#                 continue

#             allowed_nonuple.add((A, B, C, D, E, F, G, H, I))

#     return allowed_nonuple

def get_allowed_nonuples(allowed_pairs, vertical_allowed_pairs, modules):
    
    allowed_nonuple = set()
    allowed_set = set(allowed_pairs)
    vertical_set = set(vertical_allowed_pairs)     

    for (A, B), (D, E), (G, H) in product(allowed_set, repeat = 3):
        if (A, D) not in vertical_set or (D, G) not in vertical_set:
            continue
        if (B, E) not in vertical_set or (E, H) not in vertical_set:
            continue

        Cs = [C for C in modules if (B, C) in allowed_set]
        for C in Cs:
            Fs = [F for F in modules if (E, F) in allowed_set and (C, F) in vertical_set]
            for F in Fs:
                Is = [I for I in modules if (H, I) in allowed_set and (F, I) in vertical_set]
                for I in Is:
                    allowed_nonuple.add((A, B, C, D, E, F, G, H, I))
    return allowed_nonuple


def is_nonaple_consistent_with_quadruples(allowed_quadruples, A, B, C, D, E, F, G, H, I):
    if (A, B, D, E) not in allowed_quadruples: # X
        return False
    if (B, C, E, F) not in allowed_quadruples: # Y
        return False
    if (D, E, G, H) not in allowed_quadruples: # W
        return False
    if (E, F, H, I) not in allowed_quadruples: # Z
        return False
    if (A, D, E, G) not in allowed_quadruples: # P
        return False
    if (B, A, C, E) not in allowed_quadruples: # Q
        return False
    if (C, E, F, I) not in allowed_quadruples: # R
        return False
    if (E, G, I, H) not in allowed_quadruples: # S
        return False
    return True

def dict_diff(d1, d2):  #function to make difference of two tensors with module index (dictionary)
    if d1.keys() != d2.keys():
        return None
    diff = {}
    for k1, v1 in d1.items():
        diff[k1] = v1 - d2[k1]
    return diff

def dict_squared_norm(d): #function to find 1/2* L_2 norm for tensors with module index/dictionary
    squared_norm = 0
    for v in d.values():
        squared_norm += np.linalg.norm(v)**2
    return 0.5 * squared_norm

def dict_norm(d): #function to find 1/2* L_2 norm for tensors with module index/dictionary
    norm = 0
    for v in d.values():
        norm += np.linalg.norm(v)
    return norm


def loss_function():
    if consistent_nonuples:
        return dict_diff(U_tensor, V_tensor) # x^4 - y^4
    else:
        raise ValueError('No consistent module index matching')
    
def loss_function_norm():
    if consistent_nonuples:
        return dict_squared_norm(loss_function())
    else:
        raise ValueError('No consistent module index matching')

def gradient_loss_function():
    if consistent_nonuples:
        return dict_diff(K_tensor, L_tensor) # This has module index = B C A E and tensor index = i m u v
    else:
        raise ValueError('No consistent module index matching')

def gradient_loss_norm():
    if consistent_nonuples:
        return dict_norm(gradient_loss_function())
    else:
        raise ValueError('No consistent module index matching')

def build_loss_tensors(Q_dict):
    if not consistent_nonuples:
        raise ValueError("No consistent module index matching")

    U_tensor = {}
    V_tensor = {}
    shape_1 = (chi,) * 8 #eight indices

    for (A, B, C, D, E, F, G, H, I) in consistent_nonuples:

        key_1 = (A, B, C, D, F, G, H, I)
        #print(key)
        U_tensor[key_1] = np.zeros(shape_1, dtype=np.complex128)
        V_tensor[key_1] = np.zeros(shape_1, dtype=np.complex128)
        

        U_tensor[key_1] += oe.contract('i l u x, m n u v, p q v w, s t x w -> i l m n q p s t',
                                    X[A, B, D, E] , Y[B, C, E, F], Z[E, F, H, I], W[D, E, G, H]
                                    )

        V_tensor[key_1] += oe.contract('l t u x, i m u v, n p v w , s q x w -> i l m n q p s t',
                                    P[A, D, E, G], Q[B, A, C, E], R[C, E, F, I], S[E, G, I, H]
                                    ) # indices matches ipad

    return U_tensor, V_tensor


def build_grad_tensors(Q_dict):
    if not consistent_nonuples:
        raise ValueError("No consistent module index matching")

    K_tensor = {} # This capures the tensor of the form x^7
    L_tensor = {} # This captures the tensor of the form x^3*y^4
    shape_2 = (chi,) * 4 #four indices

    for (A, B, C, D, E, F, G, H, I) in consistent_nonuples:

        key_2 = (B, A, C, E)
        #print(key)
        K_tensor[key_2] = np.zeros(shape_2, dtype=np.complex128)
        L_tensor[key_2] = np.zeros(shape_2, dtype=np.complex128)

        K_tensor[key_2] += oe.contract('l t x u, n p v w, s q w x, l t a b, i m b c, n p c d, s q d a -> i m u v',# consistent with ipad diagram; the tensor matches module and tensor indices of Q
                                    P[A, D, E, G], R[C, E, F, I], S[E, G, I, H],
                                    P[A, D, E, G].conj(), Q_1[B, A, C, E].conj(), R[C, E, F, I].conj(), S[E, G, I, H].conj(),
                                    )

        L_tensor[key_2] += oe.contract('l t x u, n p v w, s q w x, i l x u, m n u v, p q v w, s t w x -> i m u v', # consistent with ipad diagram
                                    P[A, D, E, G], R[C, E, F, I], S[E, G, I, H],
                                    X[A, B, D, E].conj(), Y[B, C, E, F].conj(), Z[E, F, H, I].conj(), W[D, E, G, H].conj()
                                    )
    return K_tensor, L_tensor


def loss_and_grad(Q_override=None):
    Q_dict = Q if Q_override is None else Q_override

    U_tensor, V_tensor = build_loss_tensors(Q_dict)
    loss_terms = dict_diff(U_tensor, V_tensor)
    if loss_terms is None:
        raise ValueError("Loss tensors have incompatible keys")
    loss = 0.5 * dict_norm(loss_terms) ** 2

    K_tensor, L_tensor = build_grad_tensors(Q_dict)
    grad_terms = dict_diff(K_tensor, L_tensor)
    if grad_terms is None:
        raise ValueError("Gradient tensors have incompatible keys")

    return loss, grad_terms

chi = 4
d = 2

modules = [0, 1, 2, 3, 4, 5]
num_allowed_pairs = 15
num_vertical_allowed_pairs = 15

pairs = [c for c in product(modules, repeat = 2)]
# allowed_pairs = random.sample(pairs, num_allowed_pairs)
# allowed_pairs = pairs
# vertical_allowed_pairs = random.sample(pairs, num_vertical_allowed_pairs)
# vertical_allowed_pairs = pairs

pairs_1 = [c for c in product(modules, repeat = 2)]
allowed_pairs_1: list[tuple[int, ...]] = random.sample(pairs_1, num_allowed_pairs)
vertical_allowed_pairs_1 = random.sample(pairs_1, num_vertical_allowed_pairs)

allowed_nonuples = get_allowed_nonuples(allowed_pairs_1, vertical_allowed_pairs_1, modules)
allowed_quadruples = get_allowed_quadruples(allowed_pairs_1, vertical_allowed_pairs_1)
consistent_nonuples = [
        nonuples
        for nonuples in allowed_nonuples
        if is_nonaple_consistent_with_quadruples(allowed_quadruples, *nonuples)
    ]


X = {}
for (A, B, D, E) in allowed_quadruples:
    X[A, B, D, E] = (np.random.randn(chi, chi, chi, chi) + 1j*np.random.randn(chi, chi, chi, chi)) / np.sqrt(2)

Y = {}
for (B, C, E, F) in allowed_quadruples:
    Y[B, C, E, F] = (np.random.randn(chi, chi, chi, chi) + 1j*np.random.randn(chi, chi, chi, chi)) / np.sqrt(2)

Z = {}
for (E, F, H , I) in allowed_quadruples:
    Z[E, F, H , I] = (np.random.randn(chi, chi, chi, chi) + 1j*np.random.randn(chi, chi, chi, chi)) / np.sqrt(2)

W = {} 
for (D, E, G, H) in allowed_quadruples:
    W[D, E, G, H] = (np.random.randn(chi, chi, chi, chi) + 1j*np.random.randn(chi, chi, chi, chi)) / np.sqrt(2)

P = {}
for (A, D, E, G) in allowed_quadruples:
    P[A, D, E, G] = (np.random.randn(chi, chi, chi, chi) + 1j*np.random.randn(chi, chi, chi, chi)) / np.sqrt(2)

Q = {}
for (B, A, C, E) in allowed_quadruples:
    Q[B, A, C, E] = (np.random.randn(chi, chi, chi, chi) + 1j*np.random.randn(chi, chi, chi, chi)) / np.sqrt(2)

Q_1 = {}
for (B, A, C, E) in allowed_quadruples:
    Q_1[B, A, C, E] = Q[B, A, C, E].copy()

R = {}
for (C, E, F, I) in allowed_quadruples:
    R[C, E, F, I] = (np.random.randn(chi, chi, chi, chi) + 1j*np.random.randn(chi, chi, chi, chi)) / np.sqrt(2)

S = {}
for (E, G, I, H) in allowed_quadruples:
    S[E, G, I, H] = (np.random.randn(chi, chi, chi, chi) + 1j*np.random.randn(chi, chi, chi, chi)) / np.sqrt(2)

U_tensor = {}
V_tensor = {}
shape_1 = (chi,) * 8 #eight indices

for (A, B, C, D, E, F, G, H, I) in consistent_nonuples:

    key_1 = (A, B, C, D, F, G, H, I)
    #print(key)
    U_tensor[key_1] = np.zeros(shape_1, dtype=np.complex128)
    V_tensor[key_1] = np.zeros(shape_1, dtype=np.complex128)
    

    U_tensor[key_1] += oe.contract('i l u x, m n u v, p q v w, s t x w -> i l m n q p s t',
                                X[A, B, D, E] , Y[B, C, E, F], Z[E, F, H, I], W[D, E, G, H]
                                )

    V_tensor[key_1] += oe.contract('l t u x, i m u v, n p v w , s q x w -> i l m n q p s t',
                                P[A, D, E, G], Q[B, A, C, E], R[C, E, F, I], S[E, G, I, H]
                                ) # indices matches ipad

K_tensor = {} # This capures the tensor of the form x^7
L_tensor = {} # This captures the tensor of the form x^3*y^4
shape_2 = (chi,) * 4 #four indices

for (A, B, C, D, E, F, G, H, I) in consistent_nonuples:

    key_2 = (B, A, C, E)
    #print(key)
    K_tensor[key_2] = np.zeros(shape_2, dtype=np.complex128)
    L_tensor[key_2] = np.zeros(shape_2, dtype=np.complex128)

    K_tensor[key_2] += oe.contract('l t x u, n p v w, s q w x, l t a b, i m b c, n p c d, s q d a -> i m u v',# consistent with ipad diagram; the tensor matches module and tensor indices of Q
                                P[A, D, E, G], R[C, E, F, I], S[E, G, I, H],
                                P[A, D, E, G].conj(), Q_1[B, A, C, E].conj(), R[C, E, F, I].conj(), S[E, G, I, H].conj(),
                                )

    L_tensor[key_2] += oe.contract('l t x u, n p v w, s q w x, i l x u, m n u v, p q v w, s t w x -> i m u v', # consistent with ipad diagram
                                P[A, D, E, G], R[C, E, F, I], S[E, G, I, H],
                                X[A, B, D, E].conj(), Y[B, C, E, F].conj(), Z[E, F, H, I].conj(), W[D, E, G, H].conj()
                                )

def main():
        return print(f"loss function is {loss_function_norm()} and gradient_norm is {gradient_loss_norm()}")

if __name__ == "__main__":
    main()