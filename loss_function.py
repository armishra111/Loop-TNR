import numpy as np
import random
from itertools import product
import opt_einsum as oe

chi = 4
d = 2

modules = [0, 1, 2, 3]
num_allowed_pairs = 5
num_vertical_allowed_pairs = 5

pairs = [c for c in product(modules, repeat = 2)]
# allowed_pairs = random.sample(pairs, num_allowed_pairs)
# allowed_pairs = pairs
# vertical_allowed_pairs = random.sample(pairs, num_vertical_allowed_pairs)
# vertical_allowed_pairs = pairs

pairs_1 = [c for c in product(modules, repeat = 2)]
allowed_pairs_1: list[tuple[int, ...]] = random.sample(pairs_1, num_allowed_pairs)
vertical_allowed_pairs_1 = random.sample(pairs_1, num_vertical_allowed_pairs)

def get_allowed_quadruples(allowed_pairs, vertical_allowed_pairs):
    
    allowed_quadruples = set()

    for (A, B), (D, E) in product(allowed_pairs, repeat = 2):

        if (A, D) in vertical_allowed_pairs and (B, E) in vertical_allowed_pairs:

            allowed_quadruples.add((A, B, D, E))

    return allowed_quadruples


def get_allowed_nonuples(allowed_pairs, vertical_allowed_pairs, modules):
    
    allowed_nonuple = set()

    for (A, B), (D, E), (G, H) in product(allowed_pairs, repeat=3):
        for C, F, I in product(modules, repeat = 3):

            horizontal_pairs = [(A, B), (B, C), (D, E), (E, F), (G, H), (H, I)]
            if any(pair not in allowed_pairs for pair in horizontal_pairs):
                continue

            verticle_pairs = [(A, D), (D, G), (B, E), (E, H), (C, F), (F, I)]
            if any(pair not in vertical_allowed_pairs for pair in verticle_pairs):
                continue

            allowed_nonuple.add((A, B, C, D, E, F, G, H, I))

    return allowed_nonuple


def is_nonaple_consistent_with_quadruples(allowed_quadruples, A, B, C, D, E, F, G, H, I):
    if (A, B, D, E) not in allowed_quadruples:
        return False
    if (B, C, E, F) not in allowed_quadruples:
        return False
    if (D, E, G, H) not in allowed_quadruples:
        return False
    if (E, F, H, I) not in allowed_quadruples:
        return False
    return True

def dict_diff(d1, d2):
    if d1.keys() != d2.keys():
        return None
    diff = {}
    for k1, v1 in d1.items():
        diff[k1] = v1 - d2[k1]
    return diff

def dict_norm(d):
    norm = 0
    for v in d.values():
        norm += np.linalg.norm(v)
    return norm

allowed_nonuples = get_allowed_nonuples(allowed_pairs_1, vertical_allowed_pairs_1, modules)
allowed_quadruples = get_allowed_quadruples(allowed_pairs_1, vertical_allowed_pairs_1)

X = {}
for (A, B, E, D) in allowed_quadruples:
    X[A, B, E, D] = np.random.randn(chi, chi, chi, chi)

Y = {}
for (B, C, F, E) in allowed_quadruples:
    Y[B, C, F, E] = np.random.randn(chi, chi, chi, chi)

Z = {}
for (E, F, I , H) in allowed_quadruples:
    Z[E, F, I , H] = np.random.randn(chi, chi, chi, chi)

W = {} 
for (D, E, H, G) in allowed_quadruples:
    W[D, E, H, G] = np.random.randn(chi, chi, chi, chi)

P = {}
for (A, D, G, E) in allowed_quadruples:
    P[A, D, G, E] = np.random.randn(chi, chi, chi, chi)

Q = {}
for (A, B, E, C) in allowed_quadruples:
    Q[A, B, E, C] = np.random.randn(chi, chi, chi, chi)

R = {}
for (C, E, F, I) in allowed_quadruples:
    R[C, E, F, I] = np.random.randn(chi, chi, chi, chi)

S = {}
for (G, E, H, I) in allowed_quadruples:
    S[G, E, H, I] = np.random.randn(chi, chi, chi, chi)

# U_tensor = {}
# for (A, B, C, D, E, F, G, H, I) in allowed_nonaples:

#     key = (A, B, C, D, F, G, H, I)
#     #print(key)
#     U_tensor[key] = np.zeros((chi,)*8)

#     U_tensor[key] += oe.contract('i l j k, m j n o, o r p q, k t r s -> i l m n p q t s',  X[A, B, D, E] , Y[B, C, E, F], Z[E, F, H, I], W[D, E, G, H])

# V_tensor = {}
# for (A, B, C, D, E, F, G, H, I) in allowed_nonaples:
#     key = (A, B, C, D, F, G, H, I)
#     # print(is_nonaple_consistent_with_quadruples(allowed_quadruples, A, B, C, D, E, F, G, H, I))
#     V_tensor[key] = np.zeros((chi,)*8)
    
#     V_tensor[A, B, C, D, F, G, H, I] += oe.contract('i u m v, n v p w, q w s x , x t u l -> i m n p q s t l', P[D, E, G, H], Q[A, B, D, E], R[B, C, E, F], S[E, F, H, I])


U_tensor = {}
V_tensor = {}
for (A, B, C, D, E, F, G, H, I) in allowed_nonuples:

    key = (A, B, C, D, F, G, H, I)
    #print(key)
    U_tensor[key] = np.zeros((chi,)*8)
    V_tensor[key] = np.zeros((chi,)*8)

    U_tensor[key] += oe.contract('i l j k, m j n o, o r p q, k t r s -> i l m n p q t s',  X[A, B, D, E] , Y[B, C, E, F], Z[E, F, H, I], W[D, E, G, H])

    V_tensor[key] += oe.contract('i u m v, n v p w, q w s x , x t u l -> i m n p q s t l', P[D, E, G, H], Q[A, B, D, E], R[B, C, E, F], S[E, F, H, I])

diff_tensor = dict_diff(U_tensor, V_tensor)
norm = dict_norm(diff_tensor)

print(norm)