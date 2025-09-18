import numpy as np
import random
from itertools import product
import opt_einsum as oe

chi = 4
d = 2

modules = [0, 1, 2, 3]
num_allowed_pairs = 6
num_vertical_allowed_pairs = 6

pairs = [c for c in product(modules, repeat = 2)]
# allowed_pairs = random.sample(pairs, num_allowed_pairs)
# allowed_pairs = pairs
# vertical_allowed_pairs = random.sample(pairs, num_vertical_allowed_pairs)
# vertical_allowed_pairs = pairs

pairs_1 = [c for c in product(modules, repeat = 2)]
allowed_pairs_1: list[tuple[int, ...]] = random.sample(pairs_1, num_allowed_pairs)
vertical_allowed_pairs_1 = random.sample(pairs_1, num_vertical_allowed_pairs)

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

def get_allowed_nonuples(allowed_pairs, vertical_allowed_pairs, modules):
    
    allowed_nonuple = set()
    allowed_set = set(allowed_pairs)
    vertical_set = set(vertical_allowed_pairs)     

    for (A, B), (D, E), (G, H) in product(allowed_set, repeat = 3):
        for (C, F, I) in product(modules, repeat = 3):

            horizontal_pairs = [(A, B), (B, C), (D, E), (E, F), (G, H), (H, I)]
            if any(pair not in allowed_set for pair in horizontal_pairs):
                continue

            verticle_pairs = [(A, D), (D, G), (B, E), (E, H), (C, F), (F, I)]
            if any(pair not in vertical_set for pair in verticle_pairs):
                continue

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
    if (A, B, C, E) not in allowed_quadruples: # Q
        return False
    if (C, F, I, E) not in allowed_quadruples: # R
        return False
    if (I, H, G, E) not in allowed_quadruples: # S
        return False
    return True

def dict_diff(d1, d2):  #function to make difference of two tensors with module index (dictionary)
    if d1.keys() != d2.keys():
        return None
    diff = {}
    for k1, v1 in d1.items():
        diff[k1] = v1 - d2[k1]
    return diff

def dict_norm(d): #function to make norm for tensors with module index/dictionary
    norm = 0
    for v in d.values():
        norm += np.linalg.norm(v)
    return norm

allowed_nonuples = get_allowed_nonuples(allowed_pairs_1, vertical_allowed_pairs_1, modules)
allowed_quadruples = get_allowed_quadruples(allowed_pairs_1, vertical_allowed_pairs_1)
consistent_nonuples = [
        nonuples
        for nonuples in allowed_nonuples
        if is_nonaple_consistent_with_quadruples(allowed_quadruples, *nonuples)
    ]


X = {}
for (A, B, E, D) in allowed_quadruples:
    X[A, B, E, D] = (np.random.randn(chi, chi, chi, chi) + 1j*np.random.randn(chi, chi, chi, chi)) / np.sqrt(2)


Y = {}
for (B, C, F, E) in allowed_quadruples:
    Y[B, C, F, E] = (np.random.randn(chi, chi, chi, chi) + 1j*np.random.randn(chi, chi, chi, chi)) / np.sqrt(2)

Z = {}
for (E, F, I , H) in allowed_quadruples:
    Z[E, F, I , H] = (np.random.randn(chi, chi, chi, chi) + 1j*np.random.randn(chi, chi, chi, chi)) / np.sqrt(2)

W = {} 
for (D, E, H, G) in allowed_quadruples:
    W[D, E, H, G] = (np.random.randn(chi, chi, chi, chi) + 1j*np.random.randn(chi, chi, chi, chi)) / np.sqrt(2)

P = {}
for (A, D, G, E) in allowed_quadruples:
    P[A, D, G, E] = (np.random.randn(chi, chi, chi, chi) + 1j*np.random.randn(chi, chi, chi, chi)) / np.sqrt(2)

Q = {}
for (A, B, E, C) in allowed_quadruples:
    Q[A, B, E, C] = (np.random.randn(chi, chi, chi, chi) + 1j*np.random.randn(chi, chi, chi, chi)) / np.sqrt(2)

R = {}
for (C, E, F, I) in allowed_quadruples:
    R[C, E, F, I] = (np.random.randn(chi, chi, chi, chi) + 1j*np.random.randn(chi, chi, chi, chi)) / np.sqrt(2)

S = {}
for (G, E, H, I) in allowed_quadruples:
    S[G, E, H, I] = (np.random.randn(chi, chi, chi, chi) + 1j*np.random.randn(chi, chi, chi, chi)) / np.sqrt(2)

U_tensor = {}
V_tensor = {}

for (A, B, C, D, E, F, G, H, I) in allowed_nonuples:

    key = (A, B, C, D, F, G, H, I)
    #print(key)
    shape = (chi,) * 8 #eight indices
    U_tensor[key] = np.zeros(shape, dtype=np.complex128)
    V_tensor[key] = np.zeros(shape, dtype=np.complex128)
    

    U_tensor[key] += oe.contract('i l j k, m j n o, o r p q, k t r s -> i l m n p q t s', X[A, B, D, E] , Y[B, C, E, F], Z[E, F, H, I], W[D, E, G, H])

    V_tensor[key] += oe.contract('i u m v, n v p w, q w s x , x t u l -> i m n p q s t l', P[D, E, G, H], Q[A, B, D, E], R[B, C, E, F], S[E, F, H, I])


def loss_function():
        if consistent_nonuples:
            return dict_diff(U_tensor, V_tensor) # x^4 - y^4

def loss_norm():
    return dict_norm(loss_function())

def main():
    print(loss_norm())

if __name__ == "__main__":
    main()