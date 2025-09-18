import numpy as np
import random
from itertools import product
import opt_einsum as oe

# ==========================
# = We create derivative tensor of "lossfunction"
# === One takes derivative with respect to component of U_tensor ===
# === Hence you end up with a loss function derivative such that if original loss function is of form ===
# ==== norm(x,y) ~ x^8 - 2x^4*y^4 + y^8 then \partial norm / \partial x ~ 8x^7 - 8x^3*y^4 ===
# ==== The variation occurs wrt V_tensor (made of 'x') as that is what is being numerically being varied  ===
# ==== derivativelossfunction ~ 8(x^7 - x^3*y^4) ===
# ==========================

chi = 4
d = 2

modules = [0, 1, 2, 3, 4, 5]
num_allowed_pairs = 25
num_vertical_allowed_pairs = 25

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

# def get_allowed_decuples(allowed_pairs, vertical_allowed_pairs, modules):
    
#     allowed_decuple = set()

#     for (A, B), (C, D), (E, F), (G, H) in product(vertical_allowed_pairs, repeat = 4):
#         for (I, J) in product(modules, repeat = 2):

#             horizontal_pairs = [
#                 (I, A), (I, B), (I, C), (I, D), (I, E), (I, F), (I, G), (I, H), 
#                 (A, J), (B, J),(C, J), (D, J), (E, J), (F, J), (G, J), (H, J  )
#                 ]
#             if any(pair not in allowed_pairs for pair in horizontal_pairs):
#                 continue

#             verticle_pairs = [
#                 (A, B), (B, C), (C, D), (D, E),
#                 (E, F), (F, G), (G, H), (H, A)
#                 ]
#             if any(pair not in vertical_allowed_pairs for pair in verticle_pairs):
#                 continue

#             allowed_decuple.add((A, B, C, D, E, F, G, H, I, J))

#     return allowed_decuple

def get_allowed_decuples(allowed_pairs, vertical_allowed_pairs, modules):

    allowed = set(allowed_pairs)
    vertical = set(vertical_allowed_pairs)
    allowed_decuple = set()

    # all eight vertical checks force A = B = C ... = H = module
    # horizontal check reduces to " check if (I, module) and (module, J) in allowed_pairs"

    for module in modules:
        if (module, module) not in vertical:
            continue

        prefix = (module,) * 8  # the first eight indices must all equal module

        for I in modules:
            if (I, module) not in allowed:
                continue
            for J in modules:
                if (module, J) not in allowed:
                    continue
                allowed_decuple.add(prefix + (I, J))

    return allowed_decuple


def is_decuples_consistent_with_quadruples(allowed_quadruples, A, B, C, D, E, F, G, H, I, J) -> bool:
    if (H, A, B, I) not in allowed_quadruples: # P_1
        return False
    if (B, C, D, I) not in allowed_quadruples: # Q_1
        return False
    if (D, E, F, I) not in allowed_quadruples: # R_1
        return False
    if (F, G, H, I) not in allowed_quadruples: # S_1
        return False
    if (H, A, B, J) not in allowed_quadruples: # P_2
        return False
    if (D, E, F, J) not in allowed_quadruples: # Q_2
        return False
    if (F, G, H, J) not in allowed_quadruples: # R_2
        return False
    if (A, B, C, I) not in allowed_quadruples: # X
        return False
    if (C, D, E, I) not in allowed_quadruples: # Y
        return False
    if (E, F, G, I) not in allowed_quadruples: # Z
        return False
    if (G, H, A, I) not in allowed_quadruples: # W
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

allowed_decuples = get_allowed_decuples(allowed_pairs_1, vertical_allowed_pairs_1, modules)
allowed_quadruples = get_allowed_quadruples(allowed_pairs_1, vertical_allowed_pairs_1)
consistent_decuples = [
        decuple
        for decuple in allowed_decuples
        if is_decuples_consistent_with_quadruples(allowed_quadruples, *decuple)
    ]

P_1 = {}
for (H, A, B, I) in allowed_quadruples:
    P_1[H, A, B, I] = (np.random.randn(chi, chi, chi, chi) + 1j*np.random.randn(chi, chi, chi, chi)) / np.sqrt(2)

Q_1 = {}
for (B, C, D, I) in allowed_quadruples:
    Q_1[B, C, D, I] = (np.random.randn(chi, chi, chi, chi) + 1j*np.random.randn(chi, chi, chi, chi)) / np.sqrt(2)

R_1 = {}
for (D, E, F, I) in allowed_quadruples:
    R_1[D, E, F, I] = (np.random.randn(chi, chi, chi, chi) + 1j*np.random.randn(chi, chi, chi, chi)) / np.sqrt(2)

S_1 = {}
for (F, G, H, I) in allowed_quadruples:
    S_1[F, G, H, I] = (np.random.randn(chi, chi, chi, chi) + 1j*np.random.randn(chi, chi, chi, chi)) / np.sqrt(2)

P_2 = {}
for (H, A, B, J) in allowed_quadruples:
    P_2[H, A, B, J] = (np.random.randn(chi, chi, chi, chi) + 1j*np.random.randn(chi, chi, chi, chi)) / np.sqrt(2)

R_2 = {}
for (D, E, F, J) in allowed_quadruples:
    R_2[D, E, F, J] = (np.random.randn(chi, chi, chi, chi) + 1j*np.random.randn(chi, chi, chi, chi)) / np.sqrt(2)

S_2 = {}
for (F, G, H, J) in allowed_quadruples:
    S_2[F, G, H, J] = (np.random.randn(chi, chi, chi, chi) + 1j*np.random.randn(chi, chi, chi, chi)) / np.sqrt(2)

X = {}
for (A, B, C, I) in allowed_quadruples:
    X[A, B, C, I] = (np.random.randn(chi, chi, chi, chi) + 1j*np.random.randn(chi, chi, chi, chi)) / np.sqrt(2)

Y = {}
for (C, D, E, I) in allowed_quadruples:
    Y[C, D, E, I] = (np.random.randn(chi, chi, chi, chi) + 1j*np.random.randn(chi, chi, chi, chi)) / np.sqrt(2)

Z = {}
for (E, F, G, I) in allowed_quadruples:
    Z[E, F, G, I] = (np.random.randn(chi, chi, chi, chi) + 1j*np.random.randn(chi, chi, chi, chi)) / np.sqrt(2)

W = {} 
for (G, H, A, I) in allowed_quadruples:
    W[G, H, A, I] = (np.random.randn(chi, chi, chi, chi) + 1j*np.random.randn(chi, chi, chi, chi)) / np.sqrt(2)

K_tensor = {} # This capures the tensor of the form x^7
L_tensor = {} # This captures the tensor of the form x^3*y^4

for (A, B, C, D, E, F, G, H, I, J) in consistent_decuples:

    key = (B, C, D, J)
    #print(key)

    shape = (chi,) * 4 #four indices
    K_tensor[key] = np.zeros(shape, dtype=np.complex128)
    L_tensor[key] = np.zeros(shape, dtype=np.complex128)

    K_tensor[key] += oe.contract('i l j k, l o m n, o r p q, r i s t, j k u x, p q w v, s t v u -> m n x w',
                                 P_1[H, A, B, I] , Q_1[B, C, D, I], R_1[D, E, F, I], S_1[F, G, H, I], P_2[H, A, B, J], R_2[D, E, F, J], S_2[F, G, H, J])

    L_tensor[key] += oe.contract('i l j k, l o m n, o r p q, r i s t, t j u x, n p w v, q s v u -> k m x w', 
                                 X[A, B, C, I], Y[C, D, E, I], Z[E, F, G, I], W[G, H, A, I], P_2[H, A, B, J], R_2[D, E, F, J], S_2[F, G, H, J])
    
def derivative_loss_function():
        if consistent_decuples:
            return dict_diff(K_tensor, L_tensor) # x^4 - y^4

def derivative_loss_norm():
    return dict_norm(derivative_loss_function())

def main():
    print(derivative_loss_norm())

if __name__ == "__main__":
    main()