"""
Z_Q Potts model plaquette tensor for 2D Loop-TNR.

Constructs the partition-function plaquette tensor for the Q-state Potts
model on the square lattice, parameterized by inverse temperature beta.

Symmetry group: Z_Q  (default Q=5, supports any Q >= 2)
Module labels:  {0, 1, ..., Q-1}
Cayley table:   C[i,j] = (i + j) mod Q
F-symbols:      Abelian associator (trivial 3-cocycle of Z_Q)
Boltzmann wts:  Geometrically normalized, w_q = (1+sqrt(Q))^{(beta/beta_c)(q-(Q-1)/2)/(Q-1)}

Output contract (matches loss_function_and_gradient.py):
  - modules, allowed_pairs, vertical_allowed_pairs
  - allowed_quadruples, consistent_nonuples
  - X, Y, Z, W, P, Q_tens, R, S : Dict[(A,B,C,D) -> ndarray(chi,chi,chi,chi)]
"""

import numpy as np
from itertools import product
import opt_einsum as oe
import time


# ============================================================
# 1. PARTITION-FUNCTION PLAQUETTE TENSOR
# ============================================================

Q_POTTS = 5

def build_cayley_table(Q):
    """Z_Q group multiplication: C[i,j] = (i+j) mod Q."""
    return np.fromfunction(lambda i, j: (i + j) % Q, (Q, Q), dtype=int).astype(int)


def build_f_symbols(C, Q):
    """
    Abelian F-symbols (6j-symbols) for Z_Q module category.

    F[i, j, k, c, a, b] = delta(a, C[i,j]) * delta(b, C[j,k]) * delta(c, C[a,k])

    Shape: (Q, Q, Q, Q, Q, Q)
    Non-zero entries: Q^3 (one per (i,j,k) triple).
    """
    F = np.zeros((Q, Q, Q, Q, Q, Q), dtype=complex)
    for i, j, k in product(range(Q), repeat=3):
        a = C[i, j]
        b = C[j, k]
        c = C[a, k]
        F[i, j, k, c, a, b] = 1.0
    return F


def build_boltzmann_weights(Q):
    """
    Geometrically normalized Boltzmann weights at criticality.

    w_q = (1 + sqrt(Q))^((q - (Q-1)/2) / (Q-1))

    Properties:
      - prod(w) = 1  (geometric mean = 1)
      - w_{Q-1}/w_0 = 1 + sqrt(Q) = e^{beta_c}
      - Monotonically increasing in q

    Verified: exact match with reference get_Ising_T("Z2") at Q=2.
    """
    base = 1.0 + np.sqrt(Q)
    return np.array([base ** ((q - (Q - 1) / 2.0) / (Q - 1)) for q in range(Q)])


def build_boltzmann_weights_at_beta(Q, beta):
    """
    Geometrically normalized Boltzmann weights at arbitrary inverse temperature.

    Same convention as build_boltzmann_weights (monotone increasing in q):
        w_q(beta) = (1+sqrt(Q))^{ (beta/beta_c) * (q-(Q-1)/2) / (Q-1) }
    where beta_c = ln(1+sqrt(Q)).

    At beta = beta_c: recovers build_boltzmann_weights(Q) exactly.
    At beta = 0: all weights = 1 (infinite temperature).
    """
    base = 1.0 + np.sqrt(Q)
    beta_c = np.log(base)
    ratio = beta / beta_c
    return np.array([base ** (ratio * (q - (Q - 1) / 2.0) / (Q - 1)) for q in range(Q)])


def build_zq_f_symbols(Q):
    """
    API name from MASTER.md §12.1. Returns (F, C) for the trivial-cocycle
    Vec_{Z_Q} fusion category.

    F has shape (Q,)*6 with non-zero count exactly Q^3 (one entry per
    (i, j, k) input triple). C is the additive Z_Q Cayley table.
    Pentagon equation holds automatically because F is the trivial
    3-cocycle of Z_Q.
    """
    C = build_cayley_table(Q)
    F = build_f_symbols(C, Q)
    return F, C


def get_model_T(f_symbols, boltzmann_weights, trace_physical=True):
    """
    API name from MASTER.md §12.1. Generic abelian plaquette assembly.

    nM is inferred from f_symbols.shape[0] and d from len(boltzmann_weights).

    Stage 1: T[A,B,C,D,i,j,k,l] = sum_m F[A,j,l,D,B,m] * conj(F[A,i,k,D,C,m])
                                       * sqrt(w_i w_j w_k w_l)
    Stage 2: trace_physical=True traces the four physical legs to give a
             (1,1,1,1) scalar per sector (Z_Q reduction); False keeps them
             at (d,d,d,d) (Vec module case).

    Returns dict keyed by (A,B,C,D); sectors with |max| < 1e-15 are dropped.
    """
    nM = f_symbols.shape[0]
    d = len(boltzmann_weights)
    sqrt_b = np.sqrt(boltzmann_weights)

    F_conj = np.conj(f_symbols)
    F_upper = f_symbols.transpose(0, 3, 4, 1, 2, 5)   # (A, D, B, j, l, m)
    F_lower = F_conj.transpose(0, 3, 4, 1, 2, 5)      # (A, D, C, i, k, m)
    T_raw = np.einsum('adbJLm,adcIKm->adcbIJKL', F_upper, F_lower)
    T = T_raw.transpose(0, 3, 2, 1, 4, 5, 6, 7)       # (A, B, C, D, i, j, k, l)

    boltz_4 = np.einsum('i,j,k,l->ijkl', sqrt_b, sqrt_b, sqrt_b, sqrt_b)
    T *= boltz_4[None, None, None, None, :, :, :, :]

    out = {}
    for A, B, C_, D in product(range(nM), repeat=4):
        if trace_physical:
            value = np.sum(T[A, B, C_, D, :, :, :, :]).reshape(1, 1, 1, 1)
        else:
            value = T[A, B, C_, D, :, :, :, :]
        if np.abs(value).max() > 1e-15:
            out[(A, B, C_, D)] = value
    return out


def get_Potts_T(Q):
    """Convenience wrapper: build plaquette tensor at criticality via get_model_T."""
    F, C = build_zq_f_symbols(Q)
    boltz = build_boltzmann_weights(Q)
    return get_model_T(F, boltz, trace_physical=True)


# ============================================================
# 2. MODULE INFRASTRUCTURE
# ============================================================

def get_allowed_pairs_from_T(T_dict):
    """
    Derive horizontal and vertical allowed pairs from T_dict keys.

    For Z_Q Potts, all Q^2 pairs are allowed in both directions
    (all sectors are non-zero).
    """
    horizontal = set()
    vertical = set()
    for (A, B, C, D) in T_dict.keys():
        horizontal.add((A, B))
        horizontal.add((C, D))
        vertical.add((A, C))
        vertical.add((B, D))
    return sorted(horizontal), sorted(vertical)


def get_allowed_quadruples(allowed_pairs, vertical_allowed_pairs):
    """Plaquette module configurations consistent with pair constraints."""
    vertical = set(vertical_allowed_pairs)
    allowed_quadruples = set()
    for (A, B) in allowed_pairs:
        for (D, E) in allowed_pairs:
            if (A, D) in vertical and (B, E) in vertical:
                allowed_quadruples.add((A, B, D, E))
    return allowed_quadruples


def get_allowed_nonuples(allowed_pairs, vertical_allowed_pairs, modules):
    """3x3 lattice patch configurations (optimized pruning)."""
    allowed_nonuple = set()
    allowed_set = set(allowed_pairs)
    vertical_set = set(vertical_allowed_pairs)

    for (A, B), (D, E), (G, H) in product(allowed_set, repeat=3):
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


def is_nonuple_consistent_with_quadruples(allowed_quadruples, A, B, C, D, E, F, G, H, I):
    """Check that all 8 sub-plaquettes of a 3x3 patch are allowed."""
    if (A, B, D, E) not in allowed_quadruples:  # X
        return False
    if (B, C, E, F) not in allowed_quadruples:  # Y
        return False
    if (D, E, G, H) not in allowed_quadruples:  # W
        return False
    if (E, F, H, I) not in allowed_quadruples:  # Z
        return False
    if (A, D, E, G) not in allowed_quadruples:  # P
        return False
    if (B, A, C, E) not in allowed_quadruples:  # Q
        return False
    if (C, E, F, I) not in allowed_quadruples:  # R
        return False
    if (E, G, I, H) not in allowed_quadruples:  # S
        return False
    return True


# ============================================================
# 3. LOOP-TNR TENSOR INITIALIZATION
# ============================================================

def initialize_loop_tnr_tensors(T_dict, allowed_quadruples):
    """
    Initialize all 8 Loop-TNR tensors from the plaquette tensor T_dict.

    At the initial RG step (chi=1), all plaquettes are identical
    (translation invariance). Each tensor is a copy of T_dict with
    keys matching the position-specific variable ordering used in
    loss_function_and_gradient.py.

    Returns:
        X, Y, Z, W  : plaquette tensors
        P, Q_tens, R, S : dual projector tensors (Q_tens avoids shadowing)
        Q_1 : copy of Q_tens (used in gradient computation)
    """
    X = {}
    Y = {}
    Z = {}
    W = {}
    P = {}
    Q_tens = {}
    R = {}
    S = {}

    for quad in allowed_quadruples:
        val = T_dict[quad].copy()
        # All 8 tensors use the same value for any given quadruple key.
        # The key-variable ordering (A,B,D,E) vs (B,C,E,F) etc. is handled
        # by the nonuple loop which assigns the correct quad to each tensor.
        X[quad] = val.copy()
        Y[quad] = val.copy()
        Z[quad] = val.copy()
        W[quad] = val.copy()
        P[quad] = val.copy()
        Q_tens[quad] = val.copy()
        R[quad] = val.copy()
        S[quad] = val.copy()

    Q_1 = {k: v.copy() for k, v in Q_tens.items()}

    return X, Y, Z, W, P, Q_tens, R, S, Q_1


# ============================================================
# 4. LOSS FUNCTION (self-contained for validation)
# ============================================================

from utils import dict_diff, dict_squared_norm


def compute_loss(X, Y, Z, W, P, Q_tens, R, S, consistent_nonuples, chi):
    """Compute ||U - V||^2 at the current tensor values."""
    U_tensor = {}
    V_tensor = {}
    shape = (chi,) * 8

    for (A, B, C, D, E, F, G, H, I) in consistent_nonuples:
        key = (A, B, C, D, F, G, H, I)
        if key not in U_tensor:
            U_tensor[key] = np.zeros(shape, dtype=np.complex128)
            V_tensor[key] = np.zeros(shape, dtype=np.complex128)

        U_tensor[key] += oe.contract(
            'i l u x, m n u v, p q v w, s t x w -> i l m n q p s t',
            X[A, B, D, E], Y[B, C, E, F], Z[E, F, H, I], W[D, E, G, H]
        )
        V_tensor[key] += oe.contract(
            'l t u x, i m u v, n p v w, s q x w -> i l m n q p s t',
            P[A, D, E, G], Q_tens[B, A, C, E], R[C, E, F, I], S[E, G, I, H]
        )

    diff = dict_diff(U_tensor, V_tensor)
    if diff is None:
        return float('inf')
    return dict_squared_norm(diff)


# ============================================================
# 5. BUILD EVERYTHING
# ============================================================

def build_q5_potts(Q=Q_POTTS, verbose=True):
    """
    Master builder: constructs all data structures for Q-state Potts Loop-TNR
    at the critical point beta_c = ln(1+sqrt(Q)).

    The default Q=5 matches the file name and the original research target;
    smaller Q values (Q=2 Ising, Q=3) run dramatically faster because the
    nonuple count scales as Q^9 (Q=5 -> 1.95M, Q=3 -> 19,683, Q=2 -> 512).

    Returns a dict with all fields needed by the Loop-TNR algorithm.
    """
    chi = 1

    if verbose:
        print(f"=== Q={Q} Potts Model at criticality ===")
        print(f"beta_c = ln(1 + sqrt({Q})) = {np.log(1 + np.sqrt(Q)):.6f}")
        print()

    # 1. Plaquette tensor
    t0 = time.time()
    T_dict = get_Potts_T(Q)
    t_plaq = time.time() - t0
    if verbose:
        print(f"Plaquette tensor: {len(T_dict)} sectors, built in {t_plaq:.2f}s")

    boltz = build_boltzmann_weights(Q)
    if verbose:
        print(f"Boltzmann weights: {boltz}")
        print()

    # 2. Module infrastructure
    modules = list(range(Q))
    allowed_pairs, vertical_allowed_pairs = get_allowed_pairs_from_T(T_dict)
    if verbose:
        print(f"Modules: {modules}")
        print(f"Horizontal pairs: {len(allowed_pairs)}")
        print(f"Vertical pairs:   {len(vertical_allowed_pairs)}")

    t0 = time.time()
    allowed_quadruples = get_allowed_quadruples(allowed_pairs, vertical_allowed_pairs)
    t_quad = time.time() - t0
    if verbose:
        print(f"Allowed quadruples: {len(allowed_quadruples)} (computed in {t_quad:.2f}s)")

    t0 = time.time()
    allowed_nonuples = get_allowed_nonuples(allowed_pairs, vertical_allowed_pairs, modules)
    t_non = time.time() - t0
    if verbose:
        print(f"Allowed nonuples:   {len(allowed_nonuples)} (computed in {t_non:.2f}s)")

    t0 = time.time()
    consistent_nonuples = [
        n for n in allowed_nonuples
        if is_nonuple_consistent_with_quadruples(allowed_quadruples, *n)
    ]
    t_con = time.time() - t0
    if verbose:
        print(f"Consistent nonuples: {len(consistent_nonuples)} (filtered in {t_con:.2f}s)")
        print()

    # 3. Tensor initialization
    X, Y, Z, W, P, Q_tens, R, S, Q_1 = initialize_loop_tnr_tensors(
        T_dict, allowed_quadruples
    )
    if verbose:
        print(f"Tensors initialized at chi={chi}")
        print(f"  X sectors: {len(X)}, shape per sector: {next(iter(X.values())).shape}")
        print()

        # 4. Sample sector values
        sample_keys = sorted(T_dict.keys())[:5]
        print("Sample plaquette values:")
        for k in sample_keys:
            print(f"  T{k} = {T_dict[k].item():.6f}")
        print()

    return {
        'Q': Q,
        'chi': chi,
        'modules': modules,
        'allowed_pairs': allowed_pairs,
        'vertical_allowed_pairs': vertical_allowed_pairs,
        'allowed_quadruples': allowed_quadruples,
        'consistent_nonuples': consistent_nonuples,
        'T_dict': T_dict,
        'X': X, 'Y': Y, 'Z': Z, 'W': W,
        'P': P, 'Q_tens': Q_tens, 'R': R, 'S': S, 'Q_1': Q_1,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    data = build_q5_potts()

    # Validate: loss should be 0 at initialization (U = V by symmetry)
    print("=== Validation ===")
    loss = compute_loss(
        data['X'], data['Y'], data['Z'], data['W'],
        data['P'], data['Q_tens'], data['R'], data['S'],
        data['consistent_nonuples'], data['chi']
    )
    print(f"Loss at initialization: {loss:.2e}")
    print(f"  (Expected: 0 by translation invariance)")


if __name__ == "__main__":
    main()
