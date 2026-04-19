"""
Loop-TNR loss function, gradient, and L-BFGS optimiser.

The loss is eps^2 = 0.5 * ||U(X,Y,Z,W) - V(P,Q,R,S)||_F^2 where
  U = pre-RG-step primal contraction (target, frozen during optimisation)
  V = dual projector contraction (the approximation, varied)

The optimiser adjusts the dual projectors P,Q,R,S to minimise eps^2,
i.e. to make V match U as closely as possible. X,Y,Z,W are never
modified — they come from the previous RG layer.

All state is passed explicitly — no module-level globals.
"""
import numpy as np
import opt_einsum as oe

from potts_q5_hamiltonian import build_q5_potts
from scipy.optimize import minimize as scipy_minimize
from utils import (dict_diff, dict_squared_norm, dict_frobenius_norm,
                   dict_to_vec, vec_to_dict)


# =====================================================================
# Constants
# =====================================================================

DUAL_NAMES = ('P', 'Q', 'R', 'S')

# State-dict key for each tensor (Q is stored as 'Q_tens' to avoid
# shadowing Python's built-in Q).
STATE_KEY = {'X': 'X', 'Y': 'Y', 'Z': 'Z', 'W': 'W',
             'P': 'P', 'Q': 'Q_tens', 'R': 'R', 'S': 'S'}

# Einsum strings (consistent with iPad diagram)
U_EINSUM = 'i l u x, m n u v, p q v w, s t x w -> i l m n q p s t'
V_EINSUM = 'l t u x, i m u v, n p v w, s q x w -> i l m n q p s t'


# =====================================================================
# Core tensor assembly
# =====================================================================

def build_loss_tensors(chi, consistent_nonuples, X, Y, Z, W, P, Q, R, S):
    """Assemble U (primal) and V (dual) 8-index loss tensors."""
    U_tensor = {}
    V_tensor = {}
    shape = (chi,) * 8

    for nonu in consistent_nonuples:
        A, B, C, D, E, F, G, H, I = nonu
        key = (A, B, C, D, F, G, H, I)
        if key not in U_tensor:
            U_tensor[key] = np.zeros(shape, dtype=np.complex128)
            V_tensor[key] = np.zeros(shape, dtype=np.complex128)

        U_tensor[key] += oe.contract(U_EINSUM,
            X[A,B,D,E], Y[B,C,E,F], Z[E,F,H,I], W[D,E,G,H])
        V_tensor[key] += oe.contract(V_EINSUM,
            P[A,D,E,G], Q[B,A,C,E], R[C,E,F,I], S[E,G,I,H])

    return U_tensor, V_tensor


# =====================================================================
# Gradient computation (any subset of 8 tensors)
# =====================================================================

def build_grads(chi, consistent_nonuples, diff_tensor,
                P, Q, R, S, which='PQRS'):
    """Compute ∂(eps²)/∂T for each dual projector T named in `which`.

    eps² = 0.5 * ||U - V||_F^2.  U is frozen (target); only the dual
    side V(P,Q,R,S) is differentiated.

    For dual tensors: grad_T = -conj(diff) contracted with the 3
    remaining V-side tensors (Q removed for ∂/∂Q, etc.).

    `which`: subset of 'PQRS'. Returns {name: grad_dict}.

    # Primal gradients (∂L/∂X etc.) are not computed — X,Y,Z,W define
    # the pre-RG target and are never optimised. If needed in future,
    # the formula is +diff (not -conj(diff)) contracted with the 3
    # remaining U-side tensors.
    """
    which_set = set(which.upper()) & set('PQRS')
    grads = {name: {} for name in which_set}
    shape4 = (chi,) * 4

    for nonu in consistent_nonuples:
        A, B, C, D, E, F, G, H, I = nonu
        diff_key = (A, B, C, D, F, G, H, I)
        if diff_key not in diff_tensor:
            continue

        Pv = P[A,D,E,G]; Qv = Q[B,A,C,E]; Rv = R[C,E,F,I]; Sv = S[E,G,I,H]
        d_neg = -diff_tensor[diff_key].conj()

        if 'P' in which_set:
            k = (A,D,E,G)
            if k not in grads['P']:
                grads['P'][k] = np.zeros(shape4, dtype=np.complex128)
            grads['P'][k] += oe.contract(
                'i l m n q p s t, i m u v, n p v w, s q x w -> l t u x',
                d_neg, Qv, Rv, Sv)

        if 'Q' in which_set:
            k = (B,A,C,E)
            if k not in grads['Q']:
                grads['Q'][k] = np.zeros(shape4, dtype=np.complex128)
            grads['Q'][k] += oe.contract(
                'i l m n q p s t, l t u x, n p v w, s q x w -> i m u v',
                d_neg, Pv, Rv, Sv)

        if 'R' in which_set:
            k = (C,E,F,I)
            if k not in grads['R']:
                grads['R'][k] = np.zeros(shape4, dtype=np.complex128)
            grads['R'][k] += oe.contract(
                'i l m n q p s t, l t u x, i m u v, s q x w -> n p v w',
                d_neg, Pv, Qv, Sv)

        if 'S' in which_set:
            k = (E,G,I,H)
            if k not in grads['S']:
                grads['S'][k] = np.zeros(shape4, dtype=np.complex128)
            grads['S'][k] += oe.contract(
                'i l m n q p s t, l t u x, i m u v, n p v w -> s q x w',
                d_neg, Pv, Qv, Rv)

    return grads


# =====================================================================
# High-level loss + gradient
# =====================================================================

def loss_and_grad(state, overrides=None, which='PQRS'):
    """Compute (eps², grad_dicts) from a state dict.

    eps² = 0.5 * ||U(X,Y,Z,W) - V(P,Q,R,S)||_F^2.
    U is frozen (pre-RG target). Only dual projectors P,Q,R,S can be varied.

    overrides: dict mapping dual tensor name -> replacement dict-tensor.
               e.g. {'Q': Q_new} or {'P': P_new, 'Q': Q_new, 'R':..., 'S':...}
    which: subset of 'PQRS' to compute gradients for.

    Returns (loss_scalar, grads) where grads = {name: grad_dict_tensor}.
    """
    # Primal side: always from state (frozen)
    X, Y, Z, W = state['X'], state['Y'], state['Z'], state['W']

    # Dual side: apply overrides
    dual = {}
    for name in DUAL_NAMES:
        sk = STATE_KEY[name]
        if overrides and name in overrides:
            dual[name] = overrides[name]
        else:
            dual[name] = state[sk]

    U, V = build_loss_tensors(
        state['chi'], state['consistent_nonuples'],
        X, Y, Z, W, dual['P'], dual['Q'], dual['R'], dual['S'])

    diff = dict_diff(U, V)
    if diff is None:
        raise ValueError("Loss tensors have incompatible keys")
    loss = dict_squared_norm(diff)

    grads = build_grads(
        state['chi'], state['consistent_nonuples'], diff,
        dual['P'], dual['Q'], dual['R'], dual['S'],
        which=which)

    return loss, grads


# =====================================================================
# Setup helper
# =====================================================================

def setup_potts(Q_potts=5, verbose=True):
    """Build the full Potts state dict at criticality."""
    state = build_q5_potts(Q=Q_potts, verbose=verbose)

    state['U_tensor'], state['V_tensor'] = build_loss_tensors(
        state['chi'], state['consistent_nonuples'],
        state['X'], state['Y'], state['Z'], state['W'],
        state['P'], state['Q_tens'], state['R'], state['S'])

    state['diff_tensor'] = dict_diff(state['U_tensor'], state['V_tensor'])
    return state


def compute_loss_from_state(state):
    """0.5 * ||U - V||_F^2 from pre-built diff_tensor."""
    return dict_squared_norm(state['diff_tensor'])


# =====================================================================
# L-BFGS optimiser (any subset of 8 tensors)
# =====================================================================

def optimize(state, which='PQRS', method='L-BFGS-B', maxiter=200,
             tol=1e-12, verbose=True):
    """Optimise dual projectors to minimise eps^2 = 0.5*||U-V||_F^2.

    U(X,Y,Z,W) is the pre-RG target (always frozen).
    V(P,Q,R,S) is the dual approximation (varied).

    which: subset of 'PQRS' to optimise. Default = all 4 dual projectors.
           (X,Y,Z,W are never varied — they define the target U.)

    # Historical note: Q-only optimisation converges to a local minimum
    # (35.5% decrease) because P,R,S are frozen at wrong values.
    # All 4 dual projectors needed to reach global zero (machine eps).

    Returns (optimised_state, scipy.OptimizeResult).
    """
    which_upper = which.upper()
    tensor_names = [c for c in which_upper if c in set('PQRS')]
    if not tensor_names:
        raise ValueError(f"which='{which}' must contain at least one of P,Q,R,S")

    # Build sorted keys and shapes for each varied tensor
    sorted_keys = sorted(state['allowed_quadruples'])
    meta = {}  # name -> (shapes, vec_len)
    total_len = 0
    for name in tensor_names:
        sk = STATE_KEY[name]
        shapes = [state[sk][k].shape for k in sorted_keys]
        vec_len = sum(2 * int(np.prod(s)) for s in shapes)  # Re+Im
        meta[name] = (shapes, vec_len)
        total_len += vec_len

    def pack(overrides):
        parts = []
        for name in tensor_names:
            parts.append(dict_to_vec(overrides[name], sorted_keys))
        return np.concatenate(parts)

    def unpack(x):
        overrides = {}
        i = 0
        for name in tensor_names:
            shapes, vlen = meta[name]
            overrides[name] = vec_to_dict(x[i:i+vlen], sorted_keys, shapes)
            i += vlen
        return overrides

    # Initial vector
    x0 = pack({name: state[STATE_KEY[name]] for name in tensor_names})
    call_count = [0]

    def objective(x):
        overrides = unpack(x)
        loss, grads = loss_and_grad(state, overrides=overrides, which=which_upper)
        grad_parts = []
        for name in tensor_names:
            grad_parts.append(dict_to_vec(grads[name], sorted_keys))
        grad_vec = np.concatenate(grad_parts)
        # TODO: negate Im(grad) for complex β — moot for real Potts
        call_count[0] += 1
        if verbose and call_count[0] % 10 == 1:
            print(f"  eval {call_count[0]:4d}: loss = {loss:.6e}")
        return float(loss), grad_vec.astype(np.float64)

    if verbose:
        init_loss = compute_loss_from_state(state)
        print(f"  optimising: {', '.join(tensor_names)}")
        print(f"  x0 dim = {total_len}, initial loss = {init_loss:.6e}")

    result = scipy_minimize(objective, x0, method=method, jac=True,
                            options={'maxiter': maxiter, 'disp': verbose,
                                     'ftol': tol, 'gtol': 1e-10})

    # Rebuild state at optimum
    opt_overrides = unpack(result.x)
    opt_state = dict(state)
    for name in tensor_names:
        opt_state[STATE_KEY[name]] = opt_overrides[name]

    U, V = build_loss_tensors(
        state['chi'], state['consistent_nonuples'],
        opt_state['X'], opt_state['Y'], opt_state['Z'], opt_state['W'],
        opt_state['P'], opt_state['Q_tens'], opt_state['R'], opt_state['S'])
    opt_state['U_tensor'] = U
    opt_state['V_tensor'] = V
    opt_state['diff_tensor'] = dict_diff(U, V)

    return opt_state, result


# =====================================================================
# CLI entry point
# =====================================================================

def main(Q_potts=2):
    state = setup_potts(Q_potts=Q_potts, verbose=True)
    loss = compute_loss_from_state(state)
    print(f"\nQ={Q_potts} Potts, chi={state['chi']}, "
          f"{len(state['consistent_nonuples'])} nonuples.")
    print(f"Initial eps^2 = ||U(X,Y,Z,W) - V(P,Q,R,S)||_F^2 = {loss:.6e}")

    print("\n=== L-BFGS: optimise V(P,Q,R,S) to match U(X,Y,Z,W) ===")
    opt, res = optimize(state, which='PQRS', maxiter=300, verbose=True)
    fl = compute_loss_from_state(opt)
    print(f"\n  Converged: {res.success}, iters: {res.nit}, evals: {res.nfev}")
    print(f"  Final eps^2: {fl:.6e} ({(1-fl/loss)*100:.1f}% decrease)")


if __name__ == "__main__":
    main()