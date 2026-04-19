"""
Loop-TNR RG iteration.

Each RG step:
  1. Build U(X,Y,Z,W) = pre-RG contraction on 3×3 nonuple patch
  2. Optimise V(P,Q,R,S) ≈ U via L-BFGS (dual projectors)
  3. Optimised P*,Q*,R*,S* become the new X',Y',Z',W' for the next
     step on the 45°-rotated lattice (TRG-style alternation)

After 2 steps the lattice returns to original orientation at coarser
scale. The loss eps² = 0.5*||U-V||_F² should stay near zero if the
tensor complexity (chi) is sufficient; growth of eps² signals the
need for higher chi.

Usage:
    python rg_iteration.py                        # free energy scan (default)
    python rg_iteration.py mode=flow Q=2 N=10     # RG flow at beta_c
    python rg_iteration.py mode=scan Q=2 N=6      # eps² beta scan
    python rg_iteration.py mode=free_energy N=10   # free energy scan
"""
import numpy as np
import time

from potts_q5_hamiltonian import (build_q5_potts, build_zq_f_symbols,
                                  build_boltzmann_weights_at_beta, get_model_T,
                                  get_allowed_pairs_from_T, get_allowed_quadruples,
                                  get_allowed_nonuples,
                                  is_nonuple_consistent_with_quadruples,
                                  initialize_loop_tnr_tensors)
from loss_function_and_gradient import (
    setup_potts, optimize, compute_loss_from_state,
    build_loss_tensors,
)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location('central_gauge_mod', 'central-gauge.py')
_cg = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_cg)
central_gauge = _cg.central_gauge
svd_redefining = _cg.svd_redefining
from utils import dict_diff, dict_squared_norm


def normalize_state(state):
    """Rescale all primal + dual tensors by max element of primals.

    Returns ln(scale_factor). Modifies state in-place, including
    rebuilding U, V, diff tensors.

    This keeps tensors O(1) across RG steps and the log of the scale
    factor is used to accumulate the free energy:
        ln(Z)/N_sites = Σ_n (1/4^n) × ln(λ_n)
    """
    max_val = 0.0
    for name in ('X', 'Y', 'Z', 'W'):
        for v in state[name].values():
            max_val = max(max_val, float(np.abs(v).max()))

    if max_val < 1e-300:
        return 0.0

    ln_scale = np.log(max_val)

    for name in ('X', 'Y', 'Z', 'W', 'P', 'Q_tens', 'R', 'S'):
        state[name] = {k: v / max_val for k, v in state[name].items()}

    # Rebuild loss tensors at rescaled values
    U, V = build_loss_tensors(
        state['chi'], state['consistent_nonuples'],
        state['X'], state['Y'], state['Z'], state['W'],
        state['P'], state['Q_tens'], state['R'], state['S'])
    state['U_tensor'] = U
    state['V_tensor'] = V
    state['diff_tensor'] = dict_diff(U, V)

    return ln_scale


def rg_step(state, maxiter=300, tol=1e-12, verbose=False):
    """One Loop-TNR RG step.

    1. Optimise P,Q,R,S to minimise eps² = 0.5*||U-V||²
    2. Return (opt_state, result, eps²_final)

    The caller then extracts the optimised dual projectors and feeds
    them as the new primal tensors for the next step.
    """
    opt_state, result = optimize(
        state, which='PQRS', method='L-BFGS-B',
        maxiter=maxiter, tol=tol, verbose=verbose)
    eps2 = compute_loss_from_state(opt_state)
    return opt_state, result, eps2


def gauge_fix_dict(T_dict, allowed_quadruples=None):
    """Apply central gauge + SVD to a dict-tensor, then reconstruct.

    Per-sector decomposition:
        T[key] = A[key] @ A_c[key] @ B[key]          (central gauge)
        A_c    = U_svd @ diag(sigma) @ Vh             (SVD)
        A_clean = A @ U_svd,  B_clean = Vh @ B        (absorb unitaries)

    Full factorisation: T = A_clean · diag(sigma) · B_clean
    where shapes are:
        A_clean: (a, d1, d2, rank)
        sigma:   (rank,)
        B_clean: (rank, b)

    Reconstruction via einsum 'ijkr, r, rl -> ijkl'.

    This doesn't change T (exact reconstruction) but puts it in
    canonical form: the polar isometry is bi-unitary, singular values
    are sorted, and small entries are zeroed. Keeps tensors
    well-conditioned across RG iterations.
    """
    A_dict, A_c_dict, B_dict = central_gauge(T_dict, allowed_quadruples)
    A_clean, sigma_clean, B_clean = svd_redefining(
        A_dict, A_c_dict, B_dict, allowed_quadruples)

    T_fixed = {}
    for key in A_clean:
        # T[i,j,k,l] = sum_r A_clean[i,j,k,r] * sigma[r] * B_clean[r,l]
        T_fixed[key] = np.einsum('ijkr, r, rl -> ijkl',
                                  A_clean[key], sigma_clean[key], B_clean[key])
    return T_fixed


def rotate_state(opt_state, gauge_fix=True):
    """Promote optimised dual projectors to the primal side for the next step.

    After optimisation: V(P*,Q*,R*,S*) ≈ U(X,Y,Z,W).
    For the next RG step on the 45°-rotated lattice:
        X' = P*,  Y' = Q*,  Z' = R*,  W' = S*
    and the new P',Q',R',S' are initialised to the same values
    (translation invariance at the start of each step).

    If gauge_fix=True (default), apply central gauge + SVD to each new
    primal tensor to keep the decomposition well-conditioned.
    """
    new_state = dict(opt_state)
    quads = list(opt_state['allowed_quadruples'])

    # Optimised duals become new primals
    new_X = opt_state['P']
    new_Y = opt_state['Q_tens']
    new_Z = opt_state['R']
    new_W = opt_state['S']

    # Gauge fix each tensor independently
    if gauge_fix:
        new_X = gauge_fix_dict(new_X, quads)
        new_Y = gauge_fix_dict(new_Y, quads)
        new_Z = gauge_fix_dict(new_Z, quads)
        new_W = gauge_fix_dict(new_W, quads)

    new_state['X'] = new_X
    new_state['Y'] = new_Y
    new_state['Z'] = new_Z
    new_state['W'] = new_W

    # Reinitialise duals to match primals (translation invariance)
    new_state['P'] = {k: v.copy() for k, v in new_X.items()}
    new_state['Q_tens'] = {k: v.copy() for k, v in new_Y.items()}
    new_state['R'] = {k: v.copy() for k, v in new_Z.items()}
    new_state['S'] = {k: v.copy() for k, v in new_W.items()}

    # Rebuild loss tensors at the new starting point
    U, V = build_loss_tensors(
        new_state['chi'], new_state['consistent_nonuples'],
        new_state['X'], new_state['Y'], new_state['Z'], new_state['W'],
        new_state['P'], new_state['Q_tens'], new_state['R'], new_state['S'])
    new_state['U_tensor'] = U
    new_state['V_tensor'] = V
    new_state['diff_tensor'] = dict_diff(U, V)

    return new_state


def rg_flow(Q_potts=2, n_steps=10, maxiter=300, tol=1e-12, verbose=True):
    """Run n_steps of Loop-TNR RG iteration.

    Returns list of (step, eps2_before, eps2_after, n_iters, walltime).
    """
    state = setup_potts(Q_potts=Q_potts, verbose=verbose)
    eps2_init = compute_loss_from_state(state)

    if verbose:
        print(f"\nQ={Q_potts} Potts, chi={state['chi']}, "
              f"{len(state['consistent_nonuples'])} nonuples")
        print(f"{'step':>4}  {'eps2_before':>12}  {'eps2_after':>12}  "
              f"{'iters':>5}  {'time':>6}")
        print("-" * 55)

    history = []

    for step in range(n_steps):
        eps2_before = compute_loss_from_state(state)
        t0 = time.time()
        opt_state, result, eps2_after = rg_step(
            state, maxiter=maxiter, tol=tol, verbose=False)
        dt = time.time() - t0

        history.append((step, eps2_before, eps2_after, result.nit, dt))

        if verbose:
            print(f"{step:4d}  {eps2_before:12.3e}  {eps2_after:12.3e}  "
                  f"{result.nit:5d}  {dt:5.1f}s")

        # Rotate: optimised duals → new primals for next step
        state = rotate_state(opt_state)

    if verbose:
        print("-" * 55)
        final = compute_loss_from_state(state)
        print(f"Final eps² at start of step {n_steps}: {final:.3e}")

        # Sample tensor values at the final RG scale
        T_final = state['X']  # primals = last step's optimised duals
        sample_keys = sorted(T_final.keys())[:5]
        print(f"\nSample tensor values at RG step {n_steps}:")
        for k in sample_keys:
            print(f"  T'{k} = {T_final[k].item():.6f}")

    return history, state


def setup_at_beta(Q, beta, verbose=False):
    """Build a state dict at arbitrary inverse temperature beta.

    Same structure as setup_potts() but with Boltzmann weights
    parameterized by beta instead of hardcoded at beta_c.
    """
    F, C = build_zq_f_symbols(Q)
    boltz = build_boltzmann_weights_at_beta(Q, beta)
    T_dict = get_model_T(F, boltz, trace_physical=True)

    modules = list(range(Q))
    allowed_pairs, vertical_pairs = get_allowed_pairs_from_T(T_dict)
    allowed_quads = get_allowed_quadruples(allowed_pairs, vertical_pairs)
    allowed_nonus = get_allowed_nonuples(allowed_pairs, vertical_pairs, modules)
    consistent = [n for n in allowed_nonus
                  if is_nonuple_consistent_with_quadruples(allowed_quads, *n)]

    X, Y, Z, W, P, Q_tens, R, S, Q_1 = initialize_loop_tnr_tensors(
        T_dict, allowed_quads)

    state = {
        'Q': Q, 'chi': 1, 'modules': modules,
        'allowed_pairs': allowed_pairs,
        'vertical_allowed_pairs': vertical_pairs,
        'allowed_quadruples': allowed_quads,
        'consistent_nonuples': consistent,
        'T_dict': T_dict,
        'X': X, 'Y': Y, 'Z': Z, 'W': W,
        'P': P, 'Q_tens': Q_tens, 'R': R, 'S': S, 'Q_1': Q_1,
    }

    U, V = build_loss_tensors(
        1, consistent, X, Y, Z, W, P, Q_tens, R, S)
    state['U_tensor'] = U
    state['V_tensor'] = V
    state['diff_tensor'] = dict_diff(U, V)

    if verbose:
        print(f"Q={Q}, beta={beta:.6f}, "
              f"beta_c={np.log(1+np.sqrt(Q)):.6f}, "
              f"{len(consistent)} nonuples")
    return state


def beta_scan(Q=2, betas=None, n_steps=6, maxiter=300, verbose=True):
    """Scan beta values and run RG iteration at each.

    Reports the eps²_after at the last RG step for each beta.
    The critical beta should be identifiable as the point where
    the RG flow behaves most cleanly (loss stays near zero).
    """
    if betas is None:
        beta_c = np.log(1 + np.sqrt(Q))
        betas = np.linspace(0.5 * beta_c, 1.5 * beta_c, 11)

    if verbose:
        beta_c = np.log(1 + np.sqrt(Q))
        print(f"Q={Q} Ising beta scan, beta_c = {beta_c:.6f}")
        print(f"{n_steps} RG steps per beta, {len(betas)} beta values")
        header = f"{'beta':>8}  " + "  ".join(f"{'step'+str(i):>10}" for i in range(n_steps))
        print(header)
        print("-" * len(header))

    results = {}
    for beta in betas:
        state = setup_at_beta(Q, beta)
        eps_history = []
        for step in range(n_steps):
            opt_state, result, eps2 = rg_step(state, maxiter=maxiter, verbose=False)
            eps_history.append(eps2)
            state = rotate_state(opt_state)

        results[beta] = eps_history
        if verbose:
            row = f"{beta:8.5f}  " + "  ".join(f"{e:10.2e}" for e in eps_history)
            print(row)

    if verbose:
        beta_c = np.log(1 + np.sqrt(Q))
        # Find beta with smallest total eps² across all steps
        total_eps = {b: sum(h) for b, h in results.items()}
        best_beta = min(total_eps, key=total_eps.get)
        print(f"\nBest beta = {best_beta:.6f} (total eps² = {total_eps[best_beta]:.3e})")
        print(f"Known beta_c = {beta_c:.6f}")
        print(f"Relative error = {abs(best_beta - beta_c)/beta_c:.4f}")

    return results


def free_energy_scan(Q=2, betas=None, n_steps=10, maxiter=300, verbose=True):
    """Compute free energy f(β) from tensor-norm tracking across RG steps.

    At each step, primals are rescaled to O(1) and ln(scale) is accumulated
    with geometric weight 1/4^n (2D area factor per coarse-graining step).

    Returns (betas, f_arr, specific_heat) where specific_heat = -β² d²f/dβ².
    The peak of specific_heat locates β_c.
    """
    beta_c = np.log(1 + np.sqrt(Q))
    if betas is None:
        betas = np.linspace(0.3 * beta_c, 1.7 * beta_c, 51)

    if verbose:
        print(f"Q={Q} free energy scan, beta_c(exact) = {beta_c:.6f}")
        print(f"{n_steps} RG steps, {len(betas)} beta values\n")

    f_values = {}

    for idx, beta in enumerate(betas):
        state = setup_at_beta(Q, beta)
        ln_Z_per_site = 0.0

        for step in range(n_steps):
            ln_scale = normalize_state(state)
            ln_Z_per_site += ln_scale / 4**step

            opt_state, result, eps2 = rg_step(
                state, maxiter=maxiter, verbose=False)
            state = rotate_state(opt_state)

        # Final tensor norm contribution
        ln_scale_final = normalize_state(state)
        ln_Z_per_site += ln_scale_final / 4**n_steps

        f = -ln_Z_per_site / beta
        f_values[beta] = f

        if verbose and (idx % 10 == 0 or idx == len(betas) - 1):
            print(f"  [{idx+1:3d}/{len(betas)}] beta={beta:.5f}  "
                  f"f={f:.8f}  ln_Z/N={ln_Z_per_site:.8f}")

    # Convert to arrays
    betas_arr = np.array(sorted(f_values.keys()))
    f_arr = np.array([f_values[b] for b in betas_arr])

    # Numerical derivatives: energy E = d(βf)/dβ, specific heat C = dE/dβ
    bf = betas_arr * f_arr
    energy = np.gradient(bf, betas_arr)
    specific_heat = np.gradient(energy, betas_arr)

    # Find specific heat peak
    peak_idx = np.argmax(np.abs(specific_heat))
    beta_peak = betas_arr[peak_idx]

    if verbose:
        print(f"\n{'beta':>8}  {'f(beta)':>12}  {'E(beta)':>12}  {'C(beta)':>12}")
        print("-" * 52)
        for i, b in enumerate(betas_arr):
            markers = ""
            if i == peak_idx:
                markers += " <-- C peak"
            if abs(b - beta_c) < (betas_arr[1] - betas_arr[0]) * 0.6:
                markers += " <-- beta_c"
            print(f"{b:8.5f}  {f_arr[i]:12.8f}  {energy[i]:12.8f}  "
                  f"{specific_heat[i]:12.6f}{markers}")

        print(f"\nSpecific heat peak at beta = {beta_peak:.6f}")
        print(f"Exact beta_c = {beta_c:.6f}")
        print(f"Relative error = {abs(beta_peak - beta_c)/beta_c:.4f}")

    return betas_arr, f_arr, specific_heat


if __name__ == "__main__":
    import sys
    args = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            k, v = arg.split('=', 1)
            args[k] = v

    mode = args.get('mode', 'free_energy')

    if mode == 'flow':
        Q = int(args.get('Q', '2'))
        N = int(args.get('N', '10'))
        rg_flow(Q_potts=Q, n_steps=N)
    elif mode == 'scan':
        Q = int(args.get('Q', '2'))
        N = int(args.get('N', '6'))
        beta_scan(Q=Q, n_steps=N)
    elif mode == 'free_energy':
        Q = int(args.get('Q', '2'))
        N = int(args.get('N', '10'))
        free_energy_scan(Q=Q, n_steps=N)