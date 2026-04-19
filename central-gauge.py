"""
Central gauge of a Loop-TNR plaquette tensor — closes G5.

`left_normalize` and `right_normalize` perform iterative polar
decomposition **block-by-block over the allowed_quadruples dictionary
keys**, exactly as specified in MASTER.md §12.3. There is no
flattened-tensor surface: every operation is per-sector, the dict
structure (and therefore the module decomposition) is preserved end
to end. `central_gauge` chains the two normalisers; `svd_redefining`
applies a per-sector SVD to redistribute the bond.

Rationale (MASTER.md §12.3): mapping the plaquette `T_dict` keyed by
4-tuples `(A, B, C, D)` into the 1D MPS infinite-cylinder pipeline of
`module-tensor-networks/util/gauge_fixing.py:157-188` would `KeyError`
because that pipeline is keyed by *vertical pairs* `(A, C)`. Instead
we keep the iterative polar-decomposition structure of the original
sandbox (raw 4-leg blocks) and apply it block-wise to each sector.

At the initial RG step (chi=1) every sector is shape (1, 1, 1, 1) and
the polar decomposition is trivial (`polar(z) = (z/|z|, |z|)`), so
A_c carries the unit phase and the gauge factors carry the modulus.
The lift becomes substantive once chi > 1.
"""
import numpy as np
from scipy.linalg import polar
from scipy.linalg import svd
import opt_einsum as oe

from potts_q5_hamiltonian import build_q5_potts


# =====================================================================
# Per-sector polar decompositions
# =====================================================================

def left_normalize(T_dict, allowed_quadruples=None):
    """
    Per-sector left polar decomposition of a Loop-TNR plaquette dict.

    Iterates the allowed_quadruples (or `T_dict.keys()` if not given)
    and for each sector reshapes the (a, d1, d2, b) block to a matrix
    of shape (a, d1 * d2 * b), runs `scipy.linalg.polar(side='left')`,
    and stores the positive factor in `A_dict[key]` and the unitary
    factor (reshaped back to (a, d1, d2, b)) in `A_c_dict[key]`.

    The decomposition is local to each sector — no inter-sector
    contraction or flattening.
    """
    keys = allowed_quadruples if allowed_quadruples is not None else T_dict.keys()
    A_dict = {}
    A_c_dict = {}
    for key in keys:
        block = T_dict[key]
        a, d1, d2, b = block.shape
        block_mat = block.reshape(a, d1 * d2 * b)
        U, P = polar(block_mat, side='left')
        A_dict[key] = P
        A_c_dict[key] = U.reshape(a, d1, d2, b)
    return A_dict, A_c_dict


def right_normalize(T_dict, allowed_quadruples=None):
    """
    Per-sector right polar decomposition of a Loop-TNR plaquette dict.

    Mirror of `left_normalize`: reshapes each (a, d1, d2, b) block to
    a (a * d1 * d2, b) matrix, runs `scipy.linalg.polar(side='right')`,
    stores the unitary factor (reshaped back) in `A_c_dict[key]` and
    the positive factor in `B_dict[key]`.
    """
    keys = allowed_quadruples if allowed_quadruples is not None else T_dict.keys()
    B_dict = {}
    A_c_dict = {}
    for key in keys:
        block = T_dict[key]
        a, d1, d2, b = block.shape
        block_mat = block.reshape(a * d1 * d2, b)
        U, P = polar(block_mat, side='right')
        B_dict[key] = P
        A_c_dict[key] = U.reshape(a, d1, d2, b)
    return B_dict, A_c_dict


# =====================================================================
# Iterative central gauge (per-sector lift of MASTER.md §12.3 template)
# =====================================================================

def central_gauge(T_dict, allowed_quadruples=None, tol=1e-10, max_iter=500):
    """
    Iterative polar decomposition on a dict of 4-leg plaquette tensors.

    Closes G5. Each iteration applies `left_normalize` and
    `right_normalize` (both per-sector), then measures the convergence
    error as the maximum sector-wise polar residual. Returns
    (A_dict, A_c_dict, B_dict).
    """
    A_c_dict = {k: v.copy() for k, v in T_dict.items()}
    A_dict = None
    B_dict = None

    for _ in range(max_iter):
        A_dict, A_c_dict = left_normalize(A_c_dict, allowed_quadruples)
        B_dict, A_c_dict = right_normalize(A_c_dict, allowed_quadruples)

        error = 0.0
        for key, A_c in A_c_dict.items():
            err_left = np.linalg.norm(
                np.eye(A_c.shape[3]) - oe.contract('aijb, aijc -> bc', A_c, A_c.conj())
            )
            err_right = np.linalg.norm(
                np.eye(A_c.shape[0]) - oe.contract('aijb, cijb -> ac', A_c, A_c.conj())
            )
            error = max(error, err_left, err_right)
        if error < tol:
            break
    else:
        raise RuntimeError(
            f"central_gauge did not converge in {max_iter} iterations (last error={error:.3e})"
        )

    if A_dict is None or B_dict is None:
        raise ValueError("internal error: A_dict or B_dict not assigned in central_gauge")
    return A_dict, A_c_dict, B_dict


# =====================================================================
# Per-sector SVD redefinition
# =====================================================================

def svd_redefining(A_dict, A_c_dict, B_dict, allowed_quadruples=None, tol=1e-9):
    """
    Per-sector SVD redistribution of the central tensor.

    For each sector key, reshapes A_c[key] to a matrix and runs SVD
    block-wise. The unitary factors are absorbed back into A and B
    via per-sector tensordot. Sector keys are independent of one
    another — no global block-matrix assembly.

    Returns (A_clean_dict, sigma_clean_dict, B_clean_dict).
    """
    keys = allowed_quadruples if allowed_quadruples is not None else A_c_dict.keys()
    A_clean_dict = {}
    sigma_clean_dict = {}
    B_clean_dict = {}

    for key in keys:
        A = A_dict[key]
        A_c = A_c_dict[key]
        B = B_dict[key]

        a, d1, d2, b = A_c.shape
        A_c_mat = A_c.reshape(a * d1 * d2, b)
        U, sigma, Vh = svd(A_c_mat, full_matrices=False)
        rank = sigma.shape[0]
        U_tensor = U.reshape(a, d1, d2, rank)

        A_updated = np.tensordot(A, U_tensor, axes=1)
        B_updated = np.tensordot(Vh, B, axes=1)

        A_clean = np.where(np.abs(A_updated) < tol, 0.0, A_updated)
        sigma_clean = np.where(np.abs(sigma) < tol, 0.0, sigma)
        B_clean = np.where(np.abs(B_updated) < tol, 0.0, B_updated)

        A_clean_dict[key] = A_clean
        sigma_clean_dict[key] = sigma_clean
        B_clean_dict[key] = B_clean

    return A_clean_dict, sigma_clean_dict, B_clean_dict


# =====================================================================
# Smoke test on Q-state Potts (chi=1, every sector is (1,1,1,1))
# =====================================================================

def _smoke_test(Q=2):
    print(f"=== central-gauge.py smoke test (Q={Q} Potts) ===")
    data = build_q5_potts(Q=Q, verbose=False)
    T_dict = data['T_dict']
    allowed_quadruples = data['allowed_quadruples']
    print(f"Loaded T_dict with {len(T_dict)} sectors, "
          f"shape per sector = {next(iter(T_dict.values())).shape}")

    A_dict, A_c_dict, B_dict = central_gauge(T_dict, allowed_quadruples)
    print(f"central_gauge converged: "
          f"A sectors = {len(A_dict)}, "
          f"A_c sectors = {len(A_c_dict)}, "
          f"B sectors = {len(B_dict)}")

    # At chi=1 the polar U should be a unit phase per sector.
    max_unit_dev = 0.0
    for key, A_c in A_c_dict.items():
        max_unit_dev = max(max_unit_dev, abs(abs(A_c.flat[0]) - 1.0))
    print(f"max ||A_c[k]| - 1| across sectors = {max_unit_dev:.3e}  "
          f"(expected 0 for chi=1: every isometry is a unit phase)")

    A_clean_dict, sigma_clean_dict, B_clean_dict = svd_redefining(
        A_dict, A_c_dict, B_dict, allowed_quadruples
    )
    print(f"svd_redefining produced {len(sigma_clean_dict)} sigma vectors")

    max_amp_diff = 0.0
    for key, T in T_dict.items():
        diff = float(np.max(np.abs(A_clean_dict[key].reshape(T.shape) - T)))
        if diff > max_amp_diff:
            max_amp_diff = diff
    print(f"max |A_clean - T_dict| across sectors = {max_amp_diff:.3e}  "
          f"(expected ~0: at chi=1 the amplitude lives in A_clean)")

    max_sigma_dev = 0.0
    for key, sig in sigma_clean_dict.items():
        max_sigma_dev = max(max_sigma_dev, abs(float(sig[0]) - 1.0))
    print(f"max |sigma - 1| across sectors = {max_sigma_dev:.3e}  "
          f"(expected 0: chi=1 SVD is the trivial 1x1 case)")
    print()


def _smoke_test_chi_gt_1():
    """
    chi > 1 sanity check: build a random per-sector dict at chi=4,
    run central_gauge + svd_redefining, verify the polar isometries
    are exact (right and left contractions give identity).
    """
    print("=== central-gauge.py smoke test (random chi=4, 6 sectors) ===")
    rng = np.random.default_rng(seed=20260416)
    chi = 4
    d1 = d2 = 2
    keys = [(0, 0, 0, 0), (0, 0, 1, 1), (1, 1, 0, 0),
            (1, 1, 1, 1), (0, 1, 1, 0), (1, 0, 0, 1)]
    T_dict = {
        k: (rng.standard_normal((chi, d1, d2, chi))
            + 1j * rng.standard_normal((chi, d1, d2, chi))) / np.sqrt(2)
        for k in keys
    }

    A_dict, A_c_dict, B_dict = central_gauge(T_dict, allowed_quadruples=keys)
    print(f"central_gauge converged for {len(A_dict)} chi={chi} sectors")

    max_left_dev = 0.0
    max_right_dev = 0.0
    for key, A_c in A_c_dict.items():
        left = oe.contract('aijb, aijc -> bc', A_c, A_c.conj())
        right = oe.contract('aijb, cijb -> ac', A_c, A_c.conj())
        max_left_dev = max(max_left_dev, np.linalg.norm(left - np.eye(left.shape[0])))
        max_right_dev = max(max_right_dev, np.linalg.norm(right - np.eye(right.shape[0])))
    print(f"max ||A_c^dag A_c - I|| = {max_left_dev:.3e}   "
          f"max ||A_c A_c^dag - I|| = {max_right_dev:.3e}")
    print(f"  (both expected ~0 — A_c is bi-unitary after iterative polar)")

    A_clean_dict, sigma_clean_dict, B_clean_dict = svd_redefining(
        A_dict, A_c_dict, B_dict, allowed_quadruples=keys
    )
    print(f"svd_redefining produced {len(sigma_clean_dict)} sigma vectors of shape "
          f"{next(iter(sigma_clean_dict.values())).shape}")
    print()


if __name__ == "__main__":
    _smoke_test(Q=2)
    _smoke_test(Q=3)
    _smoke_test_chi_gt_1()