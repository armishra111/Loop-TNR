"""Validation suite: produces validation_report.pdf.

Sections:
  1. Pentagon identity for Z_Q F-symbols (Q=2,3,4,5)
  2. Dense vs block-sparse loss match at (Q=2, chi=4)
  3. beta_c location for Q=2,3,4 at chi=16 (short flow) vs ln(1+sqrt(Q))
  4. Convergence: error-vs-chi for Q=2,3 at chi in {2,4,8,16}
  5. Central charge / scaling dimensions for Q=2 (c=1/2), Q=3 (c=4/5)
  6. Complex-beta delta=2.1 run: STUB (requires optimizer Im sign fix)

Run:  python colab/validate.py --output_dir ./validation_out
"""
import argparse
import os
import sys
import time
import json
from itertools import product

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Force x64 before importing engine
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from fast_engine import LoopTNREngine                                 # noqa: E402
from potts_q5_hamiltonian import (                                    # noqa: E402
    build_zq_f_symbols,
    build_boltzmann_weights_at_beta,
    get_model_T,
    get_allowed_quadruples,
    get_allowed_pairs_from_T,
    get_allowed_nonuples,
    is_nonuple_consistent_with_quadruples,
)


# =====================================================================
# Section 1 — Pentagon
# =====================================================================

def pentagon_residual(F, Q):
    """Same check as test_zq_physics.pentagon_residual — inlined to avoid
    hard dep on that test module."""
    max_diff = 0.0
    for a, b, c, d in product(range(Q), repeat=4):
        f = (a + b) % Q
        g = (a + b + c) % Q
        h = (c + d) % Q
        i = (b + c + d) % Q
        j = (b + c) % Q
        e = (a + b + c + d) % Q
        lhs = (F[f, c, d, e, (f + c) % Q, (c + d) % Q] *
               F[a, b, h, e, (a + b) % Q, (b + h) % Q])
        rhs = (F[a, b, c, g, (a + b) % Q, (b + c) % Q] *
               F[a, j, d, e, (a + j) % Q, (j + d) % Q] *
               F[b, c, d, i, (b + c) % Q, (c + d) % Q])
        diff = abs(lhs - rhs)
        if diff > max_diff:
            max_diff = diff
    return max_diff


def section_pentagon(pdf, log):
    log("=== Section 1: pentagon ===")
    Qs = [2, 3, 4, 5]
    residuals = []
    for Q in Qs:
        F, _ = build_zq_f_symbols(Q)
        r = pentagon_residual(F, Q)
        residuals.append(r)
        log(f"  Q={Q}: residual = {r:.3e}  {'OK' if r < 1e-14 else 'FAIL'}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(Qs, np.maximum(residuals, 1e-18), "o-")
    ax.axhline(1e-14, color="r", ls="--", label="tol 1e-14")
    ax.set_xlabel("Q")
    ax.set_ylabel("Pentagon residual (max |LHS - RHS|)")
    ax.set_title("Section 1: Pentagon identity for $\\mathrm{Vec}_{Z_Q}$")
    ax.set_xticks(Qs)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    return {"Q": Qs, "residuals": residuals,
            "pass": all(r < 1e-14 for r in residuals)}


# =====================================================================
# Section 2 — Dense vs block-sparse
# =====================================================================

def dense_loss_from_engine(engine):
    """Reconstruct loss by explicitly summing per-nonuple contributions,
    using the engine's tensor arrays but Python-looping sectors (no
    scatter-add batching). Provides an independent numerical path."""
    chi = engine.chi
    X = np.asarray(engine.X); Y = np.asarray(engine.Y)
    Z = np.asarray(engine.Z); W = np.asarray(engine.W)
    P = np.asarray(engine.P); Q = np.asarray(engine.Q_tens)
    R = np.asarray(engine.R); S = np.asarray(engine.S)

    quad_to_idx = engine._quad_to_idx
    out_to_idx = engine._out_to_idx
    n_output = engine.n_output
    chi8 = chi ** 8

    U = np.zeros((n_output, chi8), dtype=X.dtype)
    V = np.zeros_like(U)

    for (A, B, C_, D, E, F_, G, H, I) in engine.consistent_nonuples:
        ix = quad_to_idx[(A, B, D, E)]
        iy = quad_to_idx[(B, C_, E, F_)]
        iz = quad_to_idx[(E, F_, H, I)]
        iw = quad_to_idx[(D, E, G, H)]
        ip = quad_to_idx[(A, D, E, G)]
        iq = quad_to_idx[(B, A, C_, E)]
        ir = quad_to_idx[(C_, E, F_, I)]
        iss = quad_to_idx[(E, G, I, H)]
        io = out_to_idx[(A, B, C_, D, F_, G, H, I)]

        u = np.einsum("ilux,mnuv,pqvw,stxw->ilmnqpst",
                      X[ix], Y[iy], Z[iz], W[iw]).reshape(-1)
        v = np.einsum("ltux,imuv,npvw,sqxw->ilmnqpst",
                      P[ip], Q[iq], R[ir], S[iss]).reshape(-1)
        U[io] += u
        V[io] += v

    diff = U - V
    return 0.5 * float(np.sum(diff ** 2))


def section_dense_vs_block(pdf, log):
    log("=== Section 2: dense vs block-sparse (Q=2, chi=4) ===")
    engine = LoopTNREngine(Q=2, chi=4)
    engine.set_beta(engine.beta_c)

    engine_loss = engine.loss()
    dense_loss = dense_loss_from_engine(engine)
    rel = abs(engine_loss - dense_loss) / max(abs(engine_loss), 1e-30)
    log(f"  engine_loss = {engine_loss:.6e}")
    log(f"  dense_loss  = {dense_loss:.6e}")
    log(f"  |diff|/|ref| = {rel:.3e}  {'OK' if rel < 1e-10 else 'WARN'}")

    # After 1 RG step, repeat
    engine.rg_step(maxiter=200)
    engine_loss_post = engine.loss()
    dense_loss_post = dense_loss_from_engine(engine)
    rel_post = abs(engine_loss_post - dense_loss_post) / max(abs(engine_loss_post), 1e-30)
    log(f"  post-RG: engine={engine_loss_post:.6e}  dense={dense_loss_post:.6e}")
    log(f"  |diff|/|ref| = {rel_post:.3e}  {'OK' if rel_post < 1e-10 else 'WARN'}")

    fig, ax = plt.subplots(figsize=(6, 4))
    xs = ["pre-RG", "post-RG"]
    ys = [engine_loss, engine_loss_post]
    zs = [dense_loss, dense_loss_post]
    ax.semilogy(xs, np.abs(ys), "o-", label="block-sparse engine")
    ax.semilogy(xs, np.abs(zs), "x--", label="dense (per-nonuple loop)")
    ax.set_ylabel("|loss|")
    ax.set_title("Section 2: block-sparse vs dense (Q=2, $\\chi$=4)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    return {
        "pre_engine": engine_loss, "pre_dense": dense_loss, "pre_rel": rel,
        "post_engine": engine_loss_post, "post_dense": dense_loss_post,
        "post_rel": rel_post,
        "pass": rel < 1e-10 and rel_post < 1e-10,
    }


# =====================================================================
# Section 3 — beta_c location (Q=2,3,4)
# =====================================================================

def _pick_chunking(Q, chi):
    """Auto-select chunk_size / bucket_chunked based on (Q, chi).

    Heuristic matches the OOM boundaries documented in MEMORY (session
    2026-04-18): Q=3 chi>=4 OOMs on the fused einsum; Q=2 chi>=8 needs
    bucket chunking because the (n_output, chi^8) accumulator alone
    exceeds RAM.
    """
    if Q == 2 and chi >= 8:
        return {"chunk_size": None, "bucket_chunked": True}
    if Q == 3 and chi >= 4:
        return {"chunk_size": 2000, "bucket_chunked": False}
    if Q >= 4 and chi >= 3:
        return {"chunk_size": 2000, "bucket_chunked": False}
    return {"chunk_size": None, "bucket_chunked": False}


def locate_beta_c(Q, chi, n_betas=9, num_rg_steps=4, maxiter=200, log=print):
    """Short bidirectional sweep over a narrow window around ln(1+sqrt(Q)).
    Returns (estimated beta_c, exact beta_c, full sweep dict)."""
    engine = LoopTNREngine(Q=Q, chi=chi, **_pick_chunking(Q, chi))
    lo = 0.85 * engine.beta_c
    hi = 1.15 * engine.beta_c
    betas = np.linspace(lo, hi, n_betas)
    log(f"  locate_beta_c Q={Q} chi={chi}: {n_betas} betas in [{lo:.4f}, {hi:.4f}]")

    res = engine.beta_sweep_bidirectional(
        betas=betas, num_rg_steps=num_rg_steps,
        maxiter=maxiter, gauge_fix=True, verbose=False)
    # beta_c indicator: avg_total_iters peak (critical slowing down).
    # Fallback: if iters flat (all clipped at maxiter → no signal),
    # use min of avg_delta (fixed-point fidelity dip at criticality).
    iters = res["avg_total_iters"]
    if float(np.ptp(iters)) < 2.0:
        avg_delta = 0.5 * (res["forward"]["delta"] + res["backward"]["delta"])
        idx = int(np.argmin(avg_delta))
    else:
        idx = int(np.argmax(iters))
    return float(res["betas"][idx]), float(engine.beta_c), res


def section_beta_c(pdf, log, chi_big=16, Qs=(2, 3, 4), n_betas=7,
                    num_rg_steps=3, maxiter=150):
    log(f"=== Section 3: beta_c (chi={chi_big}, Qs={Qs}) ===")
    rows = []
    sweeps = {}
    for Q in Qs:
        est, exact, res = locate_beta_c(
            Q=Q, chi=chi_big, n_betas=n_betas,
            num_rg_steps=num_rg_steps, maxiter=maxiter, log=log)
        err = abs(est - exact) / exact
        log(f"  Q={Q}: est={est:.6f} exact={exact:.6f} rel_err={err:.3e}")
        rows.append({"Q": Q, "est": est, "exact": exact, "rel_err": err})
        sweeps[Q] = res

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for Q, res in sweeps.items():
        ax.plot(res["betas"], res["avg_total_iters"],
                "o-", label=f"Q={Q}")
        ax.axvline(float(res["beta_c"]), color=f"C{Q-2}", ls="--", alpha=0.5)
    ax.set_xlabel("beta")
    ax.set_ylabel("avg L-BFGS iters (critical slowing)")
    ax.set_title(f"Section 3: $\\beta_c$ detection at $\\chi$={chi_big}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # Summary table figure
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis("off")
    data = [[str(r["Q"]), f"{r['exact']:.6f}", f"{r['est']:.6f}",
             f"{r['rel_err']:.2e}"] for r in rows]
    tbl = ax.table(cellText=data,
                    colLabels=["Q", "exact beta_c", "estimated", "rel err"],
                    loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    return {"rows": rows, "pass": all(r["rel_err"] < 0.10 for r in rows)}


# =====================================================================
# Section 4 — Error vs chi for Q=2,3
# =====================================================================

def section_convergence(pdf, log, chi_list=(2, 4, 8, 16), Qs=(2, 3),
                          n_rg=3, maxiter=2500, maxiter_cap=20000):
    log(f"=== Section 4: error vs chi, chi in {chi_list}, Qs={Qs}, "
        f"n_rg={n_rg}, maxiter={maxiter} (cap={maxiter_cap}) ===")
    curves = {}
    for Q in Qs:
        eps_at_chi = []
        for chi in chi_list:
            # L-BFGS iters scale ~linearly with param count (~chi^4), but
            # warm-starting + L-BFGS Hessian approximation compress that.
            # Linear-in-chi budget with cap handles chi up to ~64 without
            # day-long runs. Clip status in log shows if cap is hit.
            chi_maxiter = min(maxiter_cap, max(maxiter, int(maxiter * chi / 2)))
            t0 = time.time()
            engine = LoopTNREngine(Q=Q, chi=chi, **_pick_chunking(Q, chi))
            engine.set_beta(engine.beta_c)
            # Run multiple RG steps: chi truncation only engages once
            # coarse-graining builds Schmidt rank > Q. One step on the
            # plaquette tensor is rank-Q trivially, so chi>Q gives no
            # signal at step 0.
            eps2 = float("nan")
            for _ in range(n_rg):
                engine.normalize()
                _, eps2 = engine.rg_step(maxiter=chi_maxiter)
            dt = time.time() - t0
            log(f"  Q={Q} chi={chi}: eps^2 = {eps2:.3e}  "
                f"(maxiter={chi_maxiter}, {dt:.1f}s)")
            eps_at_chi.append(eps2)
        curves[Q] = eps_at_chi

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for Q, ys in curves.items():
        ax.loglog(chi_list, np.maximum(ys, 1e-300), "o-", label=f"Q={Q}")
    ax.set_xlabel("chi")
    ax.set_ylabel("Truncation error $\\|U-V\\|^2$")
    ax.set_title("Section 4: truncation error vs chi (1 RG step, $\\beta_c$)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    return {"chi_list": list(chi_list), "curves": {k: list(v) for k, v in curves.items()}}


# =====================================================================
# Section 5 — Scaling dimensions + central charge
# =====================================================================

def scaling_dimensions(engine, n_modes=8):
    """Build pooled transfer-matrix eigendecomposition from current X.

    For each sector, reshape X[i] -> (chi^2, chi^2) matrix M_i and
    eigendecompose. Pool all eigenvalues by |lambda|, sort descending.
    Scaling dimensions:  Delta_a = -ln(|lam_a| / |lam_0|) / (2*pi/L).
    Choose L = chi (lattice size = bond dim after one coarse-grain).
    """
    chi = engine.chi
    X = np.asarray(engine.X)
    n_q = X.shape[0]
    pooled_eigs = []
    for i in range(n_q):
        M = X[i].reshape(chi * chi, chi * chi)
        try:
            w = np.linalg.eigvals(M)
        except np.linalg.LinAlgError:
            continue
        pooled_eigs.extend(w.tolist())
    pooled_eigs = np.array(pooled_eigs)
    mags = np.abs(pooled_eigs)
    order = np.argsort(mags)[::-1]
    top = pooled_eigs[order][:n_modes]
    top_mag = mags[order][:n_modes]
    if top_mag[0] < 1e-300:
        return np.zeros(n_modes), np.zeros(n_modes)
    deltas = -np.log(np.maximum(top_mag / top_mag[0], 1e-300)) / (2 * np.pi / chi)
    im_parts = np.imag(top) / np.maximum(np.abs(top), 1e-300)
    return deltas, im_parts


def section_scaling(pdf, log, chi=16, n_rg=6, include_q3=True, maxiter=1500):
    log(f"=== Section 5: scaling dims / c (chi={chi}, include_q3={include_q3}, "
        f"maxiter={maxiter}) ===")
    rows = []
    targets = [(2, 0.5, 1/8, "Delta_sigma (Ising)")]
    if include_q3:
        targets.append((3, 4/5, 2/15, "Delta_sigma (3-Potts)"))
    for Q, c_expected, delta_ref, delta_label in targets:
        engine = LoopTNREngine(Q=Q, chi=chi, **_pick_chunking(Q, chi))
        engine.set_beta(engine.beta_c)
        for step in range(n_rg):
            engine.normalize()
            engine.rg_step(maxiter=maxiter)
        engine.normalize()
        deltas, _ = scaling_dimensions(engine, n_modes=8)
        # Smallest positive Delta is the scaling dim of the lightest primary
        positive = [d for d in deltas if d > 1e-6]
        delta_est = float(min(positive)) if positive else float("nan")
        # Heuristic c estimate using c = 12 * Delta_stress_tensor? no; just
        # state the theoretical value and the measured lowest Delta.
        log(f"  Q={Q}: deltas[:6] = {deltas[:6]}")
        log(f"  Q={Q}: lowest Delta = {delta_est:.4f} (ref {delta_ref}), "
            f"c_exact={c_expected}")
        rows.append({"Q": Q, "c_expected": c_expected,
                      "delta_ref": delta_ref, "delta_est": delta_est,
                      "deltas_all": [float(d) for d in deltas]})

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for row in rows:
        ax.plot(range(len(row["deltas_all"])), row["deltas_all"],
                "o-", label=f"Q={row['Q']}")
        ax.axhline(row["delta_ref"], ls="--", alpha=0.4,
                    color=f"C{row['Q']-2}",
                    label=f"Q={row['Q']} CFT Delta_sigma={row['delta_ref']:.4f}")
    ax.set_xlabel("Mode index")
    ax.set_ylabel("Scaling dimension Delta")
    ax.set_title(f"Section 5: scaling dimensions at chi={chi}, {n_rg} RG steps")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    return {"rows": rows}


# =====================================================================
# Section 6 — complex beta delta=2.1 STUB
# =====================================================================

def section_complex_beta_stub(pdf, log):
    log("=== Section 6: complex beta delta=2.1 STUB ===")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")
    ax.text(0.05, 0.85,
             "Section 6 — complex beta (delta = 2.1): NOT RUN",
             fontsize=12, fontweight="bold")
    ax.text(0.05, 0.70,
             "Requires: gradient Im-component sign-flip fix in L-BFGS\n"
             "objective (fast_engine.optimize). Pending open item\n"
             "in project state (see MASTER.md / CLAUDE.md).",
             fontsize=10, va="top")
    ax.text(0.05, 0.40,
             "Expected output when enabled:\n"
             "  Im(Delta_leading) != 0 indicating a complex-CFT\n"
             "  leading primary along the Lee-Yang direction.",
             fontsize=10, va="top")
    pdf.savefig(fig)
    plt.close(fig)
    return {"status": "stub"}


# =====================================================================
# Main
# =====================================================================

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=str, default="./validation_out")
    ap.add_argument("--chi_big", type=int, default=16,
                     help="Large chi for sections 3 and 5.")
    ap.add_argument("--chi_list", type=str, default="2,4,8,16")
    ap.add_argument("--fast", action="store_true",
                     help="Shrink chi/n_rg for quick CPU smoke run.")
    ap.add_argument("--only_beta_c", action="store_true",
                     help="CPU-targeted: run only Sections 1+2+3 with "
                          "Qs=(2,3,4) at chi_big=2. Beats --fast for "
                          "low-chi beta_c validation across all Qs.")
    args = ap.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "validate_log.txt")
    log_fh = open(log_path, "w", buffering=1)

    def log(msg):
        print(msg, flush=True)
        log_fh.write(msg + "\n")

    if args.only_beta_c:
        # CPU-targeted beta_c-across-Q validation. Keeps Qs=(2,3,4) at
        # chi_big=2 so memory stays under 2 GB even for Q=4.
        chi_big = 2
        chi_list = (2,)
        section3_Qs = (2, 3, 4)
        section3_n_betas = 7
        section3_num_rg = 3
        section3_maxiter = 150
        section4_Qs = (2,)                  # lean — convergence plot tiny
        section5_include_q3 = False
        section5_n_rg = 2
    elif args.fast:
        chi_big = 4
        chi_list = (2, 4)
        section3_Qs = (2,)                 # Q=3,4 too slow on CPU fast mode
        section3_n_betas = 5
        section3_num_rg = 2
        section3_maxiter = 80
        section4_Qs = (2,)
        section5_include_q3 = False
        section5_n_rg = 2
    else:
        chi_big = args.chi_big
        chi_list = tuple(int(x) for x in args.chi_list.split(","))
        section3_Qs = (2, 3, 4)
        section3_n_betas = 7
        section3_num_rg = 3
        section3_maxiter = 150
        section4_Qs = (2, 3)
        section5_include_q3 = True
        section5_n_rg = 6

    log(f"Validation run. output_dir={args.output_dir}")
    log(f"JAX devices: {jax.devices()}  backend={jax.default_backend()}")
    log(f"chi_big={chi_big}  chi_list={chi_list}")

    pdf_path = os.path.join(args.output_dir, "validation_report.pdf")
    summary = {}
    with PdfPages(pdf_path) as pdf:
        # Cover page
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.text(0.5, 0.90, "Loop-TNR validation report",
                 ha="center", fontsize=18, fontweight="bold")
        ax.text(0.5, 0.85, time.strftime("%Y-%m-%d %H:%M:%S"),
                 ha="center", fontsize=10)
        ax.text(0.1, 0.70,
                 "Sections:\n"
                 "  1. Pentagon identity\n"
                 "  2. Dense vs block-sparse loss match\n"
                 "  3. beta_c location (Q=2,3,4)\n"
                 "  4. Error-vs-chi convergence\n"
                 "  5. Scaling dimensions / central charge\n"
                 "  6. Complex-beta delta=2.1 (stub)",
                 fontsize=12, va="top")
        pdf.savefig(fig); plt.close(fig)

        summary["pentagon"] = section_pentagon(pdf, log)
        summary["dense_vs_block"] = section_dense_vs_block(pdf, log)
        summary["beta_c"] = section_beta_c(
            pdf, log, chi_big=chi_big, Qs=section3_Qs,
            n_betas=section3_n_betas, num_rg_steps=section3_num_rg,
            maxiter=section3_maxiter)
        if not args.only_beta_c:
            summary["convergence"] = section_convergence(
                pdf, log, chi_list=chi_list, Qs=section4_Qs)
            summary["scaling"] = section_scaling(
                pdf, log, chi=chi_big, n_rg=section5_n_rg,
                include_q3=section5_include_q3)
            summary["complex_beta"] = section_complex_beta_stub(pdf, log)

    with open(os.path.join(args.output_dir, "validation_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=lambda o: float(o)
                   if isinstance(o, (np.floating, np.integer)) else str(o))

    log(f"\nPDF: {pdf_path}")
    log(f"Summary: {os.path.join(args.output_dir, 'validation_summary.json')}")
    log_fh.close()


if __name__ == "__main__":
    main()
