"""
Vectorized Loop-TNR engine using JAX.

Same physics as the explicit code in loss_function_and_gradient.py and
rg_iteration.py, but with stacked arrays + jnp.einsum + jax.jit for
speedup. Supports chi >= 1.

Usage:
    engine = LoopTNREngine(Q=2, chi=4)
    engine.set_beta(0.881374)
    engine.rg_step()
    betas, f, C = engine.free_energy_scan(betas, n_steps=8)
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit
from scipy.optimize import minimize as scipy_minimize
from functools import partial

from potts_q5_hamiltonian import (
    build_zq_f_symbols, build_boltzmann_weights_at_beta, get_model_T,
    get_allowed_pairs_from_T, get_allowed_quadruples,
    get_allowed_nonuples, is_nonuple_consistent_with_quadruples,
)
# No scipy gauge-fixing dependency — gauge fix is pure JAX (see _gauge_fix_jax)


# =====================================================================
# Einsum strings (same as loss_function_and_gradient.py)
# =====================================================================

# U contraction: X(i,l,u,x) Y(m,n,u,v) Z(p,q,v,w) W(s,t,x,w)
U_EIN = 'ilux,mnuv,pqvw,stxw->ilmnqpst'
V_EIN = 'ltux,imuv,npvw,sqxw->ilmnqpst'

# Gradient einsums: contract diff(8-leg) with 3 remaining duals
# Remove P(l,t,u,x): diff contracted with Q(i,m,u,v), R(n,p,v,w), S(s,q,x,w)
GRAD_P_EIN = 'ilmnqpst,imuv,npvw,sqxw->ltux'
GRAD_Q_EIN = 'ilmnqpst,ltux,npvw,sqxw->imuv'
GRAD_R_EIN = 'ilmnqpst,ltux,imuv,sqxw->npvw'
GRAD_S_EIN = 'ilmnqpst,ltux,imuv,npvw->sqxw'


class LoopTNREngine:
    """Vectorized Loop-TNR engine. Supports chi >= 1.

    At chi=1: scalar multiply (fastest).
    At chi>1: batched jnp.einsum with 4-leg tensor contractions.
    Gradients hand-derived (same formulas as explicit code).
    """

    def __init__(self, Q, chi=1, chunk_size=None, bucket_chunked=False):
        """chunk_size: if set, run _compute_loss_and_grad via jax.lax.scan
        over nonuple-chunks of this size. Required for chi>=4 at Q=3
        where the fused batched einsum otherwise materializes
        (n_nonus, chi^8) buffers (~10GB per tensor) and OOMs.

        bucket_chunked: if True, run _compute_loss_and_grad via
        jax.lax.scan over OUTPUT buckets. Needed when chi is large
        enough that the (n_output, chi^8) accumulator itself exceeds
        memory (e.g. Q=2 chi=8 → 34 GB/tensor). Bucket chunking never
        allocates the full accumulator: each bucket iteration builds
        U_bucket, V_bucket (chi^8), computes diff + loss contribution
        + gradient contributions, then discards the buffers. Takes
        precedence over chunk_size when both are set.
        """
        self.Q = Q
        self.chi = chi
        self.chunk_size = chunk_size
        self.bucket_chunked = bucket_chunked
        self.beta_c = float(np.log(1 + np.sqrt(Q)))

        # F-symbols (beta-independent)
        self.F, self.C = build_zq_f_symbols(Q)

        # Module infrastructure (beta-independent)
        dummy_boltz = build_boltzmann_weights_at_beta(Q, self.beta_c)
        dummy_T = get_model_T(self.F, dummy_boltz, trace_physical=True)

        modules = list(range(Q))
        allowed_pairs, vert_pairs = get_allowed_pairs_from_T(dummy_T)
        allowed_quads = get_allowed_quadruples(allowed_pairs, vert_pairs)
        allowed_nonus = get_allowed_nonuples(allowed_pairs, vert_pairs, modules)
        consistent = [n for n in allowed_nonus
                      if is_nonuple_consistent_with_quadruples(allowed_quads, *n)]

        self.allowed_quadruples = sorted(allowed_quads)
        self.consistent_nonuples = consistent
        self.n_quads = len(self.allowed_quadruples)
        self.n_nonus = len(consistent)

        self._quad_to_idx = {q: i for i, q in enumerate(self.allowed_quadruples)}

        # Index maps: nonuple → tensor array indices
        x_idx = np.empty(self.n_nonus, dtype=np.int32)
        y_idx = np.empty(self.n_nonus, dtype=np.int32)
        z_idx = np.empty(self.n_nonus, dtype=np.int32)
        w_idx = np.empty(self.n_nonus, dtype=np.int32)
        p_idx = np.empty(self.n_nonus, dtype=np.int32)
        q_idx = np.empty(self.n_nonus, dtype=np.int32)
        r_idx = np.empty(self.n_nonus, dtype=np.int32)
        s_idx = np.empty(self.n_nonus, dtype=np.int32)

        output_keys_set = set()
        for A, B, C, D, E, F, G, H, I in consistent:
            output_keys_set.add((A, B, C, D, F, G, H, I))
        self.output_keys = sorted(output_keys_set)
        self.n_output = len(self.output_keys)
        self._out_to_idx = {k: i for i, k in enumerate(self.output_keys)}

        out_idx = np.empty(self.n_nonus, dtype=np.int32)

        for n, (A, B, C, D, E, F, G, H, I) in enumerate(consistent):
            x_idx[n] = self._quad_to_idx[(A, B, D, E)]
            y_idx[n] = self._quad_to_idx[(B, C, E, F)]
            z_idx[n] = self._quad_to_idx[(E, F, H, I)]
            w_idx[n] = self._quad_to_idx[(D, E, G, H)]
            p_idx[n] = self._quad_to_idx[(A, D, E, G)]
            q_idx[n] = self._quad_to_idx[(B, A, C, E)]
            r_idx[n] = self._quad_to_idx[(C, E, F, I)]
            s_idx[n] = self._quad_to_idx[(E, G, I, H)]
            out_idx[n] = self._out_to_idx[(A, B, C, D, F, G, H, I)]

        self.x_idx = jnp.array(x_idx)
        self.y_idx = jnp.array(y_idx)
        self.z_idx = jnp.array(z_idx)
        self.w_idx = jnp.array(w_idx)
        self.p_idx = jnp.array(p_idx)
        self.q_idx = jnp.array(q_idx)
        self.r_idx = jnp.array(r_idx)
        self.s_idx = jnp.array(s_idx)
        self.out_idx = jnp.array(out_idx)

        # Padded index arrays for chunked loss/grad via jax.lax.scan.
        # Pad gather-indices with 0 (valid slot, junk data dropped at sentinel
        # scatter row); pad scatter-indices with a sentinel slot appended past
        # the real data (n_quads for grad-scatter, n_output for U/V-scatter).
        # The sentinel row is sliced off after scan.
        if chunk_size is not None:
            import math
            self.n_chunks = math.ceil(self.n_nonus / chunk_size)
            self.n_padded = self.n_chunks * chunk_size

            def _pad(arr_np, fill):
                out = np.full(self.n_padded, fill, dtype=np.int32)
                out[:self.n_nonus] = arr_np
                return jnp.array(out)

            # Gather indices (pad with 0 — valid slot, contribution discarded
            # because scatter targets the sentinel row).
            self.x_idx_pad = _pad(x_idx, 0)
            self.y_idx_pad = _pad(y_idx, 0)
            self.z_idx_pad = _pad(z_idx, 0)
            self.w_idx_pad = _pad(w_idx, 0)
            self.p_idx_gather_pad = _pad(p_idx, 0)
            self.q_idx_gather_pad = _pad(q_idx, 0)
            self.r_idx_gather_pad = _pad(r_idx, 0)
            self.s_idx_gather_pad = _pad(s_idx, 0)
            # out_idx acts as scatter (pass 1) AND gather (pass 2).
            # Sentinel = n_output in both roles: pass-1 scatter dumps padded
            # rows into the sentinel slot; pass-2 gather on the sentinel slot
            # reads the zero row we append to diff_flat.
            self.out_idx_pad = _pad(out_idx, self.n_output)
            # Scatter indices for gradient accumulation: sentinel = n_quads.
            self.p_idx_scatter_pad = _pad(p_idx, self.n_quads)
            self.q_idx_scatter_pad = _pad(q_idx, self.n_quads)
            self.r_idx_scatter_pad = _pad(r_idx, self.n_quads)
            self.s_idx_scatter_pad = _pad(s_idx, self.n_quads)

        # Bucket table for output-axis chunking. Each output key k has
        # bucket_nonuples[k] = list of nonuple indices that scatter-add into
        # its (chi^8) slot. Pad to max_bucket so the bucket-scan body has
        # uniform shape. For Z_Q the bucket size is exactly Q (empirically
        # uniform at Q=2,3), so padding is a no-op in practice.
        if bucket_chunked:
            bucket_nonuples = [[] for _ in range(self.n_output)]
            for n in range(self.n_nonus):
                bucket_nonuples[int(out_idx[n])].append(n)
            max_bucket = max(len(b) for b in bucket_nonuples)
            self.max_bucket = max_bucket

            # Padded gather / scatter tables, shape (n_output, max_bucket).
            # Gather slots default to index 0 (valid, contribution masked off).
            # Scatter slots default to n_quads sentinel (row sliced off).
            bx = np.zeros((self.n_output, max_bucket), dtype=np.int32)
            by = np.zeros_like(bx); bz = np.zeros_like(bx); bw = np.zeros_like(bx)
            bpg = np.zeros_like(bx); bqg = np.zeros_like(bx)
            brg = np.zeros_like(bx); bsg = np.zeros_like(bx)
            bps = np.full_like(bx, self.n_quads)
            bqs = np.full_like(bx, self.n_quads)
            brs = np.full_like(bx, self.n_quads)
            bss = np.full_like(bx, self.n_quads)
            bmask = np.zeros((self.n_output, max_bucket), dtype=np.bool_)

            for k in range(self.n_output):
                for slot, n in enumerate(bucket_nonuples[k]):
                    bx[k, slot]  = x_idx[n]
                    by[k, slot]  = y_idx[n]
                    bz[k, slot]  = z_idx[n]
                    bw[k, slot]  = w_idx[n]
                    bpg[k, slot] = p_idx[n]; bps[k, slot] = p_idx[n]
                    bqg[k, slot] = q_idx[n]; bqs[k, slot] = q_idx[n]
                    brg[k, slot] = r_idx[n]; brs[k, slot] = r_idx[n]
                    bsg[k, slot] = s_idx[n]; bss[k, slot] = s_idx[n]
                    bmask[k, slot] = True

            self.bucket_x  = jnp.array(bx)
            self.bucket_y  = jnp.array(by)
            self.bucket_z  = jnp.array(bz)
            self.bucket_w  = jnp.array(bw)
            self.bucket_pg = jnp.array(bpg)
            self.bucket_qg = jnp.array(bqg)
            self.bucket_rg = jnp.array(brg)
            self.bucket_sg = jnp.array(bsg)
            self.bucket_ps = jnp.array(bps)
            self.bucket_qs = jnp.array(bqs)
            self.bucket_rs = jnp.array(brs)
            self.bucket_ss = jnp.array(bss)
            self.bucket_mask = jnp.array(bmask)

        # Tensor arrays: shape (n_quads, chi, chi, chi, chi)
        self.X = None
        self.Y = None
        self.Z = None
        self.W = None
        self.P = None
        self.Q_tens = None
        self.R = None
        self.S = None

        print(f"LoopTNREngine(Q={Q}, chi={chi}): {self.n_quads} quads, "
              f"{self.n_nonus} nonuples, {self.n_output} output keys")

    def set_beta(self, beta):
        """Set temperature. Rebuilds tensor values from Boltzmann weights.

        At chi=1: trace_physical=True → (1,1,1,1) scalars per sector.
        At chi>1: trace_physical=False → (Q,Q,Q,Q) tensors with physical
                  indices exposed. Actual entanglement structure, not noise.
                  Padded to (chi,chi,chi,chi) if chi > Q.
        """
        self.beta = beta
        boltz = build_boltzmann_weights_at_beta(self.Q, beta)

        chi = self.chi
        if chi == 1:
            T_dict = get_model_T(self.F, boltz, trace_physical=True)
        else:
            T_dict = get_model_T(self.F, boltz, trace_physical=False)

        T_arr = np.zeros((self.n_quads, chi, chi, chi, chi))
        for i, q in enumerate(self.allowed_quadruples):
            # get_model_T drops sectors whose max|value| < 1e-15. At extreme
            # beta (e.g. beta=100) whole sectors fall below that threshold;
            # leave those entries at zero instead of KeyError-ing out.
            if q not in T_dict:
                continue
            block = T_dict[q].real
            d = min(block.shape[0], chi)
            T_arr[i, :d, :d, :d, :d] = block[:d, :d, :d, :d]

        T_jax = jnp.array(T_arr)
        self.X = T_jax
        self.Y = T_jax.copy()
        self.Z = T_jax.copy()
        self.W = T_jax.copy()

        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, 4)
        noise = 1e-3 * float(jnp.max(jnp.abs(T_jax))) 
        
        self.P      = T_jax + noise * jax.random.normal(keys[0], T_jax.shape, dtype=T_jax.dtype)
        self.Q_tens = T_jax + noise * jax.random.normal(keys[1], T_jax.shape, dtype=T_jax.dtype)
        self.R      = T_jax + noise * jax.random.normal(keys[2], T_jax.shape, dtype=T_jax.dtype)
        self.S      = T_jax + noise * jax.random.normal(keys[3], T_jax.shape, dtype=T_jax.dtype)

    # =================================================================
    # Core JIT-compiled operations
    # =================================================================

    @partial(jit, static_argnums=(0,))
    def _compute_loss_and_grad(self, X, Y, Z, W, P, Q, R, S):
        """Loss + hand-derived gradient. Works at any chi.

        Batched gather → einsum → scatter-add.
        Gradient formulas identical to build_grads in explicit code.
        """
        chi = self.chi

        # Gather: shape (n_nonus, chi, chi, chi, chi)
        Xb = X[self.x_idx]; Yb = Y[self.y_idx]
        Zb = Z[self.z_idx]; Wb = W[self.w_idx]
        Pb = P[self.p_idx]; Qb = Q[self.q_idx]
        Rb = R[self.r_idx]; Sb = S[self.s_idx]

        # Batched contraction: add N (batch) dimension to einsum
        # U: 'Nilux,Nmnuv,Npqvw,Nstxw -> Nilmnqpst'
        U_batch = jnp.einsum('Nilux,Nmnuv,Npqvw,Nstxw->Nilmnqpst',
                             Xb, Yb, Zb, Wb)
        V_batch = jnp.einsum('Nltux,Nimuv,Nnpvw,Nsqxw->Nilmnqpst',
                             Pb, Qb, Rb, Sb)

        # Flatten spatial dims for scatter-add: (n_nonus, chi^8)
        flat_shape = (self.n_nonus, chi**8)
        U_flat = U_batch.reshape(flat_shape)
        V_flat = V_batch.reshape(flat_shape)

        # Accumulate into output keys
        out_shape = (self.n_output, chi**8)
        U_out = jnp.zeros(out_shape).at[self.out_idx].add(U_flat)
        V_out = jnp.zeros(out_shape).at[self.out_idx].add(V_flat)

        diff_flat = U_out - V_out
        loss = 0.5 * jnp.sum(diff_flat ** 2)

        # Gather diff back to per-nonuple: (n_nonus, chi^8)
        diff_per = diff_flat[self.out_idx]
        neg_diff = -jnp.conj(diff_per)

        # Reshape to 8-leg tensor: (n_nonus, chi, chi, chi, chi, chi, chi, chi, chi)
        d8 = neg_diff.reshape(self.n_nonus, chi, chi, chi, chi, chi, chi, chi, chi)

        # Hand-derived gradients (same einsums as explicit code, with N batch dim)
        gP_batch = jnp.einsum('Nilmnqpst,Nimuv,Nnpvw,Nsqxw->Nltux', d8, Qb, Rb, Sb)
        gQ_batch = jnp.einsum('Nilmnqpst,Nltux,Nnpvw,Nsqxw->Nimuv', d8, Pb, Rb, Sb)
        gR_batch = jnp.einsum('Nilmnqpst,Nltux,Nimuv,Nsqxw->Nnpvw', d8, Pb, Qb, Sb)
        gS_batch = jnp.einsum('Nilmnqpst,Nltux,Nimuv,Nnpvw->Nsqxw', d8, Pb, Qb, Rb)

        # Scatter-add gradients: (n_quads, chi, chi, chi, chi)
        z4 = jnp.zeros_like(P)
        grad_P = z4.at[self.p_idx].add(gP_batch)
        grad_Q = z4.at[self.q_idx].add(gQ_batch)
        grad_R = z4.at[self.r_idx].add(gR_batch)
        grad_S = z4.at[self.s_idx].add(gS_batch)

        return loss, grad_P, grad_Q, grad_R, grad_S

    @partial(jit, static_argnums=(0,))
    def _compute_loss_and_grad_chunked(self, X, Y, Z, W, P, Q, R, S):
        """Same math as `_compute_loss_and_grad`, run in chunks of
        `self.chunk_size` over the nonuple axis via `jax.lax.scan`.

        Peak working buffer drops from (n_nonus, chi^8) to
        (chunk_size, chi^8). Required at chi>=4 (fused path OOMs at 43GB
        for Q=3 chi=4).

        Sentinel rows: padded nonuples scatter into extra output slots
        (row n_output for U/V; row n_quads for gradients). After scan
        those rows are sliced off. Padded gather indices use slot 0
        (junk data, harmless because the scatter target is the sentinel).
        For pass-2 diff gather we append a zero row at slot n_output so
        the sentinel produces a zero contribution.
        """
        chi = self.chi
        cs = self.chunk_size
        n_chunks = self.n_chunks
        chi8 = chi ** 8
        dtype = X.dtype

        # -------- Pass 1: accumulate U_out, V_out --------
        U_out0 = jnp.zeros((self.n_output + 1, chi8), dtype=dtype)
        V_out0 = jnp.zeros((self.n_output + 1, chi8), dtype=dtype)

        def body1(carry, ci):
            Uo, Vo = carry
            start = ci * cs
            xs = jax.lax.dynamic_slice_in_dim(self.x_idx_pad, start, cs)
            ys = jax.lax.dynamic_slice_in_dim(self.y_idx_pad, start, cs)
            zs = jax.lax.dynamic_slice_in_dim(self.z_idx_pad, start, cs)
            ws = jax.lax.dynamic_slice_in_dim(self.w_idx_pad, start, cs)
            ps = jax.lax.dynamic_slice_in_dim(self.p_idx_gather_pad, start, cs)
            qs = jax.lax.dynamic_slice_in_dim(self.q_idx_gather_pad, start, cs)
            rs = jax.lax.dynamic_slice_in_dim(self.r_idx_gather_pad, start, cs)
            ss_ = jax.lax.dynamic_slice_in_dim(self.s_idx_gather_pad, start, cs)
            os = jax.lax.dynamic_slice_in_dim(self.out_idx_pad, start, cs)

            Xb = X[xs]; Yb = Y[ys]; Zb = Z[zs]; Wb = W[ws]
            Pb = P[ps]; Qb = Q[qs]; Rb = R[rs]; Sb = S[ss_]

            Uc = jnp.einsum('Nilux,Nmnuv,Npqvw,Nstxw->Nilmnqpst',
                            Xb, Yb, Zb, Wb)
            Vc = jnp.einsum('Nltux,Nimuv,Nnpvw,Nsqxw->Nilmnqpst',
                            Pb, Qb, Rb, Sb)

            Uf = Uc.reshape(cs, chi8)
            Vf = Vc.reshape(cs, chi8)

            Uo = Uo.at[os].add(Uf)
            Vo = Vo.at[os].add(Vf)
            return (Uo, Vo), None

        (U_out, V_out), _ = jax.lax.scan(
            body1, (U_out0, V_out0), jnp.arange(n_chunks))

        # Drop sentinel row, compute diff + loss
        diff_real = (U_out - V_out)[:-1]                # (n_output, chi^8)
        loss = 0.5 * jnp.sum(diff_real ** 2)

        # Append zero row so sentinel gather produces zero contribution
        zero_row = jnp.zeros((1, chi8), dtype=dtype)
        diff_padded = jnp.concatenate([diff_real, zero_row], axis=0)

        # -------- Pass 2: accumulate gradients --------
        gP0 = jnp.zeros((self.n_quads + 1, chi, chi, chi, chi), dtype=dtype)
        gQ0 = jnp.zeros_like(gP0)
        gR0 = jnp.zeros_like(gP0)
        gS0 = jnp.zeros_like(gP0)

        def body2(carry, ci):
            gP, gQ, gR, gS = carry
            start = ci * cs
            ps_g = jax.lax.dynamic_slice_in_dim(self.p_idx_gather_pad, start, cs)
            qs_g = jax.lax.dynamic_slice_in_dim(self.q_idx_gather_pad, start, cs)
            rs_g = jax.lax.dynamic_slice_in_dim(self.r_idx_gather_pad, start, cs)
            ss_g = jax.lax.dynamic_slice_in_dim(self.s_idx_gather_pad, start, cs)
            os_g = jax.lax.dynamic_slice_in_dim(self.out_idx_pad, start, cs)
            ps_s = jax.lax.dynamic_slice_in_dim(self.p_idx_scatter_pad, start, cs)
            qs_s = jax.lax.dynamic_slice_in_dim(self.q_idx_scatter_pad, start, cs)
            rs_s = jax.lax.dynamic_slice_in_dim(self.r_idx_scatter_pad, start, cs)
            ss_s = jax.lax.dynamic_slice_in_dim(self.s_idx_scatter_pad, start, cs)

            Pb = P[ps_g]; Qb = Q[qs_g]; Rb = R[rs_g]; Sb = S[ss_g]

            diff_c = diff_padded[os_g]                  # (cs, chi^8)
            d8 = diff_c.reshape(cs, chi, chi, chi, chi, chi, chi, chi, chi)
            neg_d8 = -jnp.conj(d8)

            gP_c = jnp.einsum('Nilmnqpst,Nimuv,Nnpvw,Nsqxw->Nltux',
                              neg_d8, Qb, Rb, Sb)
            gQ_c = jnp.einsum('Nilmnqpst,Nltux,Nnpvw,Nsqxw->Nimuv',
                              neg_d8, Pb, Rb, Sb)
            gR_c = jnp.einsum('Nilmnqpst,Nltux,Nimuv,Nsqxw->Nnpvw',
                              neg_d8, Pb, Qb, Sb)
            gS_c = jnp.einsum('Nilmnqpst,Nltux,Nimuv,Nnpvw->Nsqxw',
                              neg_d8, Pb, Qb, Rb)

            gP = gP.at[ps_s].add(gP_c)
            gQ = gQ.at[qs_s].add(gQ_c)
            gR = gR.at[rs_s].add(gR_c)
            gS = gS.at[ss_s].add(gS_c)
            return (gP, gQ, gR, gS), None

        (gP, gQ, gR, gS), _ = jax.lax.scan(
            body2, (gP0, gQ0, gR0, gS0), jnp.arange(n_chunks))

        # Drop sentinel slot
        return loss, gP[:-1], gQ[:-1], gR[:-1], gS[:-1]

    @partial(jit, static_argnums=(0,))
    def _compute_loss_and_grad_bucket(self, X, Y, Z, W, P, Q, R, S):
        """Bucket-chunked loss + gradient.

        Scans over the n_output axis; each iteration handles one output
        bucket. Never allocates the full (n_output, chi^8) accumulator.
        Required at large chi where that accumulator itself exceeds RAM
        (Q=2 chi=8 → 34 GB per tensor).

        Per-iteration working set:
            U_batch / V_batch  : (max_bucket, chi^8)   contraction of
                                  (X,Y,Z,W) and (P,Q,R,S) duals for the
                                  nonuples contributing to this bucket
            U_bucket / V_bucket: (chi^8,) summed across the bucket
            diff               : (chi^8,) gives the bucket's loss + grad
                                  contribution.

        Gradient fan-out: diff is the same for every nonuple in the
        bucket, so each slot's grad contribution differs only through
        the other three duals (Qb, Rb, Sb etc.) and the scatter target
        (p_idx, q_idx, r_idx, s_idx).
        """
        chi = self.chi
        mb = self.max_bucket
        dtype = X.dtype

        gP0 = jnp.zeros((self.n_quads + 1, chi, chi, chi, chi), dtype=dtype)
        gQ0 = jnp.zeros_like(gP0)
        gR0 = jnp.zeros_like(gP0)
        gS0 = jnp.zeros_like(gP0)
        loss0 = jnp.asarray(0.0, dtype=dtype)

        def body(carry, k):
            loss, gP, gQ, gR, gS = carry
            xs  = self.bucket_x[k];  ys  = self.bucket_y[k]
            zs  = self.bucket_z[k];  ws  = self.bucket_w[k]
            pg  = self.bucket_pg[k]; qg  = self.bucket_qg[k]
            rg_ = self.bucket_rg[k]; sg  = self.bucket_sg[k]
            ps  = self.bucket_ps[k]; qs  = self.bucket_qs[k]
            rs_ = self.bucket_rs[k]; ss  = self.bucket_ss[k]
            mask = self.bucket_mask[k].astype(dtype)   # (mb,)

            Xb = X[xs]; Yb = Y[ys]; Zb = Z[zs]; Wb = W[ws]
            Pb = P[pg]; Qb = Q[qg]; Rb = R[rg_]; Sb = S[sg]

            # Batched 8-leg contractions (mb, chi^8-reshaped)
            Uc = jnp.einsum('Nilux,Nmnuv,Npqvw,Nstxw->Nilmnqpst',
                            Xb, Yb, Zb, Wb)
            Vc = jnp.einsum('Nltux,Nimuv,Nnpvw,Nsqxw->Nilmnqpst',
                            Pb, Qb, Rb, Sb)

            # Mask padded rows so they contribute nothing
            m8 = mask.reshape(mb, 1, 1, 1, 1, 1, 1, 1, 1)
            Uc = Uc * m8
            Vc = Vc * m8

            # Sum across bucket slots → (chi, chi, chi, chi, chi, chi, chi, chi)
            U_bucket = jnp.sum(Uc, axis=0)
            V_bucket = jnp.sum(Vc, axis=0)
            diff = U_bucket - V_bucket

            loss = loss + 0.5 * jnp.sum(diff ** 2)

            # Per-slot grad contributions. diff has no N dim; broadcast
            # across N via the einsum subscripts (no 'N' on neg_diff).
            neg_diff = -jnp.conj(diff)

            gP_N = jnp.einsum('ilmnqpst,Nimuv,Nnpvw,Nsqxw->Nltux',
                              neg_diff, Qb, Rb, Sb)
            gQ_N = jnp.einsum('ilmnqpst,Nltux,Nnpvw,Nsqxw->Nimuv',
                              neg_diff, Pb, Rb, Sb)
            gR_N = jnp.einsum('ilmnqpst,Nltux,Nimuv,Nsqxw->Nnpvw',
                              neg_diff, Pb, Qb, Sb)
            gS_N = jnp.einsum('ilmnqpst,Nltux,Nimuv,Nnpvw->Nsqxw',
                              neg_diff, Pb, Qb, Rb)

            m4 = mask.reshape(mb, 1, 1, 1, 1)
            gP_N = gP_N * m4
            gQ_N = gQ_N * m4
            gR_N = gR_N * m4
            gS_N = gS_N * m4

            gP = gP.at[ps].add(gP_N)
            gQ = gQ.at[qs].add(gQ_N)
            gR = gR.at[rs_].add(gR_N)
            gS = gS.at[ss].add(gS_N)
            return (loss, gP, gQ, gR, gS), None

        (loss, gP, gQ, gR, gS), _ = jax.lax.scan(
            body, (loss0, gP0, gQ0, gR0, gS0), jnp.arange(self.n_output))

        return loss, gP[:-1], gQ[:-1], gR[:-1], gS[:-1]

    def _call_loss_and_grad(self, X, Y, Z, W, P, Q, R, S):
        """Dispatch: bucket > nonuple-chunk > fused."""
        if self.bucket_chunked:
            return self._compute_loss_and_grad_bucket(X, Y, Z, W, P, Q, R, S)
        if self.chunk_size is None:
            return self._compute_loss_and_grad(X, Y, Z, W, P, Q, R, S)
        return self._compute_loss_and_grad_chunked(X, Y, Z, W, P, Q, R, S)

    def loss(self):
        loss, _, _, _, _ = self._call_loss_and_grad(
            self.X, self.Y, self.Z, self.W,
            self.P, self.Q_tens, self.R, self.S)
        return float(loss)

    def loss_and_grad(self):
        loss, gP, gQ, gR, gS = self._call_loss_and_grad(
            self.X, self.Y, self.Z, self.W,
            self.P, self.Q_tens, self.R, self.S)
        return float(loss), gP, gQ, gR, gS

    # =================================================================
    # Optimizer
    # =================================================================

    def optimize(self, maxiter=2000, tol=1e-12, verbose=False):
        """L-BFGS on dual projectors PQRS."""
        def pack(*arrays):
            return np.concatenate([np.array(a).ravel() for a in arrays])

        n_per = self.n_quads * self.chi**4

        def unpack(x):
            arrs = []
            for i in range(4):
                arrs.append(jnp.array(
                    x[i*n_per:(i+1)*n_per].reshape(
                        self.n_quads, self.chi, self.chi, self.chi, self.chi)))
            return tuple(arrs)

        x0 = pack(self.P, self.Q_tens, self.R, self.S)

        def objective(x):
            P, Q, R, S = unpack(x)
            loss, gP, gQ, gR, gS = self._call_loss_and_grad(
                self.X, self.Y, self.Z, self.W, P, Q, R, S)
            grad = pack(gP, gQ, gR, gS)
            return float(loss), grad.astype(np.float64)

        result = scipy_minimize(
            objective, x0, method='L-BFGS-B', jac=True,
            options={'maxiter': maxiter, 'ftol': tol, 'gtol': 1e-10,
                     'disp': verbose})

        self.P, self.Q_tens, self.R, self.S = unpack(result.x)
        return result

    # =================================================================
    # Normalization and gauge fixing
    # =================================================================

    def normalize(self):
        """Rescale all tensors by max primal element. Returns ln(scale)."""
        max_val = float(jnp.max(jnp.abs(jnp.concatenate(
            [self.X.ravel(), self.Y.ravel(), self.Z.ravel(), self.W.ravel()]))))
        if max_val < 1e-300:
            return 0.0
        ln_scale = np.log(max_val)
        for name in ('X', 'Y', 'Z', 'W', 'P', 'Q_tens', 'R', 'S'):
            setattr(self, name, getattr(self, name) / max_val)
        return ln_scale

    @partial(jit, static_argnums=(0,))
    def _gauge_fix_jax(self, T_arr):
        """Central-gauge fixing with accumulated polar factors.

        For a 4-leg tensor T[a, d1, d2, b] the central gauge writes
            T = L @ A_c @ R
        with A_c bi-unitary (both left-reshape and right-reshape are isometries).
        L and R are obtained by iterating left/right polar decompositions on
        A_c. Crucially, the factors across ALL iterations must be accumulated —
        otherwise the reconstruction is not a gauge transformation but loses
        information about T.

        Algorithm (per-sector, batched over n_quads):
            1. Initialize A_c = T, L = I, R = I.
            2. For N iterations:
               a. Left polar: A_c = P_l @ U_l. Update A_c := U_l, L := L @ P_l.
               b. Right polar: A_c = U_r @ P_r. Update A_c := U_r, R := P_r @ R.
            3. SVD the final (bi-unitary) A_c = U Σ V†.
            4. Absorb U into L and V† into R:
                 A_clean = L @ U     (as 4-leg tensor via reshape)
                 B_clean = V† @ R
            5. Reconstruct T_gauged = A_clean @ diag(Σ) @ B_clean.

        Result: T_gauged equals the input T up to numerical noise (pure gauge
        op), but is expressed in central-gauge form — singular values on the
        central bond are non-negative and sorted, so the canonical form is
        uniquely defined.
        """
        chi = self.chi
        n = self.n_quads

        # Batched identity on the two gauge bonds
        I_batch = jnp.broadcast_to(jnp.eye(chi, dtype=T_arr.dtype),
                                   (n, chi, chi))

        def body(_, carry):
            A_c, L, R = carry

            # --- Left polar: A_c = P_l @ U_l on matricization (chi, chi^3) ---
            mat_l = A_c.reshape(n, chi, chi**3)
            Ul, Sl, Vhl = jnp.linalg.svd(mat_l, full_matrices=False)
            # P_l = U_l @ diag(S) @ U_l^†  (hermitian positive)
            P_l = jnp.einsum('nar,nr,nbr->nab', Ul, Sl, jnp.conj(Ul))
            A_c = (Ul @ Vhl).reshape(n, chi, chi, chi, chi)
            # Accumulate LEFT on the LEFT: L_new = L_old @ P_l
            L = jnp.einsum('nab,nbc->nac', L, P_l)

            # --- Right polar: A_c = U_r @ P_r on matricization (chi^3, chi) --
            mat_r = A_c.reshape(n, chi**3, chi)
            Ur, Sr, Vhr = jnp.linalg.svd(mat_r, full_matrices=False)
            # P_r = V @ diag(S) @ V^†  (hermitian positive)
            # With V_r s.t. V_r^† = Vhr (shape (chi, chi)),
            #   V_r[a, r] = conj(Vhr[r, a]), so
            #   P_r[a, b] = Σ_r V_r[a, r] S[r] conj(V_r[b, r])
            #            = Σ_r conj(Vhr[r, a]) S[r] Vhr[r, b]
            P_r = jnp.einsum('nra,nr,nrb->nab', jnp.conj(Vhr), Sr, Vhr)
            A_c = (Ur @ Vhr).reshape(n, chi, chi, chi, chi)
            # Accumulate RIGHT on the RIGHT: R_new = P_r @ R_old
            R = jnp.einsum('nab,nbc->nac', P_r, R)

            return (A_c, L, R)

        A_c_final, L_total, R_total = jax.lax.fori_loop(
            0, 50, body, (T_arr, I_batch, I_batch))

        # SVD the final bi-unitary A_c in the right-matricization
        A_c_mat = A_c_final.reshape(n, chi**3, chi)
        U_f, sigma_f, Vh_f = jnp.linalg.svd(A_c_mat, full_matrices=False)

        # Absorb U into L (acts on the leftmost leg 'a' of the 4-leg tensor)
        # and V† into R (acts on the rightmost leg 'b').
        # U_f shape (n, chi^3, chi) → reshape to (n, chi, chi, chi, chi) with
        # the first chi being the outer (left) leg.
        U_4leg = U_f.reshape(n, chi, chi, chi, chi)
        # A_clean[n,a,b,c,r] = L[n,a,x] * U_4leg[n,x,b,c,r]
        A_clean = jnp.einsum('nax,nxbcr->nabcr', L_total, U_4leg)
        # B_clean[n,r,b] = Vh_f[n,r,y] * R[n,y,b]
        B_clean = jnp.einsum('nry,nyb->nrb', Vh_f, R_total)

        # Zero tiny singular values (keeps the reconstruction numerically clean
        # without discarding physically meaningful modes)
        sigma_safe = jnp.where(jnp.abs(sigma_f) > 1e-12, sigma_f, 0.0)

        # T_gauged[n,a,b,c,d] = Σ_r A_clean[n,a,b,c,r] * sigma[n,r] * B_clean[n,r,d]
        T_gauged = jnp.einsum('nabcr,nr,nrd->nabcd',
                              A_clean, sigma_safe, B_clean)
        return T_gauged

    def rotate(self, gauge_fix=True):
        """Promote optimized duals to primals. Gauge fix via JAX SVD."""
        new_X = self.P
        new_Y = self.Q_tens
        new_Z = self.R
        new_W = self.S

        if gauge_fix:
            new_X = self._gauge_fix_jax(new_X)
            new_Y = self._gauge_fix_jax(new_Y)
            new_Z = self._gauge_fix_jax(new_Z)
            new_W = self._gauge_fix_jax(new_W)

        self.X = new_X
        self.Y = new_Y
        self.Z = new_Z
        self.W = new_W
        self.P = new_X.copy()
        self.Q_tens = new_Y.copy()
        self.R = new_Z.copy()
        self.S = new_W.copy()

    def rg_step(self, maxiter=50, gauge_fix=True, verbose=False):
        """One RG step: optimize + rotate + gauge fix (all JAX)."""
        result = self.optimize(maxiter=maxiter, verbose=verbose)
        eps2 = self.loss()
        self.rotate(gauge_fix=gauge_fix)
        return result, eps2

    def rg_flow(self, n_steps=10, maxiter=50, verbose=True):
        """Full RG iteration."""
        import time
        history = []
        for step in range(n_steps):
            eps2_before = self.loss()
            t0 = time.time()
            result, eps2_after = self.rg_step(maxiter=maxiter)
            dt = time.time() - t0
            history.append((step, eps2_before, eps2_after, result.nit, dt))
            if verbose:
                print(f"  step {step:2d}: eps2 {eps2_before:.3e} -> {eps2_after:.3e}  "
                      f"({result.nit} iters, {dt:.2f}s)")
        return history

    def free_energy_scan(self, betas=None, n_steps=8, maxiter=50, verbose=True):
        """Free energy f(beta), energy E(beta), specific heat C(beta)."""
        if betas is None:
            betas = np.linspace(0.3 * self.beta_c, 1.7 * self.beta_c, 41)

        if verbose:
            print(f"Free energy scan: Q={self.Q}, chi={self.chi}, "
                  f"beta_c={self.beta_c:.6f}")
            print(f"  {len(betas)} betas, {n_steps} RG steps each\n")

        f_values = np.empty(len(betas))

        for idx, beta in enumerate(betas):
            self.set_beta(beta)
            ln_Z = 0.0
            for step in range(n_steps):
                ln_scale = self.normalize()
                ln_Z += ln_scale / 4**step
                self.rg_step(maxiter=maxiter)

            ln_scale_final = self.normalize()
            ln_Z += ln_scale_final / 4**n_steps
            f_values[idx] = -ln_Z / beta

            if verbose and (idx % 10 == 0 or idx == len(betas) - 1):
                print(f"  [{idx+1:3d}/{len(betas)}] beta={beta:.5f}  "
                      f"f={f_values[idx]:.8f}")

        bf = betas * f_values
        energy = np.gradient(bf, betas)
        specific_heat = np.gradient(energy, betas)

        peak_idx = np.argmax(np.abs(specific_heat))
        if verbose:
            print(f"\n  Specific heat peak at beta = {betas[peak_idx]:.6f}")
            print(f"  Exact beta_c = {self.beta_c:.6f}")
            print(f"  Relative error = "
                  f"{abs(betas[peak_idx] - self.beta_c)/self.beta_c:.4f}")

        return betas, f_values, specific_heat

    # =================================================================
    # Warm-started flow + bidirectional beta sweep
    # =================================================================

    def _final_grad_norm(self):
        """L2 norm of (gP, gQ, gR, gS) at current state.

        Used as a post-L-BFGS diagnostic: tiny if the optimizer converged
        below gtol, larger if it hit the maxiter cap (critical slowing down).
        """
        _, gP, gQ, gR, gS = self._call_loss_and_grad(
            self.X, self.Y, self.Z, self.W,
            self.P, self.Q_tens, self.R, self.S)
        total = jnp.concatenate([gP.ravel(), gQ.ravel(),
                                 gR.ravel(), gS.ravel()])
        return float(jnp.linalg.norm(total))

    def run_full_tnr_flow(self, beta, init_unitaries, num_rg_steps,
                          maxiter=2000, gauge_fix=True, tol=1e-13):
        """One beta, multi-step RG flow with PyTree warm-start.

        Interface (as requested):
            final_tensor, prev_tensor, optimized_unitaries,
            total_lbfgs_iters, final_grad_norm = run_full_tnr_flow(
                beta, init_unitaries_pytree, num_rg_steps, maxiter)

        - `final_tensor`  : primal X after the final RG step (normalized)
        - `prev_tensor`   : primal X entering the final RG step (normalized)
                            → Delta = ||final − prev||_F measures the fixed-point
                            fidelity (small ⇒ scale-invariant flow, large ⇒
                            not yet at a fixed point)
        - `optimized_unitaries` : `(P, Q_tens, R, S)` tuple — PyTree for warm
                            starting the next beta
        - `total_lbfgs_iters` : sum over all RG steps; spikes at beta_c from
                            critical slowing down
        - `final_grad_norm` : ‖∇ loss‖ at the end of the LAST L-BFGS call
                            (fallback diagnostic when maxiter is clipped)

        If `init_unitaries is None`, duals are left at the plaquette-tensor
        values set by `set_beta` (cold start). Otherwise the 4-tuple is
        unpacked into `self.P, self.Q_tens, self.R, self.S` to warm-start the
        optimizer at this beta.
        """
        self.set_beta(beta)

        if init_unitaries is not None:
            P0, Q0, R0, S0 = init_unitaries
            expected_shape = (self.n_quads, self.chi, self.chi,
                              self.chi, self.chi)
            for name, arr in (('P', P0), ('Q', Q0), ('R', R0), ('S', S0)):
                if tuple(arr.shape) != expected_shape:
                    raise ValueError(
                        f"init_unitaries[{name}] shape {tuple(arr.shape)} "
                        f"!= expected {expected_shape}")
            self.P = jnp.asarray(P0)
            self.Q_tens = jnp.asarray(Q0)
            self.R = jnp.asarray(R0)
            self.S = jnp.asarray(S0)

        # Scale to O(1). Done AFTER overriding duals so the rescale applies
        # uniformly (normalize() divides every tensor by max-primal).
        self.normalize()

        total_iters = 0
        prev_tensor = self.X

        for step in range(num_rg_steps):
            # Snapshot primal entering this RG step (will be the "prev" for
            # the last step's fidelity metric).
            prev_tensor = self.X

            # Optimize P,Q,R,S → min ‖U−V‖², then rotate (+ gauge_fix).
            result, _ = self.rg_step(maxiter=maxiter, gauge_fix=gauge_fix)
            total_iters += int(result.nit)

            # Keep tensors O(1) between RG depths so ‖X^(n) − X^(n-1)‖ is
            # comparable across steps and across betas.
            self.normalize()

        final_tensor = self.X
        final_grad_norm = self._final_grad_norm()

        optimized_unitaries = (self.P, self.Q_tens, self.R, self.S)
        return (final_tensor, prev_tensor, optimized_unitaries,
                total_iters, final_grad_norm)

    def beta_sweep_bidirectional(self, betas, num_rg_steps=6, maxiter=2000,
                                  gauge_fix=True, tol=1e-13,
                                  init_unitaries=None, verbose=True):
        """Forward-backward beta sweep to locate beta_c without hysteresis.

        Passes the PyTree of optimized duals between adjacent beta in both
        directions. Returns a dict with the three requested metrics computed
        separately for the forward (low→high beta) and backward (high→low beta)
        passes, plus averages and a hysteresis diagnostic.

        Metrics (per beta, per direction):
            delta       : ‖X^(N) − X^(N-1)‖_F on normalized primals
            total_iters : Σ L-BFGS iterations across the N RG steps
            grad_norms  : ‖∇ loss‖ at the end of the last L-BFGS call
            eps2        : loss at the end of the last L-BFGS call

        Usage:
            results = engine.beta_sweep_bidirectional(betas, num_rg_steps=6)
            # plot results['betas'] vs results['forward']['total_iters'], etc.
        """
        import time

        betas_sorted = np.asarray(sorted(np.asarray(betas).tolist()))
        n = len(betas_sorted)

        def empty_metrics():
            return {
                'delta':       np.empty(n),
                'total_iters': np.empty(n, dtype=np.int64),
                'grad_norms':  np.empty(n),
                'eps2':        np.empty(n),
            }

        fwd = empty_metrics()
        bwd = empty_metrics()

        if verbose:
            print(f"Bidirectional beta-sweep: Q={self.Q}, chi={self.chi}, "
                  f"{n} betas, {num_rg_steps} RG steps, maxiter={maxiter}")
            print(f"  beta_c (exact) = {self.beta_c:.6f}")

        # -------- Forward pass: low beta → high beta --------
        t_fwd = time.time()
        unitaries = init_unitaries
        if verbose:
            print(f"\n--- Forward pass (low beta -> high beta) ---")
            print(f"  {'beta':>8} | {'Delta':>10} | {'iters':>7} | "
                  f"{'|grad|':>10} | {'eps2':>10}")
            print("  " + "-" * 58)
        for i in range(n):
            fX, pX, unitaries, n_it, gnorm = self.run_full_tnr_flow(
                float(betas_sorted[i]), unitaries, num_rg_steps,
                maxiter=maxiter, gauge_fix=gauge_fix, tol=tol)
            fwd['delta'][i]       = float(jnp.linalg.norm(fX - pX))
            fwd['total_iters'][i] = n_it
            fwd['grad_norms'][i]  = gnorm
            fwd['eps2'][i]        = float(self.loss())
            if verbose:
                print(f"  {betas_sorted[i]:8.5f} | "
                      f"{fwd['delta'][i]:10.3e} | "
                      f"{fwd['total_iters'][i]:7d} | "
                      f"{fwd['grad_norms'][i]:10.3e} | "
                      f"{fwd['eps2'][i]:10.3e}")
        t_fwd = time.time() - t_fwd

        # -------- Backward pass: high beta → low beta --------
        # Start from the pytree at the END of the forward pass (continuity at
        # the high-beta endpoint), not from the final-beta-cold-start, so there's a
        # chance for hysteresis detection at the critical boundary.
        t_bwd = time.time()
        if verbose:
            print(f"\n--- Backward pass (high beta -> low beta) ---")
            print(f"  {'beta':>8} | {'Delta':>10} | {'iters':>7} | "
                  f"{'|grad|':>10} | {'eps2':>10}")
            print("  " + "-" * 58)
        for j in range(n - 1, -1, -1):
            fX, pX, unitaries, n_it, gnorm = self.run_full_tnr_flow(
                float(betas_sorted[j]), unitaries, num_rg_steps,
                maxiter=maxiter, gauge_fix=gauge_fix, tol=tol)
            bwd['delta'][j]       = float(jnp.linalg.norm(fX - pX))
            bwd['total_iters'][j] = n_it
            bwd['grad_norms'][j]  = gnorm
            bwd['eps2'][j]        = float(self.loss())
            if verbose:
                print(f"  {betas_sorted[j]:8.5f} | "
                      f"{bwd['delta'][j]:10.3e} | "
                      f"{bwd['total_iters'][j]:7d} | "
                      f"{bwd['grad_norms'][j]:10.3e} | "
                      f"{bwd['eps2'][j]:10.3e}")
        t_bwd = time.time() - t_bwd

        # Summary metrics
        avg_delta = 0.5 * (fwd['delta'] + bwd['delta'])
        avg_iters = 0.5 * (fwd['total_iters'].astype(float) +
                           bwd['total_iters'].astype(float))
        hysteresis = np.abs(fwd['delta'] - bwd['delta'])

        # beta_c indicators
        i_delta_min_fwd  = int(np.argmin(fwd['delta']))
        i_delta_min_bwd  = int(np.argmin(bwd['delta']))
        i_iter_peak_fwd  = int(np.argmax(fwd['total_iters']))
        i_iter_peak_bwd  = int(np.argmax(bwd['total_iters']))
        i_avg_iter_peak  = int(np.argmax(avg_iters))

        if verbose:
            print(f"\n=== beta_c indicators ===")
            print(f"  Exact beta_c                      : {self.beta_c:.6f}")
            print(f"  Delta(fwd) minimum at beta            : "
                  f"{betas_sorted[i_delta_min_fwd]:.6f}")
            print(f"  Delta(bwd) minimum at beta            : "
                  f"{betas_sorted[i_delta_min_bwd]:.6f}")
            print(f"  total_iters(fwd) peak at beta     : "
                  f"{betas_sorted[i_iter_peak_fwd]:.6f}")
            print(f"  total_iters(bwd) peak at beta     : "
                  f"{betas_sorted[i_iter_peak_bwd]:.6f}")
            print(f"  avg_total_iters peak at beta      : "
                  f"{betas_sorted[i_avg_iter_peak]:.6f}")
            print(f"  Forward pass wall-time         : {t_fwd:.1f}s")
            print(f"  Backward pass wall-time        : {t_bwd:.1f}s")

        return {
            'betas':            betas_sorted,
            'forward':          fwd,
            'backward':         bwd,
            'avg_delta':        avg_delta,
            'avg_total_iters':  avg_iters,
            'hysteresis':       hysteresis,
            'beta_c':           self.beta_c,
            'wall_time_fwd':    t_fwd,
            'wall_time_bwd':    t_bwd,
        }

    def beta_scan(self, betas=None, n_steps=4, maxiter=50, verbose=True):
        """Scan betas, return eps^2 history."""
        if betas is None:
            betas = np.linspace(0.5 * self.beta_c, 1.5 * self.beta_c, 21)

        results = {}
        for beta in betas:
            self.set_beta(beta)
            eps_history = []
            for step in range(n_steps):
                _, eps2 = self.rg_step(maxiter=maxiter)
                eps_history.append(eps2)
            results[beta] = eps_history

            if verbose:
                row = (f"  beta={beta:.5f}  " +
                       "  ".join(f"{e:.2e}" for e in eps_history))
                print(row)

        return results

    # =================================================================
    # Interop
    # =================================================================

    def to_dict_state(self):
        def to_dict(arr):
            return {q: np.array(arr[i]) for i, q in
                    enumerate(self.allowed_quadruples)}
        return {
            'Q': self.Q, 'chi': self.chi,
            'allowed_quadruples': set(self.allowed_quadruples),
            'consistent_nonuples': self.consistent_nonuples,
            'X': to_dict(self.X), 'Y': to_dict(self.Y),
            'Z': to_dict(self.Z), 'W': to_dict(self.W),
            'P': to_dict(self.P), 'Q_tens': to_dict(self.Q_tens),
            'R': to_dict(self.R), 'S': to_dict(self.S),
        }


# =====================================================================
# Smoke test
# =====================================================================

if __name__ == "__main__":
    import time
    from loss_function_and_gradient import setup_potts, compute_loss_from_state

    print("=== Chi=1 validation ===\n")
    engine = LoopTNREngine(Q=2, chi=1)
    engine.set_beta(engine.beta_c)

    state = setup_potts(Q_potts=2, verbose=False)
    explicit_loss = compute_loss_from_state(state)
    engine_loss = engine.loss()
    print(f"Explicit: {explicit_loss:.6e}  Engine: {engine_loss:.6e}  "
          f"Match: {abs(explicit_loss - engine_loss) < 1e-8}")

    result, eps2 = engine.rg_step()
    print(f"RG step: eps2={eps2:.3e}, {result.nit} iters\n")

    print("=== Chi=2 test (physical tensors, JAX gauge fix) ===\n")
    engine2 = LoopTNREngine(Q=2, chi=2)
    engine2.set_beta(engine2.beta_c)
    print(f"Initial loss: {engine2.loss():.6e}")

    t0 = time.time()
    result, eps2 = engine2.rg_step(maxiter=200)
    dt = time.time() - t0
    print(f"RG step 1: eps2={eps2:.3e}, {result.nit} iters, {dt:.2f}s (JIT compile)")

    t0 = time.time()
    result, eps2 = engine2.rg_step(maxiter=200)
    dt = time.time() - t0
    print(f"RG step 2: eps2={eps2:.3e}, {result.nit} iters, {dt:.2f}s (JIT warm)")

    print(f"\n4 RG steps at chi=2, beta_c:")
    engine2.set_beta(engine2.beta_c)
    history = engine2.rg_flow(n_steps=4, maxiter=200)

    print("\n=== Chi=2 normalized beta scan (21 betas, 3 steps) ===\n")
    betas_test = np.linspace(0.5 * engine2.beta_c, 1.5 * engine2.beta_c, 21)
    eps_s2 = []
    for beta in betas_test:
        engine2.set_beta(beta)
        eps_list = []
        for step in range(3):
            engine2.normalize()
            _, eps2 = engine2.rg_step(maxiter=200)
            eps_list.append(eps2)
        eps_s2.append(eps_list[2])
        marker = ' <--' if abs(beta - engine2.beta_c) < (betas_test[1]-betas_test[0])*0.6 else ''
        print(f"  beta={beta:.5f}  s0={eps_list[0]:.4e}  s1={eps_list[1]:.4e}  s2={eps_list[2]:.4e}{marker}")

    d_eps = np.gradient(np.array(eps_s2), betas_test)
    peak = np.argmax(np.abs(d_eps[2:-2])) + 2
    print(f"\nStep-2 |d(eps2)/d(beta)| peak: beta = {betas_test[peak]:.6f}")
    print(f"Exact beta_c = {engine2.beta_c:.6f}")
    print(f"Relative error = {abs(betas_test[peak] - engine2.beta_c)/engine2.beta_c:.4f}")