// sparse_hessian.cpp
// Definitions of the sum-to-zero block Schur helpers declared in sparse_hessian.h.

#include <RcppEigen.h>
#include "sparse_hessian.h"

namespace tulpa {

void build_s2z_block_schur_cache(
    const SparseHessianBuilder& A,
    const std::vector<SparseHessianBuilder::S2ZRank1>& r1,
    S2ZBlockSchurCache& cache
) {
    const int n_x = A.n;
    const int K = static_cast<int>(r1.size());
    cache.floc.assign(n_x, -1);
    cache.sloc.assign(n_x, -1);
    std::vector<char> is_field(n_x, 0);
    for (int k = 0; k < K; ++k)
        for (int i = 0; i < r1[k].n; ++i) is_field[r1[k].node(i)] = 1;
    int nf = 0;
    for (int g = 0; g < n_x; ++g) if (is_field[g]) cache.floc[g] = nf++;
    int ns = 0;
    for (int g = 0; g < n_x; ++g) if (cache.floc[g] < 0) cache.sloc[g] = ns++;
    cache.nf = nf; cache.ns = ns;

    std::vector<std::pair<int,int>> ffpat;
    cache.dest_kind.assign(A.nnz, 0);
    cache.dest_a.assign(A.nnz, -1);
    cache.dest_b.assign(A.nnz, -1);
    int t = 0;
    for (int c = 0; c < n_x; ++c)
        for (int p = A.col_ptr[c]; p < A.col_ptr[c + 1]; ++p, ++t) {
            const int r = A.row_idx[p];
            const int fr = cache.floc[r], fc = cache.floc[c], sr = cache.sloc[r], sc = cache.sloc[c];
            if (fr >= 0 && fc >= 0)      { cache.dest_kind[t] = 0; ffpat.emplace_back(fr, fc); }
            else if (sr >= 0 && sc >= 0) { cache.dest_kind[t] = 1; cache.dest_a[t] = sr*ns+sc;
                                           cache.dest_b[t] = (sr != sc) ? sc*ns+sr : -1; }
            else if (fr >= 0 && sc >= 0) { cache.dest_kind[t] = 2; cache.dest_a[t] = fr*ns + sc; }
            else                          { cache.dest_kind[t] = 2; cache.dest_a[t] = fc*ns + sr; }
        }
    cache.aff.init(nf, ffpat);
    // Resolve field-field value slots now the A_FF pattern (entry_map) exists.
    t = 0;
    for (int c = 0; c < n_x; ++c)
        for (int p = A.col_ptr[c]; p < A.col_ptr[c + 1]; ++p, ++t)
            if (cache.dest_kind[t] == 0)
                cache.dest_a[t] = cache.aff.lookup(cache.floc[A.row_idx[p]], cache.floc[c]);

    cache.aff_solver.reset();
    cache.aff_ldl.reset();
    cache.aff_ldl.use_simplicial_ldl();
    cholmod_sparse view = cache.aff.as_cholmod(&cache.aff_solver.common());
    cache.aff_solver.analyze(&view);

    cache.n_x = n_x; cache.a_nnz = A.nnz;
    cache.block_layout.clear(); cache.block_layout.reserve(K);
    for (int k = 0; k < K; ++k) cache.block_layout.emplace_back(s2z_block_nodes(r1[k]));
    cache.built = true;
}

bool s2z_block_schur(
    const SparseHessianBuilder& A,
    const std::vector<SparseHessianBuilder::S2ZRank1>& r1,
    const double* grad,
    double* delta,
    double* logdet_out,
    S2ZBlockSchurCache* cache,
    bool field_inertia,
    bool* field_indefinite) {
    if (field_indefinite) *field_indefinite = false;
    const int K = static_cast<int>(r1.size());
    if (K == 0) return false;
    const int n_x = A.n;

    S2ZBlockSchurCache local_cache;
    S2ZBlockSchurCache& cc = cache ? *cache : local_cache;
    if (!cc.matches(A, r1)) build_s2z_block_schur_cache(A, r1, cc);
    const int nf = cc.nf, ns = cc.ns;
    const std::vector<int>& floc = cc.floc;
    const std::vector<int>& sloc = cc.sloc;

    // Scatter A's values into the cached A_FF pattern + dense A_ss / A_fs via the
    // precomputed per-nonzero destinations (no membership recompute, no re-sort).
    cc.aff.zero();
    double* __restrict__ affv = cc.aff.values.data();
    std::vector<double> Ass((std::size_t) ns * ns, 0.0);
    std::vector<double> Afs((std::size_t) nf * ns, 0.0);
    {
        int t = 0;
        for (int c = 0; c < n_x; ++c)
            for (int p = A.col_ptr[c]; p < A.col_ptr[c + 1]; ++p, ++t) {
                const double v = A.values[p];
                const int a = cc.dest_a[t];
                switch (cc.dest_kind[t]) {
                    case 0:  scatter_slot(affv, a, v); break;
                    case 1:  Ass[a] += v; if (cc.dest_b[t] >= 0) Ass[cc.dest_b[t]] += v; break;
                    default: Afs[a] += v; break;
                }
            }
    }

    cholmod_sparse vff = cc.aff.as_cholmod(&cc.aff_solver.common());
    if (!cc.aff_solver.analyzed()) cc.aff_solver.analyze(&vff);
    double logAFF = 0.0;
    int neg_aff = 0;
    SparseCholeskySolver* sfp = &cc.aff_solver;
    if (cc.aff_solver.factorize(&vff)) {
        logAFF = cc.aff_solver.log_determinant();
    } else {
        if (field_indefinite) *field_indefinite = true;
        if (!field_inertia) return false;
        if (!cc.aff_ldl.analyzed()) cc.aff_ldl.analyze(&vff);
        if (!cc.aff_ldl.factorize(&vff)) return false;
        if (!cc.aff_ldl.ldl_inertia(neg_aff, logAFF)) return false;
        // U C U' is PSD of rank K, so B_FF keeps at least neg(A_FF) - K
        // negative eigenvalues.
        if (neg_aff > K) return false;
        sfp = &cc.aff_ldl;
    }
    const bool indefinite = (sfp == &cc.aff_ldl);
    SparseCholeskySolver& sf = *sfp;

    // W = A_FF^-1 U (U = [1_k] local). cap = C^-1 + U'A_FF^-1 U (K x K, SPD).
    std::vector<std::vector<double>> W(K, std::vector<double>(nf, 0.0));
    {
        std::vector<double> uk(nf, 0.0);
        for (int k = 0; k < K; ++k) {
            std::fill(uk.begin(), uk.end(), 0.0);
            for (int i = 0; i < r1[k].n; ++i) uk[floc[r1[k].node(i)]] = 1.0;
            sf.solve(uk.data(), W[k].data(), nf);
        }
    }
    if (indefinite) {
        // Relative backward error of A_FF W_k = 1_k, componentwise against
        // |A_FF| |W_k| + |1_k|, over the stored lower triangle.
        const std::vector<int>& cp = cc.aff.col_ptr;
        const std::vector<int>& ri = cc.aff.row_idx;
        const std::vector<double>& av = cc.aff.values;
        std::vector<double> r(nf), mag(nf);
        for (int k = 0; k < K; ++k) {
            const std::vector<double>& wk = W[k];
            std::fill(r.begin(), r.end(), 0.0);
            std::fill(mag.begin(), mag.end(), 0.0);
            for (int i = 0; i < r1[k].n; ++i) {
                const int f = floc[r1[k].node(i)];
                r[f] = -1.0; mag[f] = 1.0;
            }
            for (int c = 0; c < nf; ++c)
                for (int p = cp[c]; p < cp[c + 1]; ++p) {
                    const int rr = ri[p];
                    const double v = av[p];
                    r[rr] += v * wk[c];  mag[rr] += std::fabs(v * wk[c]);
                    if (rr != c) { r[c] += v * wk[rr]; mag[c] += std::fabs(v * wk[rr]); }
                }
            double num = 0.0, den = 0.0;
            for (int i = 0; i < nf; ++i) {
                num = std::max(num, std::fabs(r[i]));
                den = std::max(den, mag[i]);
            }
            if (!(num <= S2Z_LDL_BACKWARD_TOL * den)) return false;
        }
    }
    // D enters only as D^{-1} in the cap and log|D| in the determinant, so a
    // Kronecker-coupled field costs nothing here beyond a dense K x K inverse:
    // no cross-block fill, no extra factorization.
    std::vector<double> Dinv; double logC = 0.0;
    if (!s2z_build_Dinv(r1, A.s2z_coupling, Dinv, logC)) return false;
    std::vector<double> cap((std::size_t) K * K, 0.0);
    for (int a = 0; a < K; ++a)
        for (int b = 0; b < K; ++b) {
            double m = 0.0;   // 1_a' W_b
            for (int i = 0; i < r1[a].n; ++i) m += W[b][floc[r1[a].node(i)]];
            cap[(std::size_t) a*K + b] = m + Dinv[(std::size_t) a * K + b];
        }
    // A PD field block gives an SPD cap; an indefinite one a cap whose negative
    // count has to match A_FF's for B_FF to be PD.
    SmallChol capL;
    SmallSymEig capE;
    double logcap = 0.0;
    if (!indefinite) {
        capL.factor(cap, K);
        if (!capL.ok) return false;
        logcap = capL.logdet();
    } else {
        capE.factor(cap, K);
        if (!capE.ok || capE.n_neg() != neg_aff) return false;
        logcap = capE.logabsdet();
    }
    auto cap_solve = [&](const double* b, double* x) {
        if (indefinite) capE.solve(b, x); else capL.solve(b, x);
    };
    const double logBFF = logAFF + logC + logcap;

    // Schur S = A_ss - A_sf B_FF^-1 A_fs, with B_FF^-1 y = A_FF^-1 y - W cap^-1 (W'y).
    // Z[j] = B_FF^-1 A_fs[:,j] is reused for the step's field back-substitution.
    std::vector<std::vector<double>> Z(ns, std::vector<double>(nf, 0.0));
    std::vector<double> S = Ass;
    {
        std::vector<double> col(nf), ainv(nf), wty(K), capy(K);
        for (int j = 0; j < ns; ++j) {
            for (int i = 0; i < nf; ++i) col[i] = Afs[(std::size_t) i*ns + j];
            sf.solve(col.data(), ainv.data(), nf);                 // A_FF^-1 A_fs[:,j]
            for (int k = 0; k < K; ++k) {                           // W' col = U' A_FF^-1 col
                double s = 0.0;
                for (int i = 0; i < nf; ++i) s += W[k][i] * col[i];
                wty[k] = s;
            }
            cap_solve(wty.data(), capy.data());
            for (int i = 0; i < nf; ++i) {
                double corr = 0.0;
                for (int k = 0; k < K; ++k) corr += W[k][i] * capy[k];
                Z[j][i] = ainv[i] - corr;
            }
        }
        for (int i = 0; i < ns; ++i)
            for (int j = 0; j < ns; ++j) {
                double m = 0.0;                                    // A_fs[:,i]' Z[j]
                for (int t = 0; t < nf; ++t) m += Afs[(std::size_t) t*ns + i] * Z[j][t];
                S[(std::size_t) i*ns + j] -= m;
            }
    }
    SmallChol SL;
    double logS = 0.0;
    if (ns > 0) {
        SL.factor(S, ns);
        if (!SL.ok) return false;
        logS = SL.logdet();
    }
    if (logdet_out) {
        const double ld = logBFF + logS;
        if (!std::isfinite(ld)) return false;
        *logdet_out = ld;
    }

    // Step: delta = B^-1 grad via block elimination.
    //   bf_g = B_FF^-1 g_f ;  S s = g_s - A_fs' bf_g ;  f = bf_g - sum_j Z[j] s_j.
    if (grad && delta) {
        std::vector<double> gf(nf, 0.0), gs(ns, 0.0);
        for (int g = 0; g < n_x; ++g) {
            if (floc[g] >= 0) gf[floc[g]] = grad[g];
            else              gs[sloc[g]] = grad[g];
        }
        std::vector<double> ainv(nf, 0.0), bf_g(nf, 0.0), wty(K), capy(K);
        sf.solve(gf.data(), ainv.data(), nf);
        for (int k = 0; k < K; ++k) {
            double s = 0.0;
            for (int i = 0; i < nf; ++i) s += W[k][i] * gf[i];
            wty[k] = s;
        }
        cap_solve(wty.data(), capy.data());
        for (int i = 0; i < nf; ++i) {
            double corr = 0.0;
            for (int k = 0; k < K; ++k) corr += W[k][i] * capy[k];
            bf_g[i] = ainv[i] - corr;
        }
        std::vector<double> svec(ns, 0.0);
        if (ns > 0) {
            std::vector<double> rhs(ns, 0.0);
            for (int j = 0; j < ns; ++j) {
                double m = 0.0;
                for (int t = 0; t < nf; ++t) m += Afs[(std::size_t) t*ns + j] * bf_g[t];
                rhs[j] = gs[j] - m;
            }
            SL.solve(rhs.data(), svec.data());
        }
        for (int g = 0; g < n_x; ++g) {
            if (floc[g] >= 0) {
                const int i = floc[g];
                double acc = bf_g[i];
                for (int j = 0; j < ns; ++j) acc -= Z[j][i] * svec[j];
                delta[g] = acc;
            } else {
                delta[g] = svec[sloc[g]];
            }
        }
        for (int g = 0; g < n_x; ++g) if (!std::isfinite(delta[g])) return false;
    }
    return true;
}

} // namespace tulpa
