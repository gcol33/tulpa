// sparse_hessian.cpp
// Definitions of the sum-to-zero block Schur helpers declared in sparse_hessian.h.

#include <RcppEigen.h>
#include "sparse_hessian.h"
#include "inv_block_extract.h"  // InvBlockConstraint

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

    // A coordinate is a dense scalar when its row of A couples to more nodes
    // than s2z_dense_degree(); every other coordinate, and every pinned node
    // (the fold anchors the pins inside A_FF), is factored sparsely.
    std::vector<int> degree(n_x, 0);
    for (int c = 0; c < n_x; ++c)
        for (int p = A.col_ptr[c]; p < A.col_ptr[c + 1]; ++p) {
            const int r = A.row_idx[p];
            if (r != c) { ++degree[r]; ++degree[c]; }
        }
    const int dense_deg = s2z_dense_degree(n_x);
    std::vector<char> is_field(n_x, 0);
    for (int g = 0; g < n_x; ++g) is_field[g] = degree[g] <= dense_deg;
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
            else if (fr >= 0 && sc >= 0) { cache.dest_kind[t] = 2; cache.dest_a[t] = sc*nf + fr; }
            else                          { cache.dest_kind[t] = 2; cache.dest_a[t] = sr*nf + fc; }
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
    cache.pattern_generation = A.pattern_generation;
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
    bool* field_indefinite,
    int n_rhs) {
    if (field_indefinite) *field_indefinite = false;
    const int K = static_cast<int>(r1.size());
    if (K == 0) return false;
    const int n_x = A.n;

    S2ZBlockSchurCache local_cache;
    S2ZBlockSchurCache& cc = cache ? *cache : local_cache;
    if (!cc.matches(A, r1))
        build_s2z_block_schur_cache(A, r1, cc);
    const int nf = cc.nf, ns = cc.ns;
    const std::size_t NF = static_cast<std::size_t>(nf);
    const std::vector<int>& floc = cc.floc;
    const std::vector<int>& sloc = cc.sloc;

    // Scatter A's values into the cached A_FF pattern, the dense A_ss and the
    // column-major nf x ns A_fs via the precomputed per-nonzero destinations
    // (no membership recompute, no re-sort).
    cc.aff.zero();
    double* __restrict__ affv = cc.aff.values.data();
    std::vector<double> Ass((std::size_t) ns * ns, 0.0);
    std::vector<double> Afs(NF * ns, 0.0);
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

    // The pins are folded against the field block anchored at one node per pin
    // (InvBlockConstraint): A_FF alone holds a direction the pins identify only
    // through the uniform ridge whenever two intrinsic fields load the same
    // rows, and a fold read off it is the difference of O(1 / ridge) terms.
    InvBlockConstraint constr;
    {
        std::vector<std::vector<int>> groups(K);
        for (int k = 0; k < K; ++k) {
            groups[k].resize(r1[k].n);
            for (int i = 0; i < r1[k].n; ++i) groups[k][i] = floc[r1[k].node(i)];
        }
        constr.anchor(nf, indicator_rows(groups, nf), [&](int j) {
            const int slot = cc.aff.lookup(j, j);
            return slot >= 0 ? affv[slot]
                             : std::numeric_limits<double>::quiet_NaN();
        });
        for (std::size_t a = 0; a < constr.anchor_node.size(); ++a) {
            const int j = constr.anchor_node[a];
            affv[cc.aff.lookup(j, j)] += constr.anchor_weight[a];
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
    auto solve_ff = [&](const double* rhs, double* out, int ncol) {
        sf.solve(rhs, out, nf, ncol);
    };

    // D enters only as D^{-1} in the capacitance and log|D| in the
    // determinant, so a Kronecker-coupled field costs nothing here beyond a
    // dense K x K inverse: no cross-block fill, no extra factorization.
    std::vector<double> Dinv; double logC = 0.0;
    if (!s2z_build_Dinv(r1, A.s2z_coupling, Dinv, logC)) return false;
    constr.build(solve_ff, nf, &Dinv, neg_aff);
    if (indefinite) {
        // Relative backward error of the anchored A_FF W_i = Vb_i,
        // componentwise against |A_FF| |W_i| + |Vb_i|, over the stored lower
        // triangle.
        const std::vector<int>& cp = cc.aff.col_ptr;
        const std::vector<int>& ri = cc.aff.row_idx;
        const std::vector<double>& av = cc.aff.values;
        std::vector<double> r(nf), mag(nf);
        for (std::size_t i = 0; i < constr.W.size(); ++i) {
            const std::vector<double>& wk = constr.W[i];
            std::fill(r.begin(), r.end(), 0.0);
            std::fill(mag.begin(), mag.end(), 0.0);
            constr.for_each_vb_node(static_cast<int>(i), [&](int f) {
                r[f] = -1.0; mag[f] = 1.0;
            });
            for (int c = 0; c < nf; ++c)
                for (int p = cp[c]; p < cp[c + 1]; ++p) {
                    const int rr = ri[p];
                    const double v = av[p];
                    r[rr] += v * wk[c];  mag[rr] += std::fabs(v * wk[c]);
                    if (rr != c) { r[c] += v * wk[rr]; mag[c] += std::fabs(v * wk[rr]); }
                }
            double num = 0.0, den = 0.0;
            for (int t = 0; t < nf; ++t) {
                num = std::max(num, std::fabs(r[t]));
                den = std::max(den, mag[t]);
            }
            if (!(num <= S2Z_LDL_BACKWARD_TOL * den)) return false;
        }
    }
    // B_FF is PD exactly when the capacitance's inertia matches (build()).
    if (!constr.usable) return false;
    const double logBFF = logAFF + logC + constr.log_det_fold();

    // B_FF^-1 Y = A_a^-1 Y - W Cap^-1 (W' Y), A_a the anchored field block,
    // for the ncol columns of Y in one solve against A_a.
    auto bff_solve = [&](const double* Y, double* X, int ncol) {
        if (ncol <= 0) return;
        std::vector<double> ainv(NF * ncol);
        sf.solve(Y, ainv.data(), nf, ncol);
        for (int c = 0; c < ncol; ++c)
            constr.apply(Y + NF * c, ainv.data() + NF * c, X + NF * c, nf);
    };

    // Schur S = A_ss - A_sf B_FF^-1 A_fs. Z = B_FF^-1 A_fs (column-major
    // nf x ns) is reused for the step's field back-substitution.
    std::vector<double> Z(NF * ns, 0.0);
    std::vector<double> S = Ass;
    if (ns > 0) {
        bff_solve(Afs.data(), Z.data(), ns);
        for (int i = 0; i < ns; ++i) {
            const double* ai = Afs.data() + NF * i;
            for (int j = 0; j < ns; ++j) {
                const double* zj = Z.data() + NF * j;
                double m = 0.0;                                    // A_fs[:,i]' Z[:,j]
                for (std::size_t t = 0; t < NF; ++t) m += ai[t] * zj[t];
                S[(std::size_t) i*ns + j] -= m;
            }
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
    if (!grad || !delta || n_rhs <= 0) return true;

    // Step: delta = B^-1 grad by block elimination, every right-hand side
    // through one solve against the field block:
    //   bf_g = B_FF^-1 g_f ;  S s = g_s - A_fs' bf_g ;  f = bf_g - Z s.
    std::vector<double> GF(NF * n_rhs, 0.0), BF(NF * n_rhs, 0.0);
    for (int rh = 0; rh < n_rhs; ++rh) {
        const double* grad_r = grad + (std::size_t) rh * n_x;
        double* gf = GF.data() + NF * rh;
        for (int g = 0; g < n_x; ++g) if (floc[g] >= 0) gf[floc[g]] = grad_r[g];
    }
    bff_solve(GF.data(), BF.data(), n_rhs);
    std::vector<double> rhs(ns, 0.0), svec(ns, 0.0);
    for (int rh = 0; rh < n_rhs; ++rh) {
        const double* grad_r = grad + (std::size_t) rh * n_x;
        double* delta_r = delta + (std::size_t) rh * n_x;
        const double* bf_g = BF.data() + NF * rh;
        if (ns > 0) {
            for (int g = 0; g < n_x; ++g) if (sloc[g] >= 0) rhs[sloc[g]] = grad_r[g];
            for (int j = 0; j < ns; ++j) {
                const double* aj = Afs.data() + NF * j;
                double m = 0.0;
                for (std::size_t t = 0; t < NF; ++t) m += aj[t] * bf_g[t];
                rhs[j] -= m;
            }
            SL.solve(rhs.data(), svec.data());
        }
        for (int g = 0; g < n_x; ++g) {
            if (floc[g] >= 0) {
                const int i = floc[g];
                double acc = bf_g[i];
                for (int j = 0; j < ns; ++j) acc -= Z[NF * j + i] * svec[j];
                delta_r[g] = acc;
            } else {
                delta_r[g] = svec[sloc[g]];
            }
        }
        for (int g = 0; g < n_x; ++g) if (!std::isfinite(delta_r[g])) return false;
    }
    return true;
}

} // namespace tulpa
