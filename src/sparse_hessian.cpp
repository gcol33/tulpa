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

bool S2ZBlockSchurFactor::factor(
    const SparseHessianBuilder& A,
    const std::vector<SparseHessianBuilder::S2ZRank1>& r1,
    S2ZBlockSchurCache* cache,
    bool field_inertia,
    bool* field_indefinite) {
    if (field_indefinite) *field_indefinite = false;
    ok_ = false;
    sqrt_ok_ = false;
    logdet_ = std::numeric_limits<double>::quiet_NaN();
    const int K = static_cast<int>(r1.size());
    if (K == 0) return false;
    const int n_x = A.n;
    n_x_ = n_x;

    cc_ = cache ? cache : &own_;
    S2ZBlockSchurCache& cc = *cc_;
    if (!cc.matches(A, r1))
        build_s2z_block_schur_cache(A, r1, cc);
    const int nf = cc.nf, ns = cc.ns;
    nf_ = nf; ns_ = ns;
    const std::size_t NF = static_cast<std::size_t>(nf);
    const std::vector<int>& floc = cc.floc;

    // Scatter A's values into the cached A_FF pattern, the dense A_ss and the
    // column-major nf x ns A_fs via the precomputed per-nonzero destinations
    // (no membership recompute, no re-sort).
    cc.aff.zero();
    double* __restrict__ affv = cc.aff.values.data();
    std::vector<double> Ass((std::size_t) ns * ns, 0.0);
    Afs_.assign(NF * ns, 0.0);
    {
        int t = 0;
        for (int c = 0; c < n_x; ++c)
            for (int p = A.col_ptr[c]; p < A.col_ptr[c + 1]; ++p, ++t) {
                const double v = A.values[p];
                const int a = cc.dest_a[t];
                switch (cc.dest_kind[t]) {
                    case 0:  scatter_slot(affv, a, v); break;
                    case 1:  Ass[a] += v; if (cc.dest_b[t] >= 0) Ass[cc.dest_b[t]] += v; break;
                    default: Afs_[a] += v; break;
                }
            }
    }

    // The pins are folded against the field block anchored at one node per pin
    // (InvBlockConstraint): A_FF alone holds a direction the pins identify only
    // through the uniform ridge whenever two intrinsic fields load the same
    // rows, and a fold read off it is the difference of O(1 / ridge) terms.
    constr_ = InvBlockConstraint();
    {
        std::vector<std::vector<int>> groups(K);
        for (int k = 0; k < K; ++k) {
            groups[k].resize(r1[k].n);
            for (int i = 0; i < r1[k].n; ++i) groups[k][i] = floc[r1[k].node(i)];
        }
        constr_.anchor(nf, indicator_rows(groups, nf), [&](int j) {
            const int slot = cc.aff.lookup(j, j);
            return slot >= 0 ? affv[slot]
                             : std::numeric_limits<double>::quiet_NaN();
        });
        for (std::size_t a = 0; a < constr_.anchor_node.size(); ++a) {
            const int j = constr_.anchor_node[a];
            affv[cc.aff.lookup(j, j)] += constr_.anchor_weight[a];
        }
    }

    cholmod_sparse vff = cc.aff.as_cholmod(&cc.aff_solver.common());
    if (!cc.aff_solver.analyzed()) cc.aff_solver.analyze(&vff);
    double logAFF = 0.0;
    int neg_aff = 0;
    sf_ = &cc.aff_solver;
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
        sf_ = &cc.aff_ldl;
    }
    const bool indefinite = (sf_ == &cc.aff_ldl);
    SparseCholeskySolver& sf = *sf_;
    auto solve_ff = [&](const double* rhs, double* out, int ncol) {
        sf.solve(rhs, out, nf, ncol);
    };

    // D enters only as D^{-1} in the capacitance and log|D| in the
    // determinant, so a Kronecker-coupled field costs nothing here beyond a
    // dense K x K inverse: no cross-block fill, no extra factorization.
    std::vector<double> Dinv; double logC = 0.0;
    if (!s2z_build_Dinv(r1, A.s2z_coupling, Dinv, logC)) return false;
    constr_.build(solve_ff, nf, &Dinv, neg_aff);
    if (indefinite) {
        // Relative backward error of the anchored A_FF W_i = Vb_i,
        // componentwise against |A_FF| |W_i| + |Vb_i|, over the stored lower
        // triangle.
        const std::vector<int>& cp = cc.aff.col_ptr;
        const std::vector<int>& ri = cc.aff.row_idx;
        const std::vector<double>& av = cc.aff.values;
        std::vector<double> r(nf), mag(nf);
        for (std::size_t i = 0; i < constr_.W.size(); ++i) {
            const std::vector<double>& wk = constr_.W[i];
            std::fill(r.begin(), r.end(), 0.0);
            std::fill(mag.begin(), mag.end(), 0.0);
            constr_.for_each_vb_node(static_cast<int>(i), [&](int f) {
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
    if (!constr_.usable) return false;
    const double logBFF = logAFF + logC + constr_.log_det_fold();

    // Schur S = A_ss - A_sf B_FF^-1 A_fs. Z = B_FF^-1 A_fs (column-major
    // nf x ns) is reused for the step's field back-substitution.
    Z_.assign(NF * ns, 0.0);
    std::vector<double> S = Ass;
    if (ns > 0) {
        bff_solve(Afs_.data(), Z_.data(), ns);
        for (int i = 0; i < ns; ++i) {
            const double* ai = Afs_.data() + NF * i;
            for (int j = 0; j < ns; ++j) {
                const double* zj = Z_.data() + NF * j;
                double m = 0.0;                                    // A_fs[:,i]' Z[:,j]
                for (std::size_t t = 0; t < NF; ++t) m += ai[t] * zj[t];
                S[(std::size_t) i*ns + j] -= m;
            }
        }
    }
    SL_ = SmallChol();
    double logS = 0.0;
    if (ns > 0) {
        SL_.factor(S, ns);
        if (!SL_.ok) return false;
        logS = SL_.logdet();
    }
    logdet_ = logBFF + logS;
    ok_ = true;
    return true;
}

// B_FF^-1 Y = A_a^-1 Y - W Cap^-1 (W' Y), A_a the anchored field block,
// for the ncol columns of Y in one solve against A_a.
void S2ZBlockSchurFactor::bff_solve(const double* Y, double* X, int ncol) {
    if (ncol <= 0) return;
    const std::size_t NF = static_cast<std::size_t>(nf_);
    std::vector<double> ainv(NF * ncol);
    sf_->solve(Y, ainv.data(), nf_, ncol);
    for (int c = 0; c < ncol; ++c)
        constr_.apply(Y + NF * c, ainv.data() + NF * c, X + NF * c, nf_);
}

bool S2ZBlockSchurFactor::solve(const double* rhs, double* out, int ncol) {
    if (!ok_) return false;
    if (ncol <= 0) return true;
    const int n_x = n_x_, nf = nf_, ns = ns_;
    const std::size_t NF = static_cast<std::size_t>(nf);
    const std::vector<int>& floc = cc_->floc;
    const std::vector<int>& sloc = cc_->sloc;

    // delta = B^-1 grad by block elimination, every right-hand side through
    // one solve against the field block:
    //   bf_g = B_FF^-1 g_f ;  S s = g_s - A_fs' bf_g ;  f = bf_g - Z s.
    std::vector<double> GF(NF * ncol, 0.0), BF(NF * ncol, 0.0);
    for (int rh = 0; rh < ncol; ++rh) {
        const double* grad_r = rhs + (std::size_t) rh * n_x;
        double* gf = GF.data() + NF * rh;
        for (int g = 0; g < n_x; ++g) if (floc[g] >= 0) gf[floc[g]] = grad_r[g];
    }
    bff_solve(GF.data(), BF.data(), ncol);
    std::vector<double> r(ns, 0.0), svec(ns, 0.0);
    for (int rh = 0; rh < ncol; ++rh) {
        const double* grad_r = rhs + (std::size_t) rh * n_x;
        double* delta_r = out + (std::size_t) rh * n_x;
        const double* bf_g = BF.data() + NF * rh;
        if (ns > 0) {
            for (int g = 0; g < n_x; ++g) if (sloc[g] >= 0) r[sloc[g]] = grad_r[g];
            for (int j = 0; j < ns; ++j) {
                const double* aj = Afs_.data() + NF * j;
                double m = 0.0;
                for (std::size_t t = 0; t < NF; ++t) m += aj[t] * bf_g[t];
                r[j] -= m;
            }
            SL_.solve(r.data(), svec.data());
        }
        for (int g = 0; g < n_x; ++g) {
            if (floc[g] >= 0) {
                const int i = floc[g];
                double acc = bf_g[i];
                for (int j = 0; j < ns; ++j) acc -= Z_[NF * j + i] * svec[j];
                delta_r[g] = acc;
            } else {
                delta_r[g] = svec[sloc[g]];
            }
        }
        for (int g = 0; g < n_x; ++g) if (!std::isfinite(delta_r[g])) return false;
    }
    return true;
}

bool S2ZBlockSchurFactor::prepare_sqrt() {
    sqrt_ok_ = false;
    if (!ok_ || sf_ != &cc_->aff_solver) return false;
    const InvBlockConstraint::DrawTransform T = constr_.draw_transform();
    const int m = constr_.order();
    if (!T.ok || static_cast<int>(T.cols.size()) != m) return false;
    for (int a = 0; a < m; ++a) if (T.cols[a] != a) return false;
    const std::size_t NF = static_cast<std::size_t>(nf_);
    m_ = m;

    // Y = L^-1 P Vb.
    std::vector<double> vb;
    constr_.vb_columns(nf_, vb);
    Yf_.assign(NF * m, 0.0);
    if (m > 0 && !sf_->apply_chol_factor_inv(vb.data(), Yf_.data(), nf_, m))
        return false;

    // Gm = Gam (I + S Gam)^-1 = (I + Gam S)^-1 Gam, S = Y'Y: solve
    // (I + Gam S) Gm = Gam by elimination with partial pivoting (m is twice
    // the pin count).
    std::vector<double> Sm((std::size_t) m * m, 0.0);
    for (int a = 0; a < m; ++a)
        for (int b = 0; b <= a; ++b) {
            double s = 0.0;
            const double* ya = Yf_.data() + NF * a;
            const double* yb = Yf_.data() + NF * b;
            for (std::size_t t = 0; t < NF; ++t) s += ya[t] * yb[t];
            Sm[(std::size_t) a * m + b] = Sm[(std::size_t) b * m + a] = s;
        }
    std::vector<double> X((std::size_t) m * m, 0.0);
    for (int a = 0; a < m; ++a) {
        for (int b = 0; b < m; ++b) {
            double s = (a == b) ? 1.0 : 0.0;
            for (int c = 0; c < m; ++c)
                s += T.Gam[(std::size_t) a * m + c] * Sm[(std::size_t) c * m + b];
            X[(std::size_t) a * m + b] = s;
        }
    }
    Gm_ = T.Gam;
    for (int col = 0; col < m; ++col) {
        int piv = col;
        for (int r = col + 1; r < m; ++r)
            if (std::fabs(X[(std::size_t) r * m + col]) >
                std::fabs(X[(std::size_t) piv * m + col])) piv = r;
        if (!(std::fabs(X[(std::size_t) piv * m + col]) > 0.0)) return false;
        if (piv != col)
            for (int c = 0; c < m; ++c) {
                std::swap(X[(std::size_t) piv * m + c], X[(std::size_t) col * m + c]);
                std::swap(Gm_[(std::size_t) piv * m + c], Gm_[(std::size_t) col * m + c]);
            }
        const double d = X[(std::size_t) col * m + col];
        for (int r = 0; r < m; ++r) {
            if (r == col) continue;
            const double f = X[(std::size_t) r * m + col] / d;
            if (f == 0.0) continue;
            for (int c = 0; c < m; ++c) {
                X[(std::size_t) r * m + c] -= f * X[(std::size_t) col * m + c];
                Gm_[(std::size_t) r * m + c] -= f * Gm_[(std::size_t) col * m + c];
            }
        }
    }
    for (int r = 0; r < m; ++r) {
        const double d = X[(std::size_t) r * m + r];
        for (int c = 0; c < m; ++c) Gm_[(std::size_t) r * m + c] /= d;
    }

    // Pz = R_F^-1 A_fs.
    sqrt_ok_ = true;
    Pz_ = Afs_;
    if (ns_ > 0 && !rf_inv(Pz_.data(), ns_)) { sqrt_ok_ = false; return false; }
    return true;
}

// v <- T^-1 L^-1 P v on `ncol` field columns.
bool S2ZBlockSchurFactor::rf_inv(double* v, int ncol) {
    const std::size_t NF = static_cast<std::size_t>(nf_);
    if (!sf_->apply_chol_factor_inv(v, v, nf_, ncol)) return false;
    std::vector<double> t(m_), s(m_);
    for (int c = 0; c < ncol; ++c) {
        double* vc = v + NF * c;
        for (int a = 0; a < m_; ++a) {
            const double* ya = Yf_.data() + NF * a;
            double acc = 0.0;
            for (std::size_t i = 0; i < NF; ++i) acc += ya[i] * vc[i];
            t[a] = acc;
        }
        for (int a = 0; a < m_; ++a) {
            double acc = 0.0;
            for (int b = 0; b < m_; ++b) acc += Gm_[(std::size_t) a * m_ + b] * t[b];
            s[a] = acc;
        }
        for (int a = 0; a < m_; ++a) {
            const double* ya = Yf_.data() + NF * a;
            for (std::size_t i = 0; i < NF; ++i) vc[i] -= ya[i] * s[a];
        }
    }
    return true;
}

// w <- P' L^-T T^-1 w on `ncol` field columns (T^-1 is symmetric).
bool S2ZBlockSchurFactor::rf_inv_t(double* w, int ncol) {
    const std::size_t NF = static_cast<std::size_t>(nf_);
    std::vector<double> t(m_), s(m_);
    for (int c = 0; c < ncol; ++c) {
        double* wc = w + NF * c;
        for (int a = 0; a < m_; ++a) {
            const double* ya = Yf_.data() + NF * a;
            double acc = 0.0;
            for (std::size_t i = 0; i < NF; ++i) acc += ya[i] * wc[i];
            t[a] = acc;
        }
        for (int a = 0; a < m_; ++a) {
            double acc = 0.0;
            for (int b = 0; b < m_; ++b) acc += Gm_[(std::size_t) a * m_ + b] * t[b];
            s[a] = acc;
        }
        for (int a = 0; a < m_; ++a) {
            const double* ya = Yf_.data() + NF * a;
            for (std::size_t i = 0; i < NF; ++i) wc[i] -= ya[i] * s[a];
        }
    }
    return sf_->apply_inv_chol_factor(w, w, nf_, ncol);
}

bool S2ZBlockSchurFactor::forward(const double* v, double* y, int ncol) {
    if (!sqrt_ok_) return false;
    if (ncol <= 0) return true;
    const int n_x = n_x_, ns = ns_;
    const std::size_t NF = static_cast<std::size_t>(nf_);
    const std::vector<int>& floc = cc_->floc;
    const std::vector<int>& sloc = cc_->sloc;
    std::vector<double> VF(NF * ncol, 0.0);
    for (int c = 0; c < ncol; ++c) {
        const double* vc = v + (std::size_t) c * n_x;
        double* vf = VF.data() + NF * c;
        for (int g = 0; g < n_x; ++g) if (floc[g] >= 0) vf[floc[g]] = vc[g];
    }
    if (!rf_inv(VF.data(), ncol)) return false;
    // y_s = L_S^-1 (v_s - Pz' y_f).
    std::vector<double> ys(ns);
    const std::vector<double>& L = SL_.L;
    for (int c = 0; c < ncol; ++c) {
        const double* vc = v + (std::size_t) c * n_x;
        const double* yf = VF.data() + NF * c;
        double* yc = y + (std::size_t) c * n_x;
        for (int g = 0; g < n_x; ++g) if (sloc[g] >= 0) ys[sloc[g]] = vc[g];
        for (int j = 0; j < ns; ++j) {
            const double* pj = Pz_.data() + NF * j;
            double acc = 0.0;
            for (std::size_t i = 0; i < NF; ++i) acc += pj[i] * yf[i];
            ys[j] -= acc;
        }
        for (int i = 0; i < ns; ++i) {
            double acc = ys[i];
            for (int p = 0; p < i; ++p) acc -= L[(std::size_t) i * ns + p] * ys[p];
            ys[i] = acc / L[(std::size_t) i * ns + i];
        }
        for (int g = 0; g < n_x; ++g)
            yc[g] = floc[g] >= 0 ? yf[floc[g]] : ys[sloc[g]];
        for (int g = 0; g < n_x; ++g) if (!std::isfinite(yc[g])) return false;
    }
    return true;
}

bool S2ZBlockSchurFactor::backward(const double* u, double* x, int ncol) {
    if (!sqrt_ok_) return false;
    if (ncol <= 0) return true;
    const int n_x = n_x_, ns = ns_;
    const std::size_t NF = static_cast<std::size_t>(nf_);
    const std::vector<int>& floc = cc_->floc;
    const std::vector<int>& sloc = cc_->sloc;
    const std::vector<double>& L = SL_.L;
    std::vector<double> UF(NF * ncol, 0.0), XS((std::size_t) ns * ncol, 0.0);
    for (int c = 0; c < ncol; ++c) {
        const double* uc = u + (std::size_t) c * n_x;
        double* uf = UF.data() + NF * c;
        double* xs = XS.data() + (std::size_t) ns * c;
        for (int g = 0; g < n_x; ++g) {
            if (floc[g] >= 0) uf[floc[g]] = uc[g];
            else xs[sloc[g]] = uc[g];
        }
        // x_s = L_S^-T u_s, then u_f - Pz x_s.
        for (int i = ns - 1; i >= 0; --i) {
            double acc = xs[i];
            for (int p = i + 1; p < ns; ++p) acc -= L[(std::size_t) p * ns + i] * xs[p];
            xs[i] = acc / L[(std::size_t) i * ns + i];
        }
        for (int j = 0; j < ns; ++j) {
            const double* pj = Pz_.data() + NF * j;
            for (std::size_t i = 0; i < NF; ++i) uf[i] -= pj[i] * xs[j];
        }
    }
    if (!rf_inv_t(UF.data(), ncol)) return false;
    for (int c = 0; c < ncol; ++c) {
        const double* xf = UF.data() + NF * c;
        const double* xs = XS.data() + (std::size_t) ns * c;
        double* xc = x + (std::size_t) c * n_x;
        for (int g = 0; g < n_x; ++g)
            xc[g] = floc[g] >= 0 ? xf[floc[g]] : xs[sloc[g]];
        for (int g = 0; g < n_x; ++g) if (!std::isfinite(xc[g])) return false;
    }
    return true;
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
    S2ZBlockSchurFactor F;
    if (!F.factor(A, r1, cache, field_inertia, field_indefinite)) return false;
    if (logdet_out) {
        if (!std::isfinite(F.log_det())) return false;
        *logdet_out = F.log_det();
    }
    if (!grad || !delta || n_rhs <= 0) return true;
    return F.solve(grad, delta, n_rhs);
}

} // namespace tulpa
