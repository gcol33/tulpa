// mcar_block_factory.cpp
// Definitions of make_mcar_block and make_miid_block, declared in mcar_block_factory.h.

#include <RcppEigen.h>
#include "mcar_block_factory.h"
#include "laplace_spatial_priors.h"     // icar_log_pdet

namespace tulpa {

LatentBlock make_mcar_block(
    int start, int n, int p, int axis0,
    const Rcpp::NumericMatrix& theta_grid,
    std::vector<Rcpp::IntegerVector> cell_idx,
    std::vector<std::vector<Rcpp::NumericVector>> field_weight,
    Rcpp::IntegerVector adj_rp, Rcpp::IntegerVector adj_ci,
    Rcpp::IntegerVector nnbr,
    int copy_arm, int axis_alpha, const GraphPartition& partition,
    std::vector<int> field_beta_offset) {
    const int m = p * (p + 1) / 2;
    (void) m;

    LatentBlock block;
    block.start = start;
    block.size  = p * n;
    block.contrib_kind = BlockContribKind::INDEXED_MULTI;
    block.prior_kind   = PriorFillKind::ADJACENCY;
    block.d_fac = [](int) -> double { return 1.0; };

    // Cross-arm copy coupling (INLA copy=): donor arms see the field at
    // amplitude 1; the copy arm scales the whole correlated field by alpha read
    // off the outer grid. The per-field weights X_{ia} stay in obs_indices (they
    // are grid-invariant and cached at fit-time); the per-grid amplitude rides on
    // arm_scale, the one per-arm scalar the eta accumulator / scatter apply to an
    // INDEXED_MULTI block's contribution. Empty arm_scale (copy_arm < 0) is
    // byte-identical to the single-arm MCAR.
    if (copy_arm >= 0)
        block.arm_scale = mv_field_copy_arm_scale(copy_arm, axis_alpha, theta_grid);

    // eta_i += sum_a X_{ia} u_a[cell_i]: obs i touches the p field slots
    // {a*n + cell_i} (1-based block-local) with weights X_{ia}.
    block.obs_indices = mv_field_obs_indices(std::move(cell_idx),
                                             std::move(field_weight), n, p);

    // Sparse prior scatter: P = Sigma^-1 (x) Q, grad += -P x, plus p sum-to-zero
    // pins (gradient + rank-1, folded by the solver). Sigma^-1 is recomputed
    // from theta_grid(k, .) here (cheap, p x p) so the closure holds no mutable
    // state and is safe across the parallel outer grid.
    block.add_prior_sparse = [start, n, p, axis0, theta_grid,
                               adj_rp, adj_ci, nnbr, partition](
        SparseHessianBuilder& H, DenseVec& grad,
        const Rcpp::NumericVector& x, int k_grid
    ) {
        std::vector<double> Sinv; double log_det_Sigma;
        if (!mcar_cell_sigma_inv(theta_grid, k_grid, axis0, p,
                                 Sinv, log_det_Sigma)) {
            return;
        }
        // Gradient: grad[a*n+i] += -sum_b Sinv[a,b] (Q x_b)[i].
        std::vector<std::vector<double>> Qx(p, std::vector<double>(n));
        for (int b = 0; b < p; ++b)
            mcar_apply_Q(x, start, n, b, adj_rp, adj_ci, nnbr, Qx[b]);
        for (int a = 0; a < p; ++a)
            for (int i = 0; i < n; ++i) {
                double g = 0.0;
                for (int b = 0; b < p; ++b) g += Sinv[(std::size_t) a * p + b] * Qx[b][i];
                grad[start + a * n + i] += -g;
            }
        // Hessian P = Sigma^-1 (x) Q, lower triangle (stype = -1).
        for (int a = 0; a < p; ++a) {
            for (int b = 0; b <= a; ++b) {
                const double coef = Sinv[(std::size_t) a * p + b];
                const int oa = start + a * n, ob = start + b * n;
                if (a == b) {
                    for (int i = 0; i < n; ++i) {
                        H.add(oa + i, oa + i, coef * static_cast<double>(nnbr[i]));
                        for (int kk = adj_rp[i]; kk < adj_rp[i + 1]; ++kk) {
                            const int j = adj_ci[kk];
                            if (j < i) H.add(oa + i, oa + j, -coef);  // lower tri of Q
                        }
                    }
                } else {  // a > b: every (i, j) entry is globally lower-triangular
                    for (int i = 0; i < n; ++i) {
                        H.add(oa + i, ob + i, coef * static_cast<double>(nnbr[i]));
                        for (int kk = adj_rp[i]; kk < adj_rp[i + 1]; ++kk) {
                            const int j = adj_ci[kk];
                            if (j != i) H.add(oa + i, ob + j, -coef);
                        }
                    }
                }
            }
        }
        // Sum-to-zero augmentation of the separable precision: Q_aug = Q +
        // sum_c 1_c 1_c'/J_c per field, coupled across fields by Sigma^-1, so
        //   P_aug = Sigma^-1 (x) Q_aug = P + Sigma^-1 (x) sum_c 1_c 1_c'/J_c.
        // Each per-field per-component constant then carries the field's own
        // (Sinv-scaled) precision instead of a stiff soft pin -- the
        // augment-and-centre on the coupled MCAR field. It is a rank-(p L) update
        // U D U': one indicator vector 1_{(a,c)} per (field a, component c) with
        // the dense field coupling D[(a,c),(b,c)] = Sinv[a,b]/J_c, folded by the
        // sparse solver (set_s2z_coupling). The coupled gradient is added here;
        // the level is folded out by `center` into each field's aliased
        // coefficient. Vector order matches the coupling: k = ci*p + a.
        const int L = partition.n_components();
        const int K = p * L;
        std::vector<double> Daug((std::size_t) K * K, 0.0);
        {
            int ci = 0;
            tulpa::for_each_icar_component(start, partition,
                [&](int cstart, const int* idx, int csize) {
                    const double invJ = 1.0 / static_cast<double>(csize);
                    std::vector<double> S(p, 0.0);   // S[a] = sum_{i in c} u_{a,i}
                    for (int a = 0; a < p; ++a)
                        for (int i = 0; i < csize; ++i)
                            S[a] += x[tulpa::s2z_node(cstart + a * n, idx, i)];
                    // grad[a, i in c] -= (sum_b Sinv[a,b] S[b]) / J_c.
                    for (int a = 0; a < p; ++a) {
                        double g = 0.0;
                        for (int b = 0; b < p; ++b)
                            g += Sinv[(std::size_t) a * p + b] * S[b];
                        g *= invJ;
                        for (int i = 0; i < csize; ++i)
                            grad[tulpa::s2z_node(cstart + a * n, idx, i)] -= g;
                    }
                    for (int a = 0; a < p; ++a) {
                        H.add_s2z_rank1(cstart + a * n, csize, idx, 1.0);
                        const int k1 = ci * p + a;
                        for (int b = 0; b < p; ++b)
                            Daug[(std::size_t) k1 * K + (ci * p + b)] =
                                Sinv[(std::size_t) a * p + b] * invJ;
                    }
                    ++ci;
                });
        }
        H.set_s2z_coupling(std::move(Daug));
    };

    // Sparsity pattern: P's lower-triangle nonzeros (diagonal always present).
    block.add_prior_pattern = [start, n, p, adj_rp, adj_ci](
        std::vector<std::pair<int,int>>& out
    ) {
        for (int a = 0; a < p; ++a) {
            for (int b = 0; b <= a; ++b) {
                const int oa = start + a * n, ob = start + b * n;
                for (int i = 0; i < n; ++i) {
                    out.emplace_back(oa + i, ob + i);     // (a,b) diagonal-in-cell
                    for (int kk = adj_rp[i]; kk < adj_rp[i + 1]; ++kk) {
                        const int j = adj_ci[kk];
                        if (j == i) continue;
                        if (a == b) { if (j < i) out.emplace_back(oa + i, oa + j); }
                        else        { out.emplace_back(oa + i, ob + j); }
                    }
                }
            }
        }
    };

    // log p(u | Sigma): -0.5 u'Pu - 0.5 sum_a sum_c lambda_c (sum_{i in c} u_a)^2
    //   + 0.5 n log|Sigma^-1| + 0.5 p log|Q_aug| - 0.5 p n log(2 pi),
    // the augmented P_aug = Sigma^-1 (x) Q_aug having |P_aug| =
    // |Sigma^-1|^n |Q_aug|^p. log|Q_aug| is the graph's generalized
    // determinant (icar_log_pdet), the same constant a plain ICAR field carries.
    const double log_pdet_Q = tulpa::icar_log_pdet(n, adj_rp, adj_ci, nnbr,
                                                   partition);
    block.log_prior = [start, n, p, axis0, theta_grid,
                       adj_rp, adj_ci, nnbr, partition, log_pdet_Q](
        const Rcpp::NumericVector& x, int k_grid
    ) -> double {
        std::vector<double> Sinv; double log_det_Sigma;
        if (!mcar_cell_sigma_inv(theta_grid, k_grid, axis0, p,
                                 Sinv, log_det_Sigma)) {
            return -INFINITY;
        }
        std::vector<std::vector<double>> Qx(p, std::vector<double>(n));
        for (int b = 0; b < p; ++b)
            mcar_apply_Q(x, start, n, b, adj_rp, adj_ci, nnbr, Qx[b]);
        double quad = 0.0;                         // u'Pu = sum_{a,b} Sinv[a,b] u_a' Q u_b
        for (int a = 0; a < p; ++a)
            for (int b = 0; b < p; ++b) {
                double xa_Qb = 0.0;
                for (int i = 0; i < n; ++i) xa_Qb += x[start + a * n + i] * Qx[b][i];
                quad += Sinv[(std::size_t) a * p + b] * xa_Qb;
            }
        // Augmentation quadratic (matches the coupled gradient/Hessian in
        // add_prior_sparse): sum_{a,b} Sinv[a,b] sum_c S_{a,c} S_{b,c} / J_c,
        // with S_{a,c} = sum_{i in c} u_{a,i}.
        double aug = 0.0;
        tulpa::for_each_icar_component(start, partition,
            [&](int cstart, const int* idx, int csize) {
                const double invJ = 1.0 / static_cast<double>(csize);
                std::vector<double> S(p, 0.0);
                for (int a = 0; a < p; ++a)
                    for (int i = 0; i < csize; ++i)
                        S[a] += x[tulpa::s2z_node(cstart + a * n, idx, i)];
                for (int a = 0; a < p; ++a)
                    for (int b = 0; b < p; ++b)
                        aug += Sinv[(std::size_t) a * p + b] * S[a] * S[b] * invJ;
            });
        const double log_det_Sinv = -log_det_Sigma;
        // P_aug = Sigma^-1 (x) Q_aug is full rank p*n (Q_aug fills all L per-field
        // constant directions), so the Sigma-dependent normalizer takes the full
        // n log|Sigma^-1| per field (not (n - L)).
        return -0.5 * quad - 0.5 * aug
               + 0.5 * n * log_det_Sinv + 0.5 * p * log_pdet_Q
               - 0.5 * p * n * std::log(2.0 * M_PI);
    };

    // Fold each field's GLOBAL constant into its aliased coefficient so eta is
    // preserved: the driver adds arm_scale * d_fac * amount to
    // beta_start + field_beta_offset[a]. Obs i sees this block as
    // eta_i += sum_a X_{ia} u_a[cell_i], so a constant removed from field a
    // shifts eta along that field's design column X_{.a} and aliases with the
    // COEFFICIENT on it -- field_beta_offset[a]: the intercept (0) for the
    // all-ones field, the covariate's column otherwise, or -1 when the field has
    // no fixed-effect counterpart, in which case the level is left in the field
    // for the augmentation to identify. Only the global constant is folded; with
    // L > 1 the per-component contrasts stay in the field, identified by the data
    // (the augmentation keeps the prior proper over all L directions).
    block.center = [start, n, p, field_beta_offset](Rcpp::NumericVector& x)
        -> std::vector<CenterFold> {
        std::vector<CenterFold> folds;
        for (int a = 0; a < p; ++a) {
            const int off = (a < (int) field_beta_offset.size())
                                ? field_beta_offset[a] : -1;
            if (off < 0) continue;
            double m = 0.0;
            for (int i = 0; i < n; ++i) m += x[start + a * n + i];
            m /= static_cast<double>(n);
            for (int i = 0; i < n; ++i) x[start + a * n + i] -= m;
            folds.push_back(CenterFold{off, m});
        }
        return folds;
    };
    // Each folded field's level is a hard constraint of the inner solve. The
    // probes are every field's constant: P_aug (e_a (x) 1) =
    // sum_b Sinv[b,a] (e_b (x) 1), because Q_aug 1 = 1, so the per-field
    // constants span a P-invariant space holding every level row, and the rows'
    // prior covariance is n times the folded fields' block of Sigma.
    {
        std::vector<LinearConstraintRow> rows, probes;
        for (int a = 0; a < p; ++a) {
            LinearConstraintRow v;
            v.cols.resize(static_cast<std::size_t>(n));
            for (int i = 0; i < n; ++i) v.cols[i] = start + a * n + i;
            const int off = (a < (int) field_beta_offset.size())
                                ? field_beta_offset[a] : -1;
            if (off >= 0) rows.push_back(v);
            probes.push_back(std::move(v));
        }
        if (!rows.empty())
            declare_level_constraints(block, std::move(rows), std::move(probes));
    }


    // A cell whose Sigma cannot be formed is INFEASIBLE, and the drivers
    // already skip a cell whose prep() declines. Without this the four
    // closures above would each return their own "contributes nothing", and a
    // grid cell with a vanished prior reads as a finite, usable one.
    block.prep = [p, axis0, theta_grid](int k_grid) -> bool {
        std::vector<double> Sinv; double log_det_Sigma;
        return mcar_cell_sigma_inv(theta_grid, k_grid, axis0, p,
                                   Sinv, log_det_Sigma);
    };
    return block;
}

LatentBlock make_miid_block(
    int start, int n, int p, int axis0,
    const Rcpp::NumericMatrix& theta_grid,
    std::vector<Rcpp::IntegerVector> group_idx,
    std::vector<std::vector<Rcpp::NumericVector>> field_weight,
    int copy_arm, int axis_alpha) {
    LatentBlock block;
    block.start = start;
    block.size  = p * n;
    block.contrib_kind = BlockContribKind::INDEXED_MULTI;
    block.prior_kind   = PriorFillKind::ADJACENCY;   // within-group field couples
    block.d_fac = [](int) -> double { return 1.0; };

    if (copy_arm >= 0)
        block.arm_scale = mv_field_copy_arm_scale(copy_arm, axis_alpha, theta_grid);

    block.obs_indices = mv_field_obs_indices(std::move(group_idx),
                                             std::move(field_weight), n, p);

    // Sparse prior scatter: P = Sigma^-1 (x) I_n. grad += -P x; Hessian is the
    // p x p Sigma^-1 coupling among the fields replicated across the n groups
    // (cell-diagonal, no neighbor entries). Full rank => no sum-to-zero pins.
    block.add_prior_sparse = [start, n, p, axis0, theta_grid](
        SparseHessianBuilder& H, DenseVec& grad,
        const Rcpp::NumericVector& x, int k_grid
    ) {
        std::vector<double> Sinv; double log_det_Sigma;
        if (!mcar_cell_sigma_inv(theta_grid, k_grid, axis0, p,
                                 Sinv, log_det_Sigma)) {
            return;
        }
        // grad[a*n+i] += -sum_b Sinv[a,b] x[b*n+i].
        for (int a = 0; a < p; ++a)
            for (int i = 0; i < n; ++i) {
                double g = 0.0;
                for (int b = 0; b < p; ++b)
                    g += Sinv[(std::size_t) a * p + b] * x[start + b * n + i];
                grad[start + a * n + i] += -g;
            }
        // Hessian P = Sigma^-1 (x) I, lower triangle (a >= b), cell-diagonal.
        for (int a = 0; a < p; ++a)
            for (int b = 0; b <= a; ++b) {
                const double coef = Sinv[(std::size_t) a * p + b];
                const int oa = start + a * n, ob = start + b * n;
                for (int i = 0; i < n; ++i) H.add(oa + i, ob + i, coef);
            }
    };

    // Sparsity pattern: the cell-diagonal nonzeros of P (a >= b). The a == b
    // diagonal is always present via the data fill; emitting it here too is
    // harmless (the builder coalesces duplicate coordinates), matching MCAR.
    block.add_prior_pattern = [start, n, p](std::vector<std::pair<int,int>>& out) {
        for (int a = 0; a < p; ++a)
            for (int b = 0; b <= a; ++b) {
                const int oa = start + a * n, ob = start + b * n;
                for (int i = 0; i < n; ++i) out.emplace_back(oa + i, ob + i);
            }
    };

    // log p(u | Sigma): -0.5 u'Pu + 0.5 n log|Sigma^-1| - 0.5 p n log(2 pi).
    // No null space (P full rank) => no pin term and the full n (not n - L)
    // multiplies log|Sigma^-1|.
    block.log_prior = [start, n, p, axis0, theta_grid](
        const Rcpp::NumericVector& x, int k_grid
    ) -> double {
        std::vector<double> Sinv; double log_det_Sigma;
        if (!mcar_cell_sigma_inv(theta_grid, k_grid, axis0, p,
                                 Sinv, log_det_Sigma)) {
            return -INFINITY;
        }
        double quad = 0.0;          // u'Pu = sum_{a,b} Sinv[a,b] sum_i u_{a,i} u_{b,i}
        for (int a = 0; a < p; ++a)
            for (int b = 0; b < p; ++b) {
                double xa_xb = 0.0;
                for (int i = 0; i < n; ++i)
                    xa_xb += x[start + a * n + i] * x[start + b * n + i];
                quad += Sinv[(std::size_t) a * p + b] * xa_xb;
            }
        const double log_det_Sinv = -log_det_Sigma;
        return -0.5 * quad + 0.5 * n * log_det_Sinv
               - 0.5 * p * n * std::log(2.0 * M_PI);
    };

    // No center (proper prior, anchored by the per-arm intercepts) and no prep.

    // A cell whose Sigma cannot be formed is INFEASIBLE, and the drivers
    // already skip a cell whose prep() declines. Without this the four
    // closures above would each return their own "contributes nothing", and a
    // grid cell with a vanished prior reads as a finite, usable one.
    block.prep = [p, axis0, theta_grid](int k_grid) -> bool {
        std::vector<double> Sinv; double log_det_Sigma;
        return mcar_cell_sigma_inv(theta_grid, k_grid, axis0, p,
                                   Sinv, log_det_Sigma);
    };
    return block;
}

} // namespace tulpa
