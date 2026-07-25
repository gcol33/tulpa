// mcar_block_factory.h
// Separable multivariate CAR (MCAR) block for the joint nested-Laplace driver
//. p areal fields over one graph share a cross-covariance
// Sigma: the joint latent (u_1, ..., u_p) over n cells has precision
//   P = Sigma^-1 (x) Q,            Q = D - W   (intrinsic CAR, rho = 1).
//
// Natural parameterization: the fields ARE the latent (no amplitude in eta).
// Observation i in arm k contributes eta_i += sum_a X_{ia} u_a[cell_i], so the
// block is INDEXED_MULTI -- obs i touches the p slots {a*n + cell_i} with the
// design weights X_{ia} (the intercept column is all-ones, a covariate column
// is the per-row value, exactly the svc_weight values). Sigma enters ONLY the
// prior, integrated over by the outer grid in log-Cholesky coordinates.
//
// Log-determinant tractability: Q is rank n-1 (constant null space per field),
// so P is rank p(n-1) with a p-dim null space spanned by the per-field
// constants. Its pseudo-determinant factorizes,
//   logpdet(Sigma^-1 (x) Q) = (n-1) log|Sigma^-1| + p logpdet(Q),
// so the only Sigma-dependent normalizer is n log|Sigma^-1| = 2 n sum_i
// (log-Cholesky diagonal), a p x p quantity -- no large generalized determinant.
// The p L constant directions (L per field over L graph components) are
// identified by AUGMENTING the precision, Q_aug = Q + sum_c 1_c 1_c'/J_c per
// field, coupled across fields by Sigma^-1 -- the augmented Sigma^-1 (x) 11'/J_c
// folded by the sparse solver's block-Schur path (set_s2z_coupling), with the
// level folded out by `center` into each field's aliased coefficient. This is
// #241's augment-and-centre carried onto the coupled MCAR field.

#ifndef TULPA_MCAR_BLOCK_FACTORY_H
#define TULPA_MCAR_BLOCK_FACTORY_H

#include "latent_block.h"
#include "sparse_hessian.h"
#include "icar_kernel.h"                  // for_each_icar_component, GraphPartition
#include "tulpa/sum_to_zero.h"            // s2z_node
#include <Rcpp.h>
#include <vector>
#include <cmath>

namespace tulpa {

// Build Sigma^-1 (p x p, row-major) and log|Sigma| from the log-Cholesky
// coordinates of Sigma = L L'. Column-major lower-triangle order (matching
// R's .re_logchol_to_L): for j in 0..p-1, for i in j..p-1, theta holds
// L[i][j], with the diagonal stored as log L[i][i]. Returns log|Sigma| via the
// out-param (= 2 sum_i log L[i][i] = 2 sum of the diagonal coords).
inline void mcar_sigma_inv_from_logchol(
    const double* theta, int p, std::vector<double>& Sinv, double& log_det_Sigma
) {
    std::vector<double> L((std::size_t) p * p, 0.0);
    log_det_Sigma = 0.0;
    int idx = 0;
    for (int j = 0; j < p; ++j) {
        for (int i = j; i < p; ++i) {
            if (i == j) {
                const double ld = theta[idx];
                L[(std::size_t) i * p + j] = std::exp(ld);
                log_det_Sigma += 2.0 * ld;
            } else {
                L[(std::size_t) i * p + j] = theta[idx];
            }
            ++idx;
        }
    }
    // M = L^-1 (lower-triangular inverse by forward substitution).
    std::vector<double> M((std::size_t) p * p, 0.0);
    for (int col = 0; col < p; ++col) {
        M[(std::size_t) col * p + col] = 1.0 / L[(std::size_t) col * p + col];
        for (int i = col + 1; i < p; ++i) {
            double s = 0.0;
            for (int k = col; k < i; ++k)
                s += L[(std::size_t) i * p + k] * M[(std::size_t) k * p + col];
            M[(std::size_t) i * p + col] = -s / L[(std::size_t) i * p + i];
        }
    }
    // Sigma^-1 = M' M (sum over rows of M).
    Sinv.assign((std::size_t) p * p, 0.0);
    for (int a = 0; a < p; ++a)
        for (int b = 0; b < p; ++b) {
            double s = 0.0;
            for (int k = 0; k < p; ++k)
                s += M[(std::size_t) k * p + a] * M[(std::size_t) k * p + b];
            Sinv[(std::size_t) a * p + b] = s;
        }
}

// Q x_b for field b: (Q x_b)[i] = nnbr[i] x_b[i] - sum_{j~i} x_b[j].
inline void mcar_apply_Q(
    const Rcpp::NumericVector& x, int start, int n, int b,
    const Rcpp::IntegerVector& adj_rp, const Rcpp::IntegerVector& adj_ci,
    const Rcpp::IntegerVector& nnbr, std::vector<double>& out
) {
    const int off = start + b * n;
    for (int i = 0; i < n; ++i) {
        double v = static_cast<double>(nnbr[i]) * x[off + i];
        for (int k = adj_rp[i]; k < adj_rp[i + 1]; ++k) {
            const int j = adj_ci[k];
            if (j != i) v -= x[off + j];
        }
        out[i] = v;
    }
}

// Shared INDEXED_MULTI obs->field scatter for a multivariate field block
// (mcar / miid): obs i in arm k contributes eta_i += sum_a X_{ia} u_a[idx_i],
// touching the p field slots {a*n + idx_i} (1-based block-local) with the
// per-field design weights X_{ia} (intercept all-ones, covariate per-row). The
// grouping `idx` is the per-arm 1-based unit index -- cell for mcar, group for
// miid; a 0 / out-of-range index skips the obs.
inline std::function<void(int, int, std::vector<std::pair<int,double>>&)>
mv_field_obs_indices(
    std::vector<Rcpp::IntegerVector> idx,
    std::vector<std::vector<Rcpp::NumericVector>> field_weight,
    int n, int p
) {
    return [idx, field_weight, n, p](
        int i, int k_arm, std::vector<std::pair<int,double>>& out
    ) {
        out.clear();
        const Rcpp::IntegerVector& gi = idx[k_arm];
        if (i < 0 || i >= gi.size()) return;
        const int cell = gi[i];                 // 1-based; 0 => skip
        if (cell < 1 || cell > n) return;
        out.reserve(p);
        for (int a = 0; a < p; ++a)
            out.emplace_back(a * n + cell, field_weight[a][k_arm][i]);
    };
}

// Shared cross-arm copy amplitude for a multivariate field block (mcar / miid):
// donor arms see the field at amplitude 1; the copy arm scales the whole
// correlated (intercept, slope) field by one alpha read off the outer grid at
// axis_alpha (the cross-arm transfer). The free cross-covariance Sigma stays a
// within-arm property of the donor field integrated over the outer grid. Empty
// arm_scale (copy_arm < 0) is byte-identical to the single-arm field.
inline std::function<double(int, int)>
mv_field_copy_arm_scale(int copy_arm, int axis_alpha,
                        const Rcpp::NumericMatrix& theta_grid) {
    return [copy_arm, axis_alpha, theta_grid](int k_arm, int k_grid) -> double {
        return (k_arm == copy_arm) ? theta_grid(k_grid, axis_alpha) : 1.0;
    };
}

// Construct the MCAR LatentBlock.
//   start        : latent offset of the p*n field block.
//   n            : n_spatial_units (cells).
//   p            : n_fields.
//   axis0        : first log-Cholesky axis column in theta_grid; the block
//                  reads p(p+1)/2 consecutive columns there.
//   cell_idx     : per-arm 1-based cell index (length n_arms list of vectors).
//   field_weight : per-field, per-arm design column (X_{ia}); outer length p,
//                  inner length n_arms. field_weight[a][k][i] = X_{ia} on arm k.
//   adjacency    : CSR (adj_rp, adj_ci, nnbr), 0-based.
//   copy_arm     : 0-based arm index that receives a scaled COPY of the whole
//                  correlated field, or -1 for no copy. The natural-parameter
//                  field IS the latent, so a copy arm sees the same (u_1,...,u_p)
//                  scaled by a single amplitude alpha (read from theta_grid at
//                  axis_alpha); donor arms see the field at amplitude 1. The free
//                  cross-covariance Sigma stays a property of the donor field
//                  integrated over the outer grid (alpha is the cross-ARM transfer,
//                  Sigma is the within-ARM covariance among the fields).
//   axis_alpha   : theta_grid column holding the per-cell copy coefficient alpha
//                  (used only when copy_arm >= 0).
//   partition    : connected-component partition of the (per-field) graph
//                  (graph_components.h): the trivial single component for an
//                  ordinary connected graph (byte-identical to the
//                  single-component path); the true per-component node sets for
//                  a replicated MCAR over the block-diagonal I_L (x) Q (the
//                  `by =` replicated CAR) or a genuine disconnected map. Q then
//                  has rank (n - L) per field, so its constant null space is
//                  L-dimensional per field: the augmentation fills all L
//                  directions per field (Q_aug full rank n) and the
//                  Sigma-normalizer uses the full n.
//   field_beta_offset : per-field (length p) 0-based index into the arm's beta
//                  block that field a's global constant aliases with -- the
//                  intercept (0) for the all-ones field, the covariate's column
//                  for a covariate field, or -1 when the field has no
//                  fixed-effect counterpart (then its level is left in the field
//                  for the augmentation to identify, not folded). Empty (the
//                  default) is all -1.
inline LatentBlock make_mcar_block(
    int start, int n, int p, int axis0,
    const Rcpp::NumericMatrix& theta_grid,
    std::vector<Rcpp::IntegerVector> cell_idx,
    std::vector<std::vector<Rcpp::NumericVector>> field_weight,
    Rcpp::IntegerVector adj_rp, Rcpp::IntegerVector adj_ci,
    Rcpp::IntegerVector nnbr,
    int copy_arm, int axis_alpha, const GraphPartition& partition,
    std::vector<int> field_beta_offset = {}
) {
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
        {
            std::vector<double> th(p * (p + 1) / 2);
            for (int t = 0; t < (int) th.size(); ++t) th[t] = theta_grid(k_grid, axis0 + t);
            mcar_sigma_inv_from_logchol(th.data(), p, Sinv, log_det_Sigma);
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
        // (Sinv-scaled) precision instead of the old stiff soft pin -- #241's
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
    //   + 0.5 (n-L) log|Sigma^-1|  - 0.5 p (n-L) log(2 pi).
    // The constant p logpdet(Q) term (Sigma-independent) is dropped, mirroring
    // log_prior_icar; the Sigma-dependent (n-L) log|Sigma^-1| is exact (P =
    // Sigma^-1 (x) Q has rank p(n-L) when Q has L components: logpdet(P) =
    // (n-L) log|Sigma^-1| + p logpdet(Q)).
    block.log_prior = [start, n, p, axis0, theta_grid,
                       adj_rp, adj_ci, nnbr, partition](
        const Rcpp::NumericVector& x, int k_grid
    ) -> double {
        std::vector<double> Sinv; double log_det_Sigma;
        {
            std::vector<double> th(p * (p + 1) / 2);
            for (int t = 0; t < (int) th.size(); ++t) th[t] = theta_grid(k_grid, axis0 + t);
            mcar_sigma_inv_from_logchol(th.data(), p, Sinv, log_det_Sigma);
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
        // n log|Sigma^-1| per field (not (n - L)); the constant p log|Q_aug| is
        // dropped as before.
        return -0.5 * quad - 0.5 * aug
               + 0.5 * n * log_det_Sinv
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

    return block;
}

// Construct the multivariate-IID (MIID) LatentBlock: the non-spatial sibling of
// make_mcar_block with Q = I (identity in place of D - W). Per group g a
// coefficient vector b_g ~ N(0, Sigma) (block dim p = n_coefs), with the free
// cross-covariance Sigma integrated over the same p(p+1)/2 log-Cholesky outer-
// grid axes as MCAR. The joint latent (u_1, ..., u_p) over n groups has
// precision
//   P = Sigma^-1 (x) I_n,
// which is FULL RANK -- so, unlike MCAR, there is no constant null space, no
// sum-to-zero pinning, and no centering (a proper N(0, Sigma) prior, like the
// scalar iid block). The only Sigma-dependent normalizer is n log|Sigma^-1|
// (vs MCAR's (n - L) log|Sigma^-1|, which drops the L-dim per-field null space).
// This expresses a CORRELATED random slope (1 + x | g): p = 1 + n_slopes
// coefficients per group sharing a free covariance, integrated as a derived
// quantity over the outer grid. A p = 1 block is a proper random intercept
// N(0, sigma^2); it is the centered-parameterization counterpart of the scalar
// iid block (eta += u, u ~ N(0, sigma^2) vs eta += sigma * u, u ~ N(0, 1)) --
// the same model, Laplace-invariant under that affine reparameterization.
//
//   start        : latent offset of the p*n block (n = n_groups).
//   n            : n_groups.
//   p            : n_coefs (block dim).
//   axis0        : first log-Cholesky axis column in theta_grid; the block reads
//                  p(p+1)/2 consecutive columns there.
//   group_idx    : per-arm 1-based group index (the MCAR cell_idx role).
//   field_weight : per-coefficient, per-arm design column X_{ia} (intercept
//                  all-ones, slope = covariate); outer length p, inner n_arms.
//   copy_arm     : 0-based arm receiving a scaled COPY of the whole field, or -1.
//   axis_alpha   : theta_grid column holding the copy coefficient alpha (used
//                  only when copy_arm >= 0).
inline LatentBlock make_miid_block(
    int start, int n, int p, int axis0,
    const Rcpp::NumericMatrix& theta_grid,
    std::vector<Rcpp::IntegerVector> group_idx,
    std::vector<std::vector<Rcpp::NumericVector>> field_weight,
    int copy_arm = -1, int axis_alpha = -1
) {
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
        {
            std::vector<double> th(p * (p + 1) / 2);
            for (int t = 0; t < (int) th.size(); ++t) th[t] = theta_grid(k_grid, axis0 + t);
            mcar_sigma_inv_from_logchol(th.data(), p, Sinv, log_det_Sigma);
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
        {
            std::vector<double> th(p * (p + 1) / 2);
            for (int t = 0; t < (int) th.size(); ++t) th[t] = theta_grid(k_grid, axis0 + t);
            mcar_sigma_inv_from_logchol(th.data(), p, Sinv, log_det_Sigma);
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
    return block;
}

}  // namespace tulpa

#endif  // TULPA_MCAR_BLOCK_FACTORY_H
