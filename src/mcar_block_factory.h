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
// the augment-and-centre scheme carried onto the coupled MCAR field.

#ifndef TULPA_MCAR_BLOCK_FACTORY_H
#define TULPA_MCAR_BLOCK_FACTORY_H

#include "latent_block.h"
#include "sparse_hessian.h"
#include "icar_kernel.h"                  // for_each_icar_component, GraphPartition
#include "tulpa/sum_to_zero.h"            // s2z_node
#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <limits>

namespace tulpa {

// Build Sigma^-1 (p x p, row-major) and log|Sigma| from the log-Cholesky
// coordinates of Sigma = L L'. Column-major lower-triangle order (matching
// R's .re_logchol_to_L): for j in 0..p-1, for i in j..p-1, theta holds
// L[i][j], with the diagonal stored as log L[i][i]. Returns log|Sigma| via the
// out-param (= 2 sum_i log L[i][i] = 2 sum of the diagonal coords).
// A diagonal coordinate whose exponential leaves the representable range is an
// INFEASIBLE cell, not a usable one. exp(ld) = Inf makes the forward
// substitution below return 1/Inf = 0, so Sigma^-1 comes back all zeros and the
// MCAR prior contributes nothing to the gradient or the Hessian while
// log|Sigma| stays finite -- a silently vanishing prior. Underflow to 0 is the
// mirror case: 1/0 = Inf poisons the whole row. The bound is half the
// exponent range at each end, so that Sigma^-1 = M'M squares within range too.
inline bool mcar_logchol_diag_ok(double ld) {
    static const double hi = 0.5 * std::log(std::numeric_limits<double>::max());
    static const double lo = 0.5 * std::log(std::numeric_limits<double>::min());
    return std::isfinite(ld) && ld < hi && ld > lo;
}

// Returns false without touching `Sinv` / `log_det_Sigma` when any diagonal
// coordinate is outside that range; the block's prep() reports the cell
// infeasible on that answer, matching how the other factories decline.
inline bool mcar_sigma_inv_from_logchol(
    const double* theta, int p, std::vector<double>& Sinv, double& log_det_Sigma
) {
    {
        int probe = 0;
        for (int j = 0; j < p; ++j) {
            for (int i = j; i < p; ++i) {
                if (i == j && !mcar_logchol_diag_ok(theta[probe])) return false;
                ++probe;
            }
        }
    }
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
    return true;
}

// Sigma^-1 and log|Sigma| for outer-grid cell k. One reader, so the prior
// scatter, the log-prior and prep() all decide feasibility the same way.
inline bool mcar_cell_sigma_inv(
    const Rcpp::NumericMatrix& theta_grid, int k_grid, int axis0, int p,
    std::vector<double>& Sinv, double& log_det_Sigma
) {
    std::vector<double> th((std::size_t) p * (p + 1) / 2);
    for (int t = 0; t < (int) th.size(); ++t) {
        th[t] = theta_grid(k_grid, axis0 + t);
    }
    return mcar_sigma_inv_from_logchol(th.data(), p, Sinv, log_det_Sigma);
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
LatentBlock make_mcar_block(
    int start, int n, int p, int axis0,
    const Rcpp::NumericMatrix& theta_grid,
    std::vector<Rcpp::IntegerVector> cell_idx,
    std::vector<std::vector<Rcpp::NumericVector>> field_weight,
    Rcpp::IntegerVector adj_rp, Rcpp::IntegerVector adj_ci,
    Rcpp::IntegerVector nnbr,
    int copy_arm, int axis_alpha, const GraphPartition& partition,
    std::vector<int> field_beta_offset = {}
);

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
LatentBlock make_miid_block(
    int start, int n, int p, int axis0,
    const Rcpp::NumericMatrix& theta_grid,
    std::vector<Rcpp::IntegerVector> group_idx,
    std::vector<std::vector<Rcpp::NumericVector>> field_weight,
    int copy_arm = -1, int axis_alpha = -1
);

}  // namespace tulpa

#endif  // TULPA_MCAR_BLOCK_FACTORY_H
