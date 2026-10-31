// gpu_nngp_laplace.h
// GPU-accelerated NNGP for Laplace approximation.
// Batches the m×m neighbor covariance Cholesky factorizations on GPU.
// Falls back to CPU transparently if CUDA unavailable.

#ifndef TULPA_GPU_NNGP_LAPLACE_H
#define TULPA_GPU_NNGP_LAPLACE_H

#include "gpu_backend.h"  // single entry point; it owns the CUDA/stub choice
#include "linalg_fast.h"  // shared small-dense Cholesky / NNGP solve core
#include "nngp_cond.h"    // nngp_row_neighbours: the shared left-packed scan
#include "tulpa/cov_kernel.h"  // cov_value: the one kernel per cov_type code
#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <cmath>

namespace tulpa {

// Covariance value at distance `d`.
//
// PRECONDITION: sigma2 > 0 and phi > 0. Both Matern branches divide by phi, so
// phi = 0 sends x to infinity and returns inf * 0 = NaN, and a negative phi
// makes the exponential branch diverge and both Matern branches negative, which
// takes the assembled neighbour matrix out of the PSD cone before it reaches
// the Cholesky. batch_nngp_scatter, the only caller, rejects both at entry, so
// no covariance is ever formed from a non-positive parameter.
inline double nngp_cov_gpu(double d, double sigma2, double phi, int cov_type) {
    if (d < 1e-10) return sigma2;
    return cov_value(d, sigma2, phi, static_cast<CovType>(cov_type));
}

// Batch-compute all NNGP conditional means, variances, and (optionally) the
// conditional regression weights `alpha`.
//
// Uses GPU for the Cholesky step if available, CPU otherwise.
//
// Outputs:
//   cond_mean_out, cond_var_out: length n_spatial, indexed by ORIGINAL obs idx.
//   alpha_out (optional, nullable): length n_spatial * nn, flat row-major,
//       indexed by NNGP-order index (NOT by obs idx). alpha_out[i * nn + k]
//       is the conditional regression coefficient for NNGP-order-i's k-th
//       neighbor (where the k-th neighbor's obs idx is
//       nn_order[nn_idx(i, k) - 1]). Inactive slots (nn_idx(i, k) == 0) and
//       trailing pad columns are zeroed.
//
// alpha_out enables callers (e.g., laplace_mode_gp) to assemble the full
// NNGP precision matrix Λ = (I-A)' D⁻¹ (I-A) instead of the diagonal-on-w
// approximation. The conditional mean satisfies cond_mean[obs] =
//     sum_k alpha_out[i_nngp * nn + k] * w[ nn_order[nn_idx(i_nngp, k) - 1] ]
// where obs = nn_order[i_nngp].
void batch_nngp_scatter(
    const std::vector<double>& w,
    int n_spatial, int nn,
    double sigma2, double phi_gp, int cov_type,
    const Rcpp::NumericMatrix& coords,
    const Rcpp::IntegerMatrix& nn_idx,
    const Rcpp::NumericMatrix& nn_dist,
    const Rcpp::IntegerVector& nn_order,
    std::vector<double>& cond_mean_out,
    std::vector<double>& cond_var_out,
    bool& gpu_used,
    std::vector<double>* alpha_out = nullptr
);

// =====================================================================
// NNGP full-precision prior scatter.
//
// Assembles the full NNGP precision Λ = (I - A)' D⁻¹ (I - A) contribution
// into (grad, H), instead of the diagonal-on-w approximation. Here `A` is
// the lower-triangular NNGP coefficient matrix whose rows are the
// conditional regression coefficients on the conditioning set, and
// `D = diag(cv)` is the diagonal of conditional variances.
//
// The NNGP log-prior is sum_i [ -½ log(2π v_i) - ½ q_i² / v_i ] where
// q_i = w_i - sum_k a_{i,k} w_{N(i)_k}. Differentiating:
//   ∂(-½ q_i²/v_i)/∂w_i        = -q_i / v_i
//   ∂(-½ q_i²/v_i)/∂w_{N(i)_k} = +a_{i,k} q_i / v_i
//   ∂²(-½ q_i²/v_i)/∂w_∂w'     = -(1/v_i) · ∂q_i/∂w · ∂q_i/∂w'
// which yields the off-diagonal Hessian terms across each conditioning
// set. The current latent w is read through w_block (length n_spatial,
// indexed by obs idx, i.e., x[gp_start + obs_idx]); grad and H are
// indexed by the global latent layout starting at gp_start.
//
// Hessian entries go through `H.add(row, col, value)`, one call per symmetric
// entry. On a SparseHessianBuilder the sparsity pattern must include all
// (focal, neighbor_k) and (neighbor_k, neighbor_kp) pairs for every row; see
// make_nngp_prior_sparsity_pattern below.
//
// Inputs:
//   alpha    : length n_spatial * nn, flat row-major, indexed by NNGP-order
//              index. Output of batch_nngp_scatter(..., &alpha).
//   cv       : length n_spatial, indexed by obs idx. Output of the same.
//   w_block  : length n_spatial, indexed by obs idx (latent w slice).
//   nn_idx   : n_spatial × nn, 1-based NNGP-order indices of neighbors
//              (0 = no neighbor sentinel).
//   nn_order : length n_spatial, 0-based permutation NNGP-order → obs idx.
//   gp_start : offset in the global latent layout where the spatial block
//              begins. grad[gp_start + obs_idx] and H[gp_start + obs_idx][.]
//              are scattered into.
//
// The scatter reproduces Λ to ~1e-16 relative, asserted against an
// independently assembled (I - A)' D⁻¹ (I - A) in test-nngp-prior-scatter.R.
// Compare it RELATIVELY: Λ's entries scale as 1/cond_var, so a fixture where
// the conditional variance collapses puts them at 1e13 and an absolute
// comparison reads machine epsilon as a large discrepancy.
template <typename DenseVec, typename SparseBuilder>
inline void apply_nngp_full_prior_sparse(
    DenseVec& grad, SparseBuilder& H,
    const std::vector<double>& w_block,
    const std::vector<double>& alpha,
    const std::vector<double>& cv,
    const Rcpp::IntegerMatrix& nn_idx,
    const Rcpp::IntegerVector& nn_order,
    int n_spatial, int nn, int gp_start
) {
    std::vector<int> nb_obs(nn);
    std::vector<double> a_row(nn);
    for (int i_nngp = 0; i_nngp < n_spatial; i_nngp++) {
        int obs_focal = nn_order[i_nngp];
        if (obs_focal < 0 || obs_focal >= n_spatial) continue;
        double v_i = cv[obs_focal];
        if (!(v_i > 0.0)) continue;
        double tau_i = 1.0 / v_i;
        int idx_i = gp_start + obs_focal;

        // Same left-packed scan batch_nngp_scatter counted `alpha`'s columns
        // with, so a_row[k] is the coefficient that was written for column k.
        const int n_row = tulpa_nngp::nngp_row_neighbours(
            &nn_idx(i_nngp, 0), /*stride=*/nn_idx.nrow(), nn, n_spatial);
        int n_nb = 0;
        const double* arow = alpha.data() + static_cast<size_t>(i_nngp) * nn;
        for (int k = 0; k < n_row; k++) {
            int obs_k = nn_order[nn_idx(i_nngp, k) - 1];
            if (obs_k < 0 || obs_k >= n_spatial) break;
            nb_obs[n_nb] = obs_k;
            a_row[n_nb] = arow[k];
            n_nb++;
        }

        double q_i = w_block[obs_focal];
        for (int k = 0; k < n_nb; k++) q_i -= a_row[k] * w_block[nb_obs[k]];

        grad[idx_i] -= q_i * tau_i;
        for (int k = 0; k < n_nb; k++) {
            grad[gp_start + nb_obs[k]] += a_row[k] * q_i * tau_i;
        }

        H.add(idx_i, idx_i, tau_i);
        for (int k = 0; k < n_nb; k++) {
            int idx_k = gp_start + nb_obs[k];
            double a_k = a_row[k];
            H.add(idx_i, idx_k, -a_k * tau_i);  // builder symmetrises
            double ak_tau = a_k * tau_i;
            // The builder stores only the lower triangle (stype = -1) and
            // normalises (row, col) to (max, min). Iterating all (k, kp)
            // pairs would hit each off-diagonal slot twice — once as
            // (k, kp) and once as (kp, k) — doubling the value. Walk only
            // the unique pairs with kp <= k.
            for (int kp = 0; kp <= k; kp++) {
                int idx_kp = gp_start + nb_obs[kp];
                H.add(idx_k, idx_kp, ak_tau * a_row[kp]);
            }
        }
    }
}

// =====================================================================
// NNGP sparsity pattern: emits the (row, col) pairs needed by
// SparseHessianBuilder to represent the full NNGP precision matrix.
// Pushes pairs into `pattern` (the builder dedups + sorts).
void make_nngp_prior_sparsity_pattern(
    std::vector<std::pair<int,int>>& pattern,
    const Rcpp::IntegerMatrix& nn_idx,
    const Rcpp::IntegerVector& nn_order,
    int n_spatial, int nn, int gp_start
);

} // namespace tulpa

#endif // TULPA_GPU_NNGP_LAPLACE_H
