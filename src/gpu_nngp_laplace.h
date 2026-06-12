// gpu_nngp_laplace.h
// GPU-accelerated NNGP for Laplace approximation.
// Batches the m×m neighbor covariance Cholesky factorizations on GPU.
// Falls back to CPU transparently if CUDA unavailable.

#ifndef TULPA_GPU_NNGP_LAPLACE_H
#define TULPA_GPU_NNGP_LAPLACE_H

#include "gpu_cuda.h"
#include "linalg_fast.h"  // shared small-dense Cholesky / NNGP solve core
#include <Rcpp.h>
#include <vector>
#include <cmath>

namespace tulpa {

// Compute covariance value
inline double nngp_cov_gpu(double d, double sigma2, double phi, int cov_type) {
    if (d < 1e-10) return sigma2;
    if (cov_type == 0) return sigma2 * std::exp(-d / phi);
    if (cov_type == 1) {
        double x = std::sqrt(3.0) * d / phi;
        return sigma2 * (1.0 + x) * std::exp(-x);
    }
    double x = std::sqrt(5.0) * d / phi;
    return sigma2 * (1.0 + x + x * x / 3.0) * std::exp(-x);
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
inline void batch_nngp_scatter(
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
) {
    cond_mean_out.assign(n_spatial, 0.0);
    cond_var_out.assign(n_spatial, sigma2);
    if (alpha_out) alpha_out->assign(static_cast<size_t>(n_spatial) * nn, 0.0);
    gpu_used = false;

    // Phase 1: Build all C matrices and c vectors on CPU
    struct LocData {
        int nngp_idx;     // index in NNGP ordering
        int obs_idx;      // original location index
        int n_nb;         // actual neighbor count
    };
    std::vector<LocData> locs;
    std::vector<std::vector<double>> C_mats;  // flattened nn×nn
    std::vector<std::vector<double>> c_vecs;  // length nn

    for (int i = 0; i < n_spatial; i++) {
        int obs_idx = nn_order[i];
        int n_nb = 0;
        for (int j = 0; j < nn; j++) {
            if (nn_idx(i, j) > 0) n_nb++;
        }
        if (n_nb == 0) continue;

        std::vector<double> C(nn * nn, 0.0);
        std::vector<double> c(nn, 0.0);

        for (int j = 0; j < n_nb; j++) {
            c[j] = nngp_cov_gpu(nn_dist(i, j), sigma2, phi_gp, cov_type);
        }
        for (int j1 = 0; j1 < n_nb; j1++) {
            int o1 = nn_order[nn_idx(i, j1) - 1];
            C[j1 * nn + j1] = sigma2;
            for (int j2 = j1 + 1; j2 < n_nb; j2++) {
                int o2 = nn_order[nn_idx(i, j2) - 1];
                double d12 = std::sqrt(
                    std::pow(coords(o1, 0) - coords(o2, 0), 2) +
                    std::pow(coords(o1, 1) - coords(o2, 1), 2));
                double cv = nngp_cov_gpu(d12, sigma2, phi_gp, cov_type);
                C[j1 * nn + j2] = cv;
                C[j2 * nn + j1] = cv;
            }
        }
        // Pad unused diagonal for stability
        for (int j = n_nb; j < nn; j++) C[j * nn + j] = 1.0;

        locs.push_back({i, obs_idx, n_nb});
        C_mats.push_back(std::move(C));
        c_vecs.push_back(std::move(c));
    }

    int batch_size = static_cast<int>(locs.size());
    if (batch_size == 0) return;

    // Phase 2: Cholesky factorize — GPU or CPU
    // After this, C_mats[b] contains L (lower triangular)
    bool chol_ok = false;
    if (batch_size >= 50) {  // GPU overhead threshold
        chol_ok = tulpa_gpu::cuda_batched_cholesky(C_mats, nn);
        if (chol_ok) gpu_used = true;
    }

    if (!chol_ok) {
        // CPU Cholesky per matrix: shared core, in-place on the nn-strided
        // n_nb×n_nb block; zero the upper triangle afterwards as the GPU
        // path does.
        for (int b = 0; b < batch_size; b++) {
            auto& L = C_mats[b];
            int n_nb = locs[b].n_nb;
            tulpa_linalg::chol_factor_lower(L.data(), L.data(), n_nb, nn,
                                            tulpa_linalg::kCholJitter);
            for (int j = 0; j < n_nb; j++) {
                for (int k = j + 1; k < n_nb; k++) L[j * nn + k] = 0.0;
            }
        }
    }

    // Phase 3: Solve and extract conditional mean/variance (CPU, O(m²) per loc)
    for (int b = 0; b < batch_size; b++) {
        auto& L = C_mats[b];
        auto& c = c_vecs[b];
        int n_nb = locs[b].n_nb;
        int obs_idx = locs[b].obs_idx;
        int nngp_idx = locs[b].nngp_idx;

        // Gather neighbor values (0 for inactive/out-of-range slots: they
        // then contribute nothing to the conditional mean)
        std::vector<double> w_nb(n_nb, 0.0);
        for (int j = 0; j < n_nb; j++) {
            int nn_val = nn_idx(nngp_idx, j);
            if (nn_val <= 0 || nn_val > n_spatial) continue;
            int nn_orig = nn_order[nn_val - 1];
            if (nn_orig < 0 || nn_orig >= n_spatial) continue;
            w_nb[j] = w[nn_orig];
        }

        std::vector<double> alpha(n_nb);
        tulpa_linalg::nngp_moments_from_chol(
            L.data(), n_nb, nn, c.data(), w_nb.data(), sigma2,
            tulpa_linalg::kCholJitter,
            cond_mean_out[obs_idx], cond_var_out[obs_idx], alpha.data());

        if (alpha_out) {
            double* row = alpha_out->data() + static_cast<size_t>(nngp_idx) * nn;
            for (int j = 0; j < n_nb; j++) row[j] = alpha[j];
        }
    }
}

// =====================================================================
// NNGP full-precision prior scatter (dense Hessian path).
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
template <typename DenseVec, typename DenseMat>
inline void apply_nngp_full_prior_dense(
    DenseVec& grad, DenseMat& H,
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

        // Resolve neighbor obs indices and weights once per row.
        int n_nb = 0;
        const double* arow = alpha.data() + static_cast<size_t>(i_nngp) * nn;
        for (int k = 0; k < nn; k++) {
            int nnidx_k = nn_idx(i_nngp, k);
            if (nnidx_k <= 0 || nnidx_k > n_spatial) continue;
            int obs_k = nn_order[nnidx_k - 1];
            if (obs_k < 0 || obs_k >= n_spatial) continue;
            nb_obs[n_nb] = obs_k;
            a_row[n_nb] = arow[k];
            n_nb++;
        }

        // q_i = w_i - sum_k a_k * w_{N(i)_k}
        double q_i = w_block[obs_focal];
        for (int k = 0; k < n_nb; k++) q_i -= a_row[k] * w_block[nb_obs[k]];

        // Gradient: focal point and each neighbor.
        grad[idx_i] -= q_i * tau_i;
        for (int k = 0; k < n_nb; k++) {
            grad[gp_start + nb_obs[k]] += a_row[k] * q_i * tau_i;
        }

        // Hessian: Λ = (I-A)' D⁻¹ (I-A) row-i contribution.
        H[idx_i][idx_i] += tau_i;
        for (int k = 0; k < n_nb; k++) {
            int idx_k = gp_start + nb_obs[k];
            double a_k = a_row[k];
            H[idx_i][idx_k] += -a_k * tau_i;
            H[idx_k][idx_i] += -a_k * tau_i;
            double ak_tau = a_k * tau_i;
            for (int kp = 0; kp < n_nb; kp++) {
                int idx_kp = gp_start + nb_obs[kp];
                H[idx_k][idx_kp] += ak_tau * a_row[kp];
            }
        }
    }
}

// =====================================================================
// NNGP full-precision prior scatter (sparse Hessian path).
//
// Same math as apply_nngp_full_prior_dense, but routes the Hessian entries
// through SparseHessianBuilder::add(). The sparsity pattern must include
// all (focal, neighbor_k) and (neighbor_k, neighbor_kp) pairs for every
// row; see make_nngp_prior_sparsity_pattern below.
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

        int n_nb = 0;
        const double* arow = alpha.data() + static_cast<size_t>(i_nngp) * nn;
        for (int k = 0; k < nn; k++) {
            int nnidx_k = nn_idx(i_nngp, k);
            if (nnidx_k <= 0 || nnidx_k > n_spatial) continue;
            int obs_k = nn_order[nnidx_k - 1];
            if (obs_k < 0 || obs_k >= n_spatial) continue;
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
inline void make_nngp_prior_sparsity_pattern(
    std::vector<std::pair<int,int>>& pattern,
    const Rcpp::IntegerMatrix& nn_idx,
    const Rcpp::IntegerVector& nn_order,
    int n_spatial, int nn, int gp_start
) {
    for (int i_nngp = 0; i_nngp < n_spatial; i_nngp++) {
        int obs_focal = nn_order[i_nngp];
        if (obs_focal < 0 || obs_focal >= n_spatial) continue;
        int idx_i = gp_start + obs_focal;
        pattern.push_back({idx_i, idx_i});
        for (int k = 0; k < nn; k++) {
            int nnidx_k = nn_idx(i_nngp, k);
            if (nnidx_k <= 0 || nnidx_k > n_spatial) continue;
            int obs_k = nn_order[nnidx_k - 1];
            if (obs_k < 0 || obs_k >= n_spatial) continue;
            int idx_k = gp_start + obs_k;
            pattern.push_back({idx_i, idx_k});
            pattern.push_back({idx_k, idx_k});
            for (int kp = 0; kp < k; kp++) {
                int nnidx_kp = nn_idx(i_nngp, kp);
                if (nnidx_kp <= 0 || nnidx_kp > n_spatial) continue;
                int obs_kp = nn_order[nnidx_kp - 1];
                if (obs_kp < 0 || obs_kp >= n_spatial) continue;
                int idx_kp = gp_start + obs_kp;
                pattern.push_back({idx_k, idx_kp});
            }
        }
    }
}

} // namespace tulpa

#endif // TULPA_GPU_NNGP_LAPLACE_H
