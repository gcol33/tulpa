// hmc_gp_autodiff.h
// Templated GP/NNGP functions for autodiff support
// Works with both double (for evaluation) and ad::Var (for gradients)

#ifndef TULPA_HMC_GP_AUTODIFF_H
#define TULPA_HMC_GP_AUTODIFF_H

#include <vector>
#include <cmath>
#include "hmc_gp.h"
#include "hmc_svc_autodiff.h"  // Canonical templated covariance functions
#include "autodiff_utils.h"
#include "pc_prior.h"          // single-source PC prior on every sampled scale

namespace tulpa_gp {

using namespace tulpa::math;
// Use canonical covariance dispatcher from tulpa_svc_ad (single source of truth)
using tulpa_svc_ad::compute_cov;
// Canonical templated triangular solves also live in tulpa_svc_ad
using tulpa_svc_ad::solve_lower;
using tulpa_svc_ad::solve_upper;
// Alias for backward compatibility with code that uses the _t suffix
template<typename T>
inline T compute_cov_t(double d, const T& sigma2, const T& phi,
                       tulpa_svc::CovType cov_type) {
    return tulpa_svc_ad::compute_cov(d, sigma2, phi, cov_type);
}

// =============================================================================
// Templated Cholesky decomposition and solve
// For small matrices (nn typically 5-30)
// =============================================================================

// Cholesky decomposition: A = L * L^T
// kGpJitter / kGpVarFloor come from hmc_gp.h, so this autodiff copy and the
// double ones in hmc_gp_log_lik.h / hmc_gp_gradients.h read the SAME values.
// The gradients are finite-differenced from the double log-likelihood, so the
// copies must condition the neighbour covariance identically or the value and
// the gradient describe different models.
//
// This copy used to add the jitter only to an already-degenerate pivot
// (get_value(sum) < 1e-10), so on well-conditioned input it added none at all
// while the double copies added kGpJitter to every diagonal.

// Returns false if not positive definite
template<typename T>
inline bool cholesky_decompose_t(const std::vector<T>& A, int n, std::vector<T>& L) {
    return tulpa_nngp::chol_decomp(A, n, L, kGpJitter);
}

// =============================================================================
// Templated NNGP log-likelihood
// =============================================================================

// Debug flag for GP autodiff
#ifndef GP_AUTODIFF_DEBUG
#define GP_AUTODIFF_DEBUG false
#endif

// NOTE: This function has a known heisenbug with autodiff - use numerical gradients for GP
// The templated code is preserved for future optimization work
template<typename T>
T gp_nngp_log_lik_t(
    const std::vector<T>& w,
    const T& sigma2,
    const T& phi,
    const GPData& gp_data
) {
    int N = gp_data.n_obs;
    int nn = gp_data.nn;

#if AUTODIFF_DEBUG
    static int call_count = 0;
    call_count++;
    Rcpp::Rcout << "[NNGP] Call #" << call_count << ": N=" << N << ", nn=" << nn
                << ", sigma2=" << get_value(sigma2) << ", phi=" << get_value(phi)
                << ", w.size()=" << w.size() << "\n";
    R_FlushConsole();
#endif

#if GP_AUTODIFF_DEBUG
    if (call_count <= 5 || call_count % 100 == 0) {
        Rcpp::Rcout << "[GP_AD] Call #" << call_count << ": N=" << N << ", nn=" << nn
                    << ", sigma2=" << get_value(sigma2) << ", phi=" << get_value(phi) << "\n";
    }
#endif

    // Bounds validation
    if (gp_data.nn_order.size() < (size_t)N) return T(-1e10);
    if (gp_data.nn_idx.size() < (size_t)(N * nn)) return T(-1e10);
    if (gp_data.nn_dist.size() < (size_t)(N * nn)) return T(-1e10);
    if (gp_data.nn_neighbor_dist.size() < (size_t)(N * nn * nn)) return T(-1e10);  // Critical: prevents segfault
    if (w.size() < (size_t)N) return T(-1e10);
    if (gp_data.coords.size() < (size_t)(2 * N)) return T(-1e10);

    T log_lik = T(0.0);

    // First observation: marginal N(0, sigma2)
    int first_idx = gp_data.nn_order[0];
    T log_sigma2 = safe_log(sigma2);
    log_lik = log_lik - T(0.5) * safe_log(T(2.0 * M_PI)) - T(0.5) * log_sigma2;
    log_lik = log_lik - T(0.5) * w[first_idx] * w[first_idx] / sigma2;

    // Pre-allocate work vectors
    std::vector<T> c_vec(nn);
    std::vector<T> C_mat(nn * nn);
    std::vector<T> L(nn * nn);
    std::vector<T> y(nn);
    std::vector<T> alpha(nn);

    // Remaining observations: conditional on neighbors
    for (int i = 1; i < N; i++) {
#if AUTODIFF_DEBUG
        if (i <= 3 || i == N-1) {
            Rcpp::Rcout << "[NNGP] Processing obs i=" << i << "/" << N << "\n";
            R_FlushConsole();
        }
#endif
        int obs_idx = gp_data.nn_order[i];

        // Bounds check
        if (obs_idx < 0 || obs_idx >= N) return T(-1e10);

        // Count actual neighbors
        int n_neighbors = 0;
        for (int j = 0; j < nn; j++) {
            int nn_flat_idx = i * nn + j;
            if (nn_flat_idx >= (int)gp_data.nn_idx.size()) break;
            if (gp_data.nn_idx[nn_flat_idx] > 0) {
                n_neighbors++;
            }
        }

        if (n_neighbors == 0) {
            // No neighbors: marginal
            log_lik = log_lik - T(0.5) * safe_log(T(2.0 * M_PI)) - T(0.5) * log_sigma2;
            log_lik = log_lik - T(0.5) * w[obs_idx] * w[obs_idx] / sigma2;
            continue;
        }

        // c_vec: covariances between obs i and its neighbors
        for (int j = 0; j < n_neighbors; j++) {
            int nn_flat_idx = i * nn + j;
            double d = gp_data.nn_dist[nn_flat_idx];
            c_vec[j] = compute_cov_t(d, sigma2, phi, gp_data.cov_type);
        }

        // C_mat: covariances among neighbors
        for (int j1 = 0; j1 < n_neighbors; j1++) {
            int raw_nn_idx1 = gp_data.nn_idx[i * nn + j1];

            // Bounds check
            if (raw_nn_idx1 - 1 < 0 || raw_nn_idx1 - 1 >= (int)gp_data.nn_order.size()) {
                return T(-1e10);
            }

            int nn_idx1 = gp_data.nn_order[raw_nn_idx1 - 1];

            if (nn_idx1 < 0 || nn_idx1 * 2 + 1 >= (int)gp_data.coords.size()) {
                return T(-1e10);
            }

            for (int j2 = 0; j2 < n_neighbors; j2++) {
                int raw_nn_idx2 = gp_data.nn_idx[i * nn + j2];

                if (raw_nn_idx2 - 1 < 0 || raw_nn_idx2 - 1 >= (int)gp_data.nn_order.size()) {
                    return T(-1e10);
                }

                int nn_idx2 = gp_data.nn_order[raw_nn_idx2 - 1];

                if (j1 == j2) {
                    C_mat[j1 * n_neighbors + j2] = sigma2;
                } else {
                    // Phase 1.3: Use cached pairwise neighbor distances
                    double d12 = gp_data.nn_neighbor_dist[i * nn * nn + j1 * nn + j2];
                    C_mat[j1 * n_neighbors + j2] = compute_cov_t(d12, sigma2, phi, gp_data.cov_type);
                }
            }
        }

        // Gather the neighbour values in c_vec order.
        std::vector<T> c_small(c_vec.begin(), c_vec.begin() + n_neighbors);
        std::vector<T> w_nb(n_neighbors);
        for (int j = 0; j < n_neighbors; j++) {
            int raw_nn_idx = gp_data.nn_idx[i * nn + j];

            if (raw_nn_idx - 1 < 0 || raw_nn_idx - 1 >= (int)gp_data.nn_order.size()) {
                return T(-1e10);
            }

            int nn_orig_idx = gp_data.nn_order[raw_nn_idx - 1];

            if (nn_orig_idx < 0 || nn_orig_idx >= (int)w.size()) {
                return T(-1e10);
            }

            w_nb[j] = w[nn_orig_idx];
        }

        // Factor / krige / floor via the shared kernel, at the same constants
        // the double GP path uses (kGpJitter on every diagonal, kGpVarFloor
        // clamped). Those constants are the contract between this copy and
        // gp_nngp_log_lik / gp_nngp_gradients: the gradients are
        // finite-differenced from the double copy, so if the two condition the
        // matrix differently they describe different models.
        T cond_mean, cond_var;
        if (!tulpa_nngp::cond_moments(C_mat, c_small, w_nb, n_neighbors, sigma2,
                                      kGpJitter, kGpVarFloor,
                                      tulpa_nngp::VarFloor::Clamp,
                                      cond_mean, cond_var)) {
            return T(-1e10);  // Not positive definite
        }
        log_lik = log_lik + tulpa_nngp::cond_log_density(w[obs_idx], cond_mean,
                                                         cond_var);
    }

#if AUTODIFF_DEBUG
    Rcpp::Rcout << "[NNGP] Completed, log_lik=" << get_value(log_lik) << "\n";
    R_FlushConsole();
#endif

    return log_lik;
}

// =============================================================================
// Templated multi-scale GP log-likelihood
// =============================================================================

template<typename T>
T multiscale_gp_log_lik_t(
    const std::vector<T>& w_local,
    const std::vector<T>& w_regional,
    const T& sigma2_local,
    const T& phi_local,
    const T& sigma2_regional,
    const T& phi_regional,
    const MultiscaleGPData& ms_data
) {
    // Create temporary GPData structures for each scale
    GPData gp_local;
    gp_local.n_obs = ms_data.n_obs;
    gp_local.nn = ms_data.nn_local;
    gp_local.coords = ms_data.coords;
    gp_local.nn_idx = ms_data.nn_idx_local;
    gp_local.nn_dist = ms_data.nn_dist_local;
    gp_local.nn_neighbor_dist = ms_data.nn_neighbor_dist_local;
    gp_local.nn_order = ms_data.nn_order_local;
    gp_local.nn_order_inv = ms_data.nn_order_inv_local;
    gp_local.cov_type = ms_data.cov_type;

    GPData gp_regional;
    gp_regional.n_obs = ms_data.n_obs;
    gp_regional.nn = ms_data.nn_regional;
    gp_regional.coords = ms_data.coords;
    gp_regional.nn_idx = ms_data.nn_idx_regional;
    gp_regional.nn_dist = ms_data.nn_dist_regional;
    gp_regional.nn_neighbor_dist = ms_data.nn_neighbor_dist_regional;
    gp_regional.nn_order = ms_data.nn_order_regional;
    gp_regional.nn_order_inv = ms_data.nn_order_inv_regional;
    gp_regional.cov_type = ms_data.cov_type;

    // Compute log-likelihood for each scale
    T ll_local = gp_nngp_log_lik_t(w_local, sigma2_local, phi_local, gp_local);
    T ll_regional = gp_nngp_log_lik_t(w_regional, sigma2_regional, phi_regional, gp_regional);

    return ll_local + ll_regional;
}

// =============================================================================
// Templated GP priors
// =============================================================================

// PC prior on sigma2: P(sigma > U) = alpha => sigma ~ Exp(rate = -log(alpha)/U)
template<typename T>
T log_prior_sigma2_pc_t(const T& sigma2, double U, double alpha) {
    return tulpa::log_prior_sigma2_pc(sigma2, U, alpha);
}

// Uniform prior on phi within bounds
template<typename T>
T log_prior_phi_uniform_t(const T& phi, double lower, double upper) {
    double phi_val = get_value(phi);
    if (phi_val < lower || phi_val > upper) {
        return T(-1e10);
    }
    return T(-std::log(upper - lower));
}

}  // namespace tulpa_gp

#endif  // TULPA_HMC_GP_AUTODIFF_H
