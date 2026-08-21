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
// the gradient describe different models: kGpJitter is added to every diagonal
// pivot, matching the double copies.

// Returns false if not positive definite
template<typename T>
inline bool cholesky_decompose_t(const std::vector<T>& A, int n, std::vector<T>& L) {
    return tulpa_nngp::chol_decomp(A, n, L, kGpJitter);
}

// =============================================================================
// Templated NNGP log-likelihood
// =============================================================================

// The live NNGP density on the sampler path: tulpa_priors_gp.h evaluates it
// from the generic log posterior, and its templated scalar type is what carries
// the gradient through every autodiff mode.
//
// Written against the six arrays it reads rather than against GPData, because
// MultiscaleGPData holds two independent scales in members of its own and had
// to be transcribed into a temporary GPData to reach a GPData-shaped
// signature -- a copy of every neighbour array, nn_neighbor_dist included
// (n_obs * nn * nn doubles), on EVERY gradient evaluation. Both entry points
// below pass their own members straight in, so nothing is copied and both read
// exactly the same validation.
template<typename T>
T nngp_log_lik_arrays_t(
    const std::vector<T>& w,
    const T& sigma2,
    const T& phi,
    int N,
    int nn,
    const std::vector<int>& v_nn_order,
    const std::vector<int>& v_nn_idx,
    const std::vector<double>& v_nn_dist,
    const std::vector<double>& v_nn_neighbor_dist,
    const std::vector<double>& v_coords,
    CovType v_cov_type
) {

#if AUTODIFF_DEBUG
    static int call_count = 0;
    call_count++;
    Rcpp::Rcout << "[NNGP] Call #" << call_count << ": N=" << N << ", nn=" << nn
                << ", sigma2=" << get_value(sigma2) << ", phi=" << get_value(phi)
                << ", w.size()=" << w.size() << "\n";
    R_FlushConsole();
#endif

    // Bounds validation
    if (v_nn_order.size() < (size_t)N) return T(-INFINITY);
    if (v_nn_idx.size() < (size_t)(N * nn)) return T(-INFINITY);
    if (v_nn_dist.size() < (size_t)(N * nn)) return T(-INFINITY);
    if (v_nn_neighbor_dist.size() < (size_t)(N * nn * nn)) return T(-INFINITY);  // Critical: prevents segfault
    if (w.size() < (size_t)N) return T(-INFINITY);
    if (v_coords.size() < (size_t)(2 * N)) return T(-INFINITY);

    T log_lik = T(0.0);

    // First observation: marginal N(0, sigma2), through the same shared arm the
    // double twin uses so the two floor a degenerate sigma2 identically.
    int first_idx = v_nn_order[0];
    log_lik = log_lik + tulpa_nngp::marginal_log_density(w[first_idx], sigma2);

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
        int obs_idx = v_nn_order[i];

        // Bounds check
        if (obs_idx < 0 || obs_idx >= N) return T(-INFINITY);

        // Count actual neighbors, through the shared left-packed scan the double
        // twin and the analytic gradient also run.
        const int n_neighbors = tulpa_nngp::nngp_row_neighbours(
            v_nn_idx.data() + (std::size_t)i * nn, /*stride=*/1, nn,
            (int)v_nn_order.size());

        if (n_neighbors == 0) {
            // No neighbors: marginal
            log_lik = log_lik + tulpa_nngp::marginal_log_density(w[obs_idx], sigma2);
            continue;
        }

        // c_vec: covariances between obs i and its neighbors
        for (int j = 0; j < n_neighbors; j++) {
            int nn_flat_idx = i * nn + j;
            double d = v_nn_dist[nn_flat_idx];
            c_vec[j] = compute_cov_t(d, sigma2, phi, v_cov_type);
        }

        // C_mat: covariances among neighbors
        for (int j1 = 0; j1 < n_neighbors; j1++) {
            int raw_nn_idx1 = v_nn_idx[i * nn + j1];

            // Bounds check
            if (raw_nn_idx1 - 1 < 0 || raw_nn_idx1 - 1 >= (int)v_nn_order.size()) {
                return T(-INFINITY);
            }

            int nn_idx1 = v_nn_order[raw_nn_idx1 - 1];

            if (nn_idx1 < 0 || nn_idx1 * 2 + 1 >= (int)v_coords.size()) {
                return T(-INFINITY);
            }

            for (int j2 = 0; j2 < n_neighbors; j2++) {
                int raw_nn_idx2 = v_nn_idx[i * nn + j2];

                if (raw_nn_idx2 - 1 < 0 || raw_nn_idx2 - 1 >= (int)v_nn_order.size()) {
                    return T(-INFINITY);
                }

                if (j1 == j2) {
                    C_mat[j1 * n_neighbors + j2] = sigma2;
                } else {
                    // Use cached pairwise neighbor distances
                    double d12 = v_nn_neighbor_dist[i * nn * nn + j1 * nn + j2];
                    C_mat[j1 * n_neighbors + j2] = compute_cov_t(d12, sigma2, phi, v_cov_type);
                }
            }
        }

        // Gather the neighbour values in c_vec order.
        std::vector<T> c_small(c_vec.begin(), c_vec.begin() + n_neighbors);
        std::vector<T> w_nb(n_neighbors);
        for (int j = 0; j < n_neighbors; j++) {
            int raw_nn_idx = v_nn_idx[i * nn + j];

            if (raw_nn_idx - 1 < 0 || raw_nn_idx - 1 >= (int)v_nn_order.size()) {
                return T(-INFINITY);
            }

            int nn_orig_idx = v_nn_order[raw_nn_idx - 1];

            if (nn_orig_idx < 0 || nn_orig_idx >= (int)w.size()) {
                return T(-INFINITY);
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
            return T(-INFINITY);  // Not positive definite
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

// One NNGP field held in a GPData.
template<typename T>
T gp_nngp_log_lik_t(
    const std::vector<T>& w,
    const T& sigma2,
    const T& phi,
    const GPData& gp_data
) {
    return nngp_log_lik_arrays_t(w, sigma2, phi, gp_data.n_obs, gp_data.nn,
                                 gp_data.nn_order, gp_data.nn_idx,
                                 gp_data.nn_dist, gp_data.nn_neighbor_dist,
                                 gp_data.coords, gp_data.cov_type);
}

// =============================================================================
// Templated multi-scale GP log-likelihood
// =============================================================================

// The two scales share the coordinates and the covariance family and carry
// their own neighbour topology, so each is one call against its own members.
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
    T ll_local = nngp_log_lik_arrays_t(
        w_local, sigma2_local, phi_local, ms_data.n_obs, ms_data.nn_local,
        ms_data.nn_order_local, ms_data.nn_idx_local, ms_data.nn_dist_local,
        ms_data.nn_neighbor_dist_local, ms_data.coords, ms_data.cov_type);
    T ll_regional = nngp_log_lik_arrays_t(
        w_regional, sigma2_regional, phi_regional, ms_data.n_obs,
        ms_data.nn_regional, ms_data.nn_order_regional, ms_data.nn_idx_regional,
        ms_data.nn_dist_regional, ms_data.nn_neighbor_dist_regional,
        ms_data.coords, ms_data.cov_type);

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

}  // namespace tulpa_gp

#endif  // TULPA_HMC_GP_AUTODIFF_H
