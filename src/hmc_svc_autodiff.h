// hmc_svc_autodiff.h
// Templated SVC (Spatially-Varying Coefficients) with NNGP approximation
// Works with both double and ad::Var for automatic differentiation

#ifndef TULPA_HMC_SVC_AUTODIFF_H
#define TULPA_HMC_SVC_AUTODIFF_H

#define _USE_MATH_DEFINES
#include <vector>
#include <cmath>
#include <algorithm>
#include "autodiff_utils.h"
#include "hmc_svc.h"  // For SVCData and CovType
#include "nngp_cond.h"  // shared Vecchia conditional kernel (factor/krige/floor)

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace tulpa_svc_ad {

using tulpa_svc::SVCData;
using tulpa_svc::CovType;
using namespace tulpa::math;

// =============================================================================
// Templated covariance functions
// =============================================================================

// Exponential covariance: sigma^2 * exp(-d / phi)
template<typename T>
inline T cov_exponential(double d, const T& sigma2, const T& phi) {
    return sigma2 * safe_exp(-T(d) / phi);
}

// Matern 3/2 covariance: sigma^2 * (1 + sqrt(3)*d/phi) * exp(-sqrt(3)*d/phi)
template<typename T>
inline T cov_matern32(double d, const T& sigma2, const T& phi) {
    T r = T(std::sqrt(3.0) * d) / phi;
    return sigma2 * (T(1.0) + r) * safe_exp(-r);
}

// Gaussian (squared exponential) covariance: sigma^2 * exp(-(d/phi)^2)
template<typename T>
inline T cov_gaussian(double d, const T& sigma2, const T& phi) {
    T r = T(d) / phi;
    return sigma2 * safe_exp(-r * r);
}

// Spherical covariance
template<typename T>
inline T cov_spherical(double d, const T& sigma2, const T& phi) {
    double phi_val = get_value(phi);
    if (d >= phi_val) return T(0.0);
    T r = T(d) / phi;
    return sigma2 * (T(1.0) - T(1.5) * r + T(0.5) * r * r * r);
}

// Generic covariance function dispatcher
template<typename T>
inline T compute_cov(double d, const T& sigma2, const T& phi, CovType cov_type) {
    switch (cov_type) {
        case CovType::EXPONENTIAL:
            return cov_exponential(d, sigma2, phi);
        case CovType::MATERN:
            return cov_matern32(d, sigma2, phi);
        case CovType::GAUSSIAN:
            return cov_gaussian(d, sigma2, phi);
        case CovType::SPHERICAL:
            return cov_spherical(d, sigma2, phi);
        default:
            return cov_exponential(d, sigma2, phi);
    }
}

// =============================================================================
// Templated Cholesky decomposition for small matrices
// =============================================================================

// The SVC conditioning constants live in tulpa_svc (hmc_svc.h) so this autodiff
// kernel and its double twin there read the same values.
using tulpa_svc::kSvcJitter;
using tulpa_svc::kSvcVarFloor;

// Cholesky / triangular solves: thin aliases over the shared NNGP kernel, kept
// so existing SVC call sites read unchanged.
template<typename T>
inline bool cholesky_decomp(const std::vector<T>& A, int n, std::vector<T>& L) {
    return tulpa_nngp::chol_decomp(A, n, L, kSvcJitter);
}

template<typename T>
inline void solve_lower(const std::vector<T>& L, int n, const std::vector<T>& b, std::vector<T>& y) {
    tulpa_nngp::solve_lower(L, n, b, y);
}

template<typename T>
inline void solve_upper(const std::vector<T>& L, int n, const std::vector<T>& y, std::vector<T>& x) {
    tulpa_nngp::solve_upper(L, n, y, x);
}

// =============================================================================
// Templated NNGP log-likelihood
// =============================================================================

// Compute NNGP log-likelihood for a single SVC term
// w: vector of SVC values at each location (length n_obs)
// sigma2: spatial variance
// phi: spatial range parameter
// Returns log p(w | sigma2, phi) under NNGP approximation
template<typename T>
T nngp_log_lik(
    const std::vector<T>& w,
    const T& sigma2,
    const T& phi,
    const SVCData& svc_data
) {
    int N = svc_data.n_obs;
    int nn = svc_data.nn;

    // Size validation, matching the double twin: the neighbour tables arrive
    // from R and are indexed below as raw offsets.
    if ((int)svc_data.nn_order.size() < N) return T(-INFINITY);
    if ((int)svc_data.nn_idx.size() < N * nn) return T(-INFINITY);
    if ((int)svc_data.nn_dist.size() < N * nn) return T(-INFINITY);
    if ((int)w.size() < N) return T(-INFINITY);
    if ((int)svc_data.coords.size() < 2 * N) return T(-INFINITY);

    T log_lik = T(0.0);

    // First observation: marginal N(0, sigma2)
    int first_idx = svc_data.nn_order[0];
    if (first_idx < 0 || first_idx >= N) return T(-INFINITY);
    log_lik = log_lik + tulpa_nngp::marginal_log_density(w[first_idx], sigma2);

    // Remaining observations: conditional on neighbors
    for (int i = 1; i < N; i++) {
        int obs_idx = svc_data.nn_order[i];
        if (obs_idx < 0 || obs_idx >= N) return T(-INFINITY);

        // Count actual neighbors (early observations have fewer), through the
        // shared left-packed scan the double twin and the analytic gradient
        // also run.
        const int n_neighbors = tulpa_nngp::nngp_row_neighbours(
            svc_data.nn_idx.data() + (std::size_t)i * nn, /*stride=*/1, nn,
            (int)svc_data.nn_order.size());

        if (n_neighbors == 0) {
            // No neighbors: marginal
            log_lik = log_lik + tulpa_nngp::marginal_log_density(w[obs_idx], sigma2);
            continue;
        }

        // Build covariance vector c(s_i, s_{N(i)}) and matrix C(s_{N(i)}, s_{N(i)})
        std::vector<T> c_vec(n_neighbors);
        std::vector<T> C_mat(n_neighbors * n_neighbors);

        // c_vec: covariances between obs i and its neighbors
        for (int j = 0; j < n_neighbors; j++) {
            int nn_flat_idx = i * nn + j;
            double d = svc_data.nn_dist[nn_flat_idx];
            c_vec[j] = compute_cov(d, sigma2, phi, svc_data.cov_type);
        }

        // Resolve each neighbour slot to its location once. nngp_row_neighbours
        // has already bounded the nn_order lookup; what remains to check is the
        // value it returns, which indexes coords and w.
        std::vector<int> nb_loc(n_neighbors);
        for (int j = 0; j < n_neighbors; j++) {
            const int loc = svc_data.nn_order[svc_data.nn_idx[i * nn + j] - 1];
            if (loc < 0 || loc >= N) return T(-INFINITY);
            nb_loc[j] = loc;
        }

        // C_mat: covariances among neighbors
        for (int j1 = 0; j1 < n_neighbors; j1++) {
            for (int j2 = 0; j2 < n_neighbors; j2++) {
                if (j1 == j2) {
                    C_mat[j1 * n_neighbors + j2] = sigma2;
                } else {
                    // Distance between neighbors (fixed, not dependent on
                    // params). SVCData::coords is a flat stride-2 buffer, so
                    // the two columns are the whole domain here.
                    double dx = svc_data.coords[nb_loc[j1] * 2] -
                                svc_data.coords[nb_loc[j2] * 2];
                    double dy = svc_data.coords[nb_loc[j1] * 2 + 1] -
                                svc_data.coords[nb_loc[j2] * 2 + 1];
                    double d12 = std::sqrt(dx * dx + dy * dy);
                    C_mat[j1 * n_neighbors + j2] = compute_cov(d12, sigma2, phi, svc_data.cov_type);
                }
            }
        }

        // Gather the neighbour values in c_vec order, then the shared kernel:
        // factor, krige, floor. The SVC constants blend at the floor to keep a
        // little gradient (see kSvcJitter / kSvcVarFloor).
        std::vector<T> w_nb(n_neighbors);
        for (int j = 0; j < n_neighbors; j++) w_nb[j] = w[nb_loc[j]];
        T cond_mean, cond_var;
        if (!tulpa_nngp::cond_moments(C_mat, c_vec, w_nb, n_neighbors, sigma2,
                                      kSvcJitter, kSvcVarFloor,
                                      tulpa_nngp::VarFloor::Blend,
                                      cond_mean, cond_var)) {
            return T(-INFINITY);
        }
        log_lik = log_lik + tulpa_nngp::cond_log_density(w[obs_idx], cond_mean,
                                                         cond_var);
    }

    return log_lik;
}

// =============================================================================
// SVC contribution to linear predictor
// =============================================================================

// Compute SVC contribution to linear predictor for all observations
// eta_svc[i] = sum_j X_svc[i,j] * w_j[i]
template<typename T>
void compute_svc_eta(
    const std::vector<T>& w_flat,  // n_obs x n_svc flattened
    const SVCData& svc_data,
    std::vector<T>& eta_svc         // Output: length n_obs
) {
    int N = svc_data.n_obs;
    int n_svc = svc_data.n_svc;

    eta_svc.assign(N, T(0.0));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < n_svc; j++) {
            // w_flat is stored as [w1[1..N], w2[1..N], ...]
            T w_ij = w_flat[j * N + i];
            double x_ij = svc_data.X_svc[i * n_svc + j];
            eta_svc[i] = eta_svc[i] + T(x_ij) * w_ij;
        }
    }
}

// =============================================================================
// Identification of each term's global level
// =============================================================================

// beta_j and mean(w_j) are not separately identifiable: the term contributes
// eta_i += X_svc[i,j] * w_j(s_i), so w_j -> w_j + c and beta_j -> beta_j - c
// leaves eta exactly unchanged. An NNGP field is PROPER, so that direction
// already carries a prior and needs no penalty supplying one; it needs only to
// be removed from the likelihood, which is what centring w_j on its way into
// eta does. See tulpa/sum_to_zero.h for why a penalty on the sum is the wrong
// instrument on a proper field.
//
// The centring and the eta it feeds are one function so no path can compute
// svc_eta from an uncentred field.
template<typename T>
void svc_center_eta(
    std::vector<T>& w_flat,          // n_svc x n_obs, term-major; centred in place
    const SVCData& svc_data,
    std::vector<T>& eta_svc          // Output: length n_obs
) {
    tulpa::s2z_centre_blocks(w_flat.data(), svc_data.n_svc, svc_data.n_obs);
    compute_svc_eta(w_flat, svc_data, eta_svc);
}

} // namespace tulpa_svc_ad

#endif // TULPA_HMC_SVC_AUTODIFF_H
