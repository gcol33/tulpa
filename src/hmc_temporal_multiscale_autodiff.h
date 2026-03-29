// hmc_temporal_multiscale_autodiff.h
// Templated multi-scale temporal decomposition functions
// Works with both double and ad::Var for automatic differentiation

#ifndef TULPA_HMC_TEMPORAL_MULTISCALE_AUTODIFF_H
#define TULPA_HMC_TEMPORAL_MULTISCALE_AUTODIFF_H

#include <vector>
#include <cmath>
#include "autodiff_utils.h"
#include "hmc_temporal_multiscale.h"  // For MultiscaleTemporalData and TemporalType

namespace tulpa_multiscale_ad {

using tulpa_temporal::MultiscaleTemporalData;
using tulpa_temporal::TemporalType;
using namespace tulpa::math;

// =============================================================================
// Templated RW1 log-likelihood (intrinsic first-order random walk)
// =============================================================================

template<typename T>
T rw1_log_lik(const std::vector<T>& phi, const T& sigma2, bool cyclic = false) {
    int n = static_cast<int>(phi.size());
    if (n < 2) return T(0.0);

    T log_lik = T(0.0);

    // First differences
    for (int t = 1; t < n; t++) {
        T diff = phi[t] - phi[t - 1];
        log_lik = log_lik - T(0.5) * diff * diff / sigma2;
    }

    // Cyclic: add connection from last to first
    if (cyclic) {
        T diff = phi[0] - phi[n - 1];
        log_lik = log_lik - T(0.5) * diff * diff / sigma2;
    }

    // Normalizing constant
    int n_diffs = cyclic ? n : (n - 1);
    log_lik = log_lik - T(0.5 * n_diffs) * safe_log(T(2.0 * M_PI) * sigma2);

    return log_lik;
}

// =============================================================================
// Templated RW2 log-likelihood (intrinsic second-order random walk)
// =============================================================================

template<typename T>
T rw2_log_lik(const std::vector<T>& phi, const T& sigma2, bool cyclic = false) {
    int n = static_cast<int>(phi.size());
    if (n < 3) return T(0.0);

    T log_lik = T(0.0);

    // Second differences
    for (int t = 2; t < n; t++) {
        T diff2 = phi[t] - T(2.0) * phi[t - 1] + phi[t - 2];
        log_lik = log_lik - T(0.5) * diff2 * diff2 / sigma2;
    }

    // Cyclic: add wrap-around connections
    if (cyclic) {
        T diff2_1 = phi[0] - T(2.0) * phi[n - 1] + phi[n - 2];
        T diff2_2 = phi[1] - T(2.0) * phi[0] + phi[n - 1];
        log_lik = log_lik - T(0.5) * diff2_1 * diff2_1 / sigma2;
        log_lik = log_lik - T(0.5) * diff2_2 * diff2_2 / sigma2;
    }

    // Normalizing constant
    int n_diffs = cyclic ? n : (n - 2);
    log_lik = log_lik - T(0.5 * n_diffs) * safe_log(T(2.0 * M_PI) * sigma2);

    return log_lik;
}

// =============================================================================
// Templated AR1 log-likelihood (stationary first-order autoregressive)
// =============================================================================

template<typename T>
T ar1_log_lik(const std::vector<T>& phi, const T& sigma2, const T& rho) {
    int n = static_cast<int>(phi.size());
    if (n < 2) return T(0.0);

    T log_lik = T(0.0);

    // Marginal distribution of first observation
    T one_minus_rho2 = T(1.0) - rho * rho;
    T marginal_var = sigma2 / (one_minus_rho2 + T(1e-10));
    log_lik = log_lik - T(0.5) * safe_log(T(2.0 * M_PI) * marginal_var);
    log_lik = log_lik - T(0.5) * phi[0] * phi[0] / marginal_var;

    // Conditional distributions
    for (int t = 1; t < n; t++) {
        T resid = phi[t] - rho * phi[t - 1];
        log_lik = log_lik - T(0.5) * safe_log(T(2.0 * M_PI) * sigma2);
        log_lik = log_lik - T(0.5) * resid * resid / sigma2;
    }

    return log_lik;
}

// =============================================================================
// Templated IID log-likelihood (independent identically distributed)
// =============================================================================

template<typename T>
T iid_log_lik(const std::vector<T>& phi, const T& sigma2) {
    int n = static_cast<int>(phi.size());
    T log_lik = T(0.0);

    for (int t = 0; t < n; t++) {
        log_lik = log_lik - T(0.5) * safe_log(T(2.0 * M_PI) * sigma2);
        log_lik = log_lik - T(0.5) * phi[t] * phi[t] / sigma2;
    }

    return log_lik;
}

// =============================================================================
// Templated multi-scale temporal log-likelihood
// Combined log-likelihood for trend + seasonal + short-term
// =============================================================================

template<typename T>
T multiscale_temporal_log_lik(
    const std::vector<T>& trend,
    const std::vector<T>& seasonal,
    const std::vector<T>& short_term,
    const T& sigma2_trend,
    const T& sigma2_seasonal,
    const T& sigma2_short,
    const T& rho_short,
    const MultiscaleTemporalData& temp_data
) {
    T log_lik = T(0.0);

    // Trend component
    if (temp_data.trend_type == TemporalType::RW1 && !trend.empty()) {
        log_lik = log_lik + rw1_log_lik(trend, sigma2_trend, false);
    } else if (temp_data.trend_type == TemporalType::RW2 && !trend.empty()) {
        log_lik = log_lik + rw2_log_lik(trend, sigma2_trend, false);
    }

    // Seasonal component (always cyclic RW1)
    if (temp_data.seasonal_period > 0 && !seasonal.empty()) {
        log_lik = log_lik + rw1_log_lik(seasonal, sigma2_seasonal, true);  // Cyclic
    }

    // Short-term component
    if (temp_data.short_term_type == TemporalType::AR1 && !short_term.empty()) {
        log_lik = log_lik + ar1_log_lik(short_term, sigma2_short, rho_short);
    } else if (temp_data.short_term_type == TemporalType::IID && !short_term.empty()) {
        log_lik = log_lik + iid_log_lik(short_term, sigma2_short);
    }

    return log_lik;
}

// =============================================================================
// Templated PC prior for temporal variance
// =============================================================================

template<typename T>
T log_prior_sigma2_temporal_pc(const T& sigma2, double U, double alpha) {
    T rate = T(-std::log(alpha) / U);
    T sigma = safe_sqrt(sigma2);
    return safe_log(rate) - rate * sigma - safe_log(T(2.0) * sigma);
}

// =============================================================================
// Templated prior for AR1 rho: Beta(a, b) on (rho + 1) / 2
// =============================================================================

template<typename T>
T log_prior_rho(const T& rho, double a = 2.0, double b = 2.0) {
    // Transform to [0, 1]
    T x = (rho + T(1.0)) / T(2.0);
    // Beta log density (unnormalized)
    return T(a - 1.0) * safe_log(x) + T(b - 1.0) * safe_log(T(1.0) - x);
}

// =============================================================================
// Compute total temporal effect at each observation
// =============================================================================

template<typename T>
void compute_temporal_eta(
    const std::vector<T>& trend,
    const std::vector<T>& seasonal,
    const std::vector<T>& short_term,
    const MultiscaleTemporalData& temp_data,
    std::vector<T>& eta_temporal
) {
    int N = temp_data.n_obs;
    eta_temporal.resize(N);

    for (int i = 0; i < N; i++) {
        T effect = T(0.0);
        int t_idx = temp_data.time_index[i] - 1;  // Convert to 0-based

        // Trend contribution
        if (!trend.empty() && t_idx >= 0 &&
            t_idx < static_cast<int>(trend.size())) {
            effect = effect + trend[t_idx];
        }

        // Seasonal contribution (wrap around using modulo)
        if (temp_data.seasonal_period > 0 && !seasonal.empty()) {
            int s_idx = t_idx % temp_data.seasonal_period;
            if (s_idx >= 0 && s_idx < static_cast<int>(seasonal.size())) {
                effect = effect + seasonal[s_idx];
            }
        }

        // Short-term contribution
        if (!short_term.empty() && t_idx >= 0 &&
            t_idx < static_cast<int>(short_term.size())) {
            effect = effect + short_term[t_idx];
        }

        eta_temporal[i] = effect;
    }
}

}  // namespace tulpa_multiscale_ad

#endif  // TULPA_HMC_TEMPORAL_MULTISCALE_AUTODIFF_H
