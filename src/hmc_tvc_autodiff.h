// hmc_tvc_autodiff.h
// Templated TVC (Temporally-Varying Coefficients) functions
// Works with both double and ad::Var for automatic differentiation

#ifndef TULPA_HMC_TVC_AUTODIFF_H
#define TULPA_HMC_TVC_AUTODIFF_H

#include <vector>
#include <cmath>
#include "autodiff_utils.h"
#include "hmc_tvc.h"  // For TVCData and TemporalType

namespace tulpa_tvc_ad {

using tulpa_tvc::TVCData;
using tulpa_temporal::TemporalType;
using namespace tulpa::math;

// =============================================================================
// Templated RW1 quadratic form
// =============================================================================

// RW1: sum of (w[t] - w[t-1])^2
template<typename T>
T rw1_quadratic_form(const T* w, int n_times, bool cyclic = false) {
    T quad = T(0.0);
    for (int t = 1; t < n_times; t++) {
        T diff = w[t] - w[t - 1];
        quad = quad + diff * diff;
    }
    if (cyclic && n_times > 1) {
        T diff = w[0] - w[n_times - 1];
        quad = quad + diff * diff;
    }
    return quad;
}

// =============================================================================
// Templated RW2 quadratic form
// =============================================================================

// RW2: sum of (w[t] - 2*w[t-1] + w[t-2])^2
template<typename T>
T rw2_quadratic_form(const T* w, int n_times, bool cyclic = false) {
    T quad = T(0.0);
    for (int t = 2; t < n_times; t++) {
        T diff = w[t] - T(2.0) * w[t - 1] + w[t - 2];
        quad = quad + diff * diff;
    }
    if (cyclic && n_times > 2) {
        // Wrap around
        T diff1 = w[0] - T(2.0) * w[n_times - 1] + w[n_times - 2];
        T diff2 = w[1] - T(2.0) * w[0] + w[n_times - 1];
        quad = quad + diff1 * diff1 + diff2 * diff2;
    }
    return quad;
}

// =============================================================================
// Templated AR1 log-density
// =============================================================================

// AR1: w[t] | w[t-1] ~ N(rho * w[t-1], 1/tau)
// Returns log p(w | rho, tau)
template<typename T>
T ar1_log_density(const T* w, int n_times, const T& rho, const T& tau) {
    T log_dens = T(0.0);

    // Stationary variance: 1 / (tau * (1 - rho^2))
    T one_minus_rho2 = T(1.0) - rho * rho;
    T var_stationary = T(1.0) / (tau * one_minus_rho2 + T(1e-10));

    // First time point: N(0, var_stationary)
    log_dens = log_dens - T(0.5) * safe_log(T(2.0 * M_PI) * var_stationary);
    log_dens = log_dens - T(0.5) * w[0] * w[0] / var_stationary;

    // Subsequent time points: N(rho * w[t-1], 1/tau)
    T cond_var = T(1.0) / tau;
    for (int t = 1; t < n_times; t++) {
        T resid = w[t] - rho * w[t - 1];
        log_dens = log_dens - T(0.5) * safe_log(T(2.0 * M_PI) * cond_var);
        log_dens = log_dens - T(0.5) * resid * resid / cond_var;
    }

    return log_dens;
}

// =============================================================================
// Templated TVC term log-prior
// =============================================================================

// Log-prior for a single TVC term's temporal trajectory
template<typename T>
T tvc_term_log_prior(
    const T* w,
    int n_times,
    TemporalType structure,
    const T& tau,
    const T& rho,
    bool cyclic = false
) {
    T log_prior = T(0.0);

    if (structure == TemporalType::RW1) {
        T quad = rw1_quadratic_form(w, n_times, cyclic);
        int rank = cyclic ? n_times : (n_times - 1);
        log_prior = log_prior + T(0.5 * rank) * safe_log(tau);
        log_prior = log_prior - T(0.5) * tau * quad;

    } else if (structure == TemporalType::RW2) {
        T quad = rw2_quadratic_form(w, n_times, cyclic);
        int rank = cyclic ? n_times : (n_times - 2);
        log_prior = log_prior + T(0.5 * rank) * safe_log(tau);
        log_prior = log_prior - T(0.5) * tau * quad;

    } else if (structure == TemporalType::AR1) {
        log_prior = ar1_log_density(w, n_times, rho, tau);

    } else {
        // IID: N(0, 1/tau) for each time point
        log_prior = log_prior + T(0.5 * n_times) * safe_log(tau);
        T quad = T(0.0);
        for (int t = 0; t < n_times; t++) {
            quad = quad + w[t] * w[t];
        }
        log_prior = log_prior - T(0.5) * tau * quad;
    }

    return log_prior;
}

// =============================================================================
// Full TVC log-prior (all terms, all groups)
// =============================================================================

// Compute log-prior for all TVC terms
// w_flat: all TVC values (n_groups * n_tvc * n_times, flattened)
// Layout: w_flat[(g * n_tvc + j) * n_times + t]
template<typename T>
T tvc_log_prior(
    const std::vector<T>& w_flat,
    const TVCData& tvc_data,
    const std::vector<T>& tau,
    const std::vector<T>& rho
) {
    int n_times = tvc_data.n_times;
    int n_tvc = tvc_data.n_tvc;
    int n_groups = tvc_data.n_groups;

    T log_prior = T(0.0);

    for (int g = 0; g < n_groups; g++) {
        for (int j = 0; j < n_tvc; j++) {
            // Pointer to w[g,j,*]
            int offset = (g * n_tvc + j) * n_times;
            std::vector<T> w_jg(n_times);
            for (int t = 0; t < n_times; t++) {
                w_jg[t] = w_flat[offset + t];
            }

            T rho_j = (tvc_data.structure == TemporalType::AR1) ? rho[j] : T(0.0);
            log_prior = log_prior + tvc_term_log_prior(
                w_jg.data(), n_times, tvc_data.structure,
                tau[j], rho_j, tvc_data.cyclic
            );
        }
    }

    return log_prior;
}

// =============================================================================
// TVC contribution to linear predictor
// =============================================================================

// Compute TVC contribution to linear predictor for all observations
// eta_tvc[i] = sum_j X_tvc[i,j] * w[group[i], j, time[i]]
template<typename T>
void compute_tvc_eta(
    const std::vector<T>& w_flat,  // n_groups * n_tvc * n_times
    const TVCData& tvc_data,
    std::vector<T>& eta_tvc         // Output: length n_obs
) {
    int N = tvc_data.n_obs;
    int n_times = tvc_data.n_times;
    int n_tvc = tvc_data.n_tvc;

    eta_tvc.assign(N, T(0.0));

    for (int i = 0; i < N; i++) {
        int t = tvc_data.time_index[i] - 1;  // 0-based
        int g = tvc_data.group_index[i] - 1;  // 0-based

        for (int j = 0; j < n_tvc; j++) {
            // w_flat layout: [(g * n_tvc + j) * n_times + t]
            T w_jgt = w_flat[(g * n_tvc + j) * n_times + t];
            double x_ij = tvc_data.X_tvc[i * n_tvc + j];
            eta_tvc[i] = eta_tvc[i] + T(x_ij) * w_jgt;
        }
    }
}

// =============================================================================
// Sum-to-zero constraint for identifiability
// =============================================================================

// Soft sum-to-zero constraint for TVC (each term and group)
template<typename T>
T tvc_sum_to_zero_penalty(
    const std::vector<T>& w_flat,
    const TVCData& tvc_data,
    double lambda = 0.001
) {
    int n_times = tvc_data.n_times;
    int n_tvc = tvc_data.n_tvc;
    int n_groups = tvc_data.n_groups;

    T penalty = T(0.0);

    for (int g = 0; g < n_groups; g++) {
        for (int j = 0; j < n_tvc; j++) {
            T sum = T(0.0);
            for (int t = 0; t < n_times; t++) {
                sum = sum + w_flat[(g * n_tvc + j) * n_times + t];
            }
            penalty = penalty - T(0.5 * lambda) * sum * sum;
        }
    }

    return penalty;
}

// =============================================================================
// TVC hyperparameter priors
// =============================================================================

// Half-Cauchy prior on tau (precision)
// Actually this is on sigma = 1/sqrt(tau), then transformed
template<typename T>
T log_prior_tau_half_cauchy(const T& log_tau, double scale) {
    T tau = safe_exp(log_tau);
    T sigma = T(1.0) / safe_sqrt(tau);
    // Half-Cauchy: 2 / (pi * scale * (1 + (sigma/scale)^2))
    // log form + Jacobian for log_tau -> tau -> sigma
    return T(std::log(2.0 / M_PI / scale))
           - safe_log(T(1.0) + sigma * sigma / T(scale * scale))
           - T(0.5) * log_tau;  // Jacobian: d(sigma)/d(tau) * d(tau)/d(log_tau)
}

// Gamma prior on tau
template<typename T>
T log_prior_tau_gamma(const T& log_tau, double shape, double rate) {
    T tau = safe_exp(log_tau);
    // Gamma(shape, rate): tau^{shape-1} * exp(-rate*tau) * rate^shape / Gamma(shape)
    // + Jacobian for log transform
    return T(shape - 1.0) * log_tau - rate * tau + log_tau;
}

// Beta prior on rho (AR1 correlation on (-1, 1))
// Transform: u = (rho + 1) / 2, u ~ Beta(a, b)
template<typename T>
T log_prior_rho_beta(const T& logit_rho_scaled, double a, double b) {
    // logit_rho_scaled = logit((rho + 1)/2) = log((rho+1)/(1-rho))
    // rho = (2 * inv_logit(logit_rho_scaled)) - 1
    T u = inv_logit(logit_rho_scaled);
    T rho = T(2.0) * u - T(1.0);

    // Beta density on u
    T log_dens = T(a - 1.0) * safe_log(u) + T(b - 1.0) * safe_log(T(1.0) - u);
    // Jacobian for logit transform: u * (1 - u)
    log_dens = log_dens + safe_log(u) + safe_log(T(1.0) - u);

    return log_dens;
}

} // namespace tulpa_tvc_ad

#endif // TULPA_HMC_TVC_AUTODIFF_H
