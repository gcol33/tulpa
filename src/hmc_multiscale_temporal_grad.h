// hmc_multiscale_temporal_grad.h
// Hand-coded gradients for Multi-scale Temporal decomposition
// Provides O(n) analytical gradients for trend + seasonal + short-term components

#ifndef TULPA_HMC_MULTISCALE_TEMPORAL_GRAD_H
#define TULPA_HMC_MULTISCALE_TEMPORAL_GRAD_H

#include <vector>
#include <cmath>
#include <cstring>
#include "hmc_temporal_multiscale.h"
#include "hmc_tvc_grad.h"  // canonical rw1_grad_w used here in sigma2 form

namespace tulpa_temporal_grad {

using tulpa_temporal::MultiscaleTemporalData;
using tulpa_temporal::TemporalType;

// Structure to hold multiscale temporal gradient results
// Pre-allocate with init() and reuse across iterations to avoid heap churn
struct MultiscaleTemporalGradients {
    std::vector<double> grad_trend;           // Gradient w.r.t. trend effects
    std::vector<double> grad_seasonal;        // Gradient w.r.t. seasonal effects
    std::vector<double> grad_short_term;      // Gradient w.r.t. short-term effects
    double grad_log_sigma2_trend = 0.0;
    double grad_log_sigma2_seasonal = 0.0;
    double grad_log_sigma2_short = 0.0;
    double grad_logit_rho_short = 0.0;

    void init(int n_trend, int n_seasonal, int n_short) {
        if ((int)grad_trend.size() != n_trend) grad_trend.resize(n_trend);
        if ((int)grad_seasonal.size() != n_seasonal) grad_seasonal.resize(n_seasonal);
        if ((int)grad_short_term.size() != n_short) grad_short_term.resize(n_short);
    }
};

// =============================================================================
// RW1 gradients (non-cyclic)
// =============================================================================

// RW1: log p(phi|sigma2) = -0.5 * (T-1) * log(2*pi*sigma2)
//                        - 0.5 / sigma2 * sum((phi[t] - phi[t-1])^2)
//
// Same gradient as tulpa_tvc::rw1_grad_w with tau = 1/sigma2; this
// wrapper exists so multiscale temporal callers can pass sigma2 directly.
inline void rw1_grad_phi(const double* phi, int n, double sigma2, double* grad_phi) {
    tulpa_tvc::rw1_grad_w(phi, n, 1.0 / (sigma2 + 1e-10), grad_phi);
}

inline double rw1_grad_log_sigma2(const double* phi, int n, double sigma2) {
    double quad = 0.0;
    for (int t = 1; t < n; t++) {
        double diff = phi[t] - phi[t-1];
        quad += diff * diff;
    }
    // d/d(sigma2) [log_GMRF] = -0.5*(n-1)/sigma2 + 0.5*quad/sigma2^2
    // Chain rule: d/d(log_sigma2) = d/d(sigma2) * sigma2
    //           = -0.5*(n-1) + 0.5*quad/sigma2
    return -0.5 * (n - 1) + 0.5 * quad / (sigma2 + 1e-10);
}

// =============================================================================
// RW1 gradients (cyclic - for seasonal)
// =============================================================================

// Cyclic RW1: adds connection from last to first
inline void rw1_cyclic_grad_phi(const double* phi, int n, double sigma2, double* grad_phi) {
    if (n < 2) {
        for (int t = 0; t < n; t++) grad_phi[t] = 0.0;
        return;
    }

    double inv_sigma2 = 1.0 / (sigma2 + 1e-10);

    // Each diff d[t] = phi[t] - phi[t-1] contributes:
    //   grad_phi[t]   -= inv_sigma2 * d[t]   (from its own diff)
    //   grad_phi[t-1] += inv_sigma2 * d[t]   (from next diff)
    // Accumulate in-place without modulo by handling boundaries explicitly.
    std::memset(grad_phi, 0, n * sizeof(double));

    // Interior diffs: t = 1..n-1
    for (int t = 1; t < n; t++) {
        double neg_inv_d = -inv_sigma2 * (phi[t] - phi[t-1]);
        grad_phi[t]   += neg_inv_d;
        grad_phi[t-1] -= neg_inv_d;
    }
    // Cyclic wrap: diff from phi[0] - phi[n-1]
    {
        double neg_inv_d = -inv_sigma2 * (phi[0] - phi[n-1]);
        grad_phi[0]   += neg_inv_d;
        grad_phi[n-1] -= neg_inv_d;
    }
}

inline double rw1_cyclic_grad_log_sigma2(const double* phi, int n, double sigma2) {
    double quad = 0.0;
    // Interior diffs
    for (int t = 1; t < n; t++) {
        double diff = phi[t] - phi[t-1];
        quad += diff * diff;
    }
    // Cyclic wrap
    {
        double diff = phi[0] - phi[n-1];
        quad += diff * diff;
    }
    return -0.5 * n + 0.5 * quad / (sigma2 + 1e-10);
}

// =============================================================================
// RW2 gradients
// =============================================================================

// Same gradient as tulpa_tvc::rw2_grad_w with tau = 1/sigma2; this wrapper
// exists so multiscale temporal callers can pass sigma2 directly, matching
// rw1_grad_phi above.
inline void rw2_grad_phi(const double* phi, int n, double sigma2, double* grad_phi) {
    tulpa_tvc::rw2_grad_w(phi, n, 1.0 / (sigma2 + 1e-10), grad_phi);
}

inline double rw2_grad_log_sigma2(const double* phi, int n, double sigma2) {
    double quad = 0.0;
    for (int t = 2; t < n; t++) {
        double d = phi[t] - 2.0 * phi[t-1] + phi[t-2];
        quad += d * d;
    }
    // d/d(sigma2) = -0.5*(n-2)/sigma2 + 0.5*quad/sigma2^2
    // Chain rule: d/d(log_sigma2) = d/d(sigma2) * sigma2
    return -0.5 * (n - 2) + 0.5 * quad / (sigma2 + 1e-10);
}

// =============================================================================
// AR1 gradients
// =============================================================================

// Same gradient as tulpa_tvc::ar1_grad_w with tau = 1/sigma2; wrapper so
// multiscale temporal callers can pass sigma2 directly.
inline void ar1_grad_phi(const double* phi, int n, double sigma2, double rho, double* grad_phi) {
    tulpa_tvc::ar1_grad_w(phi, n, 1.0 / (sigma2 + 1e-10), rho, grad_phi);
}

inline double ar1_grad_log_sigma2(const double* phi, int n, double sigma2, double rho) {
    double one_m_rho2 = 1.0 - rho * rho + 1e-10;

    // AR1 has n terms: 1 marginal + (n-1) conditional
    // d/d(sigma2) [log_AR1] = -0.5*n/sigma2 + 0.5*quad/sigma2^2
    double grad = -0.5 * n;

    // Quadratic form: (1-rho^2)*phi[0]^2 + sum(resid^2)
    double quad = one_m_rho2 * phi[0] * phi[0];
    for (int t = 1; t < n; t++) {
        double resid = phi[t] - rho * phi[t-1];
        quad += resid * resid;
    }
    grad += 0.5 * quad / (sigma2 + 1e-10);

    // Chain rule: d/d(log_sigma2) = d/d(sigma2) * sigma2
    // Already applied: -0.5*n + 0.5*quad/sigma2 IS the result after chain rule
    return grad;
}

// Same gradient as tulpa_tvc::ar1_grad_logit_rho with tau = 1/sigma2.
inline double ar1_grad_logit_rho(const double* phi, int n, double sigma2, double rho) {
    return tulpa_tvc::ar1_grad_logit_rho(phi, n, 1.0 / (sigma2 + 1e-10), rho);
}

// =============================================================================
// IID gradients
// =============================================================================

inline void iid_grad_phi(const double* phi, int n, double sigma2, double* grad_phi) {
    double inv_sigma2 = 1.0 / (sigma2 + 1e-10);
    for (int t = 0; t < n; t++) {
        grad_phi[t] = -inv_sigma2 * phi[t];
    }
}

inline double iid_grad_log_sigma2(const double* phi, int n, double sigma2) {
    double quad = 0.0;
    for (int t = 0; t < n; t++) {
        quad += phi[t] * phi[t];
    }
    // d/d(sigma2) = -0.5*n/sigma2 + 0.5*quad/sigma2^2
    // Chain rule: d/d(log_sigma2) = d/d(sigma2) * sigma2
    return -0.5 * n + 0.5 * quad / (sigma2 + 1e-10);
}

// =============================================================================
// PC prior gradient for sigma2 (on log_sigma2 scale)
// =============================================================================

// =============================================================================
// Full multiscale temporal prior gradients
// =============================================================================

inline void multiscale_temporal_prior_gradients(
    const double* trend, int n_trend,
    const double* seasonal, int n_seasonal,
    const double* short_term, int n_short,
    double sigma2_trend, double sigma2_seasonal, double sigma2_short,
    double rho_short,
    const MultiscaleTemporalData& temp_data,
    MultiscaleTemporalGradients& grads
) {
    // Initialize gradients (reuse allocated memory)
    grads.init(n_trend, n_seasonal, n_short);
    if (n_trend > 0) std::memset(grads.grad_trend.data(), 0, n_trend * sizeof(double));
    if (n_seasonal > 0) std::memset(grads.grad_seasonal.data(), 0, n_seasonal * sizeof(double));
    if (n_short > 0) std::memset(grads.grad_short_term.data(), 0, n_short * sizeof(double));
    grads.grad_log_sigma2_trend = 0.0;
    grads.grad_log_sigma2_seasonal = 0.0;
    grads.grad_log_sigma2_short = 0.0;
    grads.grad_logit_rho_short = 0.0;

    // Trend component
    if (n_trend > 0) {
        if (temp_data.trend_type == TemporalType::RW1) {
            rw1_grad_phi(trend, n_trend, sigma2_trend, grads.grad_trend.data());
            grads.grad_log_sigma2_trend = rw1_grad_log_sigma2(trend, n_trend, sigma2_trend);
        } else if (temp_data.trend_type == TemporalType::RW2) {
            rw2_grad_phi(trend, n_trend, sigma2_trend, grads.grad_trend.data());
            grads.grad_log_sigma2_trend = rw2_grad_log_sigma2(trend, n_trend, sigma2_trend);
        }
    }

    // Seasonal component (cyclic RW1)
    if (n_seasonal > 0 && temp_data.seasonal_period > 0) {
        rw1_cyclic_grad_phi(seasonal, n_seasonal, sigma2_seasonal, grads.grad_seasonal.data());
        grads.grad_log_sigma2_seasonal = rw1_cyclic_grad_log_sigma2(seasonal, n_seasonal, sigma2_seasonal);
    }

    // Short-term component
    if (n_short > 0) {
        if (temp_data.short_term_type == TemporalType::AR1) {
            ar1_grad_phi(short_term, n_short, sigma2_short, rho_short, grads.grad_short_term.data());
            grads.grad_log_sigma2_short = ar1_grad_log_sigma2(short_term, n_short, sigma2_short, rho_short);
            grads.grad_logit_rho_short = ar1_grad_logit_rho(short_term, n_short, sigma2_short, rho_short);
        } else if (temp_data.short_term_type == TemporalType::IID) {
            iid_grad_phi(short_term, n_short, sigma2_short, grads.grad_short_term.data());
            grads.grad_log_sigma2_short = iid_grad_log_sigma2(short_term, n_short, sigma2_short);
        }
    }
}

} // namespace tulpa_temporal_grad

#endif // TULPA_HMC_MULTISCALE_TEMPORAL_GRAD_H
