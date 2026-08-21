// hmc_tvc_grad.h
// Hand-coded gradients for TVC (Temporally-Varying Coefficients)
// Provides O(n) analytical gradients for RW1, RW2, and AR1 temporal structures

#ifndef TULPA_HMC_TVC_GRAD_H
#define TULPA_HMC_TVC_GRAD_H

#include <vector>
#include <cmath>
#include "hmc_tvc.h"

namespace tulpa_tvc {

// =============================================================================
// Analytical RW1 gradients
// =============================================================================

// RW1 prior: log p(w|tau) = 0.5 * (T-1) * log(tau) - 0.5 * tau * sum((w[t] - w[t-1])^2)
// d/d(w[t]) = -tau * (2*w[t] - w[t-1] - w[t+1]) for interior
//           = -tau * (w[t] - w[t+1]) for t=0
//           = -tau * (w[t] - w[t-1]) for t=T-1
inline void rw1_grad_w(const double* w, int n_times, double tau, double* grad_w) {
    // A single level has no increment, so the RW1 prior is flat in it. Without
    // this guard the t == 0 branch below wins over t == n_times - 1 and reads
    // w[1] past the end.
    if (n_times < 2) {
        for (int t = 0; t < n_times; t++) grad_w[t] = 0.0;
        return;
    }
    for (int t = 0; t < n_times; t++) {
        if (t == 0) {
            grad_w[t] = -tau * (w[0] - w[1]);
        } else if (t == n_times - 1) {
            grad_w[t] = -tau * (w[t] - w[t-1]);
        } else {
            grad_w[t] = -tau * (2.0 * w[t] - w[t-1] - w[t+1]);
        }
    }
}

// d/d(log_tau) = 0.5 * (T-1) - 0.5 * tau * quad + log_tau_jacobian
//              = 0.5 * (T-1) - 0.5 * quad * tau  (Jacobian cancels in chain rule)
inline double rw1_grad_log_tau(const double* w, int n_times, double tau) {
    double quad = tulpa_temporal::rw1_quadratic_form(w, n_times, false);
    // d/d(log_tau) = 0.5*(T-1) - 0.5*tau*quad + tau (Jacobian: d tau / d log_tau = tau)
    // But we want d log_post / d log_tau, which includes Jacobian automatically in computation
    return 0.5 * tulpa_temporal::rw1_rank(n_times, false) - 0.5 * tau * quad;
}

// =============================================================================
// Analytical RW2 gradients
// =============================================================================

// RW2 prior: second differences d[t] = w[t] - 2*w[t-1] + w[t-2]
// log p(w|tau) = 0.5 * (T-2) * log(tau) - 0.5 * tau * sum(d[t]^2)

// The gradient is more complex. For w[t], it contributes to d[t], d[t+1], d[t+2].
// d[t] = w[t] - 2*w[t-1] + w[t-2]
// d/d(w[k]) of sum(d[t]^2) = 2 * sum over t where w[k] appears
// At t: coefficient for w[t] is 1, for w[t-1] is -2, for w[t-2] is 1
// So d/d(w[k]) = 2 * (d[k+2] * 1 - 2 * d[k+1] + d[k] * 1) if k >= 2
inline void rw2_grad_w(const double* w, int n_times, double tau, double* grad_w,
                       double* d_buf = nullptr) {
    // Fewer than three levels admit no second difference, so the RW2 prior is
    // flat. Guarding here also keeps the d[0] / d[1] initialisation below from
    // writing past a buffer sized n_times.
    if (n_times < 3) {
        for (int k = 0; k < n_times; k++) grad_w[k] = 0.0;
        return;
    }
    // Compute second differences (use pre-allocated buffer if provided)
    std::vector<double> d_local;
    double* d;
    if (d_buf) {
        d = d_buf;
    } else {
        d_local.resize(n_times);
        d = d_local.data();
    }
    d[0] = 0.0;
    d[1] = 0.0;
    for (int t = 2; t < n_times; t++) {
        d[t] = w[t] - 2.0 * w[t-1] + w[t-2];
    }

    // Gradient: d/d(w[k]) [-0.5 * tau * sum(d[t]^2)]
    // = -tau * sum_t d[t] * (d d[t] / d w[k])
    for (int k = 0; k < n_times; k++) {
        grad_w[k] = 0.0;
        // d[t] depends on w[t], w[t-1], w[t-2]
        // So w[k] affects d[k] (coef=1), d[k+1] (coef=-2), d[k+2] (coef=1)
        if (k >= 2 && k < n_times) {
            grad_w[k] += -tau * d[k] * 1.0;
        }
        if (k >= 1 && k+1 < n_times) {
            grad_w[k] += -tau * d[k+1] * (-2.0);
        }
        if (k+2 < n_times) {
            grad_w[k] += -tau * d[k+2] * 1.0;
        }
    }
}

inline double rw2_grad_log_tau(const double* w, int n_times, double tau) {
    double quad = tulpa_temporal::rw2_quadratic_form(w, n_times, false);
    return 0.5 * tulpa_temporal::rw2_rank(n_times, false) - 0.5 * tau * quad;
}

// =============================================================================
// Analytical AR1 gradients
// =============================================================================

// AR1 prior:
// w[0] ~ N(0, 1/(tau*(1-rho^2)))
// w[t] | w[t-1] ~ N(rho * w[t-1], 1/tau) for t > 0

// Gradient w.r.t. w[t]:
// t=0: d/d(w[0]) = -tau * (1-rho^2) * w[0] - tau * (w[1] - rho*w[0]) * (-rho)
// t>0, t<T-1: d/d(w[t]) = -tau * (w[t] - rho*w[t-1]) + tau * rho * (w[t+1] - rho*w[t])
// t=T-1: d/d(w[T-1]) = -tau * (w[T-1] - rho*w[T-2])
inline void ar1_grad_w(const double* w, int n_times, double tau, double rho, double* grad_w) {
    // An empty field has no w[0]; ar1_log_density is flat there too.
    if (n_times < 1) return;

    double one_m_rho2 = tulpa_temporal::ar1_one_minus_rho2(rho);

    if (n_times == 1) {
        grad_w[0] = -tau * one_m_rho2 * w[0];
        return;
    }

    // First time point
    double resid_1 = w[1] - rho * w[0];
    grad_w[0] = -tau * one_m_rho2 * w[0] + tau * rho * resid_1;

    // Interior time points
    for (int t = 1; t < n_times - 1; t++) {
        double resid_t = w[t] - rho * w[t-1];
        double resid_tp1 = w[t+1] - rho * w[t];
        grad_w[t] = -tau * resid_t + tau * rho * resid_tp1;
    }

    // Last time point
    double resid_T = w[n_times-1] - rho * w[n_times-2];
    grad_w[n_times-1] = -tau * resid_T;
}

// Gradient w.r.t. log(tau)
inline double ar1_grad_log_tau(const double* w, int n_times, double tau, double rho) {
    if (n_times < 1) return 0.0;

    double one_m_rho2 = tulpa_temporal::ar1_one_minus_rho2(rho);

    // Stationary part. var_stationary = 1/(tau*(1-rho^2)) is the precision
    // parameterization, so d/d(log tau)[-0.5*log(2*pi*var_stationary)] = +0.5.
    double grad = 0.5;
    grad += 0.5 * (n_times - 1);  // From the t>0 conditional normalizers (1/tau)

    // Quadratic terms
    double quad = one_m_rho2 * w[0] * w[0];
    for (int t = 1; t < n_times; t++) {
        double resid = w[t] - rho * w[t-1];
        quad += resid * resid;
    }
    grad -= 0.5 * tau * quad;

    return grad;
}

// Gradient w.r.t. logit(rho) where u = inv_logit(logit_rho), rho = 2*u - 1
// d/d(logit_rho) = d/d(rho) * d(rho)/d(u) * d(u)/d(logit_rho)
//                = d/d(rho) * 2 * u * (1-u)
inline double ar1_grad_logit_rho(const double* w, int n_times, double tau, double rho) {
    if (n_times < 1) return 0.0;

    // d log p / d rho from stationary distribution:
    // log p(w[0]) = 0.5*log(tau*(1-rho^2)) - 0.5*tau*(1-rho^2)*w[0]^2 + const
    // d/d(rho) = -rho/(1-rho^2) + tau*rho*w[0]^2
    // The denominator carries the shared AR1 stationary floor, the same factor
    // ar1_log_density evaluates its own normalizer at, so the two do not floor
    // 1 - rho^2 at different points.
    double one_m_rho2 = tulpa_temporal::ar1_one_minus_rho2(rho);
    double grad_rho = tau * rho * w[0] * w[0] - rho / one_m_rho2;

    // d log p / d rho from AR terms
    for (int t = 1; t < n_times; t++) {
        double resid = w[t] - rho * w[t-1];
        grad_rho += tau * resid * w[t-1];
    }

    // Transform to logit_rho
    // u = (rho + 1) / 2, logit_rho = logit(u)
    // d(rho)/d(logit_rho) = d(rho)/d(u) * d(u)/d(logit_rho)
    //                     = 2 * u * (1-u)
    double u = (rho + 1.0) / 2.0;
    double d_rho_d_logit = 2.0 * u * (1.0 - u);

    return grad_rho * d_rho_d_logit;
}

} // namespace tulpa_tvc

#endif // TULPA_HMC_TVC_GRAD_H
