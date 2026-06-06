// hmc_temporal_gp.h
// Gaussian Process temporal effects for irregularly-spaced time series
// Supports exponential, Matern, Gaussian, and periodic covariance functions

#ifndef TULPA_HMC_TEMPORAL_GP_H
#define TULPA_HMC_TEMPORAL_GP_H

#include <vector>
#include <cmath>
#include <algorithm>

// Use canonical type definitions from exported headers
#include "tulpa/temporal_data.h"
#include "tulpa/types.h"
#include "pc_prior.h"

namespace tulpa_temporal_gp {

using tulpa::TemporalCovType;
using tulpa::TemporalGPData;

// -----------------------------------------------------------------------------
// Temporal covariance functions
// -----------------------------------------------------------------------------

// Exponential covariance: sigma^2 * exp(-d / phi)
inline double temporal_cov_exponential(double d, double sigma2, double phi) {
  return sigma2 * std::exp(-d / phi);
}

// Matern 3/2 covariance: sigma^2 * (1 + sqrt(3)*d/phi) * exp(-sqrt(3)*d/phi)
inline double temporal_cov_matern32(double d, double sigma2, double phi) {
  double r = std::sqrt(3.0) * d / phi;
  return sigma2 * (1.0 + r) * std::exp(-r);
}

// Matern 5/2 covariance
inline double temporal_cov_matern52(double d, double sigma2, double phi) {
  double r = std::sqrt(5.0) * d / phi;
  return sigma2 * (1.0 + r + r * r / 3.0) * std::exp(-r);
}

// Gaussian (squared exponential) covariance: sigma^2 * exp(-(d/phi)^2)
inline double temporal_cov_gaussian(double d, double sigma2, double phi) {
  double r = d / phi;
  return sigma2 * std::exp(-r * r);
}

// Periodic covariance: sigma^2 * exp(-2 * sin^2(pi * d / period) / phi^2)
inline double temporal_cov_periodic(double d, double sigma2, double phi, double period) {
  double sin_term = std::sin(M_PI * d / period);
  return sigma2 * std::exp(-2.0 * sin_term * sin_term / (phi * phi));
}

// Generic covariance function dispatcher
inline double compute_temporal_cov(double d, double sigma2, double phi,
                                   TemporalCovType cov_type, double nu = 1.5,
                                   double period = 1.0) {
  switch (cov_type) {
    case TemporalCovType::EXPONENTIAL:
      return temporal_cov_exponential(d, sigma2, phi);
    case TemporalCovType::MATERN:
      if (nu <= 1.0) {
        return temporal_cov_exponential(d, sigma2, phi);  // nu=0.5 equivalent
      } else if (nu <= 2.0) {
        return temporal_cov_matern32(d, sigma2, phi);  // nu=1.5
      } else {
        return temporal_cov_matern52(d, sigma2, phi);  // nu=2.5
      }
    case TemporalCovType::GAUSSIAN:
      return temporal_cov_gaussian(d, sigma2, phi);
    case TemporalCovType::PERIODIC:
      return temporal_cov_periodic(d, sigma2, phi, period);
    default:
      return temporal_cov_exponential(d, sigma2, phi);
  }
}

// -----------------------------------------------------------------------------
// State-space representation for efficient O(n) inference
// (Only for exponential and Matern with half-integer nu)
// -----------------------------------------------------------------------------

// AR(1) representation for exponential covariance
// phi[t] = rho * phi[t-1] + epsilon, where rho = exp(-dt / range)
struct StateSpaceAR1 {
  double marginal_var;    // sigma^2
  double range;           // phi (range parameter)
};

// Compute AR(1) transition parameters given time gap dt
inline void ar1_transition(double dt, const StateSpaceAR1& ss,
                           double& rho, double& cond_var) {
  rho = std::exp(-dt / ss.range);
  cond_var = ss.marginal_var * (1.0 - rho * rho);
}

// Compute log-likelihood using state-space (O(n) algorithm)
// Assumes observations are sorted by time!
inline double temporal_gp_log_lik_statespace(
    const std::vector<double>& f,        // Temporal effect values
    const std::vector<double>& times,    // Time values (sorted)
    double sigma2,                        // Marginal variance
    double phi,                           // Range parameter
    TemporalCovType cov_type
) {
  int N = f.size();
  if (N == 0) return 0.0;

  // Only use state-space for exponential covariance
  if (cov_type != TemporalCovType::EXPONENTIAL) {
    // Fall back to direct computation for other covariance types
    return -1.0;  // Signal to use direct method
  }

  double log_lik = 0.0;

  // First observation: marginal N(0, sigma2)
  log_lik += -0.5 * std::log(2.0 * M_PI * sigma2) -
             0.5 * f[0] * f[0] / sigma2;

  // Remaining observations: conditional on previous
  for (int i = 1; i < N; i++) {
    double dt = times[i] - times[i-1];

    // AR(1) transition
    double rho = std::exp(-dt / phi);
    double cond_var = sigma2 * (1.0 - rho * rho);

    // Ensure positive variance
    if (cond_var < 1e-10) cond_var = 1e-10;

    double cond_mean = rho * f[i-1];
    double resid = f[i] - cond_mean;

    log_lik += -0.5 * std::log(2.0 * M_PI * cond_var) -
               0.5 * resid * resid / cond_var;
  }

  return log_lik;
}

// -----------------------------------------------------------------------------
// Direct GP log-likelihood (for general covariance)
// O(n^3) but works for any covariance function
// Use only for small n or when state-space not applicable
// -----------------------------------------------------------------------------

inline double temporal_gp_log_lik_direct(
    const std::vector<double>& f,        // Temporal effect values
    const std::vector<double>& times,    // Time values
    double sigma2,                        // Marginal variance
    double phi,                           // Range parameter
    TemporalCovType cov_type,
    double nu = 1.5,
    double period = 1.0
) {
  int N = f.size();
  if (N == 0) return 0.0;

  // Build covariance matrix
  std::vector<double> K(N * N);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j <= i; j++) {
      double d = std::abs(times[i] - times[j]);
      double cov = compute_temporal_cov(d, sigma2, phi, cov_type, nu, period);
      K[i * N + j] = cov;
      K[j * N + i] = cov;  // Symmetric
    }
    // Add small nugget for numerical stability
    K[i * N + i] += 1e-8;
  }

  // Cholesky decomposition: K = L * L^T
  std::vector<double> L(N * N, 0.0);
  for (int j = 0; j < N; j++) {
    for (int k = 0; k <= j; k++) {
      double sum = K[j * N + k];
      for (int m = 0; m < k; m++) {
        sum -= L[j * N + m] * L[k * N + m];
      }
      if (j == k) {
        L[j * N + j] = std::sqrt(std::max(1e-10, sum));
      } else {
        L[j * N + k] = sum / L[k * N + k];
      }
    }
  }

  // Solve L * y = f (forward substitution)
  std::vector<double> y(N);
  for (int j = 0; j < N; j++) {
    double sum = f[j];
    for (int k = 0; k < j; k++) {
      sum -= L[j * N + k] * y[k];
    }
    y[j] = sum / L[j * N + j];
  }

  // Log-likelihood: -0.5 * (N * log(2*pi) + log|K| + f' K^{-1} f)
  // log|K| = 2 * sum(log(L_ii))
  // f' K^{-1} f = y' y
  double log_det = 0.0;
  for (int j = 0; j < N; j++) {
    log_det += std::log(L[j * N + j]);
  }
  log_det *= 2.0;

  double quad = 0.0;
  for (int j = 0; j < N; j++) {
    quad += y[j] * y[j];
  }

  return -0.5 * (N * std::log(2.0 * M_PI) + log_det + quad);
}

// -----------------------------------------------------------------------------
// Main GP log-likelihood dispatcher
// -----------------------------------------------------------------------------

inline double temporal_gp_log_lik(
    const std::vector<double>& f,
    const TemporalGPData& gp_data,
    double sigma2,
    double phi
) {
  // Try state-space first (exponential covariance)
  if (gp_data.cov_type == TemporalCovType::EXPONENTIAL) {
    return temporal_gp_log_lik_statespace(f, gp_data.time_values,
                                           sigma2, phi, gp_data.cov_type);
  }

  // Fall back to direct computation
  return temporal_gp_log_lik_direct(f, gp_data.time_values, sigma2, phi,
                                     gp_data.cov_type, gp_data.nu, gp_data.period);
}

// -----------------------------------------------------------------------------
// Priors for GP hyperparameters
// -----------------------------------------------------------------------------

// Log prior for temporal variance (PC prior style)
inline double log_prior_sigma2_temporal_pc(double sigma2, double U, double alpha) {
  return tulpa::log_prior_sigma2_pc(sigma2, U, alpha);
}

// Log prior for temporal range (uniform or PC)
inline double log_prior_temporal_phi_uniform(double phi, double lower, double upper) {
  if (phi < lower || phi > upper) return -INFINITY;
  return -std::log(upper - lower);
}

// -----------------------------------------------------------------------------
// Gradient computation (for HMC)
// -----------------------------------------------------------------------------

// Numerical gradient of GP log-likelihood w.r.t. f (temporal effects)
inline void temporal_gp_gradient_f(
    const std::vector<double>& f,
    const TemporalGPData& gp_data,
    double sigma2,
    double phi,
    std::vector<double>& grad_f,
    double epsilon = 1e-6
) {
  int N = gp_data.n_obs;
  grad_f.resize(N);

  double base_ll = temporal_gp_log_lik(f, gp_data, sigma2, phi);

  std::vector<double> f_plus = f;
  for (int i = 0; i < N; i++) {
    f_plus[i] = f[i] + epsilon;
    double ll_plus = temporal_gp_log_lik(f_plus, gp_data, sigma2, phi);
    grad_f[i] = (ll_plus - base_ll) / epsilon;
    f_plus[i] = f[i];  // Reset
  }
}

// Parse covariance type from string
inline TemporalCovType parse_temporal_cov_type(const std::string& cov_str) {
  static const tulpa::EnumEntry<TemporalCovType> table[] = {
      {"exponential", TemporalCovType::EXPONENTIAL},
      {"matern", TemporalCovType::MATERN},
      {"gaussian", TemporalCovType::GAUSSIAN},
      {"periodic", TemporalCovType::PERIODIC}
  };
  return tulpa::parse_enum(cov_str, table, TemporalCovType::EXPONENTIAL);
}

// -----------------------------------------------------------------------------
// Non-centered parameterization helpers
// Store z ~ N(0,1) in params, reconstruct f via state-space forward pass.
// Eliminates funnel geometry → shallower NUTS trees.
// -----------------------------------------------------------------------------

// Workspace for NC forward/backward (avoids per-call allocation)
struct TemporalGPNCWorkspace {
    int T = 0;
    int n_groups = 0;
    std::vector<double> f;           // Reconstructed f[g*T + t]
    std::vector<double> rho;         // rho[t] for t=1..T-1 (length T-1, shared across groups)
    std::vector<double> a;           // scale factors a[t] for t=0..T-1 (length T)
    std::vector<double> dL_df;       // Likelihood gradient w.r.t. f[g*T + t]
    std::vector<double> adj;         // Adjoint buffer for backward pass (length T per group)

    void init(int T_times, int ng) {
        if (T_times == T && ng == n_groups) return;
        T = T_times;
        n_groups = ng;
        f.resize(ng * T);
        rho.resize(T > 0 ? T - 1 : 0);
        a.resize(T);
        dL_df.resize(ng * T);
        adj.resize(T);
    }
};

// Forward pass: z[g*T + t] -> f[g*T + t] using state-space AR(1)
// f[0] = a[0] * z[0],  where a[0] = sqrt(sigma2)
// f[t] = rho[t] * f[t-1] + a[t] * z[t],  where a[t] = sqrt(sigma2 * (1 - rho[t]^2))
// rho[t] = exp(-dt[t] / phi)
//
// rho and a are shared across groups (same time grid), computed once.
// f is group-specific because z is group-specific.
static inline void temporal_gp_nc_forward(
    const double* z,              // z[g*T + t], length n_groups * T
    int T, int n_groups,
    double sigma2, double phi,
    const std::vector<double>& time_values,
    TemporalGPNCWorkspace& ws
) {
    // Compute shared rho and a vectors
    double sigma = std::sqrt(sigma2);
    ws.a[0] = sigma;

    for (int t = 1; t < T; t++) {
        double dt = time_values[t] - time_values[t - 1];
        double rho_t = std::exp(-dt / phi);
        ws.rho[t - 1] = rho_t;
        double one_minus_rho2 = 1.0 - rho_t * rho_t;
        if (one_minus_rho2 < 1e-10) one_minus_rho2 = 1e-10;
        ws.a[t] = sigma * std::sqrt(one_minus_rho2);
    }

    // Forward pass per group
    for (int g = 0; g < n_groups; g++) {
        int off = g * T;
        ws.f[off] = ws.a[0] * z[off];
        for (int t = 1; t < T; t++) {
            ws.f[off + t] = ws.rho[t - 1] * ws.f[off + t - 1] + ws.a[t] * z[off + t];
        }
    }
}

// Backward pass: given dL/df from likelihood, compute full gradient for z, sigma2, phi.
//
// Uses adjoint method per group:
//   adj[T-1] = dL_df[T-1]
//   adj[t]   = dL_df[t] + rho[t+1] * adj[t+1]     (for t = T-2 ... 0)
//
// Then:
//   grad_z[g*T + t] = -z[t] + a[t] * adj[t]        (prior + likelihood chain rule)
//
// For hyperparameters (sigma2, phi), accumulate via forward sensitivity:
//   grad_log_sigma2 from prior: sum_g [ -0.5*T + 0.5 * sum_t z[t]^2 ]
//                                (since z ~ N(0,I), log p(z) = -0.5*sum(z^2), Jacobian = T*log(a) terms)
//   Hyperparameter gradients from likelihood are computed via chain rule through a[] and rho[].
static inline void temporal_gp_nc_backward(
    const double* z,
    int T, int n_groups,
    double sigma2, double phi,
    const std::vector<double>& time_values,
    const TemporalGPNCWorkspace& ws,
    double* grad_z,               // output: full gradient for z[g*T + t] (prior already included)
    double& grad_log_sigma2_lik,  // output: sigma2 likelihood contribution
    double& grad_log_phi_lik      // output: phi likelihood contribution
) {
    grad_log_sigma2_lik = 0.0;
    grad_log_phi_lik = 0.0;

    double sigma = std::sqrt(sigma2);
    // Precompute da/d(log_sigma2) and da/d(log_phi) for each t
    // a[0] = sqrt(sigma2), so da[0]/d(log_sigma2) = d(sqrt(sigma2))/d(log_sigma2) = 0.5*sqrt(sigma2) = a[0]/2
    // For t >= 1: a[t] = sqrt(sigma2) * sqrt(1 - rho_t^2)
    //   da[t]/d(log_sigma2) = a[t] / 2
    //   da[t]/d(log_phi) = sqrt(sigma2) * 0.5 / sqrt(1-rho^2) * (-2*rho*drho/d(log_phi))
    //     where drho/d(log_phi) = (dt/phi) * rho  (since d(log_phi) = d(phi)/phi, d(exp(-dt/phi))/d(phi) = (dt/phi^2)*rho)
    //   = sqrt(sigma2) * (-rho * (dt/phi) * rho) / sqrt(1-rho^2)
    //   = -sigma * rho^2 * (dt/phi) / sqrt(1-rho^2)

    for (int g = 0; g < n_groups; g++) {
        int off = g * T;

        // --- Backward adjoint recursion ---
        // adj[t] accumulates the total derivative of L w.r.t. f[t] including
        // indirect effects through f[t+1], f[t+2], ...
        // Build adjoint from T-1 down to 0
        // We process from t = T-1 backward. Store in a local buffer.
        // Since we need adj for computing grad_z and hyperparameter grads,
        // we do the full backward then apply.

        // Use workspace adj buffer
        double* adj = const_cast<TemporalGPNCWorkspace&>(ws).adj.data();
        adj[T - 1] = ws.dL_df[off + T - 1];
        for (int t = T - 2; t >= 0; t--) {
            adj[t] = ws.dL_df[off + t] + ws.rho[t] * adj[t + 1];
        }

        // --- z gradients: prior (-z) + likelihood (a[t] * adj[t]) ---
        for (int t = 0; t < T; t++) {
            grad_z[off + t] = -z[off + t] + ws.a[t] * adj[t];
        }

        // --- Hyperparameter gradients from likelihood via chain rule ---
        // The likelihood depends on sigma2 and phi only through f[t].
        // dL/d(log_sigma2) = sum_t (dL/df[t]) * (df[t]/d(log_sigma2))
        // But df[t]/d(log_sigma2) involves the full forward recursion sensitivity.
        //
        // Using the adjoint method:
        // dL/d(log_sigma2) = sum_t adj[t] * (da[t]/d(log_sigma2)) * z[t]
        //                  = sum_t adj[t] * (a[t]/2) * z[t]
        //                  = 0.5 * sum_t adj[t] * a[t] * z[t]
        //                  = 0.5 * sum_t adj[t] * (f[t] - rho[t]*f[t-1])  [since a[t]*z[t] = f[t] - rho[t]*f[t-1]]
        // Actually the sensitivity of a[t] to log_sigma2 is da[t]/d(log_sigma2) = a[t]/2
        // (because a[t] = sigma * sqrt(1-rho^2) and d(sigma)/d(log_sigma2) = sigma/2)
        // So: dL/d(log_sigma2) via a = sum_t adj[t] * (a[t]/2) * z[off+t]

        double g_sigma2 = 0.0;
        for (int t = 0; t < T; t++) {
            g_sigma2 += adj[t] * ws.a[t] * z[off + t] * 0.5;
        }
        grad_log_sigma2_lik += g_sigma2;

        // dL/d(log_phi): two sources:
        // 1. Through a[t] for t >= 1: da[t]/d(log_phi) * z[t]
        //    da[t]/d(log_phi) = -sigma * rho^2 * (dt/phi) / sqrt(1-rho^2)
        //                     = -(a[t] == 0 ? 0 : sigma * rho^2 * (dt/phi) / sqrt(1-rho^2))
        //    Contribution: sum_{t>=1} adj[t] * da[t]/d(log_phi) * z[off+t]
        //
        // 2. Through rho[t] in the recursion f[t] = rho[t]*f[t-1] + a[t]*z[t]:
        //    drho/d(log_phi) = (dt/phi) * rho
        //    Contribution: sum_{t>=1} adj[t] * (dt/phi) * rho[t] * f[t-1]
        //    (This uses the adjoint directly: the partial derivative of f[t] w.r.t. rho[t] is f[t-1])

        double g_phi = 0.0;
        for (int t = 1; t < T; t++) {
            double dt = time_values[t] - time_values[t - 1];
            double dt_over_phi = dt / phi;
            double rho_t = ws.rho[t - 1];
            double rho2 = rho_t * rho_t;

            // Source 1: through a[t]
            // da[t]/d(log_phi) = -sigma * rho^2 * (dt/phi) / sqrt(1-rho^2)
            // Reuse cached ws.a[t] = sigma * sqrt(1-rho^2), so sqrt(1-rho^2) = ws.a[t]/sigma
            double sqrt_1mr2 = (sigma > 1e-15) ? ws.a[t] / sigma : 0.0;
            double da_dphi = (sqrt_1mr2 > 1e-15)
                ? -sigma * rho2 * dt_over_phi / sqrt_1mr2
                : 0.0;
            g_phi += adj[t] * da_dphi * z[off + t];

            // Source 2: through rho[t]
            // drho/d(log_phi) = (dt/phi) * rho, and df[t]/drho = f[t-1]
            g_phi += adj[t] * dt_over_phi * rho_t * ws.f[off + t - 1];
        }
        grad_log_phi_lik += g_phi;
    }
}

} // namespace tulpa_temporal_gp

#endif // TULPA_HMC_TEMPORAL_GP_H
