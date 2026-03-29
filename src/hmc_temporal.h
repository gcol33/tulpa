// hmc_temporal.h
// Temporal random effects support for HMC backend
// Supports RW1, RW2, and AR1 temporal structures

#ifndef TULPA_HMC_TEMPORAL_H
#define TULPA_HMC_TEMPORAL_H

#define _USE_MATH_DEFINES  // For M_PI on Windows
#include <vector>
#include <cmath>
#include <utility>

// Fallback definition of M_PI if not provided by <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Use canonical type definitions from exported headers
#include "tulpa/temporal_data.h"
#include "tulpa/types.h"

namespace tulpa_temporal {

using tulpa::TemporalType;
using tulpa::TemporalData;
using tulpa::MultiscaleTemporalData;

// =====================================================================
// RW1 precision matrix quadratic form
// =====================================================================

// Compute phi' Q_RW1 phi for RW1 prior
// Q_RW1 is the first-order random walk precision matrix
inline double rw1_quadratic_form(
    const double* phi,
    int T,
    bool cyclic
) {
  double quad = 0.0;

  // Sum of squared differences
  for (int t = 0; t < T - 1; t++) {
    double diff = phi[t + 1] - phi[t];
    quad += diff * diff;
  }

  // Cyclic: add edge from T to 1
  if (cyclic) {
    double diff = phi[0] - phi[T - 1];
    quad += diff * diff;
  }

  return quad;
}

// =====================================================================
// RW2 precision matrix quadratic form
// =====================================================================

// Compute phi' Q_RW2 phi for RW2 prior
// Q_RW2 penalizes second differences (curvature)
inline double rw2_quadratic_form(
    const double* phi,
    int T,
    bool cyclic
) {
  if (T < 3) return 0.0;

  double quad = 0.0;

  // Sum of squared second differences
  for (int t = 0; t < T - 2; t++) {
    double diff2 = phi[t] - 2.0 * phi[t + 1] + phi[t + 2];
    quad += diff2 * diff2;
  }

  // Cyclic: wrap around
  if (cyclic && T >= 3) {
    // t = T-2
    double diff2_1 = phi[T - 2] - 2.0 * phi[T - 1] + phi[0];
    quad += diff2_1 * diff2_1;
    // t = T-1
    double diff2_2 = phi[T - 1] - 2.0 * phi[0] + phi[1];
    quad += diff2_2 * diff2_2;
  }

  return quad;
}

// =====================================================================
// AR1 log-density
// =====================================================================

// Compute log-density for AR1 process
// phi[t] | phi[t-1] ~ N(rho * phi[t-1], sigma^2)
// Marginal variance: sigma^2 / (1 - rho^2)
inline double ar1_log_density(
    const double* phi,
    int T,
    double rho,
    double tau  // precision = 1/sigma^2
) {
  if (T < 2) return 0.0;

  double log_dens = 0.0;

  // First observation: phi[0] ~ N(0, sigma^2 / (1 - rho^2))
  double marginal_var = 1.0 / (tau * (1.0 - rho * rho));
  log_dens -= 0.5 * phi[0] * phi[0] / marginal_var;
  log_dens -= 0.5 * std::log(2.0 * M_PI * marginal_var);

  // Conditional: phi[t] | phi[t-1] ~ N(rho * phi[t-1], sigma^2)
  double sigma2 = 1.0 / tau;
  for (int t = 1; t < T; t++) {
    double resid = phi[t] - rho * phi[t - 1];
    log_dens -= 0.5 * resid * resid / sigma2;
    log_dens -= 0.5 * std::log(2.0 * M_PI * sigma2);
  }

  return log_dens;
}

// =====================================================================
// Non-centered AR1 parameterization helpers
// =====================================================================

// Forward recursion: z (N(0,1) innovations) → phi (actual effects)
// phi[0] = z[0] / sqrt(tau * (1 - rho^2))
// phi[t] = rho * phi[t-1] + z[t] / sqrt(tau)   for t >= 1
inline void ar1_nc_forward(
    const double* z, double* phi, int T,
    double rho, double tau
) {
  if (T <= 0) return;
  double omr2 = 1.0 - rho * rho;
  if (omr2 < 1e-12) omr2 = 1e-12;  // Guard against rho ≈ ±1
  double inv_sqrt_tau = 1.0 / std::sqrt(tau);
  double inv_sqrt_tau_omr2 = inv_sqrt_tau / std::sqrt(omr2);

  phi[0] = z[0] * inv_sqrt_tau_omr2;
  for (int t = 1; t < T; t++) {
    phi[t] = rho * phi[t - 1] + z[t] * inv_sqrt_tau;
  }
}

// Non-centered AR1 log-prior on z: z ~ N(0, I)
// Returns log p(z) = -0.5 * sum(z[t]^2) - 0.5*T*log(2*pi)
inline double ar1_nc_log_prior(const double* z, int T) {
  double lp = -0.5 * T * std::log(2.0 * M_PI);
  for (int t = 0; t < T; t++) {
    lp -= 0.5 * z[t] * z[t];
  }
  return lp;
}

// Compute gradients for non-centered AR1 parameterization.
// Given grad_phi_lik = d(likelihood)/d(phi), computes:
//   grad_z[t] = d(log_post)/d(z[t]) including N(0,1) prior
//   returns (grad_log_tau_from_lik, grad_logit_rho_from_lik)
//     — likelihood contributions to hyperparameter gradients via chain rule
//
// The caller is responsible for adding the tau and rho PRIOR gradients separately.
inline std::pair<double, double> ar1_nc_gradient(
    const double* z,           // NC parameters from param vector
    const double* phi,         // Reconstructed effects from ar1_nc_forward
    const double* grad_phi_lik,// d(likelihood)/d(phi)
    double* grad_z,            // OUTPUT: final d(log_post)/d(z)
    int T,
    double rho,
    double tau
) {
  if (T <= 0) return {0.0, 0.0};
  if (T == 1) {
    // Single time point: phi[0] = z[0] / sqrt(tau*(1-rho^2))
    double omr2 = 1.0 - rho * rho;
    if (omr2 < 1e-12) omr2 = 1e-12;
    double inv_sqrt_tau_omr2 = 1.0 / std::sqrt(tau * omr2);
    grad_z[0] = grad_phi_lik[0] * inv_sqrt_tau_omr2 - z[0];
    // tau gradient: d(phi[0])/d(log_tau) = -0.5 * phi[0]
    double glt = grad_phi_lik[0] * (-0.5 * phi[0]);
    // rho gradient: d(phi[0])/d(rho) = phi[0] * rho / (1-rho^2)
    double gr = grad_phi_lik[0] * (phi[0] * rho / omr2);
    double glr = gr * rho * (1.0 - rho);
    return {glt, glr};
  }

  double omr2 = 1.0 - rho * rho;
  if (omr2 < 1e-12) omr2 = 1e-12;
  double inv_sqrt_tau = 1.0 / std::sqrt(tau);
  double inv_sqrt_tau_omr2 = inv_sqrt_tau / std::sqrt(omr2);

  // ---- Backward adjoint: adj[t] accumulates all downstream effects ----
  // adj[t] = grad_phi_lik[t] + rho * adj[t+1]
  std::vector<double> adj(T);
  adj[T - 1] = grad_phi_lik[T - 1];
  for (int t = T - 2; t >= 0; t--) {
    adj[t] = grad_phi_lik[t] + rho * adj[t + 1];
  }

  // ---- z gradients: adj * d(phi)/d(z) + N(0,1) prior ----
  grad_z[0] = adj[0] * inv_sqrt_tau_omr2 - z[0];
  for (int t = 1; t < T; t++) {
    grad_z[t] = adj[t] * inv_sqrt_tau - z[t];
  }

  // ---- log_tau gradient via chain rule through phi transformation ----
  // d(phi[0])/d(log_tau) = -0.5 * phi[0]
  // d(phi[t])/d(log_tau) = rho * d(phi[t-1])/d(log_tau) - 0.5 * (phi[t] - rho*phi[t-1])
  double dphi_dlt = -0.5 * phi[0];
  double grad_log_tau = adj[0] * dphi_dlt;
  for (int t = 1; t < T; t++) {
    dphi_dlt = rho * dphi_dlt - 0.5 * (phi[t] - rho * phi[t - 1]);
    grad_log_tau += adj[t] * dphi_dlt;
  }

  // ---- rho gradient via chain rule through phi transformation ----
  // d(phi[0])/d(rho) = phi[0] * rho / (1 - rho^2)
  // d(phi[t])/d(rho) = phi[t-1] + rho * d(phi[t-1])/d(rho)
  double dphi_drho = phi[0] * rho / omr2;
  double grad_rho = adj[0] * dphi_drho;
  for (int t = 1; t < T; t++) {
    dphi_drho = phi[t - 1] + rho * dphi_drho;
    grad_rho += adj[t] * dphi_drho;
  }
  // Chain rule: d/d(logit_rho) = d/d(rho) * rho * (1-rho)
  double grad_logit_rho = grad_rho * rho * (1.0 - rho);

  return {grad_log_tau, grad_logit_rho};
}

// =====================================================================
// Temporal log-prior contribution
// =====================================================================

// Compute log-prior for temporal effects (single group)
inline double temporal_log_prior(
    const double* phi,
    int T,
    TemporalType type,
    double tau,           // precision for RW1/RW2, or conditional precision for AR1
    double rho,           // AR1 autocorrelation (ignored for RW)
    bool cyclic
) {
  double log_prior = 0.0;

  if (type == TemporalType::RW1) {
    // RW1: p(phi|tau) propto tau^{(T-1)/2} exp(-0.5 * tau * phi' Q phi)
    double quad = rw1_quadratic_form(phi, T, cyclic);
    int rank = cyclic ? T : T - 1;  // Rank of precision matrix
    log_prior += 0.5 * rank * std::log(tau);
    log_prior -= 0.5 * tau * quad;

  } else if (type == TemporalType::RW2) {
    // RW2: p(phi|tau) propto tau^{(T-2)/2} exp(-0.5 * tau * phi' Q phi)
    double quad = rw2_quadratic_form(phi, T, cyclic);
    int rank = cyclic ? T : T - 2;  // Rank of precision matrix
    log_prior += 0.5 * rank * std::log(tau);
    log_prior -= 0.5 * tau * quad;

  } else if (type == TemporalType::AR1) {
    // AR1: proper prior
    log_prior += ar1_log_density(phi, T, rho, tau);

  } else if (type == TemporalType::IID) {
    // IID N(0, 1/tau): sum of independent normal log-densities
    double sigma2 = 1.0 / tau;
    double log_norm = -0.5 * std::log(2.0 * M_PI * sigma2);
    for (int t = 0; t < T; t++) {
      log_prior += log_norm - 0.5 * phi[t] * phi[t] / sigma2;
    }
  }

  return log_prior;
}

// =====================================================================
// Sum-to-zero constraint (soft)
// =====================================================================

// Apply soft sum-to-zero constraint penalty for RW models
inline double sum_to_zero_penalty(const double* phi, int T, double lambda = 0.001) {
  double sum = 0.0;
  for (int t = 0; t < T; t++) {
    sum += phi[t];
  }
  return -0.5 * lambda * sum * sum;
}

} // namespace tulpa_temporal

#endif // QUOTR_HMC_TEMPORAL_H
