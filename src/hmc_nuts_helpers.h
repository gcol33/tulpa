// hmc_nuts_helpers.h
// Fragment of hmc_nuts_sampler.cpp. Included from the umbrella
// translation unit inside namespace tulpa_hmc; do NOT add a
// namespace wrapper here; do not list this file in the package SRCS —
// it is not a standalone translation unit.
// NUTS helpers: log-sum-exp, leapfrog_step_with_grad, U-turn check, etc.
#ifndef TULPA_HMC_NUTS_HELPERS_H
#define TULPA_HMC_NUTS_HELPERS_H


// =====================================================================
// NUTS (No-U-Turn Sampler) helper functions
// =====================================================================

double nuts_log_sum_exp(double a, double b) {
  double m = std::max(a, b);
  if (!std::isfinite(m)) return m;
  return m + std::log(std::exp(a - m) + std::exp(b - m));
}

double nuts_compute_hamiltonian(double log_prob, const std::vector<double>& p,
                                const std::vector<double>& inv_mass, int n) {
  double kinetic = 0.0;
  for (int i = 0; i < n; i++) {
    kinetic += p[i] * p[i] * inv_mass[i];
  }
  return -log_prob + 0.5 * kinetic;
}

bool nuts_check_uturn(const std::vector<double>& q_minus, const std::vector<double>& q_plus,
                      const std::vector<double>& p_minus, const std::vector<double>& p_plus,
                      const std::vector<double>& inv_mass, int n) {
  // Generalized U-turn criterion (Betancourt 2017, Section 3.2)
  // Check both directions: (q+ - q-) . (M^-1 p-) and (q+ - q-) . (M^-1 p+)
  double dot_fwd = 0.0, dot_bwd = 0.0;
  for (int i = 0; i < n; i++) {
    double dq = q_plus[i] - q_minus[i];
    dot_fwd += dq * (inv_mass[i] * p_plus[i]);
    dot_bwd += dq * (inv_mass[i] * p_minus[i]);
  }
  return (dot_fwd < 0.0) || (dot_bwd < 0.0);
}

LeapfrogResultWithGrad leapfrog_step_with_grad(
    const std::vector<double>& q, const std::vector<double>& p,
    const std::vector<double>& grad,
    double epsilon, const std::vector<double>& inv_mass,
    bool use_mass, const ModelData& data, const ParamLayout& layout) {

  int n = q.size();
  LeapfrogResultWithGrad result;
  result.q = q;
  result.p = p;
  result.grad.resize(n);
  result.divergent = false;

  // Half step for momentum using provided gradient
  for (int i = 0; i < n; i++) {
    result.p[i] += 0.5 * epsilon * grad[i];
  }

  // Full step for position
  if (use_mass) {
    for (int i = 0; i < n; i++) {
      result.q[i] += epsilon * inv_mass[i] * result.p[i];
    }
  } else {
    for (int i = 0; i < n; i++) {
      result.q[i] += epsilon * result.p[i];
    }
  }

  // Compute gradient and log_prob at new position (fused: single O(N) pass)
  compute_gradient(result.q, data, layout, result.grad, &result.log_prob);

  // Half step for momentum using new gradient
  for (int i = 0; i < n; i++) {
    result.p[i] += 0.5 * epsilon * result.grad[i];
  }

  // Check for divergence
  if (!std::isfinite(result.log_prob)) {
    result.divergent = true;
  }
  for (int i = 0; i < n; i++) {
    if (std::abs(result.q[i]) > 1e10 || !std::isfinite(result.q[i])) {
      result.divergent = true;
      break;
    }
  }

  return result;
}

#endif  // TULPA_HMC_NUTS_HELPERS_H
