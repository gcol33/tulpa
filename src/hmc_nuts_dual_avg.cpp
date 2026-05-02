// hmc_nuts_dual_avg.cpp
// DualAveraging method definitions for the HMC/NUTS step-size adaptation.

#include <algorithm>
#include <cmath>

#include "hmc_gp_collapsed.h"
#include "hmc_icar_collapsed.h"
#include "hmc_sampler.h"

namespace tulpa_hmc {

// g_gradient_mode is declared in hmc_sampler_decls.h.
extern thread_local CollapsedGPWorkspace collapsed_gp_ws;
extern thread_local CollapsedICARWorkspace collapsed_icar_ws;

// Dual averaging for step size adaptation
// =====================================================================

DualAveraging::DualAveraging(double epsilon_init, int n_params, double target_boost)
  : mu(std::log(10.0 * epsilon_init)), log_epsilon_bar(std::log(epsilon_init)), H_bar(0.0),
    gamma(0.05), t0(10.0), kappa(0.75),
    target_accept(compute_target(n_params, target_boost)), m(0) {}

double DualAveraging::update(double alpha) {
  m++;
  double w = 1.0 / (m + t0);
  H_bar = (1.0 - w) * H_bar + w * (target_accept - alpha);
  double log_epsilon = mu - std::sqrt((double)m) / gamma * H_bar;
  // Clamp log_epsilon to reasonable range
  // Lower bound: exp(-14) ? 8e-7, Upper bound: exp(2) ? 7.4
  log_epsilon = std::max(-14.0, std::min(log_epsilon, 2.0));
  double epsilon = std::exp(log_epsilon);
  double m_w = std::pow((double)m, -kappa);
  log_epsilon_bar = m_w * log_epsilon + (1.0 - m_w) * log_epsilon_bar;
  return epsilon;
}

double DualAveraging::final_epsilon() const {
  return std::exp(log_epsilon_bar);
}

}  // namespace tulpa_hmc
