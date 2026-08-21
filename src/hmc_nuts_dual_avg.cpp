// hmc_nuts_dual_avg.cpp
// DualAveraging method definitions for the HMC/NUTS step-size adaptation.

#include <algorithm>
#include <cmath>

#include "hmc_sampler.h"

namespace tulpa_hmc {

// g_gradient_mode is declared in hmc_sampler_decls.h.

// Dual averaging for step size adaptation
// =====================================================================

DualAveraging::DualAveraging(double epsilon_init, double target_accept_)
  : mu(std::log(10.0 * epsilon_init)), log_epsilon_bar(std::log(epsilon_init)), H_bar(0.0),
    gamma(0.05), t0(10.0), kappa(0.75),
    target_accept(target_accept_), m(0) {}

// Target acceptance rate, resolved once per chain from the model structure.
// A caller-supplied adapt_delta is an override and is taken as given; otherwise
// the target rises with the posterior correlation the geometry carries.
double resolve_target_accept(
    const ModelData& data,
    const ParamLayout& layout,
    double adapt_delta
) {
  if (adapt_delta > 0) return adapt_delta;

  double target = 0.80;  // Stan default base

  // BYM2: high correlation between ICAR phi + unstructured theta
  if (data.spatial_type == SpatialType::BYM2) {
    target = 0.90;
  }
  // ICAR: correlated spatial params need slightly higher target
  else if (data.spatial_type == SpatialType::ICAR) {
    target = 0.85;
  }

  // Correlated random slopes add funnel geometry
  if (data.has_re_correlated_slopes) {
    target = std::max(target, 0.90);
  }

  // Temporal GP NC: z ~ N(0,1) decorrelates the field, so a lower target
  // reaches the same accuracy with fewer leapfrog steps.
  if (layout.is_temporal_gp && target > 0.70) {
    target = 0.70;
  }

  return std::min(0.99, target);
}

double DualAveraging::update(double alpha) {
  m++;
  double w = 1.0 / (m + t0);
  H_bar = (1.0 - w) * H_bar + w * (target_accept - alpha);
  double log_epsilon = mu - std::sqrt((double)m) / gamma * H_bar;
  // Clamp log_epsilon to reasonable range
  // Lower bound: exp(-14) ~ 8e-7, Upper bound: exp(2) ~ 7.4
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
