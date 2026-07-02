// hmc_nuts_leapfrog.cpp
// Leapfrog integrator for HMC/NUTS.

#include <vector>

#include "hmc_sampler.h"

namespace tulpa_hmc {

// =====================================================================
// Leapfrog integrator
// =====================================================================

// Unified leapfrog step: identity mass when inv_mass is nullptr.
//
// The step walks the active SIMP scheme's op sequence. A Kick recomputes the
// gradient at the current position and advances momentum by the force
// (grad = d(log_post)/dq); a Drift advances position (scaled by the inverse
// mass if provided). The last Kick fuses log_prob, so its position -- the
// trajectory endpoint -- carries the returned log_prob. For the default
// leapfrog scheme (kick 1/2, drift 1, kick 1/2) this reduces to the classic
// three-line velocity-Verlet step, coefficient for coefficient.
LeapfrogResult leapfrog_step(
    const std::vector<double>& q,
    const std::vector<double>& p,
    double epsilon,
    const ModelData& data,
    const ParamLayout& layout,
    const double* inv_mass
) {
  int n = q.size();
  LeapfrogResult result;
  result.q = q;
  result.p = p;
  result.divergent = false;

  const simp::Scheme& scheme = get_integrator_scheme();
  int last_kick = -1;
  for (int j = 0; j < static_cast<int>(scheme.ops.size()); j++) {
    if (scheme.ops[j].first == simp::Op::Kick) last_kick = j;
  }

  std::vector<double> grad(n);

  for (int j = 0; j < static_cast<int>(scheme.ops.size()); j++) {
    double c = scheme.ops[j].second * epsilon;
    if (scheme.ops[j].first == simp::Op::Kick) {
      if (j == last_kick) {
        compute_gradient(result.q, data, layout, grad, &result.log_prob);
      } else {
        compute_gradient(result.q, data, layout, grad);
      }
      for (int i = 0; i < n; i++) {
        result.p[i] += c * grad[i];
      }
    } else {
      if (inv_mass) {
        for (int i = 0; i < n; i++) {
          result.q[i] += c * inv_mass[i] * result.p[i];
        }
      } else {
        for (int i = 0; i < n; i++) {
          result.q[i] += c * result.p[i];
        }
      }
    }
  }

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

// =====================================================================
// Find reasonable initial step size
// =====================================================================

// Compute diagonal mass matrix from gradient magnitudes
std::vector<double> compute_diagonal_mass(
    const std::vector<double>& q,
    const ModelData& data,
    const ParamLayout& layout
) {
  int n = q.size();
  std::vector<double> grad(n);
  compute_gradient(q, data, layout, grad);

  std::vector<double> mass(n);
  for (int i = 0; i < n; i++) {
    double abs_grad = std::abs(grad[i]);
    mass[i] = std::max(1.0, std::min(abs_grad, 1000.0));
  }

  return mass;
}

}  // namespace tulpa_hmc
