// hmc_nuts_leapfrog.cpp
// Leapfrog integrator for HMC/NUTS.

#include <vector>

#include "hmc_sampler.h"

namespace tulpa_hmc {

// =====================================================================
// Leapfrog integrator
// =====================================================================

// Unified leapfrog step.
//
// The step walks the active SIMP scheme's op sequence. A Kick recomputes the
// gradient at the current position and advances momentum by the force
// (grad = d(log_post)/dq); a Drift advances position by the metric applied to
// the momentum, q += c * M^{-1} p. The last Kick fuses log_prob, so its
// position -- the trajectory endpoint -- carries the returned log_prob. For the
// default leapfrog scheme (kick 1/2, drift 1, kick 1/2) this reduces to the
// classic three-line velocity-Verlet step, coefficient for coefficient.
//
// The metric is whatever the caller drew and scores the momentum with:
// `dense_mass` covers dense, block-diagonal and the structured precision /
// Kronecker / sparse-GMRF blocks; `inv_mass` is the plain diagonal; neither is
// the identity. A drift under a different metric than the momentum draw and
// the kinetic energy is not a Hamiltonian flow, so the Metropolis ratio would
// not correct it.
LeapfrogResult leapfrog_step(
    const std::vector<double>& q,
    const std::vector<double>& p,
    double epsilon,
    const ModelData& data,
    const ParamLayout& layout,
    const double* inv_mass,
    const DenseMassMatrix* dense_mass
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
  std::vector<double> Mp;
  if (dense_mass) Mp.resize(n);

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
      if (dense_mass) {
        dense_mass->inv_mass_times_p(result.p.data(), Mp.data());
        for (int i = 0; i < n; i++) {
          result.q[i] += c * Mp[i];
        }
      } else if (inv_mass) {
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

  // The fused log_prob above is at the last Kick's position. Every shipped
  // scheme ends in a Kick, so that is already the trajectory endpoint; the
  // guard covers a drift-terminated scheme, and a scheme with no Kick at all.
  if (last_kick < 0 || last_kick + 1 != static_cast<int>(scheme.ops.size())) {
    result.log_prob = compute_log_post(result.q, data, layout);
  }

  result.divergent = leapfrog_state_nonfinite(
      result.log_prob, result.q.data(), result.p.data(), n);

  return result;
}

}  // namespace tulpa_hmc
