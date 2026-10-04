// hmc_walnuts.cpp
// The engine's model for the WALNUTS transition (see hmc_walnuts.h): the
// metric is DenseMassMatrix, whose fused drift apply_drift() also serves the
// NUTS leapfrog, and the gradient is the dispatched GradientFn.

#include <random>
#include <vector>

#include "hmc_mass_drift.h"
#include "hmc_walnuts.h"
#include "laplace_profile.h"  // TULPA_PROFILE_PHASE (gradient)

namespace tulpa_hmc {

namespace {

struct EngineWalnutsModel {
  const DenseMassMatrix& mass;
  GradientFn gradient_fn;
  const ModelData& data;
  const ParamLayout& layout;
  int n;

  void gradient(const std::vector<double>& theta, std::vector<double>& grad,
                double* log_density) {
    TULPA_PROFILE_PHASE(::tulpa::PHASE_GRADIENT);
    gradient_fn(theta, data, layout, grad, log_density);
  }
  double kinetic_energy(const double* rho) const {
    return mass.kinetic_energy(rho);
  }
  void inv_mass_times_p(const double* rho, double* out) const {
    mass.inv_mass_times_p(rho, out);
  }
  void drift(double step, double* theta, const double* rho,
             double* scratch) const {
    apply_drift(step, theta, rho, mass, scratch, n);
  }
  template <class Rng>
  void sample_momentum(double* rho, Rng& rng) const {
    mass.sample_momentum(rho, rng);
  }
};

}  // namespace

tulpa::WalnutsTransitionResult walnuts_transition(
    std::vector<double>& q, std::vector<double>& grad, double& log_post,
    double step, int max_depth, const tulpa::WalnutsConfig& cfg,
    const DenseMassMatrix& mass, GradientFn gradient_fn,
    const ModelData& data, const ParamLayout& layout,
    tulpa::WalnutsWorkspace& ws, std::mt19937& rng) {
  EngineWalnutsModel model{mass, gradient_fn, data, layout,
                           static_cast<int>(q.size())};
  return tulpa::walnuts_transition(q, grad, log_post, step, max_depth, cfg,
                                   model, ws, rng);
}

}  // namespace tulpa_hmc
