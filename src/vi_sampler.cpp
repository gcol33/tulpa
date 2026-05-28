// vi_sampler.cpp
// Variational Inference dispatcher (mean-field / low-rank / full-rank).
// The legacy ratio Rcpp entry points (cpp_vi_fit, cpp_vi_get_n_params) were
// removed in Phase D of the tulpaRatio migration (gcol33/tulpa#15); VI is
// reached from downstream packages via the C-callable shim `tulpa_fit_vi`
// (tulpa_shims_vi_ess.h) and the generic-layout LikelihoodSpec path.

#include "hmc_sampler.h"
#include "vi_types.h"
#include "vi_optimizer.h"
#include "vi_meanfield.h"
#include "vi_fullrank.h"
#include "vi_lowrank.h"
#include "autodiff.h"
#include <Rcpp.h>

using namespace Rcpp;
using namespace tulpa_hmc;

namespace tulpa {
namespace vi {

// =====================================================================
// Bridge function: compute log joint and gradient
// =====================================================================

// This function bridges VI with the existing HMC gradient infrastructure
double compute_log_joint_grad(
    const Eigen::VectorXd& params_eigen,
    const ModelData& data,
    const ParamLayout& layout,
    Eigen::VectorXd& grad_eigen
) {
  int D = params_eigen.size();

  // Map Eigen to std::vector (zero-copy where possible)
  std::vector<double> params(params_eigen.data(), params_eigen.data() + D);

  // Compute gradient (H-mode returns log_post as byproduct via pointer)
  std::vector<double> grad(D);
  double log_post = 0.0;
  compute_gradient(params, data, layout, grad, &log_post);

  // Map back to Eigen
  grad_eigen = Eigen::Map<Eigen::VectorXd>(grad.data(), D);

  return log_post;
}

// =====================================================================
// VI Dispatcher
// =====================================================================

VIResult fit_vi(
    const ModelData& data,
    const ParamLayout& layout,
    int D,
    const VIConfig& config,
    const Eigen::VectorXd* init_mu = nullptr
) {
  // Select variant
  VIVariant variant = select_variant(D, config);

  if (config.verbose) {
    Rcpp::Rcout << "VI variant: " << variant_to_string(variant)
                << " (D=" << D << ")\n";
  }

  // Dispatch to appropriate implementation
  switch (variant) {
    case VIVariant::MEANFIELD:
      return fit_meanfield(data, layout, D, config, init_mu);

    case VIVariant::LOWRANK:
      return fit_lowrank(data, layout, D, config, init_mu);

    case VIVariant::FULLRANK:
      return fit_fullrank(data, layout, D, config, init_mu, nullptr);

    default:
      Rcpp::stop("Unknown VI variant");
  }

  return VIResult();  // Never reached
}

} // namespace vi
} // namespace tulpa
