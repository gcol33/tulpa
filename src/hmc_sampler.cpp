// hmc_sampler.cpp
// HMC/NUTS log-posterior orchestrators.
//
// LikelihoodSpec supplies observations for joint and marginal field targets.

#include "hmc_sampler.h"
#include "hmc_field_coordinates.h"
#include "hmc_spatial_marginal.h"
#include "log_post_impl.h"
#include "tulpa/likelihood.h"

#include <Rcpp.h>
#include <cmath>
#include <limits>

using namespace Rcpp;

namespace tulpa_hmc {

// =====================================================================
// Orchestrators: compute_log_post / compute_log_prior / compute_log_lik_only
//
// Every caller has n_processes > 0 and a non-null likelihood_spec. The
// orchestrators evaluate the joint or marginal field target; `skip_obs_loop`
// selects its prior contribution.
// =====================================================================

double compute_log_post(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    bool skip_obs_loop,
    const double* /*precomputed_st_log_prior*/,
    const double* /*precomputed_tgp_log_prior*/
) {
  if (field_coordinates && !field_coordinates->empty()) {
    std::vector<double> x;
    const double jacobian = field_coordinates->forward(params, x);
    if (!std::isfinite(jacobian)) return -INFINITY;
    FieldCoordinateScope raw(nullptr);
    return compute_log_post(x, data, layout, skip_obs_loop) + jacobian;
  }
  if (has_collapsed_spatial(layout)) return compute_collapsed_spatial(params, data, layout, skip_obs_loop);
  if (data.n_processes == 0 || data.likelihood_spec == nullptr) {
    Rcpp::stop("tulpa: compute_log_post requires a generic LikelihoodSpec "
               "ModelData (n_processes > 0 and data.likelihood_spec set).");
  }
  return tulpa::compute_log_post_generic_spec_double(
      params, data, layout, skip_obs_loop);
}

double compute_log_prior(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout
) {
  return compute_log_post(params, data, layout, /*skip_obs_loop=*/true);
}

double compute_log_lik_only(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout
) {
  if (field_coordinates && !field_coordinates->empty()) {
    std::vector<double> x;
    if (!std::isfinite(field_coordinates->forward(params, x))) return -INFINITY;
    FieldCoordinateScope raw(nullptr);
    return compute_log_lik_only(x, data, layout);
  }
  if (has_collapsed_spatial(layout)) return compute_collapsed_spatial(params, data, layout) -
      compute_collapsed_spatial(params, data, layout, true);
  if (data.n_processes == 0 || data.likelihood_spec == nullptr) {
    Rcpp::stop("tulpa: compute_log_lik_only requires a generic LikelihoodSpec "
               "ModelData (n_processes > 0 and data.likelihood_spec set).");
  }
  return tulpa::compute_log_post_generic_spec_double(
      params, data, layout, /*skip_obs_loop=*/false, /*skip_prior=*/true);
}

double compute_log_prior_normalizer(
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<std::string>* uncarried
) {
  if (has_collapsed_spatial(layout)) return compute_collapsed_spatial_normalizer(data, layout, uncarried);
  return tulpa::compute_log_prior_normalizer_generic_spec(data, layout, uncarried);
}

} // namespace tulpa_hmc
