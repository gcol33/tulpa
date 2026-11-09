// hmc_sampler.cpp
// HMC/NUTS log-posterior orchestrators.
//
// The one supported path is the generic LikelihoodSpec interface, evaluated
// via tulpa::compute_log_post_generic_spec_double.

#include "hmc_sampler.h"
#include "hmc_field_coordinates.h"
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
// orchestrators forward to the generic-spec evaluator; the `skip_obs_loop`
// argument controls whether the observation loop is executed.
// =====================================================================

double compute_log_post(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    bool skip_obs_loop,
    const double* /*precomputed_st_log_prior*/,
    const double* /*precomputed_tgp_log_prior*/
) {
  if (field_coordinates && !field_coordinates->entries.empty()) {
    std::vector<double> x;
    const double jacobian = field_coordinates->forward(params, x);
    FieldCoordinateScope raw(nullptr);
    return compute_log_post(x, data, layout, skip_obs_loop) + jacobian;
  }
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
  if (field_coordinates && !field_coordinates->entries.empty()) {
    std::vector<double> x;
    field_coordinates->forward(params, x);
    FieldCoordinateScope raw(nullptr);
    return compute_log_lik_only(x, data, layout);
  }
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
  return tulpa::compute_log_prior_normalizer_generic_spec(data, layout, uncarried);
}

} // namespace tulpa_hmc
