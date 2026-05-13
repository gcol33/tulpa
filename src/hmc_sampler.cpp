// hmc_sampler.cpp
// HMC/NUTS log-posterior orchestrators.
//
// Phase D simplification (gcol33/tulpa#15): the legacy ratio body of
// compute_log_post (accumulate_log_prior_and_state + accumulate_obs_log_lik
// + their fragments) was deleted along with the ratio entry points. The
// only supported path is the generic LikelihoodSpec interface, which is
// evaluated via tulpa::compute_log_post_generic_spec_double.

#include "hmc_sampler.h"
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
// After Phase D, every caller has n_processes > 0 and a non-null
// likelihood_spec. The orchestrators forward to the generic-spec
// evaluator; the `skip_obs_loop` argument controls whether the
// observation loop is executed.
// =====================================================================

double compute_log_post(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    bool skip_obs_loop,
    const double* /*precomputed_st_log_prior*/,
    const double* /*precomputed_tgp_log_prior*/
) {
  if (data.n_processes == 0 || data.likelihood_spec == nullptr) {
    Rcpp::stop("tulpa: compute_log_post requires a generic LikelihoodSpec "
               "ModelData (n_processes > 0 and data.likelihood_spec set). "
               "The legacy ratio body was removed in Phase D "
               "(gcol33/tulpa#15).");
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
  if (data.n_processes == 0 || data.likelihood_spec == nullptr) {
    Rcpp::stop("tulpa: compute_log_lik_only requires a generic LikelihoodSpec "
               "ModelData (n_processes > 0 and data.likelihood_spec set). "
               "The legacy ratio body was removed in Phase D "
               "(gcol33/tulpa#15).");
  }
  const double lp = tulpa::compute_log_post_generic_spec_double(
      params, data, layout, /*skip_obs_loop=*/false);
  const double lpr = tulpa::compute_log_post_generic_spec_double(
      params, data, layout, /*skip_obs_loop=*/true);
  return lp - lpr;
}

} // namespace tulpa_hmc
