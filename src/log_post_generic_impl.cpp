// log_post_generic_impl.cpp
// Out-of-line definition of the generic LikelihoodSpec evaluator plus an
// explicit instantiation of compute_log_post_generic<double>. The matching
// extern-template declaration in log_post_generic_impl.h keeps every other
// TU that includes log_post_impl.h from re-instantiating the heaviest
// template chain in the engine (prior dispatch + autodiff math dispatch +
// Eigen ops).

#include "hmc_sampler.h"   // tulpa_hmc::ModelData / ParamLayout
#include "log_post_impl.h"

namespace tulpa {

double compute_log_post_generic_spec_double(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    bool skip_obs_loop,
    bool skip_prior
) {
    // A malformed ModelData is raised rather than answered with a sentinel:
    // 0 is a valid flat log-posterior and -Inf a rejected proposal, so either
    // reads as a value and a sampler cannot tell it from one. The parallel
    // chain loop wraps each chain in an exception barrier, so the raise
    // reaches the caller from any thread.
    if (data.n_processes <= 0 || data.likelihood_spec == nullptr) {
        Rcpp::stop("tulpa: the log-posterior requires a generic LikelihoodSpec "
                   "ModelData -- n_processes > 0 (got %d) and "
                   "data.likelihood_spec set (%s).",
                   data.n_processes,
                   data.likelihood_spec == nullptr ? "null" : "set");
    }

    const auto* spec = static_cast<const tulpa::LikelihoodSpec*>(data.likelihood_spec);
    if (spec->ll_double == nullptr) {
        Rcpp::stop("tulpa: LikelihoodSpec '%s' ships no double-precision "
                   "log-likelihood (ll_double), which the double log-posterior "
                   "evaluator requires.", spec->name.c_str());
    }

    double log_post = compute_log_post_generic<double>(
        params, data, layout, spec->ll_double, data.model_response_data,
        skip_obs_loop, skip_prior);
    if (!skip_prior && spec->extra_prior != nullptr) {
        log_post += spec->extra_prior(params, layout, data.model_response_data);
    }
    return log_post;
}

template double compute_log_post_generic<double>(
    const std::vector<double>&,
    const ModelData&,
    const ParamLayout&,
    LikelihoodFnT<double>,
    const void*,
    bool,
    bool);

}  // namespace tulpa
