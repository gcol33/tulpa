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
    bool skip_obs_loop
) {
    if (data.n_processes <= 0 || data.likelihood_spec == nullptr) {
        return -INFINITY;
    }

    const auto* spec = static_cast<const tulpa::LikelihoodSpec*>(data.likelihood_spec);
    if (spec->ll_double == nullptr) {
        return -INFINITY;
    }

    double log_post = compute_log_post_generic<double>(
        params, data, layout, spec->ll_double, data.model_response_data, skip_obs_loop);
    if (spec->extra_prior != nullptr) {
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
    bool);

}  // namespace tulpa
