// log_post_impl.h
// Templated log-posterior computation. Phase D simplification
// (gcol33/tulpa#15): the only supported path is the generic
// LikelihoodSpec interface, so the template now just forwards to
// the generic-spec evaluator for T = double and is a defensive no-op
// for autodiff T (those callers should route through
// compute_gradient_generic_arena instead, which builds its own
// arena-AD log_post via compute_log_post_generic<Var>).

#ifndef TULPA_LOG_POST_IMPL_H
#define TULPA_LOG_POST_IMPL_H

#include <type_traits>
#include <vector>

#include "autodiff_utils.h"
#include "tulpa_priors.h"      // priors::* helpers used by log_post_generic_impl.h
#include "tulpa/likelihood.h"
#include "spde_nc_apply.h"     // apply_spde_nc_transform_{double,arena};
                                // included here (outside namespace tulpa)
                                // so log_post_generic_impl.h's nested
                                // include is a guard-hit no-op.

// Expects hmc_sampler.h to have been included first by the umbrella TU,
// defining tulpa_hmc::ModelData / tulpa_hmc::ParamLayout.

using tulpa_hmc::ModelData;
using tulpa_hmc::ParamLayout;

namespace tulpa {

using namespace math;

// Generic-LikelihoodSpec evaluator (compute_log_post_generic +
// compute_log_post_generic_spec_double). Self-contained, idempotent.
#include "log_post_generic_impl.h"

template<typename T>
T compute_log_post_impl(
    const std::vector<T>& params,
    const ModelData& data,
    const ParamLayout& layout
) {
    if (data.n_processes == 0 || data.likelihood_spec == nullptr) {
        // Legacy ratio body was deleted in Phase D. Reaching this branch
        // means a caller built ModelData without a LikelihoodSpec, which
        // is no longer supported in tulpa core. Return T(0) as a defensive
        // no-op so callers that defer error reporting to a downstream
        // check (e.g. resolve_gradient_fn's Rcpp::stop) see a finite
        // value rather than UB on layout.legacy.*.
        return T(0);
    }

    if constexpr (std::is_same_v<T, double>) {
        return compute_log_post_generic_spec_double(params, data, layout);
    } else {
        // Autodiff (arena, tape, forward) for the generic-spec path goes
        // through compute_gradient_generic_arena, which calls
        // compute_log_post_generic<Var>(...) directly. compute_log_post_impl
        // is therefore only reached with T = double in production; an
        // autodiff call here is a logic error in the caller.
        return T(0);
    }
}

}  // namespace tulpa

#endif // TULPA_LOG_POST_IMPL_H
