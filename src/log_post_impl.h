// log_post_impl.h
// Templated log-posterior computation. The one supported path is the generic
// LikelihoodSpec interface, so the template forwards to the generic-spec
// evaluator for T = double and is a defensive no-op for autodiff T (those
// callers route through compute_gradient_generic_arena instead, which builds
// its own arena-AD log_post via compute_log_post_generic<Var>).

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
#include "gp_nc_apply.h"        // apply_gp_nc_transform_{double,arena};
                                // same rationale -- keeps the GP kernel / Eigen
                                // chain out of log_post_generic_impl.h's TUs.
#include "svc_nc_apply.h"       // apply_svc_nc_transform_{double,arena};
                                // same rationale, one custom_backward per term.
#include "msgp_nc_apply.h"      // apply_msgp_nc_transform_{double,arena};
                                // same rationale, one custom_backward per scale.

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
    // A malformed ModelData (no LikelihoodSpec, or no process) is raised by
    // compute_log_post_generic_spec_double, so the predicate is asked in one
    // place rather than answered with a different sentinel at each entry point.
    if constexpr (std::is_same_v<T, double>) {
        return compute_log_post_generic_spec_double(params, data, layout);
    } else {
        // Autodiff (arena, tape, forward) for the generic-spec path goes
        // through compute_gradient_generic_arena, which calls
        // compute_log_post_generic<Var>(...) directly, so an autodiff call
        // here is a caller error and not a point to reject.
        Rcpp::stop("tulpa: compute_log_post_impl is the double-precision "
                   "log-posterior entry point; autodiff callers route through "
                   "compute_gradient_generic_arena.");
    }
}

}  // namespace tulpa

#endif // TULPA_LOG_POST_IMPL_H
