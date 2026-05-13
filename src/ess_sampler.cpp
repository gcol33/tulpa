// ess_sampler.cpp
// Definition of tulpa_ess::compute_log_post_double, the closure target
// used by run_ess_sampler (ess_sampler.h). The legacy ratio Rcpp entry
// points (cpp_ess_fit, cpp_ess_get_n_params) were removed in Phase D of
// the tulpaRatio migration (gcol33/tulpa#15); ESS is reached from
// downstream packages via the C-callable shim `tulpa_run_ess_sampler`
// (tulpa_shims_vi_ess.h) and the generic-layout LikelihoodSpec path.

#include <Rcpp.h>
#include <RcppEigen.h>
#include "hmc_sampler.h"
#include "ess_sampler.h"
#include "log_post_impl.h"

namespace tulpa_ess {

double compute_log_post_double(
    const std::vector<double>& params,
    const tulpa_hmc::ModelData& data,
    const tulpa_hmc::ParamLayout& layout
) {
    return tulpa::compute_log_post_impl<double>(params, data, layout);
}

} // namespace tulpa_ess
