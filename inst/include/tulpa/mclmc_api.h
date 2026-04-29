// mclmc_api.h
// Cross-package MCLMC / MAMCLMC API for model packages.
//
// Internally the samplers in src/mclmc.h take a std::function log_prob_grad
// callback, which cannot cross a DLL boundary. The shim entry point
// tulpa_mclmc_fit takes ModelData + ParamLayout instead and reconstructs
// the closure from compute_log_post + compute_gradient inside tulpa
// (see src/mclmc_modeldata.h::run_mclmc_sampler).
//
// Both MCLMC (unadjusted) and MAMCLMC (MH-adjusted) are reached through
// the same entry point — toggle via the `adjusted` flag (0 / 1).
//
// Result shape mirrors SGSamplerShimResult: samples matrix, log-lik vector,
// and an "epsilon history" that for MCLMC degenerates to a single-element
// {step_size_final} (MCLMC adapts internally but does not expose the per-iter
// trajectory). final_epsilon is populated from that single element.

#ifndef TULPA_MCLMC_API_H
#define TULPA_MCLMC_API_H

#include <R_ext/Rdynload.h>
#include "model_data.h"
#include "param_layout.h"
#include "nuts_api.h"     // check_abi_version()
#include "sghmc_api.h"    // SGSamplerShimResult (reused — same flat shape)

namespace tulpa {

// ----------------------------------------------------------------------------
// MCLMC / MAMCLMC fit signature.
//
//   step_size  : initial eps; <= 0 triggers internal adaptation.
//   L          : leapfrog steps per trajectory; <= 0 picks max(5, sqrt(d)).
//   adjusted   : 0 = MCLMC (unadjusted), 1 = MAMCLMC (MH-adjusted).
//
// Mass matrix is currently identity-only across the shim (matches the SGHMC
// shim convention). To pass a custom diagonal, extend the signature and bump
// TULPA_ABI_VERSION.
// ----------------------------------------------------------------------------
typedef void (*MclmcFitFn)(
    const ModelData* data,
    const ParamLayout* layout,
    const double* init,
    int n_params,
    int n_iter,
    int n_warmup,
    double step_size,
    int L,
    unsigned int seed,
    int adjusted,
    int verbose,
    SGSamplerShimResult* result_out
);

inline MclmcFitFn get_mclmc_fit_fn() {
    static MclmcFitFn fn = nullptr;
    if (!fn) {
        check_abi_version();
        fn = (MclmcFitFn)R_GetCCallable("tulpa", "tulpa_mclmc_fit");
    }
    return fn;
}

} // namespace tulpa

#endif // TULPA_MCLMC_API_H
