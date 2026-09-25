// smc_api.h
// Cross-package SMC API for model packages (tulpaGlmm, tulpaObs, tulpaRatio).
//
// SMC runs a particle population through a tempering schedule to the
// posterior. The model package builds a populated ModelData and ParamLayout,
// supplies an `init` vector (the centre of the Gaussian reference
// q = N(init, prior_sigma^2 I) the population is drawn from), and optionally
// supplies a domain-specific `SmcMutationFn` mutation kernel.
//
// q is not the prior -- proper prior draws need per-prior closed-form
// samplers tulpa lacks generically -- and a population drawn from it cannot
// simply be tempered in the likelihood as if it were (gcol33/tulpa#876):
//   * Without a kernel (nullptr), the population tempers along the bridge
//     q^(1 - beta) (p L)^beta, mutated by a built-in HMC kernel preconditioned
//     by the population covariance, so beta = 1 is the posterior whatever q is.
//   * With a kernel, whose target is log_prior + beta * log_lik, the
//     population is first importance-corrected from q to the prior (weights
//     p / q, resampled, then mutated at beta = 0) and tempered in L alone.
//     That correction is exact but degrades when q is far from the prior.
// The log-Z accumulator does not carry every normalizing constant of the
// ModelData prior and likelihood, so SMCShimResult::log_evidence comes back
// NaN and must not be read as a model-comparison number.

#ifndef TULPA_SMC_API_H
#define TULPA_SMC_API_H

#include <R_ext/Rdynload.h>
#include "model_data.h"
#include "param_layout.h"
#include "nuts_api.h"  // for check_abi_version()

namespace tulpa {

// ----------------------------------------------------------------------------
// Flat result shape: particles row-major (n_particles x n_params),
// per-particle log-weights, and the SMC log-evidence estimate.
// Caller owns the buffers via free_buffers().
// ----------------------------------------------------------------------------
struct SMCShimResult {
    int n_particles;
    int n_params;
    double* particles;     // [n_particles * n_params] row-major
    double* log_weights;   // [n_particles]
    double  log_evidence;  // NaN (see above)
    int     success;       // 0 / 1
    char    error_msg[256];

    void free_buffers() {
        if (particles)   { delete[] particles;   particles   = nullptr; }
        if (log_weights) { delete[] log_weights; log_weights = nullptr; }
    }
};

// ----------------------------------------------------------------------------
// Pluggable mutation kernel
//   The kernel must mutate `theta` in place targeting
//   log_prior + beta * log_lik. tulpa supplies a per-call rng_seed so
//   the kernel can build a deterministic std::mt19937 inside.
// ----------------------------------------------------------------------------
typedef void (*SmcMutationFn)(
    double* theta, int n_params, double beta,
    const ModelData* data, const ParamLayout* layout,
    unsigned int rng_seed, void* user_data);

// ----------------------------------------------------------------------------
// SMC fit signature:
//   - n_particles   : population size (e.g. 500-2000).
//   - n_mcmc_steps  : MCMC mutation steps applied per particle per
//                     temperature change.
//   - ess_threshold : each temperature step is sized so the reweighted ESS
//                     is threshold * n_particles (e.g. 0.5).
//   - prior_sigma   : SD of the Gaussian reference q around `init`.
//   - mutation      : optional pluggable mutation kernel. Pass nullptr
//                     to use the built-in HMC kernel on the bridge.
//   - user_data     : opaque pointer forwarded to `mutation`.
// ----------------------------------------------------------------------------
typedef void (*SmcFitFn)(
    const ModelData* data,
    const ParamLayout* layout,
    const double* init,
    int n_params,
    int n_particles,
    int n_mcmc_steps,
    double ess_threshold,
    double prior_sigma,
    SmcMutationFn mutation,
    void* user_data,
    unsigned int seed,
    int verbose,
    SMCShimResult* result_out
);

inline SmcFitFn get_smc_fit_fn() {
    static SmcFitFn fn = nullptr;
    if (!fn) {
        check_abi_version();
        fn = (SmcFitFn)R_GetCCallable("tulpa", "tulpa_smc_fit");
    }
    return fn;
}

} // namespace tulpa

#endif // TULPA_SMC_API_H
