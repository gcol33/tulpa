// sghmc_api.h
// Cross-package SGHMC / SGLD API for model packages (tulpaGlmm, tulpaObs).
//
// Both samplers share the NUTS-style ModelData + ParamLayout interface
// (see nuts_api.h). The model package builds a populated ModelData and
// passes it through; tulpa drives the sampler internally.
//
// MCLMC / MAMCLMC have a separate cross-DLL entry — see tulpa/mclmc_api.h
// (tulpa_mclmc_fit). The shim reconstructs the std::function log_prob_grad
// closure from compute_log_post + compute_gradient inside tulpa.
//
// SMC has a separate cross-DLL entry — see tulpa/smc_api.h (tulpa_smc_fit).
// The shim builds log_prior / log_likelihood closures from compute_log_prior
// + compute_log_lik_only and supports a user-supplied SmcMutationFn function
// pointer (nullptr falls back to a built-in RWM kernel scaling by 1/sqrt(beta)).

#ifndef TULPA_SGHMC_API_H
#define TULPA_SGHMC_API_H

#include <R_ext/Rdynload.h>
#include "model_data.h"
#include "param_layout.h"
#include "nuts_api.h"  // for check_abi_version() + NUTSResult layout reuse

namespace tulpa {

// ----------------------------------------------------------------------------
// SGHMC / SGLD share the same flat result shape: a samples matrix
// (n_save x n_params, row-major), per-iteration log-likelihood, and the
// step-size history (epsilon adapts during warmup for SGHMC; follows a
// schedule for SGLD).
// ----------------------------------------------------------------------------
struct SGSamplerShimResult {
    int n_sample;
    int n_params;
    double* samples;          // [n_sample * n_params] row-major
    double* log_lik;          // [n_sample]
    double* epsilon_history;  // [n_iter]
    int     n_eps_history;
    double  final_epsilon;
    int     success;          // 0 / 1
    char    error_msg[256];

    void free_buffers() {
        if (samples)         { delete[] samples;         samples         = nullptr; }
        if (log_lik)         { delete[] log_lik;         log_lik         = nullptr; }
        if (epsilon_history) { delete[] epsilon_history; epsilon_history = nullptr; }
    }
};

// ----------------------------------------------------------------------------
// SGHMC: stochastic gradient HMC with friction term and minibatch noise
// correction. Tier 1 (exact MCMC) at scale.
//
//   batch_size   : minibatch size; clamped internally to N if larger.
//   epsilon      : initial step size (learning rate).
//   alpha        : friction coefficient (momentum decay; 0.01 typical).
//   L            : leapfrog steps per iteration.
//   adapt_eps    : 0 / 1; if 1, eps adapts during warmup toward
//                  target_accept = 0.65.
//   grad_clip    : gradient clipping threshold (0 disables; 100 typical).
// ----------------------------------------------------------------------------
typedef void (*SghmcFitFn)(
    const ModelData* data,
    const ParamLayout* layout,
    const double* init,
    int n_params,
    int n_iter,
    int n_warmup,
    int batch_size,
    double epsilon,
    double alpha,
    int L,
    unsigned int seed,
    int adapt_eps,
    double grad_clip,
    int verbose,
    SGSamplerShimResult* result_out
);

inline SghmcFitFn get_sghmc_fit_fn() {
    static SghmcFitFn fn = nullptr;
    if (!fn) {
        check_abi_version();
        fn = (SghmcFitFn)R_GetCCallable("tulpa", "tulpa_sghmc_fit");
    }
    return fn;
}

// ----------------------------------------------------------------------------
// SGLD: stochastic gradient Langevin dynamics. Same minibatch interface as
// SGHMC, but no momentum — the noise term plays the role of the
// HMC kinetic energy.
//
//   schedule_a / b / gamma : eps_t = a * (b + t)^(-gamma) when use_schedule=1.
//   use_schedule           : 0 / 1.
// ----------------------------------------------------------------------------
typedef void (*SgldFitFn)(
    const ModelData* data,
    const ParamLayout* layout,
    const double* init,
    int n_params,
    int n_iter,
    int n_warmup,
    int batch_size,
    double epsilon,
    double schedule_a,
    double schedule_b,
    double schedule_gamma,
    int use_schedule,
    double grad_clip,
    unsigned int seed,
    int verbose,
    SGSamplerShimResult* result_out
);

inline SgldFitFn get_sgld_fit_fn() {
    static SgldFitFn fn = nullptr;
    if (!fn) {
        check_abi_version();
        fn = (SgldFitFn)R_GetCCallable("tulpa", "tulpa_sgld_fit");
    }
    return fn;
}

} // namespace tulpa

#endif // TULPA_SGHMC_API_H
