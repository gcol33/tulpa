// nuts_api.h
// Cross-package NUTS API for model packages (tulpaObs, etc.)
//
// Uses R_RegisterCCallable / R_GetCCallable for inter-package C++ calls.
// Both tulpa and the model package see the same types via LinkingTo headers.
//
// Usage from a model package:
//   1. #include <tulpa/nuts_api.h>
//   2. In .onLoad or first call: auto fn = tulpa::get_nuts_fn();
//   3. Call: fn(data, layout, init, n_params, n_iter, ..., &result);

#ifndef TULPA_NUTS_API_H
#define TULPA_NUTS_API_H

#include <R_ext/Error.h>
#include <R_ext/Rdynload.h>
#include "model_data.h"
#include "param_layout.h"

namespace tulpa {

// ============================================================================
// ABI version check — call from model packages at init/first use.
// Compares the version compiled into the model package (from headers)
// against the version in the loaded tulpa DLL (from registration).
// Prevents silent segfaults when structs change between tulpa versions.
// ============================================================================
typedef int (*GetABIVersionFn)();

inline void check_abi_version() {
    static bool checked = false;
    if (checked) return;
    checked = true;

    auto fn = (GetABIVersionFn)R_GetCCallable("tulpa", "tulpa_get_abi_version");
    int runtime_version = fn();
    if (runtime_version != TULPA_ABI_VERSION) {
        Rf_error(
            "tulpa ABI mismatch: model package compiled against version %d "
            "but tulpa DLL is version %d. Reinstall both packages.",
            TULPA_ABI_VERSION, runtime_version
        );
    }
}

// Result struct — flat C-compatible layout, no std::vector
struct NUTSResult {
    int n_sample;
    int n_params;
    double* samples;       // [n_sample * n_params] row-major, caller must free
    double* log_prob;      // [n_sample]
    double* accept_prob;   // [n_sample]
    int* divergent;        // [n_sample]
    int* treedepth;        // [n_sample]
    double epsilon;        // Final adapted step size
    char sampler[64];      // Sampler name (e.g. "NUTS")

    // Warm-start / resume outputs (ABI v23+). Both length
    // n_params; caller must free. Together with `epsilon` they capture the
    // adapted geometry and last sampler state, so a continued chain is
    //   init = final_position, inv_metric_diag = inv_metric_out, n_warmup = 0.
    // inv_metric_out is the inverse-mass *diagonal* (the diagonal part even
    // when the run used a denser metric internally). final_position is the
    // last draw in the sampling parameterization.
    double* inv_metric_out;   // [n_params] adapted inverse-mass diagonal
    double* final_position;   // [n_params] last sampler position

    void free_buffers() {
        if (samples)        { delete[] samples;         samples = nullptr; }
        if (log_prob)       { delete[] log_prob;        log_prob = nullptr; }
        if (accept_prob)    { delete[] accept_prob;     accept_prob = nullptr; }
        if (divergent)      { delete[] divergent;       divergent = nullptr; }
        if (treedepth)      { delete[] treedepth;       treedepth = nullptr; }
        if (inv_metric_out) { delete[] inv_metric_out;  inv_metric_out = nullptr; }
        if (final_position) { delete[] final_position;  final_position = nullptr; }
    }
};

// Function signature for the registered NUTS callable.
//
// `inv_metric_diag` (ABI v11+): optional length-`n_params` vector of initial
// inverse-mass diagonal entries. Pass nullptr to use the default structural
// warm-start (the previous v10 behaviour). When supplied, the values seed
// the dual-averaging / mass-adaptation path so subsequent warmup refines
// rather than discovers the metric — e.g. for HMC warm-started from a
// Laplace approximation, pass diag(Sigma_marg) for latent slots and
// diag(H_theta_inv) for hyperparameter slots.
typedef void (*NUTSFn)(
    const ModelData* data,
    const ParamLayout* layout,
    const double* init,
    int n_params,
    int n_iter,
    int n_warmup,
    int max_treedepth,
    double adapt_delta,
    unsigned int seed,
    int verbose,
    const double* inv_metric_diag,
    NUTSResult* result_out
);

// Retrieve the NUTS function pointer from tulpa (call from model packages)
// Automatically checks ABI version on first call.
inline NUTSFn get_nuts_fn() {
    static NUTSFn fn = nullptr;
    if (!fn) {
        check_abi_version();
        fn = (NUTSFn)R_GetCCallable("tulpa", "tulpa_run_nuts_generic");
    }
    return fn;
}

// ============================================================================
// Multi-chain NUTS (ABI v24+). Runs `n_chains` chains via
// tulpa's OpenMP across-chain runner in one call, so model packages stop
// re-implementing chain orchestration (offset-seed loops / PSOCK clusters) in
// R and get the engine's thread-parallel path for free.
//
// Layout conventions (chain-major):
//   init             [n_chains * n_params]  — chain c starts at row c. For a
//                                             fresh fit, replicate one init
//                                             across rows; for a resume, pass
//                                             each chain's final_position.
//   inv_metric_diag  [n_chains * n_params]  — row c seeds chain c's inverse-
//                    or nullptr               mass diagonal (see NUTSFn doc).
//                                             nullptr -> structural default for
//                                             every chain. For a resume, pass
//                                             each chain's inv_metric_out and
//                                             n_warmup = 0.
//   results_out      [n_chains]             — caller-allocated array; entry c
//                                             is filled exactly like the single
//                                             -chain NUTSResult (samples,
//                                             diagnostics, epsilon, and the v23
//                                             inv_metric_out / final_position).
//                                             Call free_buffers() on each.
//
// `seed` is the base seed; chains are diversified internally by chain index,
// so a shared init still yields independent chains.
// ============================================================================
typedef void (*NUTSChainsFn)(
    const ModelData* data,
    const ParamLayout* layout,
    const double* init,
    int n_params,
    int n_chains,
    int n_iter,
    int n_warmup,
    int max_treedepth,
    double adapt_delta,
    unsigned int seed,
    int verbose,
    const double* inv_metric_diag,
    NUTSResult* results_out
);

// Retrieve the multi-chain NUTS function pointer from tulpa.
// Automatically checks ABI version on first call.
inline NUTSChainsFn get_nuts_chains_fn() {
    static NUTSChainsFn fn = nullptr;
    if (!fn) {
        check_abi_version();
        fn = (NUTSChainsFn)R_GetCCallable("tulpa", "tulpa_run_nuts_chains");
    }
    return fn;
}

// ============================================================================
// Set the global gradient mode used by the next NUTS run.
//
// Accepted strings (case-sensitive, matching parse_gradient_mode in tulpa):
//   "auto", "AUTO"                          — let the dispatcher pick
//   "N", "numerical"                        — finite differences
//   "A", "autodiff", "forward"              — forward-mode AD
//   "A_r", "arena", "autodiff_arena"        — arena reverse-mode AD
//   "A_t", "autodiff_tape"                  — tape reverse-mode AD
//   "H", "handcoded", "analytical"          — hand-coded gradient
//
// Returns 0 on success, non-zero on unrecognised mode (default "auto" stays).
//
// Downstream packages call this immediately before tulpa_run_nuts_generic
// when they need to pin the gradient mode (e.g. for benchmarks, for routing
// through a registered FullGradFn, or for verifying their hand-coded
// gradient against numerical). The setting is process-global; tulpa's
// own callers reset it on every fit.
// ============================================================================
typedef int (*SetGradientModeFn)(const char* mode_str);

inline SetGradientModeFn get_set_gradient_mode_fn() {
    static SetGradientModeFn fn = nullptr;
    if (!fn) {
        check_abi_version();
        fn = (SetGradientModeFn)R_GetCCallable(
            "tulpa", "tulpa_set_gradient_mode_str");
    }
    return fn;
}

inline int set_gradient_mode_str(const char* mode_str) {
    return get_set_gradient_mode_fn()(mode_str);
}

// Function signature for compute_param_layout
typedef void (*ComputeLayoutFn)(const ModelData* data, ParamLayout* layout_out);

// Retrieve compute_param_layout from tulpa
inline ComputeLayoutFn get_compute_layout_fn() {
    static ComputeLayoutFn fn = nullptr;
    if (!fn) {
        fn = (ComputeLayoutFn)R_GetCCallable("tulpa", "tulpa_compute_param_layout");
    }
    return fn;
}

// Convenience: compute param layout for a ModelData
inline ParamLayout compute_layout(const ModelData& data) {
    ParamLayout layout;
    get_compute_layout_fn()(&data, &layout);
    return layout;
}

} // namespace tulpa

#endif // TULPA_NUTS_API_H
