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
#include "walnuts_config.h"
#include "sampling_target.h"

namespace tulpa {

struct SamplerControl {
    int L = 0;
    MassMatrixType metric = MassMatrixType::DIAG;
    bool walnuts = false;
    WalnutsConfig walnuts_config;
};

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

    auto fn = (GetABIVersionFn)R_GetCCallable("tulpa", "tulpa_get_abi_version");
    int runtime_version = fn();
    if (runtime_version != TULPA_ABI_VERSION) {
        Rf_error(
            "tulpa ABI mismatch: model package compiled against version %d "
            "but tulpa DLL is version %d. Reinstall both packages.",
            TULPA_ABI_VERSION, runtime_version
        );
    }
    checked = true;
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
    //   init = final_position, inv_metric_diag = inv_metric_out,
    //   epsilon_init = epsilon, n_warmup = 0, with the same max_treedepth,
    //   fixed integrator and sampler control.
    // This preserves a diagonal-metric kernel. A dense or structured metric's
    // diagonal alone does not reproduce its transition kernel.
    // inv_metric_out is the inverse-mass *diagonal* (the diagonal part even
    // when the run used a denser metric internally). final_position is the
    // last position in the ModelData parameterization. Automatic field
    // transport is reconstructed from ModelData when the chain resumes.
    double* inv_metric_out;   // [n_params] adapted inverse-mass diagonal
    double* final_position;   // [n_params] last sampler position

    int* n_leapfrog;       // [n_sample] actual integrator step count
    double* energy;       // [n_sample] starting Hamiltonian

    int n_spatial_fields = 0;
    double* spatial_draws = nullptr;      // [n_sample * n_spatial_fields] row-major
    double* unstructured_draws = nullptr; // BYM2 conditional unstructured field

    SamplingTarget target = SamplingTarget::JOINT_POSTERIOR;
    int inference_tier = 1;

    void free_buffers() {
        if (samples)        { delete[] samples;         samples = nullptr; }
        if (log_prob)       { delete[] log_prob;        log_prob = nullptr; }
        if (accept_prob)    { delete[] accept_prob;     accept_prob = nullptr; }
        if (divergent)      { delete[] divergent;       divergent = nullptr; }
        if (treedepth)      { delete[] treedepth;       treedepth = nullptr; }
        if (inv_metric_out) { delete[] inv_metric_out;  inv_metric_out = nullptr; }
        if (final_position) { delete[] final_position;  final_position = nullptr; }
        if (n_leapfrog)     { delete[] n_leapfrog;      n_leapfrog = nullptr; }
        if (energy)         { delete[] energy;          energy = nullptr; }
        if (spatial_draws)   { delete[] spatial_draws;   spatial_draws = nullptr; }
        if (unstructured_draws) { delete[] unstructured_draws; unstructured_draws = nullptr; }
    }
};

// Function signature for the registered NUTS callable.
//
// `control` selects fixed-L HMC, NUTS or WALNUTS and the mass-matrix type.
// nullptr uses NUTS with a diagonal metric. WALNUTS requires L == 0.
// The same control applies to every chain in NUTSChainsFn.
//
// `inv_metric_diag` (ABI v11+): optional length-`n_params` vector of initial
// inverse-mass diagonal entries. Pass nullptr to use the default structural
// warm-start (the previous v10 behaviour). When supplied, the values seed
// the dual-averaging / mass-adaptation path so subsequent warmup refines
// rather than discovers the metric — e.g. for HMC warm-started from a
// Laplace approximation, pass diag(Sigma_marg) for latent slots and
// diag(H_theta_inv) for hyperparameter slots.
// `epsilon_init`: finite, positive initial step size, or 0 for the heuristic.
// With n_warmup == 0 a positive value is used unchanged, including the usual
// independent trajectory jitter. Warmup adapts from a supplied value.
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
    const SamplerControl* control,
    const double* inv_metric_diag,
    double epsilon_init,
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
//   epsilon_init     [n_chains] or nullptr  — chain c's step size, or the
//                                             heuristic for every chain.
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
    const SamplerControl* control,
    const double* inv_metric_diag,
    const double* epsilon_init,
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
//   "A_r", "arena", "autodiff_arena"        — arena reverse-mode AD
//   "H", "handcoded", "analytical"          — hand-coded gradient
//   "A", "autodiff", "forward"              — accepted; resolves to the arena
//   "A_t", "autodiff_tape"                    reverse-mode AD path (no separate
//                                             forward / tape kernel is shipped)
//
// The dispatcher exposes three distinct gradient sources -- numerical (N),
// arena reverse-mode AD (A_r, the default AD path), and a model package's
// hand-coded gradient (H). The forward (A) and tape (A_t) spellings are parsed
// for API stability but currently alias A_r rather than selecting a separate
// kernel; requesting them does not error.
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

// ============================================================================
// tulpa's OpenMP thread cap. A model package linking its own OpenMP runtime
// cannot set the width of tulpa's teams with its own omp_set_num_threads();
// this sets tulpa's cap to `n` (n > 0) and returns the cap before the call
// (n <= 0 only reads it). The multi-chain runner's chain team is
// min(n_chains, this cap). OmpCapScope restores the cap on scope exit.
// ============================================================================
typedef int (*OmpMaxThreadsFn)(int n);

inline int omp_max_threads(int n) {
    static OmpMaxThreadsFn fn = nullptr;
    if (!fn) fn = (OmpMaxThreadsFn)R_GetCCallable("tulpa", "tulpa_omp_max_threads");
    return fn(n);
}

struct OmpCapScope {
    int prev;
    explicit OmpCapScope(int n) : prev(omp_max_threads(n)) {}
    ~OmpCapScope() { omp_max_threads(prev); }
    OmpCapScope(const OmpCapScope&) = delete;
    OmpCapScope& operator=(const OmpCapScope&) = delete;
};

// Function signature for compute_param_layout
typedef void (*ComputeLayoutFn)(const ModelData* data, ParamLayout* layout_out);

// Retrieve compute_param_layout from tulpa. compute_layout is typically a
// consumer's first tulpa call and writes through a ParamLayout*, so it must
// verify the ABI before any struct is exchanged, like the sibling getters.
inline ComputeLayoutFn get_compute_layout_fn() {
    static ComputeLayoutFn fn = nullptr;
    if (!fn) {
        check_abi_version();
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
