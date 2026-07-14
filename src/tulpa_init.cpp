#include <Rcpp.h>
#include <R_ext/Rdynload.h>
#include <string>
#include <cstring>
#include "hmc_sampler.h"
#include "shim_guard.h"
#include "tulpa/likelihood.h"
#include "tulpa/nuts_api.h"

// [[Rcpp::export]]
std::string tulpa_version() {
    return "0.1.0";
}

// ============================================================================
// Copy one pure-C++ chain result into a flat NUTSResult (caller owns the
// allocated buffers via free_buffers). Single source of truth shared by the
// single-chain and multi-chain registered callables.
// ============================================================================
static void fill_nuts_result_from_cpp(
    tulpa::NUTSResult* out,
    const tulpa_hmc::HMCResultCpp& hmc,
    int n_params
) {
    out->n_sample = hmc.n_sample;
    out->n_params = n_params;
    out->epsilon = hmc.epsilon;
    std::strncpy(out->sampler, hmc.sampler.c_str(), 63);
    out->sampler[63] = '\0';

    int ns = hmc.n_sample;
    // size_t products: int * int overflows (UB) before ~2.1e9 elements, which
    // a large-latent multi-chain run can reach.
    out->samples = new double[static_cast<std::size_t>(ns) * n_params];
    out->log_prob = new double[ns];
    out->accept_prob = new double[ns];
    out->divergent = new int[ns];
    out->treedepth = new int[ns];

    for (int s = 0; s < ns; s++) {
        const double* row = hmc.sample_row(s);
        for (int j = 0; j < n_params; j++) {
            out->samples[static_cast<std::size_t>(s) * n_params + j] = row[j];
        }
        out->log_prob[s] = hmc.log_prob[s];
        out->accept_prob[s] = hmc.accept_prob[s];
        out->divergent[s] = hmc.divergent[s] ? 1 : 0;
        out->treedepth[s] = hmc.treedepth[s];
    }

    // Warm-start / resume outputs (gcol33/tulpa#29). run_hmc_chain_cpp always
    // sizes these to n_params; the bounds check is belt-and-braces so a short
    // vector defaults to the identity metric / origin rather than reading OOB.
    out->inv_metric_out = new double[n_params];
    out->final_position = new double[n_params];
    for (int j = 0; j < n_params; j++) {
        out->inv_metric_out[j] =
            (j < (int)hmc.inv_metric_diag.size()) ? hmc.inv_metric_diag[j] : 1.0;
        out->final_position[j] =
            (j < (int)hmc.final_position.size()) ? hmc.final_position[j] : 0.0;
    }
}

// ============================================================================
// Registered C callable: tulpa_run_nuts_generic
// Called by model packages (tulpaObs, etc.) via R_GetCCallable.
// Thin wrapper around tulpa_hmc::run_hmc_chain_cpp.
// ============================================================================
static void tulpa_run_nuts_generic_impl(
    const tulpa::ModelData* data,
    const tulpa::ParamLayout* layout,
    const double* init,
    int n_params,
    int n_iter,
    int n_warmup,
    int max_treedepth,
    double adapt_delta,
    unsigned int seed,
    int verbose,
    const double* inv_metric_diag,
    tulpa::NUTSResult* result_out
) {
    TULPA_SHIM_GUARD_BEGIN
    // Convert init to std::vector
    std::vector<double> q_init(init, init + n_params);

    // Convert optional inv-mass diagonal to std::vector (empty -> default warm-start)
    std::vector<double> inv_metric_init;
    if (inv_metric_diag != nullptr) {
        inv_metric_init.assign(inv_metric_diag, inv_metric_diag + n_params);
    }

    // Run full NUTS (L=0 means NUTS mode)
    tulpa_hmc::HMCResultCpp hmc = tulpa_hmc::run_hmc_chain_cpp(
        q_init, *data, *layout,
        n_iter, n_warmup,
        0,                // L=0 → NUTS
        1,                // chain_id
        seed,
        verbose != 0,
        max_treedepth,
        tulpa::MassMatrixType::DIAG,
        adapt_delta,
        0,                // riemannian=off
        inv_metric_init
    );

    fill_nuts_result_from_cpp(result_out, hmc, n_params);
    TULPA_SHIM_GUARD_END("tulpa_run_nuts_generic")
}

// ============================================================================
// Registered C callable: tulpa_run_nuts_chains (gcol33/tulpa#30)
// Multi-chain across-chain OpenMP runner. `init` is chain-major
// [n_chains * n_params]; `inv_metric_diag` is chain-major
// [n_chains * n_params] or nullptr (structural default for all chains).
// `results_out` is a caller-allocated array of n_chains NUTSResult.
// ============================================================================
static void tulpa_run_nuts_chains_impl(
    const tulpa::ModelData* data,
    const tulpa::ParamLayout* layout,
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
    tulpa::NUTSResult* results_out
) {
    TULPA_SHIM_GUARD_BEGIN
    // Per-chain init (chain-major rows).
    std::vector<std::vector<double>> q_init_per_chain(n_chains);
    for (int c = 0; c < n_chains; c++) {
        const double* row = init + (std::size_t)c * n_params;
        q_init_per_chain[c].assign(row, row + n_params);
    }

    // Per-chain inverse-mass diagonal (empty -> default for every chain).
    std::vector<std::vector<double>> inv_metric_per_chain;
    if (inv_metric_diag != nullptr) {
        inv_metric_per_chain.resize(n_chains);
        for (int c = 0; c < n_chains; c++) {
            const double* row = inv_metric_diag + (std::size_t)c * n_params;
            inv_metric_per_chain[c].assign(row, row + n_params);
        }
    }

    std::vector<tulpa_hmc::HMCResultCpp> chains = tulpa_hmc::run_hmc_parallel_chains_cpp(
        q_init_per_chain, inv_metric_per_chain, *data,
        n_iter, n_warmup,
        0,                // L=0 → NUTS
        n_chains, seed, verbose != 0, max_treedepth,
        tulpa::MassMatrixType::DIAG, adapt_delta,
        0,                // riemannian=off
        "",               // checkpoint_path
        layout            // honour the caller's layout (gcol33/tulpa#70)
    );

    for (int c = 0; c < n_chains; c++) {
        fill_nuts_result_from_cpp(&results_out[c], chains[c], n_params);
    }
    TULPA_SHIM_GUARD_END("tulpa_run_nuts_chains")
}

// ============================================================================
// Registration: called by R_init_tulpa (auto-generated in RcppExports.cpp)
// We hook into it via .onLoad to also register our C callable.
// ============================================================================

// Wrapper for compute_param_layout — registered as C callable
static void tulpa_compute_param_layout_impl(
    const tulpa::ModelData* data,
    tulpa::ParamLayout* layout_out
) {
    TULPA_SHIM_GUARD_BEGIN
    *layout_out = tulpa_hmc::compute_param_layout(*data);
    TULPA_SHIM_GUARD_END("tulpa_compute_param_layout")
}

// ABI version — returns the version baked into this DLL
static int tulpa_get_abi_version_impl() {
    return tulpa::TULPA_ABI_VERSION;
}

// Set the global gradient mode from a string. Single source of truth: parses
// via tulpa_hmc::parse_gradient_mode (the same function R-level entry points
// hmc_rcpp_fit / vi_sampler use), then forwards to set_gradient_mode. Returns
// 0 on a recognised mode, 1 if the string falls through to the AUTO default
// (so a downstream caller can detect a typo and retry).
static int tulpa_set_gradient_mode_str_impl(const char* mode_str) {
    TULPA_SHIM_GUARD_BEGIN
    if (mode_str == nullptr) {
        tulpa_hmc::set_gradient_mode(tulpa::GradientMode::AUTO);
        return 1;
    }
    std::string s(mode_str);
    tulpa::GradientMode mode = tulpa_hmc::parse_gradient_mode(s);
    tulpa_hmc::set_gradient_mode(mode);

    // parse_gradient_mode returns AUTO both for "auto" and for unrecognised
    // input. Distinguish them so a typo does not silently revert the user's
    // request to AUTO.
    if (mode == tulpa::GradientMode::AUTO &&
        s != "auto" && s != "AUTO") {
        return 1;
    }
    return 0;
    TULPA_SHIM_GUARD_END("tulpa_set_gradient_mode_str")
    return 1;  // unreachable: the guard either returned or raised
}

// Defined in tulpa_shims.cpp — registers laplace / pg / vi / ess shims.
void tulpa_register_shims(DllInfo* dll);

// Defined in tgmrf_registry.cpp — registers the tulpa_register_tgmrf
// C callable that user DLLs use to insert TgmrfSpec POD into the
// process-global registry at load time.
void tulpa_register_tgmrf_callables(DllInfo* dll);

// Defined in cell_coupling_registry.cpp — registers the
// tulpa_register_cell_coupling C callable that user DLLs use to insert
// CellCouplingSpec subclasses into tulpa's process-global registry at
// load time (gcol33/tulpa#32 Change 2b).
void tulpa_register_cell_coupling_callables(DllInfo* dll);

// [[Rcpp::init]]
void tulpa_register_callables(DllInfo* dll) {
    R_RegisterCCallable("tulpa", "tulpa_run_nuts_generic",
                        (DL_FUNC)&tulpa_run_nuts_generic_impl);
    R_RegisterCCallable("tulpa", "tulpa_run_nuts_chains",
                        (DL_FUNC)&tulpa_run_nuts_chains_impl);
    R_RegisterCCallable("tulpa", "tulpa_compute_param_layout",
                        (DL_FUNC)&tulpa_compute_param_layout_impl);
    R_RegisterCCallable("tulpa", "tulpa_get_abi_version",
                        (DL_FUNC)&tulpa_get_abi_version_impl);
    R_RegisterCCallable("tulpa", "tulpa_set_gradient_mode_str",
                        (DL_FUNC)&tulpa_set_gradient_mode_str_impl);

    tulpa_register_shims(dll);
    tulpa_register_tgmrf_callables(dll);
    tulpa_register_cell_coupling_callables(dll);
}
