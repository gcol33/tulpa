#include <Rcpp.h>
#include <R_ext/Rdynload.h>
#include <string>
#include <cstring>
#include "hmc_sampler.h"
#include "tulpa/likelihood.h"
#include "tulpa/nuts_api.h"

// [[Rcpp::export]]
std::string tulpa_version() {
    return "0.1.0";
}

// ============================================================================
// Registered C callable: tulpa_run_nuts_generic
// Called by model packages (tulpaOcc, etc.) via R_GetCCallable.
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
    tulpa::NUTSResult* result_out
) {
    // Convert init to std::vector
    std::vector<double> q_init(init, init + n_params);

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
        0                 // riemannian=off
    );

    // Fill result struct
    result_out->n_sample = hmc.n_sample;
    result_out->n_params = n_params;
    result_out->epsilon = hmc.epsilon;
    std::strncpy(result_out->sampler, hmc.sampler.c_str(), 63);
    result_out->sampler[63] = '\0';

    // Allocate and copy result buffers (caller owns these via free_buffers)
    int ns = hmc.n_sample;
    result_out->samples = new double[ns * n_params];
    result_out->log_prob = new double[ns];
    result_out->accept_prob = new double[ns];
    result_out->divergent = new int[ns];
    result_out->treedepth = new int[ns];

    for (int s = 0; s < ns; s++) {
        const double* row = hmc.sample_row(s);
        for (int j = 0; j < n_params; j++) {
            result_out->samples[s * n_params + j] = row[j];
        }
        result_out->log_prob[s] = hmc.log_prob[s];
        result_out->accept_prob[s] = hmc.accept_prob[s];
        result_out->divergent[s] = hmc.divergent[s] ? 1 : 0;
        result_out->treedepth[s] = hmc.treedepth[s];
    }
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
    *layout_out = tulpa_hmc::compute_param_layout(*data);
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
}

// Defined in tulpa_shims.cpp — registers laplace / pg / vi / ess shims.
void tulpa_register_shims(DllInfo* dll);

// [[Rcpp::init]]
void tulpa_register_callables(DllInfo* dll) {
    R_RegisterCCallable("tulpa", "tulpa_run_nuts_generic",
                        (DL_FUNC)&tulpa_run_nuts_generic_impl);
    R_RegisterCCallable("tulpa", "tulpa_compute_param_layout",
                        (DL_FUNC)&tulpa_compute_param_layout_impl);
    R_RegisterCCallable("tulpa", "tulpa_get_abi_version",
                        (DL_FUNC)&tulpa_get_abi_version_impl);
    R_RegisterCCallable("tulpa", "tulpa_set_gradient_mode_str",
                        (DL_FUNC)&tulpa_set_gradient_mode_str_impl);

    tulpa_register_shims(dll);
}
