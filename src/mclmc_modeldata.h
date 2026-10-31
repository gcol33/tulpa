// mclmc_modeldata.h
// MCLMC / MAMCLMC entry point that takes ModelData + ParamLayout (mirror of
// run_sghmc_sampler in sghmc_sampler.h). Builds the log_prob_grad closure
// from compute_log_post + compute_gradient and dispatches to the underlying
// tulpa::mclmc_sample / tulpa::mamclmc_sample. This is the form used by the
// cross-DLL shim (tulpa_mclmc_fit) — std::function callbacks cannot cross
// the DLL boundary, ModelData + ParamLayout can.

#ifndef TULPA_MCLMC_MODELDATA_H
#define TULPA_MCLMC_MODELDATA_H

#include <Rcpp.h>
#include <RcppEigen.h>
#include <vector>
#include <string>
#include <cmath>

#include "hmc_sampler.h"   // tulpa_hmc::compute_log_post / compute_gradient
#include "mclmc.h"         // tulpa::mclmc_sample / tulpa::mamclmc_sample

namespace tulpa_mclmc {

using tulpa_hmc::ModelData;
using tulpa_hmc::ParamLayout;
using tulpa_hmc::compute_log_post;
using tulpa_hmc::compute_gradient;

// ============================================================================
// MCLMC configuration: shim-friendly POD.
// ============================================================================
struct MCLMCConfig {
    int n_iter;                      // post-warmup iterations
    int n_warmup;                    // warmup iterations
    double step_size;                // initial eps; <= 0 triggers adaptation
    int L;                           // leapfrog steps per trajectory; <= 0 auto
    int adjusted;                    // 0 = MCLMC (unadjusted), 1 = MAMCLMC
    std::vector<double> mass_diag;   // diagonal mass matrix (empty = identity)
    unsigned int seed;
    bool verbose;

    MCLMCConfig()
        : n_iter(2000), n_warmup(1000), step_size(0.0), L(0),
          adjusted(0), mass_diag(), seed(42u), verbose(true) {}
};

// ============================================================================
// MCLMC result: flat shape ready for the cross-DLL shim. Mirrors
// SGSamplerShimResult so we can reuse copy_sg_sampler_result.
// ============================================================================
struct MCLMCFitResult {
    Eigen::MatrixXd samples;             // (n_save x n_params)
    std::vector<double> log_lik;         // [n_save]
    std::vector<double> epsilon_history; // {step_size_final} (no per-iter eps trace)
    bool success;
    std::string error_msg;
};

// ============================================================================
// run_mclmc_sampler: build log_prob_grad closure from ModelData/ParamLayout
// and dispatch to mclmc_sample (unadjusted) or mamclmc_sample (MH-adjusted).
// ============================================================================
MCLMCFitResult run_mclmc_sampler(
    const std::vector<double>& init_params,
    const ModelData& data,
    const ParamLayout& layout,
    const MCLMCConfig& config
);

} // namespace tulpa_mclmc

#endif // TULPA_MCLMC_MODELDATA_H
