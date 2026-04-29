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
inline MCLMCFitResult run_mclmc_sampler(
    const std::vector<double>& init_params,
    const ModelData& data,
    const ParamLayout& layout,
    const MCLMCConfig& config
) {
    MCLMCFitResult out;
    out.success = true;

    int n_params = static_cast<int>(init_params.size());

    // Sanity check: initial log-posterior must be finite.
    double lp0 = compute_log_post(init_params, data, layout);
    if (!std::isfinite(lp0)) {
        out.success = false;
        out.error_msg = "Initial log-posterior is not finite";
        out.samples.resize(0, n_params);
        return out;
    }

    if (config.verbose) {
        Rcpp::Rcout << "Running "
                    << (config.adjusted ? "MAMCLMC" : "MCLMC")
                    << " sampler...\n"
                    << "  Parameters: " << n_params << "\n"
                    << "  Initial step size: " << config.step_size
                    << (config.step_size <= 0.0 ? " (adapt)" : "") << "\n"
                    << "  Leapfrog steps L: " << config.L
                    << (config.L <= 0 ? " (auto)" : "") << "\n"
                    << "  Iterations: " << config.n_iter
                    << " (warmup: " << config.n_warmup << ")\n";
    }

    // log_prob_grad closure: fills `grad` with d(log_post)/dq and returns
    // log_post(q). Mirrors the SGHMC pattern (compute_gradient followed by
    // compute_log_post), reusing the existing tulpa_hmc machinery — no
    // duplicate gradient logic.
    auto log_prob_grad = [&](const std::vector<double>& q,
                             std::vector<double>& grad) -> double {
        double lp = 0.0;
        // Fused: compute_gradient writes log_post into lp via log_post_out.
        compute_gradient(q, data, layout, grad, &lp);
        return lp;
    };

    tulpa::MCLMCResult res;
    try {
        if (config.adjusted) {
            res = tulpa::mamclmc_sample(
                log_prob_grad, init_params, n_params,
                config.n_iter, config.n_warmup,
                config.step_size, config.L,
                config.mass_diag, config.seed);
        } else {
            res = tulpa::mclmc_sample(
                log_prob_grad, init_params, n_params,
                config.n_iter, config.n_warmup,
                config.step_size, config.L,
                config.mass_diag, config.seed);
        }
    } catch (const std::exception& e) {
        out.success = false;
        out.error_msg = std::string("MCLMC sampler threw: ") + e.what();
        out.samples.resize(0, n_params);
        return out;
    }

    int n_save = static_cast<int>(res.draws.size());
    out.samples.resize(n_save, n_params);
    for (int i = 0; i < n_save; i++) {
        const std::vector<double>& row = res.draws[i];
        for (int j = 0; j < n_params; j++) {
            out.samples(i, j) = row[j];
        }
    }
    out.log_lik = std::move(res.lp);

    // No per-iteration eps history is exposed by mclmc_sample / mamclmc_sample
    // (only the final adapted step size). Surface that as a single-element
    // history so the SGSamplerShimResult shape can carry it through.
    out.epsilon_history.assign(1, res.step_size_final);

    if (config.verbose) {
        Rcpp::Rcout << (config.adjusted ? "MAMCLMC" : "MCLMC")
                    << " complete. Final eps: " << res.step_size_final
                    << " accept_rate: " << res.accept_rate
                    << " divergences: " << res.n_divergences << "\n";
    }

    return out;
}

} // namespace tulpa_mclmc

#endif // TULPA_MCLMC_MODELDATA_H
