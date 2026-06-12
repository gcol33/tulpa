// tulpa_priors_mstemporal.h
// Sliced from tulpa_priors.h. Include via "tulpa_priors.h" or directly.
//
// Prerequisite: ModelData and ParamLayout must be defined before this
// header (normally via hmc_sampler.h).

#ifndef TULPA_PRIORS_MSTEMPORAL_H
#define TULPA_PRIORS_MSTEMPORAL_H

#include <vector>
#include <cmath>
#include "autodiff_utils.h"
#include "hmc_temporal_multiscale.h"

namespace tulpa {
namespace priors {

using namespace math;

// ============================================================================
// 7. Multiscale temporal prior
// ============================================================================

template<typename T>
T compute_multiscale_temporal_prior(const std::vector<T>& params, const ModelData& data,
                                     const ParamLayout& layout, std::vector<T>& ms_temporal_eta)
{
    T log_post = T(0.0);

    if (layout.has_multiscale_temporal) {
        const auto& ms_data = data.multiscale_temporal_data;

        std::vector<T> ms_trend;
        std::vector<T> ms_seasonal;
        std::vector<T> ms_short_term;
        T ms_sigma2_trend = T(1.0);
        T ms_sigma2_seasonal = T(1.0);
        T ms_sigma2_short = T(1.0);
        T ms_rho_short = T(0.5);

        // Trend component
        if (layout.log_sigma2_trend_idx >= 0) {
            T log_sigma2_trend = params[layout.log_sigma2_trend_idx];
            ms_sigma2_trend = safe_exp(log_sigma2_trend);

            int n_trend = layout.trend_end - layout.trend_start;
            ms_trend.resize(n_trend);
            for (int t = 0; t < n_trend; t++) {
                ms_trend[t] = params[layout.trend_start + t];
            }

            // PC prior on sigma2_trend + Jacobian for log transform
            log_post = log_post + tulpa_temporal::log_prior_sigma2_temporal_pc(
                ms_sigma2_trend, data.ms_sigma2_trend_prior_U, data.ms_sigma2_trend_prior_alpha);
            log_post = log_post + log_sigma2_trend;  // Jacobian
        }

        // Seasonal component
        if (layout.log_sigma2_seasonal_idx >= 0) {
            T log_sigma2_seasonal = params[layout.log_sigma2_seasonal_idx];
            ms_sigma2_seasonal = safe_exp(log_sigma2_seasonal);

            int n_seasonal = layout.seasonal_end - layout.seasonal_start;
            ms_seasonal.resize(n_seasonal);
            for (int t = 0; t < n_seasonal; t++) {
                ms_seasonal[t] = params[layout.seasonal_start + t];
            }

            // PC prior on sigma2_seasonal + Jacobian
            log_post = log_post + tulpa_temporal::log_prior_sigma2_temporal_pc(
                ms_sigma2_seasonal, data.ms_sigma2_seasonal_prior_U, data.ms_sigma2_seasonal_prior_alpha);
            log_post = log_post + log_sigma2_seasonal;  // Jacobian
        }

        // Short-term component
        if (layout.log_sigma2_short_idx >= 0) {
            T log_sigma2_short = params[layout.log_sigma2_short_idx];
            ms_sigma2_short = safe_exp(log_sigma2_short);

            int n_short = layout.short_term_end - layout.short_term_start;
            ms_short_term.resize(n_short);
            for (int t = 0; t < n_short; t++) {
                ms_short_term[t] = params[layout.short_term_start + t];
            }

            // PC prior on sigma2_short + Jacobian
            log_post = log_post + tulpa_temporal::log_prior_sigma2_temporal_pc(
                ms_sigma2_short, data.ms_sigma2_short_prior_U, data.ms_sigma2_short_prior_alpha);
            log_post = log_post + log_sigma2_short;  // Jacobian

            // AR1 rho parameter
            if (layout.logit_rho_short_idx >= 0) {
                T logit_rho_short = params[layout.logit_rho_short_idx];
                // Map logit to (-1, 1): rho = 2*invlogit(logit) - 1
                T u = inv_logit(logit_rho_short);
                ms_rho_short = T(2.0) * u - T(1.0);

                // Beta(2,2) prior on u + Jacobian for logit transform
                log_post = log_post + safe_log(u) + safe_log(T(1.0) - u);  // Beta(2,2)
                log_post = log_post + safe_log(u) + safe_log(T(1.0) - u);  // Jacobian
            }
        }

        // GMRF log-likelihood for all components
        log_post = log_post + tulpa_temporal::multiscale_temporal_log_lik(
            ms_trend, ms_seasonal, ms_short_term,
            ms_sigma2_trend, ms_sigma2_seasonal, ms_sigma2_short, ms_rho_short,
            ms_data);

        // Precompute multiscale temporal contribution to linear predictor
        tulpa_temporal::compute_temporal_eta(
            ms_trend, ms_seasonal, ms_short_term, ms_data, ms_temporal_eta);
    }

    return log_post;
}


} // namespace priors
} // namespace tulpa

#endif // TULPA_PRIORS_MSTEMPORAL_H
