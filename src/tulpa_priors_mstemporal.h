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
#include "intrinsic_log_pdet.h"  // rw_log_pdet
#include <limits>
#include <string>

namespace tulpa {
namespace priors {

using namespace math;

// ============================================================================
// 7. Multiscale temporal prior
// ============================================================================

// Whether the trend's sampled coordinates are the walk's standardized
// innovations (non-centred) rather than its values.
inline bool multiscale_trend_noncentered(const ModelData& data,
                                         const ParamLayout& layout) {
    return data.ms_trend_parameterization == 1 &&
           layout.log_sigma2_trend_idx >= 0 &&
           tulpa_temporal::rw_walk_order(
               data.multiscale_temporal_data.trend_type) > 0;
}

// The trend values the linear predictor reads, off the sampled coordinates:
// the coordinates themselves under the centred parameterization, the walk
// rw_walk_noncentered builds from them in the spectral basis
// (set_multiscale_trend_noncentered) under the non-centred one. Returns the
// log density of the coordinates the non-centred parameterization carries
// (0 under the centred one, whose density multiscale_temporal_log_lik adds).
// The log-posterior and the NUTS store both read the trend through here, so a
// stored draw is the field the likelihood saw.
template<typename T>
T multiscale_trend_values(const std::vector<T>& params, const ModelData& data,
                          const ParamLayout& layout, std::vector<T>& trend)
{
    const int n_trend = layout.trend_end - layout.trend_start;
    std::vector<T> coords(params.begin() + layout.trend_start,
                          params.begin() + layout.trend_start + n_trend);
    if (!multiscale_trend_noncentered(data, layout)) {
        trend.swap(coords);
        return T(0.0);
    }
    const T sigma2 = safe_exp(params[layout.log_sigma2_trend_idx]);
    return tulpa_temporal::rw_walk_noncentered(
        coords, data.multiscale_temporal_data.trend_type,
        data.ms_trend_basis, data.ms_trend_slope, sigma2, trend);
}

template<typename T>
T compute_multiscale_temporal_prior(const std::vector<T>& params, const ModelData& data,
                                     const ParamLayout& layout, std::vector<T>& ms_temporal_eta)
{
    T log_post = T(0.0);

    if (layout.has_multiscale_temporal) {
        const auto& ms_data = data.multiscale_temporal_data;

        std::vector<T> ms_trend;
        // The trend the centred density is evaluated on; empty under the
        // non-centred parameterization, whose density is the coordinates'.
        std::vector<T> ms_trend_density;
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

            log_post = log_post +
                multiscale_trend_values(params, data, layout, ms_trend);
            if (!multiscale_trend_noncentered(data, layout)) {
                ms_trend_density = ms_trend;
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

                // Beta(a, b) on u: with the logit change of variables the
                // density and its Jacobian combine to a*log(u) + b*log(1-u).
                log_post = log_post
                    + T(data.ar1_rho_prior_a) * safe_log(u)
                    + T(data.ar1_rho_prior_b) * safe_log(T(1.0) - u);
            }
        }

        // GMRF log-likelihood for all components
        log_post = log_post + tulpa_temporal::multiscale_temporal_log_lik(
            ms_trend_density, ms_seasonal, ms_short_term,
            ms_sigma2_trend, ms_sigma2_seasonal, ms_sigma2_short, ms_rho_short,
            ms_data);

        // Precompute multiscale temporal contribution to linear predictor
        tulpa_temporal::compute_temporal_eta(
            ms_trend, ms_seasonal, ms_short_term, ms_data, ms_temporal_eta);
    }

    return log_post;
}

// The constant compute_multiscale_temporal_prior leaves out, or NaN with the
// reason appended to `declined`. The PC priors, the AR1 / IID short-term
// densities and every walk's 0.5 rank log(tau / 2 pi) are carried. Left out:
// the Beta(a, b) normalizer of the short-term (rho + 1) / 2, read off the
// same ar1_rho_prior_a / ar1_rho_prior_b the density reads, and the
// 0.5 log|R|_+ of an RW1 trend and of the cyclic RW1 seasonal walk, each
// augmented on its constant null direction with eigenvalue 1. An RW2 trend
// keeps its linear null direction, and a walk too short to carry a quadratic
// form (rw1_log_lik / rw2_log_lik return 0 there) leaves its values under a
// flat prior; either is improper. A non-centred trend carries standard normal
// densities on its level and innovations, normalized as they stand, and a flat
// slope under RW2.
inline double multiscale_temporal_prior_log_normalizer(
    const ModelData& data, const ParamLayout& layout,
    std::vector<std::string>* declined) {
    if (!layout.has_multiscale_temporal) return 0.0;
    const auto& ms = data.multiscale_temporal_data;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    double c = 0.0;
    if (multiscale_trend_noncentered(data, layout)) {
        if (ms.trend_type != TemporalType::RW1) {
            if (declined) declined->push_back("multiscale_temporal:trend_improper");
            return nan;
        }
    } else if (layout.log_sigma2_trend_idx >= 0) {
        const int n = layout.trend_end - layout.trend_start;
        if (ms.trend_type != TemporalType::RW1 || n < 2) {
            if (declined) declined->push_back("multiscale_temporal:trend_improper");
            return nan;
        }
        c += 0.5 * rw_log_pdet(n, 1, false);
    }
    if (layout.log_sigma2_seasonal_idx >= 0) {
        const int n = layout.seasonal_end - layout.seasonal_start;
        if (n < 2) {
            if (declined) declined->push_back("multiscale_temporal:seasonal_improper");
            return nan;
        }
        c += 0.5 * rw_log_pdet(n, 1, true);
    }
    if (layout.logit_rho_short_idx >= 0) {
        c += log_prior_beta_logit_log_normalizer(data.ar1_rho_prior_a,
                                                 data.ar1_rho_prior_b);
    }
    return c;
}

} // namespace priors
} // namespace tulpa

#endif // TULPA_PRIORS_MSTEMPORAL_H
