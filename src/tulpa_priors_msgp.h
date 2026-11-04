// tulpa_priors_msgp.h
// Sliced from tulpa_priors.h. Include via "tulpa_priors.h" or directly.
//
// Prerequisite: ModelData and ParamLayout must be defined before this
// header (normally via hmc_sampler.h).

#ifndef TULPA_PRIORS_MSGP_H
#define TULPA_PRIORS_MSGP_H

#include <Rcpp.h>
#include <algorithm>
#include <cstddef>
#include <vector>
#include <cmath>
#include "autodiff_utils.h"
#include "hmc_gp_autodiff.h"
#include "pc_prior.h"
#include <limits>
#include <string>

namespace tulpa {
namespace priors {

using namespace math;

// ============================================================================
// 4. Multiscale GP spatial prior
// ============================================================================

template<typename T>
T compute_multiscale_gp_prior(const std::vector<T>& params, const ModelData& data,
                               const ParamLayout& layout, std::vector<T>& ms_gp_effect)
{
    T log_post = T(0.0);

    if (layout.is_multiscale_gp && data.has_multiscale_gp) {
        T log_sigma2_local = params[layout.log_sigma2_gp_local_idx];
        T log_phi_local = params[layout.log_phi_gp_local_idx];
        T log_sigma2_regional = params[layout.log_sigma2_gp_regional_idx];
        T log_phi_regional = params[layout.log_phi_gp_regional_idx];

        T sigma2_local_n = safe_exp(log_sigma2_local);
        T phi_local = safe_exp(log_phi_local);
        T sigma2_regional_n = safe_exp(log_sigma2_regional);
        T phi_regional = safe_exp(log_phi_regional);

        // PC priors on sigma2 + Jacobians
        log_post = log_post + tulpa_gp::log_prior_sigma2_pc_t(
            sigma2_local_n, data.ms_sigma2_local_prior_U, data.ms_sigma2_local_prior_alpha);
        log_post = log_post + log_sigma2_local;  // Jacobian

        log_post = log_post + tulpa_gp::log_prior_sigma2_pc_t(
            sigma2_regional_n, data.ms_sigma2_regional_prior_U, data.ms_sigma2_regional_prior_alpha);
        log_post = log_post + log_sigma2_regional;  // Jacobian

        // PC priors on each scale's range + Jacobians. phi is sampled
        // unconstrained on the log scale: the PC density is proper on
        // (0, inf) and penalizes short ranges, so it needs no bounding box.
        // Each scale anchors at its OWN declared lower bound --
        // P(range < range_*_lower) = range_*_prior_alpha -- which is what a
        // user setting range_local / range_regional is expressing (ranges
        // below this are implausible for this scale) and what keeps the two
        // scales apart, now through the prior's mass rather than a wall.
        //
        // This replaces a Uniform behind a hard `return -INFINITY` outside
        // (lower, upper). That form had the two defects since fixed on
        // the GP and SVC paths, which this block escaped only by
        // being unreachable until its front door was wired: the rejection
        // sits inside an autodiff log-posterior, so a step outside the box
        // yields no usable gradient and NUTS books it as a divergence; and
        // the `+ log_phi` Jacobian turned the flat density in log_phi into a
        // Uniform on phi itself, whose mean is the box centre -- with the
        // library defaults that put the regional prior mean at 5.5, several
        // times a unit-square domain's diameter. Measured at 82-88% of
        // post-warmup draws divergent.
        log_post = log_post + log_prior_range_pc_at_log(
            log_phi_local, data.multiscale_gp_data.range_local_lower,
            data.multiscale_gp_data.range_local_prior_alpha);
        log_post = log_post + log_phi_local;  // Jacobian

        log_post = log_post + log_prior_range_pc_at_log(
            log_phi_regional, data.multiscale_gp_data.range_regional_lower,
            data.multiscale_gp_data.range_regional_prior_alpha);
        log_post = log_post + log_phi_regional;  // Jacobian

        int n_gp_local = layout.gp_local_end - layout.gp_local_start;
        int n_gp_regional = layout.gp_regional_end - layout.gp_regional_start;

        if (data.msgp_parameterization == 1) {
            // Non-centered: params[gp_local_start..]/[gp_regional_start..]
            // are z ~ N(0, I) per scale. Each scale's field
            // w = f(z, sigma2, phi) is reconstructed by
            // apply_msgp_nc_transform_* in initialize_generic_state, which
            // also combines both scales to the per-observation
            // ms_gp_effect this branch would otherwise compute inline;
            // here we add only the two z priors. Avoids the same
            // field/hyperparameter funnel gp_parameterization documents,
            // applied independently per scale.
            //
            // RSR is not supported on this path, matching
            // compute_gp_spatial_prior's non-centered branch (RSR only
            // applies in the centered branch below).
            for (int k = 0; k < n_gp_local; k++) {
                T zk = params[layout.gp_local_start + k];
                log_post = log_post - T(0.5) * zk * zk;
            }
            for (int k = 0; k < n_gp_regional; k++) {
                T zk = params[layout.gp_regional_start + k];
                log_post = log_post - T(0.5) * zk * zk;
            }
            ms_gp_effect.clear();  // filled by apply_msgp_nc_transform_* on w
        } else {
            // Extract local GP effects
            std::vector<T> ms_gp_w_local(n_gp_local);
            for (int k = 0; k < n_gp_local; k++) {
                ms_gp_w_local[k] = params[layout.gp_local_start + k];
            }

            // Extract regional GP effects
            std::vector<T> ms_gp_w_regional(n_gp_regional);
            for (int k = 0; k < n_gp_regional; k++) {
                ms_gp_w_regional[k] = params[layout.gp_regional_start + k];
            }

            // Apply RSR projection if enabled
            if (data.has_rsr && !data.rsr_projection.empty()) {
                if (data.rsr_n != n_gp_local || data.rsr_n != n_gp_regional) {
                    Rcpp::stop("RSR projection: rsr_n (%d) must equal both "
                               "multiscale field lengths (local %d, "
                               "regional %d).",
                               data.rsr_n, n_gp_local, n_gp_regional);
                }
                if (data.rsr_projection.size() <
                        (std::size_t)data.rsr_n * (std::size_t)data.rsr_n) {
                    Rcpp::stop("RSR projection: rsr_projection holds %d "
                               "entries but rsr_n = %d needs %d.",
                               (int)data.rsr_projection.size(), data.rsr_n,
                               data.rsr_n * data.rsr_n);
                }
                std::vector<T> local_proj(data.rsr_n, T(0.0));
                std::vector<T> regional_proj(data.rsr_n, T(0.0));
                for (int ii = 0; ii < data.rsr_n; ii++) {
                    for (int jj = 0; jj < data.rsr_n; jj++) {
                        local_proj[ii] = local_proj[ii]
                            + T(data.rsr_projection[ii * data.rsr_n + jj]) * ms_gp_w_local[jj];
                        regional_proj[ii] = regional_proj[ii]
                            + T(data.rsr_projection[ii * data.rsr_n + jj]) * ms_gp_w_regional[jj];
                    }
                }
                ms_gp_w_local = local_proj;
                ms_gp_w_regional = regional_proj;
            }

            // Multiscale NNGP log-likelihood for both scales
            log_post = log_post + tulpa_gp::multiscale_gp_log_lik_t(
                ms_gp_w_local, ms_gp_w_regional,
                sigma2_local_n, phi_local, sigma2_regional_n, phi_regional,
                data.multiscale_gp_data);

            // Precompute combined effect at observation level. The
            // projection above can have changed the field length, so the
            // observation map is checked against what it now indexes.
            const std::vector<int>& obs_to_loc =
                data.multiscale_gp_data.obs_to_loc;
            if ((int)obs_to_loc.size() < data.N) {
                Rcpp::stop("Multiscale GP: obs_to_loc has %d entries but "
                           "the model has %d observations.",
                           (int)obs_to_loc.size(), data.N);
            }
            const int n_field = (int)std::min(ms_gp_w_local.size(),
                                              ms_gp_w_regional.size());
            for (int ii = 0; ii < data.N; ii++) {
                const int loc = obs_to_loc[ii];
                if (loc < 0 || loc >= n_field) {
                    Rcpp::stop("Multiscale GP: obs_to_loc[%d] is %d; it "
                               "must lie in [0, %d).",
                               ii + 1, loc, n_field);
                }
            }
            ms_gp_effect.resize(data.N, T(0.0));
            for (int ii = 0; ii < data.N; ii++) {
                int loc = obs_to_loc[ii];
                ms_gp_effect[ii] = ms_gp_w_local[loc] + ms_gp_w_regional[loc];
            }
        }
    }

    return log_post;
}

// The constant compute_multiscale_gp_prior leaves out, or NaN with the reason
// appended to `declined`. The PC priors and the centered NNGP densities are
// normalized. A non-centered NNGP omits the unit-normal constant for each z.
// A restricted field is evaluated at its projection, which is not a density
// on the sampled field.
inline double multiscale_gp_prior_log_normalizer(const ModelData& data,
                                                 const ParamLayout& layout,
                                                 std::vector<std::string>* declined) {
    if (!(layout.is_multiscale_gp && data.has_multiscale_gp)) return 0.0;
    const double unit_normal = log_prior_normal_log_normalizer(1.0);
    if (data.msgp_parameterization == 1) {
        return ((layout.gp_local_end - layout.gp_local_start) +
                (layout.gp_regional_end - layout.gp_regional_start)) * unit_normal;
    }
    if (data.has_rsr && !data.rsr_projection.empty()) {
        if (declined) declined->push_back("multiscale_gp:rsr_projection");
        return std::numeric_limits<double>::quiet_NaN();
    }
    return 0.0;
}

} // namespace priors
} // namespace tulpa

#endif // TULPA_PRIORS_MSGP_H
