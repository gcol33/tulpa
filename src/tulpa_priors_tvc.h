// tulpa_priors_tvc.h
// Sliced from tulpa_priors.h. Include via "tulpa_priors.h" or directly.
//
// Prerequisite: ModelData and ParamLayout must be defined before this
// header (normally via hmc_sampler.h).

#ifndef TULPA_PRIORS_TVC_H
#define TULPA_PRIORS_TVC_H

#include <vector>
#include <cmath>
#include "autodiff_utils.h"
#include "hmc_tvc.h"
#include "pc_prior.h"

namespace tulpa {
namespace priors {

using namespace math;

// ============================================================================
// 9. TVC (Temporally-Varying Coefficients) prior
// ============================================================================

template<typename T>
T compute_tvc_prior(const std::vector<T>& params, const ModelData& data,
                     const ParamLayout& layout, std::vector<T>& tvc_eta)
{
    T log_post = T(0.0);

    if (layout.has_tvc && data.tvc_data.n_tvc > 0) {
        int n_tvc = data.tvc_data.n_tvc;
        int n_groups = data.tvc_data.n_groups;
        int n_times = data.tvc_data.n_times;
        int n_obs = data.tvc_data.n_obs;
        const bool is_gp =
            data.tvc_data.structure == tulpa_temporal::TemporalType::GP;

        // Hyperparameters, on the coordinates the structure samples them on:
        // a log-precision (+ AR1's correlation) for the discrete structures, an
        // amplitude and a lengthscale for the continuous-time GP.
        std::vector<T> tvc_tau(n_tvc), tvc_rho(n_tvc, T(0.0));
        std::vector<T> tvc_sigma2(n_tvc), tvc_phi(n_tvc);

        if (is_gp) {
            for (int j = 0; j < n_tvc; j++) {
                const T log_sigma2 = params[layout.log_sigma2_tvc_gp_start + j];
                tvc_sigma2[j] = safe_exp(log_sigma2);
                // The SAME anchor pair the discrete structures read, on the
                // log-variance coordinate this one samples: "P(sigma > U) =
                // alpha" is one prior with two parameterizations, not two
                // priors (pc_prior.h).
                log_post = log_post + log_prior_log_sigma2_pc(
                    log_sigma2, data.tvc_sigma_prior_U,
                    data.tvc_sigma_prior_alpha);

                tvc_phi[j] = bounded_from_logit(
                    params[layout.logit_phi_tvc_gp_start + j],
                    data.tvc_gp_phi_prior_lower, data.tvc_gp_phi_prior_upper);
                // Uniform on (lower, upper); the bounded map holds the
                // interval, so only its Jacobian is contributed here, exactly
                // as on temporal_gp()'s lengthscale.
                log_post = log_post + log_jacobian_bounded(
                    tvc_phi[j], data.tvc_gp_phi_prior_lower,
                    data.tvc_gp_phi_prior_upper);
            }
        } else {
            for (int j = 0; j < n_tvc; j++) {
                T log_tau = params[layout.log_tau_tvc_start + j];
                tvc_tau[j] = safe_exp(log_tau);

                // PC prior on sigma = 1/sqrt(tau), on the sampled log-precision
                // scale, at the spec's own anchors: P(sigma > U) = alpha.
                log_post = log_post + log_prior_log_tau_pc(
                    log_tau, data.tvc_sigma_prior_U, data.tvc_sigma_prior_alpha);
            }

            if (data.tvc_data.structure == tulpa_temporal::TemporalType::AR1) {
                for (int j = 0; j < n_tvc; j++) {
                    T logit_rho = params[layout.logit_rho_tvc_start + j];
                    // Map logit to (-1, 1): rho = 2*invlogit(logit) - 1
                    T u = inv_logit(logit_rho);
                    tvc_rho[j] = T(2.0) * u - T(1.0);

                    // Uniform(-1, 1) prior on rho
                    // Jacobian for logit((rho+1)/2) transform
                    log_post = log_post + safe_log(u) + safe_log(T(1.0) - u);
                }
            }
        }

        // Extract TVC values
        int n_tvc_params = n_groups * n_tvc * n_times;
        std::vector<T> tvc_w_flat(n_tvc_params);
        for (int k = 0; k < n_tvc_params; k++) {
            tvc_w_flat[k] = params[layout.tvc_w_start + k];
        }

        if (is_gp) {
            // A continuous-time GP over the distinct instants. Proper, so no
            // augmentation; the level is removed by centring below like every
            // other block's. A coordinate whose covariance is not numerically
            // PD is declined as -Inf rather than carried as a NaN gradient.
            T gp_lp = T(0.0);
            if (!tulpa_tvc::tvc_log_prior_gp(tvc_w_flat, data.tvc_data,
                                             tvc_sigma2, tvc_phi, gp_lp)) {
                return T(-INFINITY);
            }
            log_post = log_post + gp_lp;
        } else {
            // TVC temporal prior (RW1, RW2, or AR1), carrying the sum-to-zero
            // AUGMENTATION for the intrinsic structures.
            log_post = log_post + tulpa_tvc::tvc_log_prior(
                tvc_w_flat, data.tvc_data, tvc_tau, tvc_rho
            );
        }

        // Each block's level is removed from eta by centring, the other half
        // of that construction. The soft penalty this replaced left the level
        // in the likelihood and stiffened it instead (gcol33/tulpa#844).
        tvc_eta.resize(n_obs, T(0.0));
        tulpa_tvc::tvc_center_eta(tvc_w_flat, data.tvc_data, tvc_eta);
    }

    return log_post;
}


} // namespace priors
} // namespace tulpa

#endif // TULPA_PRIORS_TVC_H
