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
#include "intrinsic_log_pdet.h"  // rw_log_pdet
#include <limits>
#include <string>

namespace tulpa {
namespace priors {

using namespace math;

// ============================================================================
// 9. TVC (Temporally-Varying Coefficients) prior
// ============================================================================

// The AR1 correlation and the GP lengthscale on the scale the density reads
// them, from the unconstrained coordinates the sampler carries. The prior below
// and every reader of a stored draw map through these, so the two cannot place a
// coordinate differently.
template<typename T>
inline T tvc_ar1_rho_from(const T& logit_rho) {
    return T(2.0) * inv_logit(logit_rho) - T(1.0);
}

template<typename T>
inline T tvc_gp_phi_from(const T& logit_phi, const ModelData& data) {
    return bounded_from_logit(logit_phi, data.tvc_gp_phi_prior_lower,
                              data.tvc_gp_phi_prior_upper);
}

// `tvc_w` receives the field the likelihood reads: every (group, term) block,
// centred, in the flat layout tvc_block_offset() addresses.
template<typename T>
T compute_tvc_prior(const std::vector<T>& params, const ModelData& data,
                     const ParamLayout& layout, std::vector<T>& tvc_eta,
                     std::vector<T>& tvc_w)
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

                tvc_phi[j] = tvc_gp_phi_from(
                    params[layout.logit_phi_tvc_gp_start + j], data);
                // Uniform on (lower, upper); the bounded map holds the
                // interval, so only its Jacobian is contributed here, exactly
                // as on temporal_gp()'s lengthscale.
                log_post = log_post + log_jacobian_bounded(
                    params[layout.logit_phi_tvc_gp_start + j], data.tvc_gp_phi_prior_lower,
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
                    tvc_rho[j] = tvc_ar1_rho_from(logit_rho);

                    // Uniform(-1, 1) prior on rho
                    // Jacobian for logit((rho+1)/2) transform
                    log_post = log_post + T(data.tvc_rho_prior_a) * safe_log(u)
                        + T(data.tvc_rho_prior_b) * safe_log(T(1.0) - u);
                }
            }
        }

        // Extract TVC values
        int n_tvc_params = n_groups * n_tvc * n_times;
        std::vector<T>& tvc_w_flat = tvc_w;
        tvc_w_flat.resize(n_tvc_params);
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
                                             tvc_sigma2, tvc_phi, gp_lp,
                                             data.tvc_gp_parameterization != 0)) {
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

// The constant compute_tvc_prior leaves out, or NaN with the reason appended to
// `declined`. The PC priors, the Uniform(-1, 1) on an AR1 rho carried to its
// logit, the AR1 and GP field densities are normalized; a GP lengthscale omits
// its Uniform normalizer. An RW1 / cyclic RW2 block (one per group and term)
// omits the 0.5 log|R|_+ of its walk, the augmentation filling the constant
// null direction with eigenvalue 1; an acyclic RW2 block keeps its linear null
// direction, which leaves the prior improper.
inline double tvc_prior_log_normalizer(const ModelData& data,
                                       const ParamLayout& layout,
                                       std::vector<std::string>* declined) {
    if (!layout.has_tvc || data.tvc_data.n_tvc <= 0) return 0.0;
    const auto& tv = data.tvc_data;
    switch (tv.structure) {
    case tulpa_temporal::TemporalType::GP:
        return tv.n_tvc * log_prior_uniform_log_normalizer(
            data.tvc_gp_phi_prior_lower, data.tvc_gp_phi_prior_upper)
            + (data.tvc_gp_parameterization != 0 ? tv.n_groups * tv.n_tvc * tv.n_times
               * log_prior_normal_log_normalizer(1.0) : 0.0);
    case tulpa_temporal::TemporalType::AR1:
        return tv.n_tvc * log_prior_beta_logit_log_normalizer(data.tvc_rho_prior_a, data.tvc_rho_prior_b);
    case tulpa_temporal::TemporalType::IID:
        return 0.0;
    case tulpa_temporal::TemporalType::RW1:
        return 0.5 * tv.n_groups * tv.n_tvc * rw_log_pdet(tv.n_times, 1, tv.cyclic);
    case tulpa_temporal::TemporalType::RW2:
        if (tv.cyclic) {
            return 0.5 * tv.n_groups * tv.n_tvc * rw_log_pdet(tv.n_times, 2, true);
        }
        if (declined) declined->push_back("tvc:rw2_improper");
        return std::numeric_limits<double>::quiet_NaN();
    default:
        if (declined) declined->push_back("tvc");
        return std::numeric_limits<double>::quiet_NaN();
    }
}


} // namespace priors
} // namespace tulpa

#endif // TULPA_PRIORS_TVC_H
