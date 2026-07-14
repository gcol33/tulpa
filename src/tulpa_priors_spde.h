// tulpa_priors_spde.h
// SPDE Matern prior on the latent mesh-node block.
//
// Two modes, dispatched on data.spde_data.joint_hypers:
//
//   joint_hypers = false (legacy fixed-hyper / inner-Laplace mode):
//     params[spde_w_start..spde_w_end) is w directly. Q is built once at
//     ModelData setup from the fixed (kappa, tau_spde) and reused every
//     gradient call. Prior contribution is
//         log p(w | theta) = -0.5 * w' Q w   (+ const log|Q|/2)
//     The constant log|Q|/2 cancels in NUTS acceptance ratios and is
//     dropped from the returned log-density. compute_spde_prior populates
//     `spde_w_out` so the eta path can pull w from a contiguous buffer.
//
//   joint_hypers = true (joint-NUTS mode):
//     params[spde_w_start..spde_w_end) is z (non-centered draws). The
//     field w = L^{-T}(theta) z is computed downstream (Step 3 of the
//     (a.iii) arc: log_post_generic_impl.h applies SpdeNcTransform after
//     this prior call and stashes the result on state.spde_w). The prior
//     contribution here is just the unit Gaussian on z:
//         log p(z) = -0.5 * sum_j z_j^2
//     The change-of-variable Jacobian from z to w cancels exactly against
//     log|Q(theta)|/2 in the original w-parameterization — see (a.ii) in
//     spde_nc_transform.{h,cpp}, where the implicit-function-theorem
//     adjoint already accounts for this. No explicit log-det term here.
//     compute_spde_prior leaves `spde_w_out` empty in this mode; the NC
//     transform in initialize_generic_state fills it before the eta loop.
//
// PC prior on (range, sigma) per Fuglstad et al. 2019 (JASA) is applied
// separately in compute_spde_hyper_prior and arrives populated in Step 4
// of the (a.iii) arc; it currently returns T(0) so this header builds
// cleanly between Step 2 and Step 4.
//
// Prerequisite: ModelData and ParamLayout must be defined before this
// header (normally via hmc_sampler.h).

#ifndef TULPA_PRIORS_SPDE_H
#define TULPA_PRIORS_SPDE_H

#include <cmath>
#include <vector>
#include "autodiff_utils.h"

namespace tulpa {
namespace priors {

using namespace math;

// SPDE latent-block prior. Returns T(0) when SPDE is inactive.
template<typename T>
T compute_spde_prior(const std::vector<T>& params, const ModelData& data,
                      const ParamLayout& layout, std::vector<T>& spde_w_out)
{
    if (!layout.is_spde || !data.has_spde) {
        spde_w_out.clear();
        return T(0.0);
    }

    const auto& spde = data.spde_data;
    const int  n_mesh = spde.n_mesh;
    const int  w0     = layout.spde_w_start;

    if (spde.joint_hypers || spde.nc_fixed) {
        // Non-centered (joint-NUTS or fixed-hyper #87): params[w0..) is z.
        // Unit Gaussian prior on z; no Q involvement and no Jacobian term.
        // spde_w_out is left empty — log_post_generic_impl computes the field
        // v = L^{-T} z and fills state.spde_w before the eta accumulator runs.
        // Under fixed hypers the z->v Jacobian and log|Q|/2 are both constant
        // and drop, so -0.5 z'z is the centered prior -0.5 v'Q v up to that
        // dropped constant.
        spde_w_out.clear();
        T qf = T(0.0);
        for (int j = 0; j < n_mesh; j++) {
            const T z_j = params[w0 + j];
            qf = qf + z_j * z_j;
        }
        return T(-0.5) * qf;
    }

    // Legacy fixed-hyper: params[w0..) is w directly. -0.5 * w' Q w via
    // CSC iteration. Q is symmetric and stored full (both triangles) —
    // matches the SpdeQBuilder convention in spde_qbuilder.h, so we sum
    // every (row, col) entry once. log|Q|/2 is constant under fixed
    // (kappa, tau_spde) and is dropped (cancels in NUTS / VI / SMC).
    spde_w_out.resize(n_mesh);
    for (int j = 0; j < n_mesh; j++) {
        spde_w_out[j] = params[w0 + j];
    }
    T qf = T(0.0);
    for (int col = 0; col < n_mesh; col++) {
        const int beg = spde.Q_p[col];
        const int end = spde.Q_p[col + 1];
        const T   w_col = spde_w_out[col];
        for (int idx = beg; idx < end; idx++) {
            const int row = spde.Q_i[idx];
            qf = qf + spde_w_out[row] * T(spde.Q_x[idx]) * w_col;
        }
    }
    return T(-0.5) * qf;
}

// PC prior on (range, sigma) expressed directly in (log_kappa, log_tau).
//
// Fuglstad et al. 2019 (JASA) "Constructing priors that penalize the
// complexity of Gaussian random fields":
//   pi(range) = (lambda_r / 2) * range^{-3/2} * exp(-lambda_r * range^{-1/2})
//   pi(sigma) = lambda_s * exp(-lambda_s * sigma)
//   lambda_r  = -log(prior_range_alpha) * sqrt(prior_range_0)
//   lambda_s  = -log(prior_sigma_alpha) / prior_sigma_0
//
// SPDE Matern map (nu = data.spde_data.nu, d = 2):
//   range = sqrt(8 nu) / kappa            = sqrt(8 nu) * exp(-log_kappa)
//   sigma = 1 / (sqrt(4 pi) * kappa * tau) = exp(-log_kappa - log_tau) / sqrt(4 pi)
// hence
//   log_range = 0.5 * log(8 nu) - log_kappa
//   log_sigma = -0.5 * log(4 pi) - log_kappa - log_tau
//
// Jacobian of (log_kappa, log_tau) -> (range, sigma):
//   d range / d log_kappa = -range,  d range / d log_tau = 0
//   d sigma / d log_kappa = -sigma,  d sigma / d log_tau = -sigma
// so |det J| = range * sigma and log|J| = log_range + log_sigma.
//
// Joint hyper-density in (log_kappa, log_tau):
//   log p(log_kappa, log_tau) = log pi(range) + log pi(sigma)
//                             + log_range + log_sigma
//
// The range^{-1/2} factor is computed as (8 nu)^{-1/4} * exp(0.5 log_kappa)
// to avoid division-by-power-of-range under autodiff; likewise sigma is
// evaluated via exp(log_sigma) instead of 1/(sqrt(4 pi) kappa tau).
//
// Returns T(0) when joint_hypers == false, when the layout has no hyper
// slots, or when any of the four PC anchors is non-positive (improper
// flat hyper-prior — floor-check / gradient-verification only).
template<typename T>
T compute_spde_hyper_prior(const std::vector<T>& params,
                            const ModelData& data,
                            const ParamLayout& layout)
{
    if (!layout.is_spde || !data.has_spde) return T(0.0);
    if (!data.spde_data.joint_hypers)       return T(0.0);
    if (layout.log_kappa_spde_idx < 0)      return T(0.0);

    const auto& spde = data.spde_data;
    if (spde.prior_range_0     <= 0.0 || spde.prior_range_alpha <= 0.0 ||
        spde.prior_range_alpha >= 1.0 ||
        spde.prior_sigma_0     <= 0.0 || spde.prior_sigma_alpha <= 0.0 ||
        spde.prior_sigma_alpha >= 1.0) {
        return T(0.0);
    }

    // Compile-time-stable constants (no <cmath> M_PI dependency on MSVC).
    constexpr double k_pi = 3.14159265358979323846;

    const double nu        = spde.nu;
    const double eight_nu  = 8.0 * nu;
    const double log_8nu   = std::log(eight_nu);
    const double log_4pi   = std::log(4.0 * k_pi);

    // d = 2 range PC prior (P(range < range_0) = alpha): lambda_r =
    // -log(alpha) * range_0, density lambda_r * range^{-2} * exp(-lambda_r/range).
    // The earlier code used the d = 1 form (sqrt(range_0), range^{-3/2}); it kept
    // the tail anchor but had the wrong penalization shape and disagreed with the
    // dimensionally-correct R nested path (fit_spde_nested).
    const double lambda_r  = -std::log(spde.prior_range_alpha)
                             * spde.prior_range_0;
    const double lambda_s  = -std::log(spde.prior_sigma_alpha)
                             / spde.prior_sigma_0;

    const T log_kappa = params[layout.log_kappa_spde_idx];
    const T log_tau   = params[layout.log_tau_spde_idx];

    const T log_range = T(0.5 * log_8nu) - log_kappa;
    const T log_sigma = T(-0.5 * log_4pi) - log_kappa - log_tau;

    // log pi(range) = log(lambda_r) - 2 * log_range - lambda_r / range  (d = 2)
    const T log_pi_range =
          T(std::log(lambda_r))
        - T(2.0) * log_range
        - T(lambda_r) * safe_exp(-log_range);

    // log pi(sigma) = log(lambda_s) - lambda_s * exp(log_sigma)
    const T log_pi_sigma =
          T(std::log(lambda_s))
        - T(lambda_s) * safe_exp(log_sigma);

    // Joint density in (log_kappa, log_tau) = PC densities + log|J|.
    return log_pi_range + log_pi_sigma + log_range + log_sigma;
}

} // namespace priors
} // namespace tulpa

#endif // TULPA_PRIORS_SPDE_H
