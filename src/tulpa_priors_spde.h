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
#include "pc_prior.h"
#include "spde_matern_map.h"

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
        // Non-centered (joint-NUTS or fixed-hyper): params[w0..) is z.
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
// The two PC densities themselves (Fuglstad et al. 2019, JASA, "Constructing
// priors that penalize the complexity of Gaussian random fields") live in
// pc_prior.h; this function contributes the SPDE-specific coordinate map and
// its Jacobian.
//
// SPDE Matern map (nu = data.spde_data.nu, d = 2), spde_matern_map.h:
//   log_range = 0.5 * log(8 nu) - log_kappa
//   log_sigma = -0.5 * log(4 pi nu) - nu * log_kappa - log_tau
//
// Jacobian of (log_kappa, log_tau) -> (range, sigma):
//   d range / d log_kappa = -range,     d range / d log_tau = 0
//   d sigma / d log_kappa = -nu*sigma,  d sigma / d log_tau = -sigma
// J is triangular, so |det J| = range * sigma for any nu and log|J| =
// log_range + log_sigma.
//
// Joint hyper-density in (log_kappa, log_tau):
//   log p(log_kappa, log_tau) = log pi(range) + log pi(sigma)
//                             + log_range + log_sigma
//
// The range density is evaluated from log_range, so no power-of-range division
// is taped under autodiff; the sigma density is linear in sigma and takes it on
// the natural scale.
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
    if (layout.log_tau_spde_idx < 0)        return T(0.0);

    // Anchors the PC calibration cannot represent fall back to a flat
    // hyper-prior rather than emitting a -Inf or NaN inside a gradient; this is
    // the one predicate every PC anchor in the package is read through
    // (pc_prior.h), and the R and sampler doors that can name the offending
    // argument check it before the fit starts.
    const auto& spde = data.spde_data;
    if (!pc_anchors_valid(spde.prior_range_0, spde.prior_range_alpha) ||
        !pc_anchors_valid(spde.prior_sigma_0, spde.prior_sigma_alpha)) {
        return T(0.0);
    }

    const double nu = spde.nu;

    const T log_kappa = params[layout.log_kappa_spde_idx];
    const T log_tau   = params[layout.log_tau_spde_idx];

    T log_range, log_sigma;
    spde_log_kappa_tau_to_log_range_sigma(log_kappa, log_tau, nu,
                                          log_range, log_sigma);

    const T log_pi_range = log_prior_range_pc_at_log(
        log_range, spde.prior_range_0, spde.prior_range_alpha);

    const T log_pi_sigma = log_prior_sigma_pc(
        safe_exp(log_sigma), spde.prior_sigma_0, spde.prior_sigma_alpha);

    // Joint density in (log_kappa, log_tau) = PC densities + log|J|.
    return log_pi_range + log_pi_sigma + log_range + log_sigma;
}

} // namespace priors
} // namespace tulpa

#endif // TULPA_PRIORS_SPDE_H
