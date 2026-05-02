// hmc_icar_collapsed_log_post.h
// High-level log_post integration wrappers + sample storage.
// Self-contained: defines symbols inside namespace tulpa_hmc.

#ifndef TULPA_HMC_ICAR_COLLAPSED_LOG_POST_H
#define TULPA_HMC_ICAR_COLLAPSED_LOG_POST_H

#include <cmath>
#include <cstring>
#include <vector>

#include "hmc_icar_collapsed_kernels.h"
#include "hmc_icar_collapsed_mode.h"
#include "hmc_icar_collapsed_workspace.h"
#include "hmc_sampler.h"

namespace tulpa_hmc {

// =========================================================================
// High-level wrappers for compute_log_post integration
// These encapsulate all collapsed ICAR/BYM2 logic so the main log_post
// function only needs a single call instead of 80+ lines of inline code.
// =========================================================================

struct CollapsedICARLogPostResult {
    double log_post_contribution = 0.0;  // Prior + Laplace correction
    const double* phi_spatial = nullptr;  // Points into ws.phi_star
    const double* theta_bym2 = nullptr;   // Points into ws.theta_star (BYM2 only)
};

// Compute the collapsed ICAR/BYM2 contribution to log_post:
//   1. Find phi* (and theta* for BYM2) via Newton
//   2. Evaluate ICAR/BYM2 prior at mode
//   3. Add Laplace correction
//   4. Return pointers to mode values for use in observation loop
//
// re_vals: actual RE values (after NC transform), or nullptr if no RE.
// Caller adds result.log_post_contribution to log_post and uses
// result.phi_spatial / result.theta_bym2 in the observation loop.
inline CollapsedICARLogPostResult collapsed_icar_log_post_contribution(
    bool is_bym2,
    double tau_spatial,        // exp(log_tau) for ICAR
    double sigma_total,        // exp(log_sigma_bym2) for BYM2
    double logit_rho,          // logit_rho for BYM2
    double bym2_scale_factor,  // scale factor for BYM2
    double phi_num, double phi_denom,
    const double* beta_num, const double* beta_denom,
    const double* re_vals,     // nullptr if no RE
    const ModelData& data,
    CollapsedICARWorkspace& ws) {

    CollapsedICARLogPostResult res;
    int S = data.n_spatial_units;

    if (is_bym2) {
        double rho = 1.0 / (1.0 + std::exp(-logit_rho));

        collapsed_bym2_find_mode(
            beta_num, beta_denom, sigma_total, rho, bym2_scale_factor,
            phi_num, phi_denom, re_vals, data, ws);

        res.phi_spatial = ws.phi_star.data();
        res.theta_bym2 = ws.theta_star.data();

        // ICAR prior on phi*: -0.5 * phi*^T Q phi* - 0.5 * lambda * (sum phi*)^2
        std::vector<double> Qphi(S);
        icar_precision_matvec(ws.phi_star.data(), Qphi.data(), S,
                              data.adj_row_ptr, data.adj_col_idx, data.n_neighbors);
        double phiQphi = 0.0, sum_phi = 0.0;
        for (int i = 0; i < S; i++) {
            phiQphi += ws.phi_star[i] * Qphi[i];
            sum_phi += ws.phi_star[i];
        }
        res.log_post_contribution += -0.5 * phiQphi - 0.5 * 0.001 * sum_phi * sum_phi;

        // N(0,1) prior on theta*
        for (int i = 0; i < S; i++) {
            res.log_post_contribution -= 0.5 * ws.theta_star[i] * ws.theta_star[i];
        }
    } else {
        // Collapsed ICAR
        collapsed_icar_find_mode(
            beta_num, beta_denom, tau_spatial, phi_num, phi_denom,
            re_vals, data, ws);

        res.phi_spatial = ws.phi_star.data();

        // ICAR prior at phi*: 0.5*(J-1)*log(tau) - 0.5*tau*phi*^T Q phi* - lambda*(sum phi*)^2
        std::vector<double> Qphi(S);
        icar_precision_matvec(ws.phi_star.data(), Qphi.data(), S,
                              data.adj_row_ptr, data.adj_col_idx, data.n_neighbors);
        double phiQphi = 0.0, sum_phi = 0.0;
        for (int i = 0; i < S; i++) {
            phiQphi += ws.phi_star[i] * Qphi[i];
            sum_phi += ws.phi_star[i];
        }
        res.log_post_contribution += -0.5 * tau_spatial * phiQphi
                                   + 0.5 * (S - 1) * std::log(tau_spatial)
                                   - 0.5 * 0.001 * sum_phi * sum_phi;
    }

    // Laplace correction: -0.5 * log det(H)
    res.log_post_contribution += ws.laplace_log_det;

    return res;
}

// Store collapsed ICAR/BYM2 mode values (phi*, theta*) into result buffer
// Called once per sampling iteration after accept/reject
inline void collapsed_icar_store_sample(
    int sample_idx,
    const ModelData& data,
    const CollapsedICARWorkspace& ws,
    std::vector<double>& icar_phi_star_flat,
    std::vector<double>& bym2_theta_star_flat,
    int S) {
    std::memcpy(&icar_phi_star_flat[sample_idx * S],
                ws.phi_star.data(), S * sizeof(double));
    if (data.bym2_collapsed && !bym2_theta_star_flat.empty()) {
        std::memcpy(&bym2_theta_star_flat[sample_idx * S],
                    ws.theta_star.data(), S * sizeof(double));
    }
}

}  // namespace tulpa_hmc

#endif // TULPA_HMC_ICAR_COLLAPSED_LOG_POST_H
