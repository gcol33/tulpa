// svc_nc_apply.h
// Apply the non-centered NNGP transform z -> w inside the log-post
// evaluation for the NNGP spatially-varying-coefficient (SVC) terms. Mirrors
// gp_nc_apply.h's single-field contract, run once PER SVC TERM: every term
// shares the SVCData neighbour topology and gets its own (sigma2_j, phi_j).
//
//   double      : forward only. No AD, no adjoint recording. Used by the
//                 plain log-post evaluator and by the central-difference
//                 gradient.
//   arena::Var  : records one custom_backward block per term into the arena
//                 so reverse-mode AD threads (dw_j, dlog_sigma2_j, dlog_phi_j)
//                 back to (dz_j, dlog_sigma2_j, dlog_phi_j) on the backward
//                 sweep, using the shared hand-derived nngp_nc_forward /
//                 nngp_nc_backward pair (hmc_gp_nc.h).
//
// The forward map w_j = f(z_j, sigma2_j, phi_j) is the same NNGP
// autoregressive whitening inverse gp_nc_apply.cpp uses; see that file's
// header comment for the no-Jacobian reasoning. The N(0,I)
// prior on each z_j and the hyperparameter priors are added in
// compute_svc_prior (templated); this file only fills svc_w_out with the
// reconstructed terms and wires each term's likelihood gradient back onto
// (z_j, log_sigma2_svc_j, log_phi_svc_j).
//
// Separate header (not folded into hmc_svc_autodiff.h) for the same reason
// gp_nc_apply.h is separate from hmc_gp_autodiff.h: callers should not pull
// the NNGP transform / Eigen chain into their translation units; only
// svc_nc_apply.cpp does.

#ifndef TULPA_SVC_NC_APPLY_H
#define TULPA_SVC_NC_APPLY_H

#include <vector>

#include "tulpa/autodiff_arena.h"

namespace tulpa {

// Forward decls -- apply functions take these by const&, so no complete
// types are needed in this header.
struct ModelData;
struct ParamLayout;

// Value-only: fills svc_w_out (flat [n_svc x n_obs], term-major --
// w_flat[j * n_obs + i], matching compute_svc_eta's layout) with
// w_j = f(z_j, sigma2_j, phi_j) reconstructed from the z block at
// params[layout.svc_w_start .. svc_w_end). Value-only; the central-difference
// gradient re-evaluates this per perturbed parameter.
void apply_svc_nc_transform_double(
    const std::vector<double>& params,
    const ModelData&           data,
    const ParamLayout&         layout,
    std::vector<double>&       svc_w_out);

// Reverse-mode: fills svc_w_out with w_j = f(z_j, sigma2_j, phi_j) as arena
// outputs, one custom_backward block per term with inputs
// [z_j[0..n_obs-1], log_sigma2_svc_j, log_phi_svc_j] so each term's
// likelihood gradient flows back to its own z_j / hyperparameters. The N(0,I)
// prior on each z_j is NOT added here (compute_svc_prior adds it in templated
// T); the backward injects only the likelihood-through-w gradient.
void apply_svc_nc_transform_arena(
    const std::vector<arena::Var>& params,
    const ModelData&               data,
    const ParamLayout&             layout,
    std::vector<arena::Var>&       svc_w_out);

} // namespace tulpa

#endif // TULPA_SVC_NC_APPLY_H
