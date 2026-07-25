// gp_nc_apply.h
// Apply the non-centered NNGP transform z -> w inside the log-post
// evaluation for a single-scale GP spatial field. Two flavours, dispatched
// at the call site (mirrors spde_nc_apply.h):
//
//   double      : forward only. No AD, no adjoint recording. Used by the
//                 plain log-post evaluator and by the central-difference
//                 gradient.
//   arena::Var  : records a custom_backward block into the arena so
//                 reverse-mode AD threads (dw, dlog_sigma2, dlog_phi) back to
//                 (dz, dlog_sigma2, dlog_phi) on the backward sweep, using the
//                 hand-derived nngp_nc_forward / nngp_nc_backward pair.
//
// The forward map w = f(z, sigma2, phi) is the NNGP autoregressive whitening
// inverse: with z ~ N(0, I) the field w has the marginal NNGP(0, sigma2, phi)
// distribution. The N(0, I) prior on z and the hyperparameter priors are
// added in compute_gp_spatial_prior (templated); this file only fills gp_w
// with the reconstructed field and wires the field's likelihood gradient
// back onto (z, log_sigma2, log_phi). No z -> w Jacobian enters the target
// (pure non-centered reparameterization; the temporal GP path documents the
// same reasoning).
//
// Separate header (not folded into hmc_gp_nc.h) because the callers --
// log_post_generic_impl.h and the gradient backends -- should not pull the
// GP kernel / Eigen chain into their translation units; only gp_nc_apply.cpp
// does.

#ifndef TULPA_GP_NC_APPLY_H
#define TULPA_GP_NC_APPLY_H

#include <vector>

#include "tulpa/autodiff_arena.h"

namespace tulpa {

// Forward decls -- apply functions take these by const&, so no complete
// types are needed in this header.
struct ModelData;
struct ParamLayout;

// Centered path: fills gp_w_out with w = f(z, sigma2, phi) reconstructed from
// the z block at params[layout.gp_w_start .. gp_w_end). Value-only; the
// central-difference gradient re-evaluates this per perturbed parameter.
void apply_gp_nc_transform_double(
    const std::vector<double>& params,
    const ModelData&           data,
    const ParamLayout&         layout,
    std::vector<double>&       gp_w_out);

// Reverse-mode: fills gp_w_out with w = f(z, sigma2, phi) as arena outputs and
// registers a custom_backward block with inputs [z[0..N-1], log_sigma2,
// log_phi] so the field's likelihood gradient flows back to those params. The
// N(0, I) prior on z is NOT added here (compute_gp_spatial_prior adds it in
// templated T); the backward injects only the likelihood-through-w gradient.
void apply_gp_nc_transform_arena(
    const std::vector<arena::Var>& params,
    const ModelData&               data,
    const ParamLayout&             layout,
    std::vector<arena::Var>&       gp_w_out);

} // namespace tulpa

#endif // TULPA_GP_NC_APPLY_H
