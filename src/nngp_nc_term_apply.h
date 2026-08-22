// nngp_nc_term_apply.h
// Shared low-level primitive behind every non-centered NNGP consumer
// (gp_nc_apply.cpp: one field; svc_nc_apply.cpp: one call per SVC term;
// msgp_nc_apply.cpp: one call per scale). A "term" is one NNGP field
// z ~ N(0, I) -> w = f(z, sigma2, phi) over a shared NNGPNCView topology; the
// custom_backward wiring (dw -> dz, dlog_sigma2, dlog_phi via the hand-derived
// tulpa_gp::nngp_nc_forward / nngp_nc_backward pair) is identical regardless
// of which model package's field the term belongs to, so it lives here once
// rather than once per caller.
//
//   double      : forward only. No AD, no adjoint recording.
//   arena::Var  : records a custom_backward block into the arena so
//                 reverse-mode AD threads (dw, dlog_sigma2, dlog_phi) back to
//                 (dz, dlog_sigma2, dlog_phi) on the backward sweep.
//
// No z -> w Jacobian is added (pure non-centered reparameterization); see
// gp_nc_apply.h for the reasoning, which applies identically here.

#ifndef TULPA_NNGP_NC_TERM_APPLY_H
#define TULPA_NNGP_NC_TERM_APPLY_H

#include <vector>

#include "tulpa/autodiff_arena.h"

namespace tulpa_gp {
struct NNGPNCView;
} // namespace tulpa_gp

namespace tulpa {

// Value-only: fills w_out (length N) with w = f(params[w_start..w_start+N),
// sigma2, phi) reconstructed via the NNGP autoregressive transform over
// `view`. sigma2 = exp(params[sigma2_idx]), phi = exp(params[phi_idx]).
void apply_nngp_nc_term_double(
    const std::vector<double>& params,
    int w_start, int sigma2_idx, int phi_idx, int N,
    const tulpa_gp::NNGPNCView& view,
    std::vector<double>&       w_out);

// Reverse-mode: same reconstruction as arena outputs, with one
// custom_backward block registered (inputs [z[0..N-1], log_sigma2, log_phi])
// so the term's likelihood gradient flows back to those params. The N(0, I)
// prior on z is NOT added here -- the caller's templated prior function adds
// it in T -- this only wires the likelihood-through-w gradient.
std::vector<arena::Var> apply_nngp_nc_term_arena(
    const std::vector<arena::Var>& params,
    int w_start, int sigma2_idx, int phi_idx, int N,
    const tulpa_gp::NNGPNCView& view);

} // namespace tulpa

#endif // TULPA_NNGP_NC_TERM_APPLY_H
