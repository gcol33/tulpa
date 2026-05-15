// spde_nc_apply.h
// Apply the non-centered SPDE transform z -> w = L^{-T}(theta) z inside the
// log-post evaluation. Two flavours, dispatched at the call site:
//
//   double      : forward only. No AD, no adjoint recording. Used by the
//                 plain log-post evaluator and by the central-difference
//                 gradient.
//   arena::Var  : records the custom_backward block from (a.i) into the
//                 arena so reverse-mode AD threads (dw, dlog_kappa,
//                 dlog_tau) back to (dz, dlog_kappa, dlog_tau) on the
//                 backward sweep.
//
// Both flavours lazily build data.spde_data.nc_transform on first call,
// initialising it from the SPDE FEM matrices (C0_diag / G1) stored on
// SpdeModelData. The Eigen-format transform persists for the lifetime of
// the ModelData, so the FEM sparsity pattern is built once per chain.
//
// Separate header (not folded into spde_nc_transform.h) because the
// callers — log_post_generic_impl.h and the gradient backends — should
// not pull Eigen into their translation units.

#ifndef TULPA_SPDE_NC_APPLY_H
#define TULPA_SPDE_NC_APPLY_H

#include <vector>

#include "tulpa/autodiff_arena.h"

namespace tulpa {

// Forward decls — apply functions take these by const&, so no complete
// types are needed in this header.
struct ModelData;
struct ParamLayout;

void apply_spde_nc_transform_double(
    const std::vector<double>& params,
    const ModelData&           data,
    const ParamLayout&         layout,
    std::vector<double>&       spde_w_out);

void apply_spde_nc_transform_arena(
    const std::vector<arena::Var>& params,
    const ModelData&               data,
    const ParamLayout&             layout,
    std::vector<arena::Var>&       spde_w_out);

} // namespace tulpa

#endif // TULPA_SPDE_NC_APPLY_H
