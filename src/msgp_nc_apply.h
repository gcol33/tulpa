// msgp_nc_apply.h
// Apply the non-centered NNGP transform z -> w inside the log-post
// evaluation for the multi-scale GP (local + regional) field. Mirrors
// gp_nc_apply.h's single-field contract, run once per SCALE: each scale has
// its own (sigma2, phi) and its own NNGP neighbour topology
// (MultiscaleGPData caches nn_neighbor_dist for both, same fast path as
// GPData), then the two reconstructed fields are combined to the
// per-observation effect the same way the centered path already does --
// ms_gp_effect[i] = w_local[obs_to_loc[i]] + w_regional[obs_to_loc[i]].
//
//   double      : forward only. No AD, no adjoint recording.
//   arena::Var  : one custom_backward block per scale (via the shared
//                 nngp_nc_term_apply.h primitive), then a plain arena
//                 addition to combine the two scales at observation level --
//                 that combination is ordinary linear arithmetic, not a
//                 custom kernel, so it needs no backward of its own.
//
// The N(0,I) prior on each scale's z and the hyperparameter priors are added
// in compute_multiscale_gp_prior (templated); this file only fills
// ms_gp_effect_out with the combined, reconstructed field. No z -> w Jacobian
// enters the target, for the same reason gp_nc_apply.h documents.

#ifndef TULPA_MSGP_NC_APPLY_H
#define TULPA_MSGP_NC_APPLY_H

#include <vector>

#include "tulpa/autodiff_arena.h"

namespace tulpa {

struct ModelData;
struct ParamLayout;

// Value-only: fills ms_gp_effect_out (length data.N, one combined
// value per observation) from the z blocks at
// params[layout.gp_local_start .. gp_local_end) and
// params[layout.gp_regional_start .. gp_regional_end). Value-only.
void apply_msgp_nc_transform_double(
    const std::vector<double>& params,
    const ModelData&           data,
    const ParamLayout&         layout,
    std::vector<double>&       ms_gp_effect_out);

// Reverse-mode: same reconstruction as arena outputs (length data.N).
void apply_msgp_nc_transform_arena(
    const std::vector<arena::Var>& params,
    const ModelData&               data,
    const ParamLayout&             layout,
    std::vector<arena::Var>&       ms_gp_effect_out);

} // namespace tulpa

#endif // TULPA_MSGP_NC_APPLY_H
