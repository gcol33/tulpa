// msgp_nc_apply.cpp
// Implementation of the non-centered NNGP transform application for the
// multi-scale (local + regional) GP field. See msgp_nc_apply.h for the
// contract. Each scale delegates to the shared per-term applier
// (nngp_nc_term_apply.h), the same primitive gp_nc_apply.cpp and
// svc_nc_apply.cpp use.

#include "msgp_nc_apply.h"

#include <vector>

#include "hmc_gp.h"              // tulpa_gp::{NNGPNCView, make_msgp_nc_view_local,
                                 // make_msgp_nc_view_regional}
#include "nngp_nc_term_apply.h"  // apply_nngp_nc_term_{double,arena}
#include "tulpa/model_data.h"
#include "tulpa/param_layout.h"

namespace tulpa {

void apply_msgp_nc_transform_double(
    const std::vector<double>& params,
    const ModelData&           data,
    const ParamLayout&         layout,
    std::vector<double>&       ms_gp_effect_out)
{
    const auto& ms = data.multiscale_gp_data;
    const int n_loc = ms.n_obs;

    std::vector<double> w_local, w_regional;
    apply_nngp_nc_term_double(
        params, layout.gp_local_start, layout.log_sigma2_gp_local_idx,
        layout.log_phi_gp_local_idx, n_loc,
        tulpa_gp::make_msgp_nc_view_local(ms), w_local);
    apply_nngp_nc_term_double(
        params, layout.gp_regional_start, layout.log_sigma2_gp_regional_idx,
        layout.log_phi_gp_regional_idx, n_loc,
        tulpa_gp::make_msgp_nc_view_regional(ms), w_regional);

    ms_gp_effect_out.resize(data.N);
    for (int ii = 0; ii < data.N; ii++) {
        int loc = ms.obs_to_loc[ii];
        ms_gp_effect_out[ii] = w_local[loc] + w_regional[loc];
    }
}

void apply_msgp_nc_transform_arena(
    const std::vector<arena::Var>& params,
    const ModelData&               data,
    const ParamLayout&             layout,
    std::vector<arena::Var>&       ms_gp_effect_out)
{
    const auto& ms = data.multiscale_gp_data;
    const int n_loc = ms.n_obs;

    std::vector<arena::Var> w_local = apply_nngp_nc_term_arena(
        params, layout.gp_local_start, layout.log_sigma2_gp_local_idx,
        layout.log_phi_gp_local_idx, n_loc, tulpa_gp::make_msgp_nc_view_local(ms));
    std::vector<arena::Var> w_regional = apply_nngp_nc_term_arena(
        params, layout.gp_regional_start, layout.log_sigma2_gp_regional_idx,
        layout.log_phi_gp_regional_idx, n_loc, tulpa_gp::make_msgp_nc_view_regional(ms));

    // Combine the two scales at observation level with plain arena
    // arithmetic -- ordinary addition of two already-differentiable Vars, so
    // no custom_backward is needed for this step; the arena's normal
    // reverse-mode handles it.
    ms_gp_effect_out.resize(data.N);
    for (int ii = 0; ii < data.N; ii++) {
        int loc = ms.obs_to_loc[ii];
        ms_gp_effect_out[ii] = w_local[loc] + w_regional[loc];
    }
}

} // namespace tulpa
