// gp_nc_apply.cpp
// Implementation of the non-centered NNGP transform application. See
// gp_nc_apply.h for the contract. Both flavours delegate to the shared
// per-term applier (nngp_nc_term_apply.h) -- GP is a single term over the
// GPData-native NNGPNCView.

#include "gp_nc_apply.h"

#include <vector>

#include "hmc_gp.h"              // tulpa_gp::{NNGPNCView, make_gp_nc_view}
#include "nngp_nc_term_apply.h"  // apply_nngp_nc_term_{double,arena}
#include "tulpa/model_data.h"
#include "tulpa/param_layout.h"

namespace tulpa {

void apply_gp_nc_transform_double(
    const std::vector<double>& params,
    const ModelData&           data,
    const ParamLayout&         layout,
    std::vector<double>&       gp_w_out)
{
    const int N = data.gp_data.n_obs;
    apply_nngp_nc_term_double(
        params, layout.gp_w_start, layout.log_sigma2_gp_idx, layout.log_phi_gp_idx,
        N, tulpa_gp::make_gp_nc_view(data.gp_data), gp_w_out);
}

void apply_gp_nc_transform_arena(
    const std::vector<arena::Var>& params,
    const ModelData&               data,
    const ParamLayout&             layout,
    std::vector<arena::Var>&       gp_w_out)
{
    const int N = data.gp_data.n_obs;
    gp_w_out = apply_nngp_nc_term_arena(
        params, layout.gp_w_start, layout.log_sigma2_gp_idx, layout.log_phi_gp_idx,
        N, tulpa_gp::make_gp_nc_view(data.gp_data));
}

} // namespace tulpa
