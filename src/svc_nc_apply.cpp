// svc_nc_apply.cpp
// Implementation of the non-centered NNGP transform application for SVC
// terms. See svc_nc_apply.h for the contract. Every term shares the SVCData
// neighbour topology and gets its own (sigma2_j, phi_j); each term delegates
// to the shared per-term applier (nngp_nc_term_apply.h), the same primitive
// gp_nc_apply.cpp uses for its single field.

#include "svc_nc_apply.h"

#include <vector>

#include "hmc_gp.h"              // tulpa_gp::{NNGPNCView, make_svc_nc_view};
                                 // pulls in hmc_svc.h for tulpa::SVCData.
#include "nngp_nc_term_apply.h"  // apply_nngp_nc_term_{double,arena}
#include "tulpa/model_data.h"
#include "tulpa/param_layout.h"

namespace tulpa {

void apply_svc_nc_transform_double(
    const std::vector<double>& params,
    const ModelData&           data,
    const ParamLayout&         layout,
    std::vector<double>&       svc_w_out)
{
    const int N     = data.svc_data.n_obs;
    const int n_svc = data.svc_data.n_svc;
    const int w0    = layout.svc_w_start;

    const tulpa_gp::NNGPNCView view = tulpa_gp::make_svc_nc_view(data.svc_data);

    svc_w_out.resize((std::size_t)n_svc * N);
    for (int j = 0; j < n_svc; j++) {
        std::vector<double> w_j;
        apply_nngp_nc_term_double(
            params, w0 + j * N, layout.log_sigma2_svc_start + j,
            layout.log_phi_svc_start + j, N, view, w_j);
        for (int i = 0; i < N; i++) svc_w_out[(std::size_t)j * N + i] = w_j[i];
    }
}

void apply_svc_nc_transform_arena(
    const std::vector<arena::Var>& params,
    const ModelData&               data,
    const ParamLayout&             layout,
    std::vector<arena::Var>&       svc_w_out)
{
    const int N     = data.svc_data.n_obs;
    const int n_svc = data.svc_data.n_svc;
    const int w0    = layout.svc_w_start;

    const tulpa_gp::NNGPNCView view = tulpa_gp::make_svc_nc_view(data.svc_data);

    svc_w_out.resize((std::size_t)n_svc * N);
    for (int j = 0; j < n_svc; j++) {
        std::vector<arena::Var> w_j = apply_nngp_nc_term_arena(
            params, w0 + j * N, layout.log_sigma2_svc_start + j,
            layout.log_phi_svc_start + j, N, view);
        for (int i = 0; i < N; i++) svc_w_out[(std::size_t)j * N + i] = w_j[i];
    }
}

} // namespace tulpa
