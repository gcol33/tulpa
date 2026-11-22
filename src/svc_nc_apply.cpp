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
#include "svc_range.h"
#include "nngp_nc_term_apply.h"  // apply_nngp_nc_term_{double,arena}
#include "tulpa/model_data.h"
#include "tulpa/param_layout.h"

namespace tulpa {

namespace {

// The per-term applier for each scalar type: value-only for double, one
// custom_backward block per term for arena::Var.
inline void svc_nc_term_into(
    const std::vector<double>& native, int w_start, int sigma2_idx,
    int phi_idx, int N, const tulpa_gp::NNGPNCView& view,
    std::vector<double>& w_out)
{
    apply_nngp_nc_term_double(native, w_start, sigma2_idx, phi_idx, N, view,
                              w_out);
}

inline void svc_nc_term_into(
    const std::vector<arena::Var>& native, int w_start, int sigma2_idx,
    int phi_idx, int N, const tulpa_gp::NNGPNCView& view,
    std::vector<arena::Var>& w_out)
{
    w_out = apply_nngp_nc_term_arena(native, w_start, sigma2_idx, phi_idx, N,
                                     view);
}

template <class T>
void apply_svc_nc_transform_impl(
    const std::vector<T>& params,
    const ModelData&      data,
    const ParamLayout&    layout,
    std::vector<T>&       svc_w_out)
{
    const auto native = svc_nc_native_coordinates(params, data, layout);
    const int N     = data.svc_data.n_obs;
    const int n_svc = data.svc_data.n_svc;
    const int w0    = layout.svc_w_start;

    const tulpa_gp::NNGPNCView view = tulpa_gp::make_svc_nc_view(data.svc_data);

    svc_w_out.resize((std::size_t)n_svc * N);
    for (int j = 0; j < n_svc; j++) {
        std::vector<T> w_j;
        svc_nc_term_into(
            native, w0 + j * N, layout.log_sigma2_svc_start + j,
            layout.log_phi_svc_start + j, N, view, w_j);
        for (int i = 0; i < N; i++) svc_w_out[(std::size_t)j * N + i] = w_j[i];
    }
}

} // namespace

void apply_svc_nc_transform_double(
    const std::vector<double>& params,
    const ModelData&           data,
    const ParamLayout&         layout,
    std::vector<double>&       svc_w_out)
{
    apply_svc_nc_transform_impl(params, data, layout, svc_w_out);
}

void apply_svc_nc_transform_arena(
    const std::vector<arena::Var>& params,
    const ModelData&               data,
    const ParamLayout&             layout,
    std::vector<arena::Var>&       svc_w_out)
{
    apply_svc_nc_transform_impl(params, data, layout, svc_w_out);
}

} // namespace tulpa
