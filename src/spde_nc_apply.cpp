// spde_nc_apply.cpp
// Implementation of the joint-NUTS NC transform application. See
// spde_nc_apply.h for the contract.

#include "spde_nc_apply.h"

#include <Eigen/Core>
#include <cmath>
#include <utility>

#include "spde_nc_transform.h"
#include "tulpa/model_data.h"
#include "tulpa/param_layout.h"

namespace tulpa {

namespace {

// Lazily build (or fetch) the cached SpdeNcTransform for this ModelData.
// The transform owns the Eigen-format FEM matrices and the running
// Cholesky factor; we want it built once per chain and reused across
// every log-post evaluation. spde_model_data.h declares `nc_transform`
// mutable so const ModelData references can populate it here.
inline SpdeNcTransform& ensure_transform(const ModelData& data) {
    auto& sm = data.spde_data;
    if (!sm.nc_transform) {
        auto t = std::make_shared<SpdeNcTransform>();
        t->init(sm.n_mesh, sm.C0_diag, sm.G1_x, sm.G1_i, sm.G1_p);
        sm.nc_transform = std::move(t);
    }
    return *sm.nc_transform;
}

} // namespace

void apply_spde_nc_transform_double(
    const std::vector<double>& params,
    const ModelData&           data,
    const ParamLayout&         layout,
    std::vector<double>&       spde_w_out)
{
    const int n_mesh = data.spde_data.n_mesh;
    const int w0     = layout.spde_w_start;

    SpdeNcTransform& tx = ensure_transform(data);

    Eigen::VectorXd z(n_mesh);
    for (int j = 0; j < n_mesh; j++) z[j] = params[w0 + j];

    const double log_kappa = params[layout.log_kappa_spde_idx];
    const double log_tau   = params[layout.log_tau_spde_idx];
    const double kappa     = std::exp(log_kappa);
    const double tau       = std::exp(log_tau);

    Eigen::VectorXd w = tx.forward(z, kappa, tau);

    spde_w_out.resize(n_mesh);
    for (int j = 0; j < n_mesh; j++) spde_w_out[j] = w[j];
}

void apply_spde_nc_transform_arena(
    const std::vector<arena::Var>& params,
    const ModelData&               data,
    const ParamLayout&             layout,
    std::vector<arena::Var>&       spde_w_out)
{
    const int n_mesh = data.spde_data.n_mesh;
    const int w0     = layout.spde_w_start;

    SpdeNcTransform& tx = ensure_transform(data);

    // params[w0..w0+n_mesh) is the z block. Copy out into a local vector
    // so spde_nc_transform_arena can take a contiguous std::vector<Var>.
    std::vector<arena::Var> z(params.begin() + w0,
                              params.begin() + w0 + n_mesh);

    // The arena is encoded on every Var; params is guaranteed non-empty
    // here because the SPDE block contributes at least one z slot.
    arena::Arena* ar = params[w0].arena_;

    spde_w_out = spde_nc_transform_arena(
        ar, z,
        params[layout.log_kappa_spde_idx],
        params[layout.log_tau_spde_idx],
        tx);
}

} // namespace tulpa
