// spde_nc_apply.cpp
// Implementation of the joint-NUTS NC transform application. See
// spde_nc_apply.h for the contract.

#include "spde_nc_apply.h"

#include <Eigen/Core>
#include <cmath>
#include <limits>
#include <stdexcept>
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
        // Pass rational poles/weights when present (fractional nu); empty
        // vectors take the integer alpha=2 fast path. The cache lives for
        // the chain's lifetime, so this dispatch is paid once.
        t->init(sm.n_mesh, sm.C0_diag, sm.G1_x, sm.G1_i, sm.G1_p,
                sm.rational_poles, sm.rational_weights);
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

    spde_w_out.resize(n_mesh);
    try {
        Eigen::VectorXd w = tx.forward(z, kappa, tau);
        for (int j = 0; j < n_mesh; j++) spde_w_out[j] = w[j];
    } catch (const std::runtime_error&) {
        // Q(kappa, tau) failed to Cholesky — log_kappa / log_tau wandered
        // somewhere numerically degenerate (e.g. extreme values during
        // warmup adaptation). Fill w with NaN so the downstream eta
        // accumulator propagates NaN to log_post; NUTS treats NaN as
        // -Inf and rejects the trajectory cleanly.
        const double nan_v = std::numeric_limits<double>::quiet_NaN();
        for (int j = 0; j < n_mesh; j++) spde_w_out[j] = nan_v;
    }
}

void apply_spde_nc_transform_fwd(
    const std::vector<::fwd::Dual>& params,
    const ModelData&                data,
    const ParamLayout&              layout,
    std::vector<::fwd::Dual>&       spde_w_out)
{
    const int n_mesh = data.spde_data.n_mesh;
    const int w0     = layout.spde_w_start;

    SpdeNcTransform& tx = ensure_transform(data);

    Eigen::VectorXd z(n_mesh), dz(n_mesh);
    for (int j = 0; j < n_mesh; j++) {
        z [j] = params[w0 + j].val;
        dz[j] = params[w0 + j].grad;
    }

    const double log_kappa  = params[layout.log_kappa_spde_idx].val;
    const double log_tau    = params[layout.log_tau_spde_idx ].val;
    const double dlog_kappa = params[layout.log_kappa_spde_idx].grad;
    const double dlog_tau   = params[layout.log_tau_spde_idx ].grad;
    const double kappa      = std::exp(log_kappa);
    const double tau        = std::exp(log_tau);

    spde_w_out.resize(n_mesh);

    try {
        Eigen::VectorXd w, dw;
        tx.forward_with_tangent(z, dz, kappa, dlog_kappa, tau, dlog_tau,
                                w, dw);
        for (int j = 0; j < n_mesh; j++) {
            spde_w_out[j] = ::fwd::Dual(w[j], dw[j]);
        }
    } catch (const std::runtime_error&) {
        // Cholesky failure during the forward (Q not PD at extreme hypers
        // during warmup adaptation). Emit NaN dual for both .val and .grad
        // so the downstream eta accumulator propagates NaN to log_post;
        // NUTS treats NaN as -Inf and rejects the trajectory cleanly.
        const double nan_v = std::numeric_limits<double>::quiet_NaN();
        for (int j = 0; j < n_mesh; j++) {
            spde_w_out[j] = ::fwd::Dual(nan_v, nan_v);
        }
    }
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

    try {
        spde_w_out = spde_nc_transform_arena(
            ar, z,
            params[layout.log_kappa_spde_idx],
            params[layout.log_tau_spde_idx],
            tx);
    } catch (const std::runtime_error&) {
        // Cholesky failure during the forward pass inside the arena hook
        // (same root cause as the double-path catch above). Emit NaN-
        // valued arena Vars not tied to any backward block. The eta
        // accumulator multiplies these into the log-post, producing NaN
        // → NUTS rejects. The lost gradient information is moot because
        // the trajectory will not be accepted anyway.
        const double nan_v = std::numeric_limits<double>::quiet_NaN();
        spde_w_out.resize(n_mesh);
        for (int j = 0; j < n_mesh; j++) {
            spde_w_out[j] = arena::Var(ar, nan_v);
        }
    }
}

} // namespace tulpa
