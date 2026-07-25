// gp_nc_apply.cpp
// Implementation of the non-centered NNGP transform application. See
// gp_nc_apply.h for the contract.

#include "gp_nc_apply.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "hmc_gp.h"                 // tulpa_gp::nngp_nc_forward / _backward,
                                    // tulpa_gp::NNGPNCWorkspace
#include "tulpa/model_data.h"
#include "tulpa/param_layout.h"

namespace tulpa {

void apply_gp_nc_transform_double(
    const std::vector<double>& params,
    const ModelData&           data,
    const ParamLayout&         layout,
    std::vector<double>&       gp_w_out)
{
    const int N  = data.gp_data.n_obs;
    const int w0 = layout.gp_w_start;

    const double sigma2 = std::exp(params[layout.log_sigma2_gp_idx]);
    const double phi    = std::exp(params[layout.log_phi_gp_idx]);

    // POD-pointer TLS (constant init, no thread-atexit destructor): a
    // lazily-initialized thread_local object here corrupts the heap under the
    // mingw toolchain when chains run in parallel. The workspace intentionally
    // leaks per thread. Same pattern as hmc_nuts_chain_iter_store.h.
    static thread_local tulpa_gp::NNGPNCWorkspace* ws_p = nullptr;
    if (!ws_p) ws_p = new tulpa_gp::NNGPNCWorkspace();

    tulpa_gp::nngp_nc_forward(&params[w0], sigma2, phi, data.gp_data, *ws_p);

    gp_w_out.resize(N);
    for (int i = 0; i < N; i++) gp_w_out[i] = ws_p->w[i];
}

void apply_gp_nc_transform_arena(
    const std::vector<arena::Var>& params,
    const ModelData&               data,
    const ParamLayout&             layout,
    std::vector<arena::Var>&       gp_w_out)
{
    const int N  = data.gp_data.n_obs;
    const int w0 = layout.gp_w_start;

    // params is guaranteed non-empty here (the GP block contributes at least
    // one z slot); the arena pointer is carried on every Var.
    arena::Arena* ar = params[w0].arena_;

    const double sigma2 = std::exp(params[layout.log_sigma2_gp_idx].val());
    const double phi    = std::exp(params[layout.log_phi_gp_idx].val());

    std::vector<double> z(N);
    for (int i = 0; i < N; i++) z[i] = params[w0 + i].val();

    // Fresh per-call workspace, kept alive by the backward lambda (the arena's
    // CustomBackwardRecord holds the closure until the backward sweep runs).
    // A plain heap alloc, not thread_local -- no atexit destructor to trip the
    // mingw parallel-chains heap bug.
    auto ws = std::make_shared<tulpa_gp::NNGPNCWorkspace>();
    tulpa_gp::nngp_nc_forward(z.data(), sigma2, phi, data.gp_data, *ws);

    std::vector<int32_t> input_indices;
    input_indices.reserve(N + 2);
    for (int i = 0; i < N; i++) input_indices.push_back(params[w0 + i].idx_);
    input_indices.push_back(params[layout.log_sigma2_gp_idx].idx_);
    input_indices.push_back(params[layout.log_phi_gp_idx].idx_);

    std::vector<double> output_values(N);
    for (int i = 0; i < N; i++) output_values[i] = ws->w[i];

    const GPData& gp_data = data.gp_data;
    auto cb = [ws, &gp_data, sigma2, phi, N](
        const double* input_vals,  int /*n_in*/,
        const double* /*output_vals*/, int /*n_out*/,
        const double* output_adjs,
        double*       input_adjs
    ) {
        // input_vals = [z[0..N-1], log_sigma2, log_phi]; output_adjs = dL/dw.
        std::vector<double> grad_z(N, 0.0);
        double g_log_sigma2_lik = 0.0;
        double g_log_phi_lik = 0.0;
        double g_log_phi_jac = 0.0;  // pure non-centered: dropped, no z->w Jacobian
        tulpa_gp::nngp_nc_backward(
            input_vals, sigma2, phi, gp_data, *ws, output_adjs,
            grad_z.data(), g_log_sigma2_lik, g_log_phi_lik, g_log_phi_jac);

        for (int i = 0; i < N; i++) input_adjs[i] = grad_z[i];
        input_adjs[N]     = g_log_sigma2_lik;  // likelihood-through-w only
        input_adjs[N + 1] = g_log_phi_lik;     // (Jacobian term intentionally unused)
    };

    std::vector<int32_t> out_indices;
    ar->add_custom_backward(input_indices, output_values, cb, out_indices);

    gp_w_out.resize(N);
    for (int i = 0; i < N; i++) {
        gp_w_out[i].arena_ = ar;
        gp_w_out[i].idx_   = out_indices[i];
    }
}

} // namespace tulpa
