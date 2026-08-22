// nngp_nc_term_apply.cpp
// Implementation of the shared per-term non-centered NNGP applier. See
// nngp_nc_term_apply.h for the contract.

#include "nngp_nc_term_apply.h"

#include <cmath>
#include <cstdint>
#include <memory>

#include "autodiff_utils.h"  // tulpa::math::safe_exp
#include "hmc_gp.h"  // tulpa_gp::{NNGPNCView, NNGPNCWorkspace, nngp_nc_forward,
                     // nngp_nc_backward}

namespace tulpa {

void apply_nngp_nc_term_double(
    const std::vector<double>& params,
    int w_start, int sigma2_idx, int phi_idx, int N,
    const tulpa_gp::NNGPNCView& view,
    std::vector<double>&       w_out)
{
    const double sigma2 = tulpa::math::safe_exp(params[sigma2_idx]);
    const double phi    = tulpa::math::safe_exp(params[phi_idx]);

    // POD-pointer TLS (constant init, no thread-atexit destructor): a
    // lazily-initialized thread_local object here corrupts the heap under the
    // mingw toolchain when chains run in parallel. The workspace intentionally
    // leaks per thread.
    static thread_local tulpa_gp::NNGPNCWorkspace* ws_p = nullptr;
    if (!ws_p) ws_p = new tulpa_gp::NNGPNCWorkspace();

    tulpa_gp::nngp_nc_forward(&params[w_start], sigma2, phi, view, *ws_p);

    w_out.resize(N);
    for (int i = 0; i < N; i++) w_out[i] = ws_p->w[i];
}

std::vector<arena::Var> apply_nngp_nc_term_arena(
    const std::vector<arena::Var>& params,
    int w_start, int sigma2_idx, int phi_idx, int N,
    const tulpa_gp::NNGPNCView& view)
{
    // params[w_start] is guaranteed to carry the arena pointer (every term
    // contributes at least one z slot).
    arena::Arena* ar = params[w_start].arena_;

    const double sigma2 = tulpa::math::safe_exp(params[sigma2_idx].val());
    const double phi    = tulpa::math::safe_exp(params[phi_idx].val());

    std::vector<double> z(N);
    for (int i = 0; i < N; i++) z[i] = params[w_start + i].val();

    // Fresh per-call workspace, kept alive by the backward lambda (the
    // arena's CustomBackwardRecord holds the closure until the backward
    // sweep runs). A plain heap alloc via shared_ptr, not thread_local -- no
    // atexit destructor to trip the mingw parallel-chains heap bug.
    auto ws = std::make_shared<tulpa_gp::NNGPNCWorkspace>();
    tulpa_gp::nngp_nc_forward(z.data(), sigma2, phi, view, *ws);

    std::vector<int32_t> input_indices;
    input_indices.reserve(N + 2);
    for (int i = 0; i < N; i++) input_indices.push_back(params[w_start + i].idx_);
    input_indices.push_back(params[sigma2_idx].idx_);
    input_indices.push_back(params[phi_idx].idx_);

    std::vector<double> output_values(N);
    for (int i = 0; i < N; i++) output_values[i] = ws->w[i];

    auto cb = [ws, view, sigma2, phi, N](
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
            input_vals, sigma2, phi, view, *ws, output_adjs,
            grad_z.data(), g_log_sigma2_lik, g_log_phi_lik, g_log_phi_jac);

        for (int i = 0; i < N; i++) input_adjs[i] = grad_z[i];
        input_adjs[N]     = g_log_sigma2_lik;  // likelihood-through-w only
        input_adjs[N + 1] = g_log_phi_lik;     // (Jacobian term intentionally unused)
    };

    std::vector<int32_t> out_indices;
    ar->add_custom_backward(input_indices, output_values, cb, out_indices);

    std::vector<arena::Var> w_out(N);
    for (int i = 0; i < N; i++) {
        w_out[i].arena_ = ar;
        w_out[i].idx_   = out_indices[i];
    }
    return w_out;
}

} // namespace tulpa
