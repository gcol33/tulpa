// tulpa_priors_spde.h
// SPDE Matern prior on the latent mesh-node block w_mesh.
//
// Phase 1 contract: (kappa, tau_spde) are held constant on
// data.spde_data; Q_x is precomputed once at ModelData setup and treated
// as a double constant array, even on the autodiff AD path. The latent
// w_mesh is sampled jointly with everything else; the prior contribution
//
//     log p(w | theta) = -0.5 * w' Q w   (+ const log|Q|/2)
//
// is templated for AD so the gradient w.r.t. w flows back through the
// quadratic form. The constant log|Q|/2 cancels in NUTS acceptance ratios
// and is dropped from the returned log-density.
//
// Phase 2 (joint NUTS over hypers, deferred) will replace the constant
// Q_x with a per-eval rebuild from log_kappa / log_tau slots and add a
// log|Q(theta)|/2 term using either (i) a spline interpolant of
// log|P(kappa)|, or (ii) a non-centered z = L^{-T} w reparameterization
// that cancels log|Q| against the Jacobian (needs arena AD with a
// custom sparse-Cholesky adjoint).
//
// Prerequisite: ModelData and ParamLayout must be defined before this
// header (normally via hmc_sampler.h).

#ifndef TULPA_PRIORS_SPDE_H
#define TULPA_PRIORS_SPDE_H

#include <vector>
#include "autodiff_utils.h"

namespace tulpa {
namespace priors {

using namespace math;

// Compute -0.5 * w' Q w with Q in CSC and w in the layout's spde_w block.
// `spde_w_out` is filled with the mesh-node values pulled from `params`
// so downstream eta accumulation can reference it without re-reading the
// flat parameter vector. Returns T(0) when SPDE is inactive.
template<typename T>
T compute_spde_prior(const std::vector<T>& params, const ModelData& data,
                      const ParamLayout& layout, std::vector<T>& spde_w_out)
{
    if (!layout.is_spde || !data.has_spde) {
        spde_w_out.clear();
        return T(0.0);
    }

    const auto& spde = data.spde_data;
    const int  n_mesh = spde.n_mesh;
    const int  w0     = layout.spde_w_start;

    // Pull w_mesh into a contiguous T-vector for both the quadratic form
    // and the downstream eta accumulator (mirrors gp_w / hsgp_f / etc.).
    spde_w_out.resize(n_mesh);
    for (int j = 0; j < n_mesh; j++) {
        spde_w_out[j] = params[w0 + j];
    }

    // -0.5 * w' Q w via CSC iteration. Q is symmetric and stored full
    // (both triangles) — matches the SpdeQBuilder convention in
    // spde_qbuilder.h, so we sum every (row, col) entry once.
    T qf = T(0.0);
    for (int col = 0; col < n_mesh; col++) {
        const int beg = spde.Q_p[col];
        const int end = spde.Q_p[col + 1];
        const T   w_col = spde_w_out[col];
        for (int idx = beg; idx < end; idx++) {
            const int row = spde.Q_i[idx];
            qf = qf + spde_w_out[row] * T(spde.Q_x[idx]) * w_col;
        }
    }

    // log|Q|/2 is a constant under fixed (kappa, tau_spde); dropping it
    // is safe in NUTS / VI / SMC where only differences matter. When the
    // follow-on arc adds joint hypers we must reinstate this term so the
    // gradient w.r.t. (log_kappa, log_tau_spde) is correct.
    return T(-0.5) * qf;
}

} // namespace priors
} // namespace tulpa

#endif // TULPA_PRIORS_SPDE_H
