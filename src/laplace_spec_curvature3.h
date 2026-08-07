// laplace_spec_curvature3.h
//
// Builds the third-log-lik-derivative oracle the inner-Laplace skewness
// diagnostic needs (see inner_laplace_skew.h) from a LikelihoodSpec. Which of
// the two oracle shapes comes back is decided by the spec's process count.
//
// SINGLE PROCESS (n_processes == 1) -- a scalar l_j'''(eta_j) per observation:
//
//  1. A built-in family wrapped via builtin_family_spec() (spec.name ==
//     "builtin:<family>", see is_builtin_family_spec()): exact, using the
//     analytic third-derivative ladder (laplace_family_curvature.h), zero extra
//     likelihood evaluations.
//  2. A genuine consumer-package LikelihoodSpec with eta_weights_fn set: a
//     central finite difference on the Newton working weight
//     w(eta) = neg_hess_eta[0] that eta_weights_fn already returns -- the only
//     generic way to reach a third derivative from an opaque value + gradient +
//     Hessian callback, reusing the SAME per-observation entry point the Newton
//     solver itself calls every iteration.
//     l'''(eta) ~= -(w(eta+h) - w(eta-h)) / (2h).
//
// SEVERAL PROCESSES (n_processes > 1) -- a zero-inflation mixture's
// (count, zi) pair, and any other spec whose per-observation log-density reads
// more than one linear predictor at once. There is no per-eta third derivative
// to return, so the oracle is the per-observation CONTRACTION of the third
// derivative tensor against the probe direction (curvature3_contract.h): the
// n_processes coordinates are the blocks, and each block's difference quotient
// is one central difference of the row-major n x n negative Hessian
// eta_weights_fn already writes. 2 * n_processes extra eta_weights_fn calls per
// observation per probed index; nothing is stored.
//
// The oracle is DECLINED (both slots empty, so every latent's gamma_3 comes back
// NaN) only when the spec exposes no way to reach a third derivative at all: a
// non-built-in spec with no eta_weights_fn. That is `curvature3_unavailable` --
// a property of what the spec ships, not of the model class. Coupling several
// processes is no longer a reason to decline.
//
// A cell-coupled likelihood (tulpaObs's occu_cover, which does not use
// LikelihoodSpec at all -- it couples psi/p/pos through tulpa's separate
// CellCouplingSpec interface) is scored by the same contraction one level up,
// over the cell's arms rather than a row's processes; see cell_curvature3.h.

#ifndef TULPA_LAPLACE_SPEC_CURVATURE3_H
#define TULPA_LAPLACE_SPEC_CURVATURE3_H

#include "curvature3_contract.h"
#include "laplace_builtin_family_spec.h"
#include "laplace_family_curvature.h"
#include "tulpa/likelihood.h"
#include "tulpa/model_data.h"
#include "tulpa/param_layout.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <vector>

namespace tulpa {

// Alias kept for the eta-space step the scalar fallback takes; the constant
// itself lives in curvature3_contract.h so the block contraction and the scalar
// difference cannot drift apart.
constexpr double SPEC_CURVATURE3_FD_STEP = CURVATURE3_FD_STEP;

// Per-observation cubic contraction for a multi-process spec: block a is
// coordinate a, so each block's direction is u_a e_a and its bilinear form over
// blocks (b, c) is just L''_{bc} u_b u_c.
inline UnitCubic3Fn build_spec_curvature3_unit_fn(
    const LikelihoodSpec& spec,
    const void* response_data,
    const ModelData& data,
    const ParamLayout& layout,
    const std::vector<double>& params,
    bool per_arm_step = true
) {
    const int np = spec.n_processes;
    EtaWeightsFn ewf = spec.eta_weights_fn;
    // Captures response_data/data/layout/params by reference/pointer: all are
    // owned by the caller's stack frame for the duration of the synchronous
    // solve this closure is used inside, so nothing outlives its owner.
    return [ewf, response_data, &data, &layout, &params, np, per_arm_step]
           (int i, const double* eta, const double* u) -> double {
        std::vector<double> eta_pert(np), grad_eta(np);
        std::vector<double> nh_p((std::size_t)np * np), nh_m((std::size_t)np * np);
        std::vector<double> bf((std::size_t)np * np * np, 0.0);
        std::vector<char>   have(np, 0);

        std::vector<double> max_eta(np), max_u(np);
        for (int a = 0; a < np; a++) {
            max_eta[a] = std::fabs(eta[a]);
            max_u[a]   = std::fabs(u[a]);
        }
        const double h_global = per_arm_step
            ? 0.0 : curvature3_global_step(max_eta, max_u);

        for (int a = 0; a < np; a++) {
            const double h = per_arm_step
                ? curvature3_block_step(max_eta[a], max_u[a]) : h_global;
            if (!(h > 0.0)) continue;

            for (int t = 0; t < np; t++) eta_pert[t] = eta[t];
            eta_pert[a] = eta[a] + h * u[a];
            ewf(i, eta_pert.data(), 0.0, 0.0, params, data, layout, response_data,
                grad_eta.data(), nh_p.data());
            eta_pert[a] = eta[a] - h * u[a];
            ewf(i, eta_pert.data(), 0.0, 0.0, params, data, layout, response_data,
                grad_eta.data(), nh_m.data());

            bool ok = true;
            for (std::size_t t = 0; t < nh_p.size(); t++) {
                if (!std::isfinite(nh_p[t]) || !std::isfinite(nh_m[t])) { ok = false; break; }
            }
            if (!ok) continue;

            // L'' = -neg_hess; the difference quotient along block a, contracted
            // with u on both remaining slots.
            const double inv = 1.0 / (2.0 * h);
            for (int b = 0; b < np; b++) {
                for (int c = 0; c < np; c++) {
                    const std::size_t t = (std::size_t)b * np + c;
                    bf[(std::size_t)(a * np + b) * np + c] =
                        -(nh_p[t] - nh_m[t]) * inv * u[b] * u[c];
                }
            }
            have[a] = 1;
        }
        return curvature3_symmetrized_sum(bf, have, np);
    };
}

inline Curvature3Oracle build_spec_curvature3_oracle(
    const LikelihoodSpec& spec,
    const void* response_data,
    const ModelData& data,
    const ParamLayout& layout,
    const std::vector<double>& params
) {
    Curvature3Oracle out;
    out.n_coords = spec.n_processes;

    if (spec.n_processes == 1) {
        if (is_builtin_family_spec(spec.name)) {
            const auto* r = static_cast<const BuiltinFamilyResponse*>(response_data);
            out.scalar = [r](int j, double eta_j) -> double {
                double w = r->weights ? r->weights[j] : 1.0;
                int nt = r->n_trials ? r->n_trials[j] : 1;
                return w * curvature3_obs_for_family(r->y[j], nt, eta_j, r->family,
                                                     r->phi, r->phi2);
            };
            return out;
        }
        if (!spec.eta_weights_fn) {
            out.declined = "curvature3_unavailable";
            return out;
        }
        // Captures response_data/data/layout/params by reference/pointer: all are
        // owned by the caller's stack frame for the duration of the synchronous
        // solve this closure is used inside, so nothing outlives its owner.
        EtaWeightsFn ewf = spec.eta_weights_fn;
        out.scalar = [ewf, response_data, &data, &layout, &params]
                     (int j, double eta_j) -> double {
            double h = SPEC_CURVATURE3_FD_STEP * std::max(1.0, std::fabs(eta_j));
            double grad_eta[1];
            double neg_hess_p[1], neg_hess_m[1];
            double eta_p = eta_j + h, eta_m = eta_j - h;
            ewf(j, &eta_p, 0.0, 0.0, params, data, layout, response_data,
                grad_eta, neg_hess_p);
            ewf(j, &eta_m, 0.0, 0.0, params, data, layout, response_data,
                grad_eta, neg_hess_m);
            if (!std::isfinite(neg_hess_p[0]) || !std::isfinite(neg_hess_m[0])) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            return -(neg_hess_p[0] - neg_hess_m[0]) / (2.0 * h);
        };
        return out;
    }

    if (spec.n_processes < 1 || !spec.eta_weights_fn) {
        out.declined = "curvature3_unavailable";
        return out;
    }
    out.unit = build_spec_curvature3_unit_fn(spec, response_data, data, layout,
                                             params);
    return out;
}

} // namespace tulpa

#endif // TULPA_LAPLACE_SPEC_CURVATURE3_H
