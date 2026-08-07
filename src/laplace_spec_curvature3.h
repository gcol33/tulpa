// laplace_spec_curvature3.h
//
// Builds the per-observation third-log-lik-derivative oracle
// (std::function<double(int,double)>, see inner_laplace_skew.h) the
// inner-Laplace skewness diagnostic needs, from a LikelihoodSpec. Two cases:
//
//  1. A built-in family wrapped via builtin_family_spec() (n_processes == 1,
//     spec.name == "builtin:<family>", see is_builtin_family_spec()): exact,
//     using the analytic third-derivative ladder
//     (laplace_family_curvature.h), zero extra likelihood evaluations.
//  2. A genuine consumer-package LikelihoodSpec with n_processes == 1 and
//     eta_weights_fn set: a central finite difference on the Newton working
//     weight w(eta) = neg_hess_eta[0] that eta_weights_fn already returns --
//     the only generic way to reach a third derivative from an opaque
//     value + gradient + Hessian callback, reusing the SAME per-observation
//     entry point the Newton solver itself calls every iteration.
//     l'''(eta) ~= -(w(eta+h) - w(eta-h)) / (2h).
//
// Declines (returns an empty std::function, so every latent's gamma_3 comes
// back NaN) for n_processes > 1 -- zero-inflated mixtures, and any other
// coupled multi-process spec (e.g. tulpaObs's occu_cover, which does not
// even use LikelihoodSpec -- it couples psi/p/pos through tulpa's separate
// CellCouplingSpec interface). A general mixed-partial third-derivative
// tensor for a coupled multi-process likelihood is a separate, larger
// derivation (see the scope note in inner_laplace_skew.h), not attempted
// here rather than shipped as an unverified guess.

#ifndef TULPA_LAPLACE_SPEC_CURVATURE3_H
#define TULPA_LAPLACE_SPEC_CURVATURE3_H

#include "laplace_builtin_family_spec.h"
#include "laplace_family_curvature.h"
#include "tulpa/likelihood.h"
#include "tulpa/model_data.h"
#include "tulpa/param_layout.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

namespace tulpa {

// Relative finite-difference step for the generic LikelihoodSpec fallback.
constexpr double SPEC_CURVATURE3_FD_STEP = 1e-4;

// `reason` (optional out-parameter, gcol33/tulpa#296) records WHY no oracle
// could be built, from the closed vocabulary the diagnostic reports:
// "coupled_likelihood" (n_processes != 1 -- a permanent property of the model
// class, never scorable by this formula) or "curvature3_unavailable" (a
// single-process spec with no eta_weights_fn to finite-difference). Set to ""
// on success. Reported here rather than re-derived by a second predicate, so
// the reason and the decision can never drift apart.
inline std::function<double(int, double)> build_spec_curvature3_fn(
    const LikelihoodSpec& spec,
    const void* response_data,
    const ModelData& data,
    const ParamLayout& layout,
    const std::vector<double>& params,
    const char** reason = nullptr
) {
    if (reason) *reason = "";
    if (spec.n_processes != 1) {
        if (reason) *reason = "coupled_likelihood";
        return nullptr;
    }

    if (is_builtin_family_spec(spec.name)) {
        const auto* r = static_cast<const BuiltinFamilyResponse*>(response_data);
        return [r](int j, double eta_j) -> double {
            double w = r->weights ? r->weights[j] : 1.0;
            int nt = r->n_trials ? r->n_trials[j] : 1;
            return w * curvature3_obs_for_family(r->y[j], nt, eta_j, r->family,
                                                 r->phi, r->phi2);
        };
    }

    if (!spec.eta_weights_fn) {
        if (reason) *reason = "curvature3_unavailable";
        return nullptr;
    }
    // Captures response_data/data/layout/params by reference/pointer: all are
    // owned by the caller's stack frame for the duration of the synchronous
    // solve this closure is used inside, so nothing outlives its owner.
    EtaWeightsFn ewf = spec.eta_weights_fn;
    return [ewf, response_data, &data, &layout, &params](int j, double eta_j) -> double {
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
}

} // namespace tulpa

#endif // TULPA_LAPLACE_SPEC_CURVATURE3_H
