// laplace_builtin_family_spec.h
// Bridges tulpa's built-in family closed forms (laplace_family_link.h) to the
// LikelihoodSpec contract, so every shipped family is expressible as a
// spec-driven Laplace likelihood.
//
// This is the single source of truth that lets the nested-Laplace kernels stop
// hardcoding the family enum WITHOUT
// duplicating the per-observation likelihood math: the spec callbacks delegate
// straight to grad_hess_for_family / log_lik_for_family. The working weight is
// the expected (Fisher) information those functions already return, which is
// what keeps the Newton Hessian positive-definite on non-canonical links.

#ifndef TULPA_LAPLACE_BUILTIN_FAMILY_SPEC_H
#define TULPA_LAPLACE_BUILTIN_FAMILY_SPEC_H

#include "tulpa/likelihood.h"
#include "tulpa/model_data.h"
#include "tulpa/param_layout.h"
#include "laplace_family_link.h"
#include "builtin_family_zi.h"
#include <limits>
#include <string>
#include <vector>

namespace tulpa {

// Per-observation response payload for a single-process built-in family.
// The arrays are borrowed; they must outlive the fit. n_trials may be null
// (treated as 1 everywhere), which is the non-binomial case.
struct BuiltinFamilyResponse {
    const double* y = nullptr;       // [N] response
    const int* n_trials = nullptr;   // [N] binomial denominators, or null (=> 1)
    int N = 0;
    std::string family;              // resolved against laplace_family_link.h
    double phi = 1.0;                // dispersion / precision / size
    const double* weights = nullptr; // [N] per-obs likelihood weights, or null (=> 1)
    // Grouped beta sufficient statistics. When non-null and
    // family == "beta", row i is an exact collapse of n_trials[i] exchangeable
    // beta observations sharing this row's linear predictor; slog_y[i] = sum
    // log(y), slog_1my[i] = sum log(1-y). Null => ungrouped per-obs path.
    const double* slog_y = nullptr;
    const double* slog_1my = nullptr;
    // Interval-censored Gaussian bounds (interval_gaussian family). When non-null
    // and family == "interval_gaussian", row i records that the latent value fell
    // in (lower[i], upper[i]] on the linear-predictor scale; -Inf / +Inf are the
    // open outer classes and phi is the latent SD. Null => not an interval arm.
    const double* lower = nullptr;
    const double* upper = nullptr;
    // Upper-truncated Gaussian ceiling (truncated_gaussian family).
    // When non-null and family == "truncated_gaussian", row i's latent log-response
    // is Normal(eta, phi^2) truncated to <= trunc_upper[i] on the predictor scale
    // (+Inf => no truncation). The point response y[i] is still read (the density is
    // evaluated at it); only the upper bound is added. Null => not a truncated arm.
    const double* trunc_upper = nullptr;
    // Second dispersion channel: the Student-t degrees of freedom (and any
    // future two-parameter family's extra parameter, e.g. the Tweedie power).
    // NaN => the family's built-in default (kStudentTDf for "t").
    double phi2 = std::numeric_limits<double>::quiet_NaN();
};

// LikelihoodFn<double>: per-obs log-likelihood for the built-in family.
//
// `data.zi_type` selects the zero-inflation channel used by the SAMPLER paths,
// where the ZI predictor arrives as the `logit_zi` argument. It stays NONE on
// the spec-driven Laplace path, which instead carries the ZI predictor as
// process 1 and uses builtin_family_zi_ll_double below -- so the branch here
// fires only when logit_zi is genuinely populated. Keeping it in step with
// builtin_family_ll_ad matters: the sampler evaluates the log-posterior VALUE
// through this callback and its GRADIENT through the AD one, and a mixture
// applied to only one of them would make the two describe different models.
inline double builtin_family_ll_double(
    int i, const double* eta, const double& logit_zi,
    const double& /*logit_oi*/, const std::vector<double>& /*params*/,
    const ModelData& data, const ParamLayout& /*layout*/,
    const void* model_data
) {
    const auto* r = static_cast<const BuiltinFamilyResponse*>(model_data);
    const int nt = r->n_trials ? r->n_trials[i] : 1;
    const double w = r->weights ? r->weights[i] : 1.0;
    if (data.zi_type != ZIType::NONE) {
        return w * zi::mixture_ll_double(r->y[i], nt, eta[0], logit_zi,
                                         r->family, r->phi, r->phi2);
    }
    const double ll =
        (r->trunc_upper && r->family == "truncated_gaussian")
        ? log_lik_truncated_gaussian(r->y[i], r->trunc_upper[i], eta[0], r->phi)
        : (r->lower && r->family == "interval_gaussian")
        ? log_lik_interval_gaussian(r->lower[i], r->upper[i], eta[0], r->phi)
        : (r->slog_y && r->family == "beta")
        ? log_lik_beta_grouped(r->slog_y[i], r->slog_1my[i], nt, eta[0], r->phi)
        : log_lik_for_family(r->y[i], nt, eta[0], r->family, r->phi, r->phi2);
    // Weight the log-lik by the SAME per-obs factor the score / Fisher Hessian
    // carry (builtin_family_eta_weights), so the Newton line search optimizes the
    // weighted objective its step direction is built from. Without this the
    // backtracking search judges a weighted step against the unweighted log-lik
    // and stalls. A no-op when weights are absent (w == 1).
    return w * ll;
}

// EtaWeightsFn: per-obs eta-space score + Fisher working weight. n_processes
// is 1, so grad_eta / neg_hess_eta are scalars. A per-obs likelihood weight w_i
// scales both the score and the Fisher information (`gh.grad *= w_i;
// gh.neg_hess *= w_i`); builtin_family_ll_double / builtin_family_ll_ad scale
// the value by the same w_i, so the line search optimizes the same weighted
// objective the step is built from.
inline void builtin_family_eta_weights(
    int i, const double* eta, double /*logit_zi*/, double /*logit_oi*/,
    const std::vector<double>& /*params*/, const ModelData& /*data*/,
    const ParamLayout& /*layout*/, const void* model_data,
    double* grad_eta, double* neg_hess_eta
) {
    const auto* r = static_cast<const BuiltinFamilyResponse*>(model_data);
    const int nt = r->n_trials ? r->n_trials[i] : 1;
    const GradHess gh =
        (r->trunc_upper && r->family == "truncated_gaussian")
        ? grad_hess_truncated_gaussian(r->y[i], r->trunc_upper[i], eta[0], r->phi)
        : (r->lower && r->family == "interval_gaussian")
        ? grad_hess_interval_gaussian(r->lower[i], r->upper[i], eta[0], r->phi)
        : (r->slog_y && r->family == "beta")
        ? grad_hess_beta_grouped(r->slog_y[i], r->slog_1my[i], nt, eta[0], r->phi)
        : grad_hess_for_family(r->y[i], nt, eta[0], r->family, r->phi, r->phi2);
    const double w = r->weights ? r->weights[i] : 1.0;
    grad_eta[0]     = w * gh.grad;
    neg_hess_eta[0] = w * gh.neg_hess;
}

// ---------------------------------------------------------------------------
// Zero-inflated variants. The spec-driven Laplace shim passes 0.0 for the
// `logit_zi` callback argument, so the ZI linear predictor is carried as
// process 1 and read from eta[1]. Process 0 remains the count predictor, and
// the RE block is shared into process 0 only (SharingSpec::re). The 2 x 2
// negative-Hessian block the EtaWeightsFn contract already specifies is exactly
// what the mixture's cross term needs -- no new curvature contract.
//
// The mixture math itself lives in builtin_family_zi.h, shared with the AD
// sampler path, and is validated against R/family_zi.R.
// ---------------------------------------------------------------------------

inline double builtin_family_zi_ll_double(
    int i, const double* eta, const double& /*logit_zi*/,
    const double& /*logit_oi*/, const std::vector<double>& /*params*/,
    const ModelData& /*data*/, const ParamLayout& /*layout*/,
    const void* model_data
) {
    const auto* r = static_cast<const BuiltinFamilyResponse*>(model_data);
    const int nt = r->n_trials ? r->n_trials[i] : 1;
    const double w = r->weights ? r->weights[i] : 1.0;
    return w * zi::mixture_ll_double(r->y[i], nt, eta[0], eta[1],
                                     r->family, r->phi, r->phi2);
}

inline void builtin_family_zi_eta_weights(
    int i, const double* eta, double /*logit_zi*/, double /*logit_oi*/,
    const std::vector<double>& /*params*/, const ModelData& /*data*/,
    const ParamLayout& /*layout*/, const void* model_data,
    double* grad_eta, double* neg_hess_eta
) {
    const auto* r = static_cast<const BuiltinFamilyResponse*>(model_data);
    const int nt = r->n_trials ? r->n_trials[i] : 1;
    zi::mixture_eta_weights_double(r->y[i], nt, eta[0], eta[1],
                                   r->family, r->phi, r->phi2,
                                   grad_eta, neg_hess_eta);
    const double w = r->weights ? r->weights[i] : 1.0;
    if (w != 1.0) {
        grad_eta[0] *= w; grad_eta[1] *= w;
        for (int k = 0; k < 4; k++) neg_hess_eta[k] *= w;
    }
}

// Build a LikelihoodSpec backed by the family-enum closed forms. `zi` selects
// the two-process zero-inflated mixture; without it the spec is single-process.
// Pair the returned spec with a BuiltinFamilyResponse (via
// ModelData.model_response_data) whose arrays outlive the fit.
inline LikelihoodSpec builtin_family_spec(const std::string& family,
                                          bool zi = false) {
    LikelihoodSpec spec;
    spec.n_processes    = zi ? 2 : 1;
    spec.name           = (zi ? "builtin_zi:" : "builtin:") + family;
    spec.ll_double      = zi ? &builtin_family_zi_ll_double
                             : &builtin_family_ll_double;
    spec.eta_weights_fn = zi ? &builtin_family_zi_eta_weights
                             : &builtin_family_eta_weights;
    spec.n_extra_params = 0;
    return spec;
}

} // namespace tulpa

#endif // TULPA_LAPLACE_BUILTIN_FAMILY_SPEC_H
