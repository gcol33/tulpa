// laplace_builtin_family_spec.h
// Bridges tulpa's built-in family closed forms (laplace_family_link.h) to the
// LikelihoodSpec contract, so every shipped family is expressible as a
// spec-driven Laplace likelihood.
//
// This is the single source of truth that lets the nested-Laplace kernels stop
// hardcoding the family enum (see dev_notes/plans/clean_migration.md, Phase L) WITHOUT
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
    // Grouped beta sufficient statistics (gcol33/tulpaObs#49). When non-null and
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
    // Upper-truncated Gaussian ceiling (truncated_gaussian family, gcol33/tulpa#122).
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
inline double builtin_family_ll_double(
    int i, const double* eta, const double& /*logit_zi*/,
    const double& /*logit_oi*/, const std::vector<double>& /*params*/,
    const ModelData& /*data*/, const ParamLayout& /*layout*/,
    const void* model_data
) {
    const auto* r = static_cast<const BuiltinFamilyResponse*>(model_data);
    const int nt = r->n_trials ? r->n_trials[i] : 1;
    const double w = r->weights ? r->weights[i] : 1.0;
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
    // and stalls (gcol33/tulpa#108). A no-op when weights are absent (w == 1).
    return w * ll;
}

// EtaWeightsFn: per-obs eta-space score + Fisher working weight. n_processes
// is 1, so grad_eta / neg_hess_eta are scalars. A per-obs likelihood weight w_i
// scales both the score and the Fisher information (`gh.grad *= w_i;
// gh.neg_hess *= w_i`); builtin_family_ll_double / builtin_family_ll_ad scale
// the value by the same w_i, so the line search optimizes the same weighted
// objective the step is built from (gcol33/tulpa#108).
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

// Build a single-process LikelihoodSpec backed by the family-enum closed forms.
// Pair the returned spec with a BuiltinFamilyResponse (via
// ModelData.model_response_data) whose arrays outlive the fit.
inline LikelihoodSpec builtin_family_spec(const std::string& family) {
    LikelihoodSpec spec;
    spec.n_processes    = 1;
    spec.name           = "builtin:" + family;
    spec.ll_double      = &builtin_family_ll_double;
    spec.eta_weights_fn = &builtin_family_eta_weights;
    spec.n_extra_params = 0;
    return spec;
}

} // namespace tulpa

#endif // TULPA_LAPLACE_BUILTIN_FAMILY_SPEC_H
