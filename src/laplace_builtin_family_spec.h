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
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace tulpa {

// Which branch of the AD likelihood ladder (builtin_family_ll_ad.h) a family
// code selects. That ladder runs once per observation per reverse-mode sweep,
// so comparing the code against up to twelve std::strings there is per-obs work
// on a quantity fixed for the whole fit. This is the same classification
// FamilyKind performs for the double path, over the AD ladder's own branches.
enum class AdFamily : int {
    UNRESOLVED,
    GAUSSIAN, POISSON, BINOMIAL, NEG_BINOMIAL_2, NEG_BINOMIAL_1,
    TRUNCATED_POISSON, TRUNCATED_NEG_BINOMIAL_2, BETA_BINOMIAL, STUDENT_T,
    GAMMA, INVERSE_GAUSSIAN, LOGNORMAL, BETA
};

inline AdFamily ad_family_kind(const std::string& code) {
    if (code == "poisson")                  return AdFamily::POISSON;
    if (code == "binomial")                 return AdFamily::BINOMIAL;
    if (code == "neg_binomial_2")           return AdFamily::NEG_BINOMIAL_2;
    if (code == "neg_binomial_1")           return AdFamily::NEG_BINOMIAL_1;
    if (code == "truncated_poisson")        return AdFamily::TRUNCATED_POISSON;
    if (code == "truncated_neg_binomial_2") return AdFamily::TRUNCATED_NEG_BINOMIAL_2;
    if (code == "beta_binomial")            return AdFamily::BETA_BINOMIAL;
    if (code == "t")                        return AdFamily::STUDENT_T;
    if (code == "gaussian")                 return AdFamily::GAUSSIAN;
    if (code == "lognormal")                return AdFamily::LOGNORMAL;
    if (code == "gamma")                    return AdFamily::GAMMA;
    if (code == "inverse_gaussian")         return AdFamily::INVERSE_GAUSSIAN;
    if (code == "beta")                     return AdFamily::BETA;
    return AdFamily::UNRESOLVED;
}

// Per-observation response payload for a single-process built-in family.
// The arrays are borrowed; they must outlive the fit. n_trials may be null
// (treated as 1 everywhere), which is the non-binomial case.
struct BuiltinFamilyResponse {
    const double* y = nullptr;       // [N] response
    const int* n_trials = nullptr;   // [N] binomial denominators, or null (=> 1)
    int N = 0;
    std::string family;              // resolved against laplace_family_link.h
    // The same code classified for the AD ladder, resolved by prepare()
    // alongside fam. Until prepare() runs it is UNRESOLVED, which the ladder
    // reports as an error rather than dispatching on a stale branch.
    AdFamily ad_kind = AdFamily::UNRESOLVED;
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

    // `family` resolved once, and the eta-independent part of each
    // observation's log-density evaluated once. Both are
    // owned, not borrowed, so a response copied into a per-arm or per-thread
    // pool carries its own -- the callbacks below read them by value.
    FamilyResolved fam;
    std::vector<double> ll_const;
    bool prepared = false;

    // Call once, after y / n_trials / family / N are set and before the solve.
    // Until it runs `prepared` is false and the callbacks take the string
    // ladder, which is the same arithmetic at the old speed -- a construction
    // site that forgets this loses the saving, never the lchoose.
    //
    // phi is NOT read here: the grid rewrites it per cell (sync_dispersion),
    // and no family's eta-independent term depends on it.
    void prepare() {
        // The tweedie variance power has no default, so tweedie_power() stops
        // on a NaN phi2 -- per observation, from inside whichever OpenMP
        // reduction the callbacks are evaluated in, where an escaping
        // exception is std::terminate rather than an R error. Raised here
        // instead: prepare() runs once, on the calling thread, before any
        // solve, and is the first point that sees the family and phi2
        // together. Same hoist as the two compute_total_log_lik entries.
        if (family == "tweedie" && std::isnan(phi2)) {
            Rcpp::stop("family 'tweedie' needs phi2 (the variance power p); "
                       "the likelihood carries none.");
        }
        // Ahead of the early return: the AD ladder dispatches on ad_kind for
        // any response it is handed, including one carrying no y array.
        ad_kind = ad_family_kind(family);
        if (!y || N <= 0) return;
        fam = resolve_family(family);
        ll_const.resize((size_t)N);
        for (int i = 0; i < N; i++) {
            const int nt = n_trials ? n_trials[i] : 1;
            ll_const[(size_t)i] = log_lik_const_for_kind(y[i], nt, fam.kind);
        }
        prepared = true;
    }
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
        : r->prepared
        ? log_lik_for_family(r->y[i], nt, eta[0], r->fam, r->phi, r->phi2,
                             r->ll_const[(size_t)i])
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
        : r->prepared
        ? grad_hess_for_family(r->y[i], nt, eta[0], r->fam, r->phi, r->phi2)
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

// The two spec-name prefixes builtin_family_spec() writes. The constructor and
// the predicates below are built from these, so the naming convention is
// defined once and the two cannot drift apart.
inline const char* builtin_spec_prefix()    { return "builtin:"; }
inline const char* builtin_zi_spec_prefix() { return "builtin_zi:"; }

// Build a LikelihoodSpec backed by the family-enum closed forms. `zi` selects
// the two-process zero-inflated mixture; without it the spec is single-process.
// Pair the returned spec with a BuiltinFamilyResponse (via
// ModelData.model_response_data) whose arrays outlive the fit.
inline LikelihoodSpec builtin_family_spec(const std::string& family,
                                          bool zi = false) {
    // The spec's callbacks dispatch on this family string per observation, and
    // the joint data log-likelihood evaluates them inside an OpenMP reduction.
    // An unregistered family reaches unknown_family_stop from there, and an
    // exception leaving a structured block is std::terminate, so the name is
    // checked here -- on the calling thread, once, at construction.
    if (!family_has_compiled_impl(family)) {
        unknown_family_stop("builtin_family_spec", family);
    }
    LikelihoodSpec spec;
    spec.n_processes    = zi ? 2 : 1;
    spec.name           = (zi ? builtin_zi_spec_prefix()
                              : builtin_spec_prefix()) + family;
    spec.ll_double      = zi ? &builtin_family_zi_ll_double
                             : &builtin_family_ll_double;
    spec.eta_weights_fn = zi ? &builtin_family_zi_eta_weights
                             : &builtin_family_eta_weights;
    spec.n_extra_params = 0;
    return spec;
}

// Whether a LikelihoodSpec was built by builtin_family_spec() above, as
// opposed to a genuine consumer-package LikelihoodSpec -- the one place
// other engine code (the inner-Laplace skew diagnostic,
// laplace_spec_curvature3.h) that needs to recover the underlying
// family/phi/phi2 from model_data checks this, so the naming convention
// stays defined in a single place.
//
// What the predicate guards is the static_cast to BuiltinFamilyResponse, so it
// has to answer true for BOTH names the constructor writes: a `builtin_zi:`
// spec carries the same response payload, and reading it as a consumer-package
// spec would take the finite-difference branch, or decline, where the closed
// form is available. The single-process / two-process distinction is
// spec.n_processes, not the prefix, so a caller that needs it reads that.
inline bool is_builtin_zi_family_spec(const std::string& spec_name) {
    return spec_name.rfind(builtin_zi_spec_prefix(), 0) == 0;
}

inline bool is_builtin_family_spec(const std::string& spec_name) {
    return spec_name.rfind(builtin_spec_prefix(), 0) == 0 ||
           is_builtin_zi_family_spec(spec_name);
}

} // namespace tulpa

#endif // TULPA_LAPLACE_BUILTIN_FAMILY_SPEC_H
